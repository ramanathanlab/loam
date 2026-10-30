"""run-campaign exports the FALDA key and pool, and the ALCF proxy on Aurora compute nodes (best effort)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POOL = "efb99ee4-c412-11f1-8af6-0affe8db6a05"
KEY = "falda_" + "r" * 40
_PROXY_VARS = ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "no_proxy", "NO_PROXY")


def _launch(tmp_path: Path, hostname: str, key: bool = True, extra: dict | None = None) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    dump = tmp_path / "env.json"
    (bin_dir / "hermes").write_text(
        f"#!/usr/bin/env python3\nimport json, os\njson.dump(dict(os.environ), open({str(dump)!r}, 'w'))\n"
    )
    (bin_dir / "hostname").write_text(f"#!/bin/sh\necho {hostname}\n")
    for name in ("hermes", "hostname"):
        (bin_dir / name).chmod(0o755)
    home = tmp_path / "home"
    keys = home / ".falda" / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    if key:
        (keys / f"loam-{POOL}.json").write_text(json.dumps({"api_key": KEY, "pool": POOL}))
    env = {k: v for k, v in os.environ.items() if k not in _PROXY_VARS and not k.startswith(("FALDA_", "LOAM_"))}
    # A throwaway checkout and campaign: run-campaign writes its shim, home and state under both.
    repo = tmp_path / "repo"
    if not repo.exists():
        shutil.copytree(ROOT / ".loam", repo / ".loam", ignore=shutil.ignore_patterns("state", "__pycache__"))
        (repo / ".hermes" / ".loam-home").mkdir(parents=True)
        shutil.copy(ROOT / ".hermes" / ".loam-home" / "config.yaml", repo / ".hermes" / ".loam-home" / "config.yaml")
    campaign = tmp_path / "campaign"
    campaign.mkdir(exist_ok=True)
    shutil.copy(ROOT / "run-campaign", campaign / "run-campaign")
    env.update({"PATH": f"{bin_dir}:{env['PATH']}", "HOME": str(home), "ALCF_PROXY": "0",
                "LOAM_REPO": str(repo), "LOAM_SKIP_PREFLIGHT": "1", **(extra or {})})
    completed = subprocess.run([str(campaign / "run-campaign")], env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    return json.loads(dump.read_text())


def test_compute_node_gets_key_pool_and_proxy_with_local_bypass(tmp_path):
    env = _launch(tmp_path, "x4102c5s3b0n0")
    assert env["FALDA_TOKEN"] == KEY
    assert env["FALDA_POOL"] == POOL
    for name in ("https_proxy", "HTTPS_PROXY"):
        assert env[name] == "http://proxy.alcf.anl.gov:3128"
    # FALDA and Cairn are HTTPS: plain-HTTP tools are left alone.
    assert "http_proxy" not in env and "HTTP_PROXY" not in env
    for name in ("no_proxy", "NO_PROXY"):
        assert {"localhost", "127.0.0.1", "::1", "x4102c5s3b0n0", ".alcf.anl.gov"} <= set(env[name].split(","))
    assert env.get("LOAM_FALDA_MCP") != "0"


def test_existing_no_proxy_is_kept(tmp_path):
    env = _launch(tmp_path, "x4102c5s3b0n0", extra={"no_proxy": ".alcf.anl.gov", "NO_PROXY": ".alcf.anl.gov"})
    assert ".alcf.anl.gov" in env["no_proxy"].split(",") and "127.0.0.1" in env["no_proxy"].split(",")


def test_login_node_gets_no_proxy(tmp_path):
    env = _launch(tmp_path, "aurora-uan-0010")
    assert env["FALDA_TOKEN"] == KEY
    assert not any(name in env for name in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"))


def test_existing_proxy_is_left_alone_but_local_services_bypass_it(tmp_path):
    env = _launch(tmp_path, "x4102c5s3b0n0", extra={"https_proxy": "http://mine:1", "no_proxy": "corp.example"})
    assert env["https_proxy"] == "http://mine:1"
    assert "HTTPS_PROXY" not in env and "HTTP_PROXY" not in env
    assert {"localhost", "127.0.0.1", "::1", "corp.example"} <= set(env["no_proxy"].split(","))


def test_no_key_still_launches_and_disables_reasoner_tools(tmp_path):
    # Budget 0: no launch-time mint (keeps the test off the network).
    env = _launch(tmp_path, "aurora-uan-0010", key=False, extra={"LOAM_FALDA_LAUNCH_BUDGET": "0"})
    assert env.get("FALDA_TOKEN", "") == ""
    assert env["FALDA_POOL"] == POOL
    assert env["LOAM_FALDA_MCP"] == "0"


def test_falda_off_exports_nothing(tmp_path):
    env = _launch(tmp_path, "aurora-uan-0010", extra={"LOAM_FALDA": "0"})
    assert env.get("FALDA_TOKEN", "") == "" and env.get("FALDA_POOL", "") == ""


def test_explicit_token_wins(tmp_path):
    env = _launch(tmp_path, "aurora-uan-0010", extra={"FALDA_TOKEN": "falda_explicit"})
    assert env["FALDA_TOKEN"] == "falda_explicit"


@pytest.mark.parametrize("path", [ROOT / "examples" / "hermes-config.yaml.example", ROOT / ".hermes" / ".loam-home" / "config.yaml"])
def test_hermes_configs_declare_the_falda_mcp_server(path):
    yaml = pytest.importorskip("yaml")
    server = yaml.safe_load(path.read_text())["mcp_servers"]["falda"]
    assert server["url"] == "https://falda.cairnscore.ai/mcp"
    assert server["headers"] == {"Authorization": "Bearer ${FALDA_TOKEN}", "X-Falda-Pool": "${FALDA_POOL}"}
    # Calls are bounded tightly; the connect runs once at launch, where 5 s lost the tools to an ordinary
    # slow start on 2026-10-30.
    assert server["timeout"] <= 5 and 10 <= server["connect_timeout"] <= 20


@pytest.mark.parametrize("path", [ROOT / "examples" / "hermes-config.yaml.example", ROOT / ".hermes" / ".loam-home" / "config.yaml"])
def test_hermes_configs_keep_mcp_tools_eager(path):
    # Hermes defers MCP tools behind tool_search once there are enough of them (FALDA adds eight), and
    # Loam's role policies block tool_search for the code-generator, which then cannot reach run_code
    # (Aurora probe, 2026-10-30).
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(path.read_text())["tools"]["tool_search"]["enabled"] is False
