"""Tests for the optional Loam profile `cairn` block."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ADAPTER = Path(__file__).resolve().parents[1] / ".loam" / "adapters" / "hermes" / "plugin"
if str(ADAPTER) not in sys.path:
    sys.path.insert(0, str(ADAPTER))

from cairn_config import cairn_config  # noqa: E402
import loam_config  # noqa: E402


@pytest.fixture(autouse=True)
def cairn_on(monkeypatch):
    # conftest turns Cairn off for every test; these tests exercise the config itself (no network here).
    monkeypatch.delenv("LOAM_CAIRN", raising=False)


def test_absent_or_disabled_block_is_none(monkeypatch):
    monkeypatch.setenv("LOAM_CAIRN", "off")
    assert cairn_config(None) is None
    assert cairn_config({"enabled": False, "entities": {"tool://lammps": {}}}) is None


def test_defaults():
    cfg = cairn_config({"enabled": True, "entities": {"tool://lammps": {"patterns": ["LMP"], "task_tags": ["md_simulation"]}}})
    assert cfg["enabled"] is True
    assert cfg["base_url"] == "https://api.cairnscore.ai"
    assert cfg["key_file"] == "~/.cairn/keys/agent-loam.key"
    assert cfg["identity"] == ""
    assert cfg["identity_prefix"] == "agent://anl-loam/loam-"
    assert cfg["auto_mint"] is True
    assert cfg["timeout_seconds"] == 5.0
    assert cfg["rate_roles"] == ["code-generator", "hypothesis-validator"]
    assert cfg["read_side"] is cfg["rate_agent_reports"] is cfg["rate_hook_fallback"] is cfg["auto_flush"] is True
    assert cfg["executor"] == "mcp://hpc#run_code"
    assert cfg["entities"]["tool://lammps"] == {"patterns": ["lmp"], "task_tags": ["md_simulation"]}
    # the executor entity is always present, added when the profile omits it
    assert cfg["entities"]["mcp://hpc#run_code"] == {"patterns": [], "task_tags": ["code_exec"]}


def test_timeout_capped_at_ten():
    assert cairn_config({"enabled": True, "timeout_seconds": 60, "entities": {"tool://x": {}}})["timeout_seconds"] == 10.0


def test_invalid_blocks_are_none():
    base = {"enabled": True}
    assert cairn_config({**base, "entities": {}}) is None
    assert cairn_config({**base, "entities": {"lammps": {}}}) is None
    assert cairn_config({**base, "entities": {"tool://x": {"task_tags": ["a"] * 11}}}) is None
    assert cairn_config({**base, "rate_roles": ["coordinator"], "entities": {"tool://x": {}}}) is None
    assert cairn_config({**base, "entities": {"tool://x": {"patterns": "lmp"}}}) is None
    assert cairn_config({**base, "read_side": "yes", "entities": {"tool://x": {}}}) is None
    assert cairn_config({**base, "timeout_seconds": 0, "entities": {"tool://x": {}}}) is None
    assert cairn_config({**base, "entities": {"tool://x": {"patterns": [""]}}}) is None


def _write_profile(tmp_path, monkeypatch, cairn):
    path = tmp_path / "loam.yaml"
    path.write_text(json.dumps({
        "enabled": True,
        "active_profile": "p",
        "profiles": {"p": {"workflow_id": "wf", "state_dir": str(tmp_path / "state"), "cairn": cairn}},
    }))
    monkeypatch.setenv("LOAM_CONFIG_PATH", str(path))


def test_profile_loads_without_cairn_when_block_invalid(tmp_path, monkeypatch):
    _write_profile(tmp_path, monkeypatch, {"enabled": True, "entities": {}})
    profile = loam_config.load_profile()
    assert profile is not None
    assert profile.cairn is None
    assert "loam_cairn" not in loam_config.payload_values(profile)


def test_payload_carries_cairn_block(tmp_path, monkeypatch):
    _write_profile(tmp_path, monkeypatch, {"enabled": True, "entities": {"tool://lammps": {"patterns": ["lmp"]}}})
    profile = loam_config.load_profile()
    assert profile is not None
    assert loam_config.payload_values(profile)["loam_cairn"]["identity_prefix"] == "agent://anl-loam/loam-"


def test_task_tags_normalised_to_snake_case():
    cfg = cairn_config({"enabled": True, "entities": {"tool://x": {"task_tags": ["MD-Simulation", " co folding ", "a:b"]}}})
    assert cfg["entities"]["tool://x"]["task_tags"] == ["md_simulation", "co_folding", "a_b"]


def test_task_tag_that_normalises_to_nothing_rejects_block():
    for bad in ("---", "42", "!!"):
        assert cairn_config({"enabled": True, "entities": {"tool://x": {"task_tags": [bad]}}}) is None, bad


def test_base_url_must_be_http_or_https():
    entities = {"tool://x": {}}
    for bad in ("api.cairnscore.ai", "ftp://c.example", "file:///etc/passwd", "HTTPS//c"):
        assert cairn_config({"enabled": True, "base_url": bad, "entities": entities}) is None, bad
    for good in ("http://localhost:8000", "https://c.example"):
        assert cairn_config({"enabled": True, "base_url": good, "entities": entities})["base_url"] == good


def test_explicit_identity_kept_and_prefix_must_be_uri_shaped():
    ents = {"tool://x": {}}
    assert cairn_config({"enabled": True, "identity": "agent://o/me", "entities": ents})["identity"] == "agent://o/me"
    assert cairn_config({"enabled": True, "identity_prefix": "nourl-", "entities": ents}) is None
    assert cairn_config({"enabled": True, "auto_mint": "yes", "entities": ents}) is None
    assert cairn_config({"enabled": True, "auto_mint": False, "entities": ents})["auto_mint"] is False


# --- repo default block (.loam/cairn.json) ---------------------------------------------------

import cairn_config as cairn_config_module  # noqa: E402


def _on(monkeypatch):
    monkeypatch.delenv("LOAM_CAIRN", raising=False)


def test_default_block_loads_when_profile_has_none(monkeypatch):
    _on(monkeypatch)
    cfg = cairn_config(None)
    assert cfg is not None and cfg["enabled"] is True and cfg["auto_mint"] is True
    assert cfg["identity_prefix"] == "agent://anl-loam/loam-" and cfg["identity"] == ""
    assert set(cfg["entities"]) == {
        "mcp://hpc#run_code", "tool://chai-1", "tool://proteinmpnn",
        "tool://openmm", "tool://chroma", "tool://lammps",
    }
    assert cfg["entities"]["tool://proteinmpnn"]["patterns"] == ["proteinmpnn", "mpnn"]


def test_profile_block_wins_over_default(monkeypatch):
    _on(monkeypatch)
    cfg = cairn_config({"enabled": True, "entities": {"tool://only": {}}})
    assert "tool://only" in cfg["entities"] and "tool://lammps" not in cfg["entities"]


def test_profile_disabled_block_wins_over_default(monkeypatch):
    _on(monkeypatch)
    assert cairn_config({"enabled": False}) is None


def test_env_kill_switch(monkeypatch):
    _on(monkeypatch)
    for off in ("0", "off", "FALSE", "No", "Off"):
        monkeypatch.setenv("LOAM_CAIRN", off)
        assert cairn_config(None) is None, off
        assert cairn_config({"enabled": True, "entities": {"tool://x": {}}}) is None, off
    monkeypatch.setenv("LOAM_CAIRN", "1")
    assert cairn_config(None) is not None


def test_invalid_or_unreadable_default_file_is_none(monkeypatch, tmp_path):
    _on(monkeypatch)
    bad = tmp_path / "cairn.json"
    monkeypatch.setattr(cairn_config_module, "DEFAULT_FILE", bad)
    assert cairn_config(None) is None  # missing
    bad.write_text("{not json")
    assert cairn_config(None) is None
    bad.write_text("[1]")
    assert cairn_config(None) is None
    bad.write_text('{"enabled": true, "entities": {}}')
    assert cairn_config(None) is None
