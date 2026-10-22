#!/usr/bin/env python3
"""Fail-closed Loam campaign readiness check with bounded HPC auto-start.

This checks real inference (not only an open TCP port) and a returned Aurora
run_code result with placement provenance. If the HPC server is stopped, it may
start one debug allocation and wait for it. It never restarts a listening server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise RuntimeError(f"configuration not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"configuration is not an object: {path}")
    return value


def _inference_roles(hermes_home: Path, loam_config: Path) -> dict[str, tuple[str, str, str]]:
    """Map each configured agent role to its (base_url, api_key, model).

    A campaign can run every role on a different model and endpoint, so the
    smoke check covers all of them rather than one representative:

      reasoner    the campaign parent (hermes `model`)
      subagents   children, when pinned via hermes `delegation`
      generation  Jnana's hypothesis generators
      ranking     the tournament judge

    Roles that are not configured separately inherit and are simply absent.
    """

    hermes = _load_yaml(hermes_home / "config.yaml")
    model_cfg = hermes.get("model") or {}
    base_url = str(model_cfg.get("base_url") or "").rstrip("/")
    api_key = str(model_cfg.get("api_key") or "")
    model = str(model_cfg.get("default") or "")
    if not all((base_url, api_key, model)):
        raise RuntimeError("incomplete reasoner base_url/api_key/model configuration")
    roles = {"reasoner": (base_url, api_key, model)}

    # Subagents only differ when delegation pins them to another endpoint/model.
    delegation = hermes.get("delegation") or {}
    deleg_model = str(delegation.get("model") or "")
    if deleg_model:
        roles["subagents"] = (
            str(delegation.get("base_url") or base_url).rstrip("/"),
            str(delegation.get("api_key") or api_key),
            deleg_model,
        )

    loam = _load_yaml(loam_config)
    profile_name = str(os.environ.get("LOAM_PROFILE") or loam.get("active_profile") or "")
    profile = (loam.get("profiles") or {}).get(profile_name) or {}
    tournament = profile.get("hypothesis_tournament") or {}
    if tournament.get("enabled") is True:
        shared_url = str(tournament.get("inference_endpoint") or "").rstrip("/") or base_url
        shared_model = str(tournament.get("model") or "") or model
        for role in ("generation", "ranking"):
            override = tournament.get(role) or {}
            roles[role] = (
                str(override.get("inference_endpoint") or shared_url).rstrip("/"),
                api_key,
                str(override.get("model") or shared_model),
            )
    return roles


def check_inference(hermes_home: Path, loam_config: Path, timeout: float) -> dict[str, Any]:
    """Smoke every configured role, skipping duplicate endpoint/model pairs."""

    results: dict[str, Any] = {}
    seen: dict[tuple[str, str], str] = {}
    for role, (base_url, api_key, model) in _inference_roles(hermes_home, loam_config).items():
        if (previous := seen.get((base_url, model))) is not None:
            results[role] = {"base_url": base_url, "model": model, "response": f"same as {previous}"}
            continue
        results[role] = _smoke_model(base_url, api_key, model, timeout)
        seen[(base_url, model)] = role
    return results


def _smoke_model(base_url: str, api_key: str, model: str, timeout: float) -> dict[str, Any]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply exactly: LOAM_PREFLIGHT_OK"}],
        # Reasoning models (e.g. inkling-bf16) spend tokens on a hidden reasoning
        # field before emitting any content; too small a budget truncates the
        # content to null and the smoke check crashes. Keep this generous.
        "max_tokens": 512,
    }).encode()
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    message = payload["choices"][0]["message"]
    # content may be null on a reasoning model; fall back to its reasoning text.
    text = (message.get("content") or message.get("reasoning") or "").strip()
    if "LOAM_PREFLIGHT_OK" not in text:
        raise RuntimeError(f"{model} smoke returned unexpected content: {text[:160]!r}")
    return {"base_url": base_url, "model": model, "response": text}


def _sourced_env(env_file: Path) -> dict[str, str]:
    result = subprocess.run(
        ["bash", "-c", 'set -a; source "$1" >/dev/null; env -0', "_", str(env_file)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return dict(item.split("=", 1) for item in result.stdout.decode().split("\0") if "=" in item)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def ensure_hpc_started(repo: Path, env_file: Path, wait_seconds: int) -> dict[str, Any]:
    if _port_open("127.0.0.1", 8000):
        return {"action": "already-listening"}

    scripts = repo / "skills" / "hpc-deployment" / "scripts"
    sys.path.insert(0, str(scripts))
    import hpc_control

    sourced = _sourced_env(env_file)
    old_env = os.environ.copy()
    try:
        os.environ.update(sourced)
        os.environ["HPC_ENV_FILE"] = str(env_file)
        controller = hpc_control.Controller()
        status = controller.status()
        if status["state"] == "stopped":
            started = controller.restart("debug", "01:00:00", 1)
            action = {"action": "started-debug-allocation", "pid": started["pid"]}
        else:
            action = {"action": "waiting", "state": status["state"]}
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _port_open("127.0.0.1", 8000):
            return action
        time.sleep(5)
    raise RuntimeError(f"HPC MCP did not listen within {wait_seconds}s after {action}")


async def check_hpc(url: str, timeout: float) -> dict[str, Any]:
    from fastmcp import Client

    code = (
        "import json, os, socket\n"
        "print(json.dumps({'hostname': socket.gethostname(), "
        "'pbs_jobid': os.environ.get('PBS_JOBID'), "
        "'device_pin': os.environ.get('ZE_AFFINITY_MASK') or os.environ.get('CUDA_VISIBLE_DEVICES')}))"
    )
    async with Client(url, timeout=timeout) as client:
        result = await client.call_tool("run_code", {"payload": code, "timeout": timeout})
    if result.is_error:
        raise RuntimeError(f"HPC run_code failed: {result}")
    data = result.data if isinstance(result.data, dict) else result.structured_content
    if not isinstance(data, dict) or str(data.get("returncode")) != "0":
        raise RuntimeError(f"HPC run_code returned invalid result: {data}")
    placement = json.loads(str(data.get("stdout") or ""))
    missing = [key for key in ("hostname", "pbs_jobid", "device_pin") if not placement.get(key)]
    if missing:
        raise RuntimeError(f"HPC result lacks placement fields {missing}: {placement}")
    return {"returncode": data["returncode"], "stderr": data.get("stderr", ""), **placement}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--hermes-home", type=Path, required=True)
    parser.add_argument("--loam-config", type=Path, default=Path.home() / ".hermes" / "loam.yaml")
    parser.add_argument("--hpc-env", type=Path)
    parser.add_argument("--hpc-url", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--no-auto-start", action="store_true")
    args = parser.parse_args()
    args.hpc_env = args.hpc_env or args.repo / "skills" / "hpc-deployment" / "set_env_akv.sh"

    report: dict[str, Any] = {"ready": False}
    try:
        report["inference"] = check_inference(args.hermes_home, args.loam_config, args.timeout)
        if not args.no_auto_start:
            report["hpc_start"] = ensure_hpc_started(args.repo, args.hpc_env, args.wait_seconds)
        report["hpc"] = asyncio.run(check_hpc(args.hpc_url, args.timeout))
        report["ready"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
