#!/usr/bin/env python3
"""Fail-closed Loam campaign readiness check with bounded auto-start.

This checks real inference (not only an open TCP port) and a returned Aurora
run_code result with placement provenance. It may start two things it depends on
and then wait for them: the local ALCF inference proxy, and -- when the HPC
server is stopped -- one debug allocation. It never restarts a listening server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.parse
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


def _expand_env(value: Any) -> Any:
    """Expand `${VAR}` in config strings exactly as Hermes does.

    Hermes resolves these references itself when it loads its config
    (`hermes_cli/config.py:_expand_env_vars`), which is how a credential stays
    out of a tracked `config.yaml`: the Minerva entry carries
    `api_key: ${ALCF_INFERENCE_TOKEN}`. This gate reads the same YAML directly,
    so without the same expansion it would send the literal string
    `${ALCF_INFERENCE_TOKEN}` as the bearer token and fail with 401 on an
    endpoint that is in fact reachable and correctly configured.

    Unresolved references are kept verbatim, as Hermes keeps them, so a missing
    export still fails -- but against the real endpoint, where the 401 says so.
    """

    if isinstance(value, str):
        return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
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

    # Expanded, because Hermes expands it: see _expand_env.
    hermes = _expand_env(_load_yaml(hermes_home / "config.yaml"))
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
    deleg_base = str(delegation.get("base_url") or base_url).rstrip("/")
    deleg_key = str(delegation.get("api_key") or api_key)
    if deleg_model:
        roles["subagents"] = (deleg_base, deleg_key, deleg_model)

    # Per-role subagent overrides (delegation.role_models, consumed by the
    # patched Hermes delegate_tool._apply_role_override). Each entry is either a
    # model string (reuse the delegation endpoint) or a mapping that may also
    # carry base_url/api_key for a per-role ENDPOINT split -- e.g. the
    # code-generator benchmarked on an ALCF cluster model while the validator
    # and reasoner stay on the default. Smoke each so an unreachable per-role
    # endpoint fails the launch HERE rather than mid-round. Values fall back to
    # the delegation defaults, mirroring _apply_role_override's resolution.
    role_models = delegation.get("role_models")
    if isinstance(role_models, dict):
        for child_role, entry in role_models.items():
            if isinstance(entry, str):
                r_base, r_key, r_model = deleg_base, deleg_key, entry.strip()
            elif isinstance(entry, dict):
                r_base = str(entry.get("base_url") or deleg_base).rstrip("/")
                r_key = str(entry.get("api_key") or deleg_key)
                r_model = str(entry.get("model") or deleg_model)
            else:
                continue
            if r_model:
                roles[f"subagents:{child_role}"] = (r_base, r_key, r_model)

    loam = _load_yaml(loam_config)
    profile_name = str(os.environ.get("LOAM_PROFILE") or loam.get("active_profile") or "")
    profile = (loam.get("profiles") or {}).get(profile_name) or {}
    tournament = profile.get("hypothesis_tournament") or {}
    if tournament.get("enabled") is True:
        shared_url = str(tournament.get("inference_endpoint") or "").rstrip("/") or base_url
        shared_model = str(tournament.get("model") or "") or model
        shared_key_env = str(tournament.get("api_key_env") or "").strip()
        for role in ("generation", "ranking"):
            override = tournament.get(role) or {}
            # Same resolution as hypothesis_tournament._llm_config, so the smoke
            # runs under the credential the tournament will really send.
            key_env = str(override.get("api_key_env") or "").strip() or shared_key_env
            roles[role] = (
                str(override.get("inference_endpoint") or shared_url).rstrip("/"),
                os.environ.get(key_env, "").strip() if key_env else "local-no-auth",
                str(override.get("model") or shared_model),
            )
    return roles


def _requires_role_endpoint_patch(hermes_home: Path) -> bool:
    """True iff config.yaml uses a delegation.role_models ENDPOINT override -- a
    mapping entry carrying base_url/provider/api_key/api_mode. A plain
    model-string entry works on unpatched Hermes, so it does not require it."""
    hermes = _load_yaml(hermes_home / "config.yaml")
    role_models = (hermes.get("delegation") or {}).get("role_models")
    if not isinstance(role_models, dict):
        return False
    endpoint_keys = {"provider", "base_url", "api_key", "api_mode"}
    for entry in role_models.values():
        if isinstance(entry, dict) and any(
            k in endpoint_keys and str(entry.get(k) or "").strip() for k in entry
        ):
            return True
    return False


def _hermes_role_endpoint_support() -> tuple[bool | None, str]:
    """Whether the installed Hermes carries the per-role-endpoint patch
    (delegate_tool._apply_role_override). Returns (supported, detail); supported
    is None when it cannot be determined -- a launch is never blocked on an
    inconclusive probe, only on a positive 'unpatched' finding."""
    import shutil

    launcher = shutil.which("hermes")
    if not launcher:
        return None, "hermes not found on PATH"
    try:
        text = Path(launcher).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"cannot read hermes launcher: {exc}"
    venv_py = ""
    for tok in text.replace('"', " ").replace("'", " ").split():
        if tok.endswith("/bin/hermes") and "/venv/" in tok:
            venv_py = tok[: -len("/bin/hermes")] + "/bin/python"
            break
    if not venv_py or not Path(venv_py).exists():
        return None, "could not resolve hermes venv python from launcher"
    probe = (
        "import tools.delegate_tool as d, sys; "
        "sys.exit(0 if hasattr(d, '_apply_role_override') else 3)"
    )
    try:
        rc = subprocess.run([venv_py, "-c", probe], capture_output=True, timeout=60).returncode
    except Exception as exc:  # noqa: BLE001
        return None, f"probe failed to run: {exc}"
    if rc == 0:
        return True, f"patched ({venv_py})"
    if rc == 3:
        return False, f"unpatched ({venv_py})"
    return None, f"probe exited {rc}"


def check_role_endpoint_capability(hermes_home: Path, repo: Path) -> dict[str, Any]:
    """Advise/enforce the Hermes per-role-endpoint patch.

    A delegation.role_models mapping entry with a per-role base_url is what
    isolates e.g. the code-generator onto its own cluster endpoint while the
    validator/reasoner stay on the default. Unpatched Hermes IGNORES such a
    mapping and silently leaves that role on the default model+endpoint -- a
    silent wrong-attribution failure for a code-gen-model benchmark. So when the
    config needs the patch and Hermes positively lacks it, fail the launch with
    the fix. Carried patch: patches/hermes-role-models-endpoints.patch.
    """
    requires = _requires_role_endpoint_patch(hermes_home)
    supported, detail = _hermes_role_endpoint_support()
    info = {"config_requires_patch": requires, "hermes_supported": supported, "detail": detail}
    if requires and supported is False:
        patch = repo / "patches" / "hermes-role-models-endpoints.patch"
        raise RuntimeError(
            "config.yaml uses a delegation.role_models ENDPOINT override, but the "
            f"installed Hermes lacks the per-role-endpoint patch ({detail}). Unpatched "
            "Hermes silently ignores the mapping and leaves that role on the DEFAULT "
            "endpoint, corrupting any per-role model comparison. Apply the carried patch "
            f"(see patches/README.md):\n    {patch}"
        )
    return info


def check_inference(hermes_home: Path, loam_config: Path, timeout: float) -> dict[str, Any]:
    """Smoke every configured role, skipping duplicate endpoint/key/model triples."""

    results: dict[str, Any] = {}
    seen: dict[tuple[str, str, str], str] = {}
    for role, (base_url, api_key, model) in _inference_roles(hermes_home, loam_config).items():
        if (previous := seen.get((base_url, api_key, model))) is not None:
            results[role] = {"base_url": base_url, "model": model, "response": f"same as {previous}"}
            continue
        results[role] = _smoke_model(base_url, api_key, model, timeout)
        seen[(base_url, api_key, model)] = role
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


def _host_port(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url)
    if not parsed.hostname:
        raise RuntimeError(f"HPC MCP URL has no host: {url!r}")
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def ensure_proxy_started(repo: Path, port: int, wait_seconds: int) -> dict[str, Any]:
    """Start the local ALCF inference proxy if nothing is listening on its port.

    The ranking role runs through this proxy, so `check_inference` below is what
    fails when it is down -- with a connection error that says nothing about the
    cause. Starting it here rather than in the launcher keeps one readiness path:
    the gate that requires a dependency is the thing that satisfies it, so the
    two cannot drift the way the launcher and this gate did while both existed.

    This is the smaller of the two auto-starts already in this file -- the other
    submits a PBS allocation on a supercomputer. Detached, so it outlives the
    campaign that started it, and skipped entirely by ALCF_PROXY=0 when the
    ranking model is served some other way.
    """

    if os.environ.get("ALCF_PROXY", "1") == "0":
        return {"action": "skipped", "reason": "ALCF_PROXY=0"}
    if _port_open("127.0.0.1", port):
        return {"action": "already-listening", "port": port}

    python = Path(
        os.environ.get("ALCF_PROXY_PYTHON") or Path.home() / ".loam-venv-312" / "bin" / "python"
    )
    if not python.is_file():
        raise RuntimeError(f"ALCF proxy interpreter is unavailable: {python}")
    # The cluster was hardcoded to "minerva" here while the rest of the launch
    # had already moved to Tara, so turning ALCF_PROXY back on would have
    # started a proxy in front of a cluster that no longer serves the
    # configured model. launch.sh exports ALCF_CLUSTER; the default stays
    # minerva so nothing that relied on it changes behavior.
    cluster = os.environ.get("ALCF_CLUSTER") or "minerva"
    log = Path(os.environ.get("ALCF_PROXY_LOG")
               or Path.home() / ".loam-hpc" / f"alcf_proxy_{cluster}.log")
    log.parent.mkdir(parents=True, exist_ok=True)

    with log.open("ab") as stream:
        subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
            [str(python), str(repo / "scripts" / "alcf_inference_proxy.py")],
            env={**os.environ, "ALCF_CLUSTER": cluster, "ALCF_PROXY_PORT": str(port)},
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _port_open("127.0.0.1", port):
            return {"action": "started", "port": port, "log": str(log)}
        time.sleep(0.5)
    tail = ""
    try:
        tail = "".join(log.read_text(errors="replace").splitlines(keepends=True)[-5:])
    except OSError:
        pass
    raise RuntimeError(
        f"ALCF proxy did not listen on {port} within {wait_seconds}s. Last log lines:\n{tail}"
        f"\nIf it is an auth error: {python.parent}/alcf-tokens login inference"
    )


def ensure_hpc_started(repo: Path, env_file: Path, wait_seconds: int, url: str) -> dict[str, Any]:
    host, port = _host_port(url)
    if _port_open(host, port):
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
            # Requisition the allocation THIS campaign's env file asks for, rather
            # than a throwaway debug block. Passing the values explicitly is what
            # makes them stick: hpc_control.current_settings lets a stale
            # allocation.json from an earlier run override the env file, so
            # restart(None, None, None) would silently reuse the last shape.
            config = json.loads(sourced["HPC_ENDPOINT_CONFIG_JSON"])
            queue = config.get("queue", "debug")
            walltime = config.get("walltime", "01:00:00")
            nodes = int(sourced.get("HPC_NNODES") or config.get("nodes_per_block") or 1)
            started = controller.restart(queue, walltime, nodes)
            action = {"action": "started-allocation", "queue": queue,
                      "walltime": walltime, "nodes": nodes, "pid": started["pid"]}
        else:
            action = {"action": "waiting", "state": status["state"]}
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _port_open(host, port):
            return action
        time.sleep(5)
    raise RuntimeError(f"HPC MCP did not listen within {wait_seconds}s after {action}")


async def check_hpc(url: str, timeout: float) -> dict[str, Any]:
    from fastmcp import Client

    code = (
        "import json, os, socket\n"
        "print(json.dumps({'hostname': socket.gethostname(), "
        "'pbs_jobid': os.environ.get('PBS_JOBID'), "
        "'slurm_jobid': os.environ.get('SLURM_JOB_ID'), "
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
    missing = [key for key in ("hostname", "device_pin") if not placement.get(key)]
    if not (placement.get("pbs_jobid") or placement.get("slurm_jobid")):
        missing.append("scheduler_jobid")
    if missing:
        raise RuntimeError(f"HPC result lacks placement fields {missing}: {placement}")
    return {"returncode": data["returncode"], "stderr": data.get("stderr", ""), **placement}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--hermes-home", type=Path, required=True)
    parser.add_argument(
        "--loam-config",
        type=Path,
        # Honor LOAM_CONFIG_PATH so a campaign-local profile is used: the
        # launcher and the driver probe invoke this without --loam-config,
        # and the global ~/.hermes/loam.yaml may not exist.
        default=Path(os.environ.get("LOAM_CONFIG_PATH") or (Path.home() / ".hermes" / "loam.yaml")).expanduser(),
    )
    parser.add_argument("--hpc-env", type=Path)
    parser.add_argument("--hpc-url", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--no-auto-start", action="store_true")
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=int(os.environ.get("ALCF_PROXY_PORT") or 51911),
        help="local ALCF inference proxy port; ALCF_PROXY=0 disables the auto-start",
    )
    parser.add_argument("--proxy-wait-seconds", type=int, default=15)
    args = parser.parse_args()
    args.hpc_env = (args.hpc_env or os.environ.get("HPC_ENV_FILE")
                    or args.repo / "skills" / "hpc-deployment" / "set_env.sh")
    args.hpc_env = Path(args.hpc_env)

    report: dict[str, Any] = {"ready": False}
    try:
        # Before the inference check, which is what depends on it.
        if not args.no_auto_start:
            report["proxy_start"] = ensure_proxy_started(
                args.repo, args.proxy_port, args.proxy_wait_seconds)
        report["role_endpoints"] = check_role_endpoint_capability(args.hermes_home, args.repo)
        report["inference"] = check_inference(args.hermes_home, args.loam_config, args.timeout)
        if not args.no_auto_start:
            report["hpc_start"] = ensure_hpc_started(
                args.repo, args.hpc_env, args.wait_seconds, args.hpc_url)
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
