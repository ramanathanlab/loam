#!/usr/bin/env python3
"""Submit a dedicated Nemotron allocation and select the Jupiter inference mode."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import yaml
from aiohttp import web

from jupiter_nemotron_api import (
    CHECKPOINT, MODEL, atomic_json, completion_check, pinned_tls, private_address,
    proxy_app, read_json, readiness, streaming_tool_check, structured_completion_check,
)

REPO = Path(__file__).resolve().parents[1]
MODE_ENV = "LOAM_JUPITER_INFERENCE_MODE"
TOKEN_ENV = "LOAM_NEMOTRON_API_KEY"
JUPITER_GCC = Path("/e/software/default/stages/2026/software/GCCcore/14.3.0")
JUPITER_BINUTILS = Path("/e/software/default/stages/2026/software/binutils/2.44")


def native_toolchain(cc: Path | None = None, cxx: Path | None = None) -> dict:
    def installed(variable, preferred, executable):
        configured = os.environ.get(variable)
        return Path(configured) if configured else (preferred if preferred.is_file()
                                                   else Path(shutil.which(executable) or preferred))
    tools = {
        "cc": cc or installed("LOAM_CC", JUPITER_GCC / "bin/gcc", "gcc"),
        "cxx": cxx or installed("LOAM_CXX", JUPITER_GCC / "bin/g++", "g++"),
        "ld": installed("LOAM_LD", JUPITER_BINUTILS / "bin/ld", "ld"),
    }
    for name, path in tools.items():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"The native vLLM toolchain is missing {name}: {path}")
    return {name: str(path.resolve()) for name, path in tools.items()}


def native_runtime_env(request: dict, inherited: dict[str, str]) -> dict[str, str]:
    env = dict(inherited)
    tools = request.get("compiler") or native_toolchain()
    for name in ("cc", "cxx", "ld"):
        path = Path(tools[name])
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"The staged native vLLM toolchain is missing {name}: {path}")
    directories = [str(Path(request["command"][0]).parent)]
    directories.extend(str(Path(tools[name]).parent) for name in ("cc", "cxx", "ld"))
    directories.extend(env.get("PATH", os.defpath).split(os.pathsep))
    env["PATH"] = os.pathsep.join(dict.fromkeys(directories))
    env.update(CC=tools["cc"], CXX=tools["cxx"])
    library = Path(tools["cc"]).parent.parent / "lib64"
    if library.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [env.get("LD_LIBRARY_PATH"), str(library)]))
    return env


def check_compiler(env: dict[str, str]) -> None:
    """Compile and run a tiny program before loading the model weights."""
    with tempfile.TemporaryDirectory(prefix="loam-nemotron-compiler-") as directory:
        source = Path(directory) / "probe.c"
        binary = Path(directory) / "probe"
        source.write_text("int main(void) { return 0; }\n")
        result = subprocess.run([env["CC"], str(source), "-o", str(binary)], env=env,
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f"Native vLLM compiler check failed: {result.stderr[-2000:]}")
        subprocess.run([str(binary)], env=env, check=True, capture_output=True, timeout=10)


def scheduler(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    # Some Slurm queries return zero with an RPC timeout on stderr.
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip() or result.returncode}")
    return result.stdout.strip()


def active_job(attempt: Path) -> tuple[str, str] | None:
    output = scheduler(["squeue", "--local", "--user", str(os.getuid()), "--noheader",
                        "--name", attempt.name, "--format", "%i|%T"])
    jobs = [line.split("|", 1) for line in output.splitlines()]
    if not jobs:
        return None
    if len(jobs) != 1 or len(jobs[0]) != 2 or not jobs[0][0].isdigit():
        raise RuntimeError(f"Cannot uniquely identify the Slurm job named {attempt.name}")
    job_id, state = jobs[0]
    receipt = attempt / "submission.json"
    if receipt.exists() and read_json(receipt)["job_id"] != job_id:
        raise RuntimeError("Scheduler job differs from the campaign submission receipt")
    if not receipt.exists():
        atomic_json(receipt, {"job_id": job_id, "reconciled": True})
    return job_id, state


def state_dir(campaign: Path) -> Path:
    return campaign / "state" / "nemotron"


def current_attempt(state: Path, pointer: str = "current.json") -> Path:
    name = read_json(state / pointer)["attempt"]
    if not re.fullmatch(r"loam-nemotron-[0-9a-f]{16}", name):
        raise ValueError("Invalid campaign inference attempt name")
    attempt = state / name
    if read_json(attempt / "request.json")["campaign"] != str(state.parents[1]):
        raise RuntimeError("Inference attempt does not belong to this campaign")
    return attempt


def model_snapshot(path: Path, revision: str) -> Path:
    path = path.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or path.name != revision:
        raise ValueError("Use a local model snapshot directory named with its full commit revision")
    config = read_json(path / "config.json")
    if config.get("model_type") != "nemotron_h_omni":
        raise ValueError("The local snapshot is not the expected Nemotron Super VL architecture")
    index = read_json(path / "model.safetensors.index.json")
    weights = set(index["weight_map"].values())
    if not weights or any(Path(name).name != name or not (path / name).is_file()
                          or (path / name).stat().st_size == 0 for name in weights):
        raise ValueError("The local Nemotron snapshot is missing model shards")
    return path


def submit(args) -> dict:
    campaign = args.campaign
    state = state_dir(campaign)
    model_dir = model_snapshot(args.model_dir, args.model_revision)
    login = private_address(args.login_address)
    with socket.socket() as probe:
        probe.bind((login, 0))
    compiler = None
    if args.image:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_digest or ""):
            raise ValueError("--image requires the pinned OCI --image-digest")
        image = args.image.resolve(strict=True)
        apptainer = shutil.which("apptainer")
        if not image.is_file() or not apptainer:
            raise ValueError("A staged ARM64 SIF and apptainer are required")
        mount = model_dir.parent.parent if model_dir.parent.name == "snapshots" else model_dir
        command = [apptainer, "exec", "--nv", "--cleanenv", "--bind", f"{mount}:{mount}:ro",
                   str(image), "vllm"]
    else:
        if args.image_digest:
            raise ValueError("--image-digest is only used with --image")
        executable = args.vllm.absolute()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("--vllm must point to an installed vLLM executable")
        command = [str(executable)]
        compiler = native_toolchain(args.cc, args.cxx)
        check_compiler(native_runtime_env({"command": command, "compiler": compiler},
                                          {"PATH": os.defpath}))
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    state.chmod(0o700)
    with (state / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        standby = getattr(args, "standby", False)
        pointer = "standby.json" if standby else "current.json"
        if standby and not (state / "current.json").exists():
            raise RuntimeError("A standby requires an existing campaign inference allocation")
        if (state / pointer).exists():
            previous = current_attempt(state, pointer)
            job = active_job(previous)
            if job:
                return {"job_id": job[0], "state": job[1], "attempt": previous.name, "reused": True}
            if not (previous / "submission.json").exists():
                raise RuntimeError(f"Submission for {previous.name} is ambiguous. Reconcile it with Slurm before retrying.")
            if not args.replace:
                raise RuntimeError("The previous inference job ended. Use --replace to start its replacement.")
        attempt = state / ("loam-nemotron-" + secrets.token_hex(8))
        attempt.mkdir(mode=0o700)
        token = attempt / "token"
        token.touch(mode=0o600)
        token.write_text(secrets.token_urlsafe(32) + "\n")
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256",
            "-noenc", "-days", "7", "-subj", "/CN=" + attempt.name,
            "-keyout", str(attempt / "tls.key"), "-out", str(attempt / "tls.pem"),
        ], check=True, capture_output=True, timeout=30)
        (attempt / "tls.key").chmod(0o600)
        atomic_json(attempt / "request.json", {
            "campaign": str(campaign), "model_dir": str(model_dir), "checkpoint": CHECKPOINT,
            "checkpoint_revision": args.model_revision, "image_digest": args.image_digest,
            "command": command, "login_address": login, "max_model_len": args.context,
            "max_num_seqs": args.sequences, "warmup_seconds": args.warmup_seconds,
            "compiler": compiler, "account": args.account, "reservation": args.reservation,
            "walltime": args.walltime,
        })
        # Persist the name before sbatch. A lost acknowledgement must never
        # trigger another submission without looking up this unique name.
        atomic_json(state / pointer, {"attempt": attempt.name})
        argv = [
            "sbatch", "--parsable", "--export=NIL", "--account", args.account, "--time", args.walltime,
            "--job-name", attempt.name, "--chdir", str(attempt),
            "--output", str(attempt / "slurm-%j.log"),
        ]
        if args.reservation:
            argv.extend(["--reservation", args.reservation])
        argv.extend([str(REPO / "ports/jupiter/nemotron.sbatch"), sys.executable,
                     str(Path(__file__).resolve()), str(attempt)])
        try:
            output = scheduler(argv)
            job_id = output.split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError("Unexpected sbatch response")
            atomic_json(attempt / "submission.json", {"job_id": job_id, "reconciled": False})
            return {"job_id": job_id, "attempt": attempt.name}
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            # Slurm may have accepted the job before the submitter timed out.
            try:
                job = active_job(attempt)
            except (RuntimeError, subprocess.TimeoutExpired):
                job = None
            if job:
                return {"job_id": job[0], "state": job[1], "attempt": attempt.name, "reconciled": True}
            raise RuntimeError(f"Unconfirmed submission for {attempt.name}. Run status before retrying. {exc}") from None


async def activate_standby(campaign: Path) -> dict:
    """Switch the stable relay only after the owned replacement serves a completion.

    Existing requests retain their old upstream connection. Leave that allocation
    running to drain; campaign cleanup releases all owned inference attempts.
    """
    state = state_dir(campaign)
    with (state / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = current_attempt(state)
        replacement = current_attempt(state, "standby.json")
        job = await asyncio.to_thread(active_job, replacement)
        if not job or job[1] != "RUNNING":
            raise RuntimeError("The standby inference allocation is not running")
        old_request = read_json(previous / "request.json")
        new_request = read_json(replacement / "request.json")
        for key in ("checkpoint", "checkpoint_revision"):
            if old_request.get(key) != new_request.get(key):
                raise RuntimeError(f"Standby differs from the active backend: {key}")
        # A larger window can serve every existing client request. Shrinking
        # it would invalidate the context length already cached by live agents.
        old_context = old_request.get("max_model_len")
        new_context = new_request.get("max_model_len")
        if (not isinstance(old_context, int) or not isinstance(new_context, int)
                or old_context <= 0 or new_context < old_context):
            raise RuntimeError("Standby max_model_len must preserve or increase the active context")
        readiness(replacement, job[0])
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=120)) as session:
            record = readiness(replacement, job[0])
            await completion_check(session, record["base_url"],
                                   (replacement / "token").read_text().strip(),
                                   tls=pinned_tls(replacement / "tls.pem"))
        result = {"previous": previous.name, "attempt": replacement.name,
                  "job_id": job[0], "activated_at": time.time()}
        # current.json is replaced atomically. The already-running relay reads
        # this pointer on each new request, without restarting Hermes or science.
        atomic_json(state / "current.json", {"attempt": replacement.name})
        atomic_json(state / "handoff.json", result)
        (state / "standby.json").unlink()
        return result


def owned_attempts(state: Path) -> list[Path]:
    """Only attempt directories whose request explicitly names this campaign."""
    attempts = []
    for path in sorted(state.glob("loam-nemotron-*")):
        if not re.fullmatch(r"loam-nemotron-[0-9a-f]{16}", path.name):
            continue
        if (path / "request.json").is_file() and read_json(path / "request.json").get("campaign") == str(state.parents[1]):
            attempts.append(path)
    return attempts


def allocation_expiry(job_id: str) -> float:
    result = os.environ.get("SLURM_JOB_END_TIME")
    if result and result.isdigit():
        return float(result)
    output = scheduler(["scontrol", "show", "job", "--oneliner", job_id])
    match = re.search(r"\bEndTime=(\S+)", output)
    if not match:
        raise RuntimeError("Slurm did not provide the inference allocation expiry")
    return datetime.fromisoformat(match[1]).timestamp()


def bound_socket(address: str) -> socket.socket:
    listener = socket.socket()
    try:
        listener.bind((address, 0))
        return listener
    except BaseException:
        listener.close()
        raise


def server_arguments(request: dict, port: int) -> list[str]:
    """The measured mathematical serving profile, with explicit device count."""
    for name, value in [('tensor_parallel_size', request.get('tensor_parallel_size', 4)),
                        ('max_model_len', request['max_model_len']),
                        ('max_num_seqs', request['max_num_seqs'])]:
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    memory = request.get('gpu_memory_utilization', 0.85)
    if not isinstance(memory, (int, float)) or not 0 < memory < 1:
        raise ValueError('gpu_memory_utilization must be between zero and one')
    return request["command"] + [
        "serve", request["model_dir"], "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", MODEL, "--tensor-parallel-size", str(request.get("tensor_parallel_size", 4)),
        "--gpu-memory-utilization", str(request.get("gpu_memory_utilization", 0.85)), "--max-model-len", str(request["max_model_len"]),
        "--max-num-seqs", str(request["max_num_seqs"]), "--language-model-only",
        "--trust-remote-code", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "nemotron_v3",
    ]


async def serve(attempt: Path) -> None:
    request = read_json(attempt / "request.json")
    job_id = os.environ["SLURM_JOB_ID"]
    if request.get("allocation_manifest"):
        from loam.allocation import validate_role
        allocation = validate_role(Path(request["allocation_manifest"]), "inference")
        if os.environ.get("SLURM_JOB_NAME") != allocation["job_name"] or socket.gethostname().split(".")[0] != request["assigned_host"]:
            raise RuntimeError("Nemotron process is outside its assigned parent job/node")
    elif not job_id.isdigit() or os.environ.get("SLURM_JOB_NAME") != attempt.name:
        raise RuntimeError("The Nemotron server must run in its dedicated Slurm allocation")
    expires_at = allocation_expiry(job_id)
    login = request["login_address"]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((login, 1))
        address = private_address(route.getsockname()[0])
    with bound_socket("127.0.0.1") as listener:
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}/v1"
    args = server_arguments(request, port)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if len(request["command"]) == 1:
        env = native_runtime_env(request, env)
        check_compiler(env)
        print(f"Native vLLM compiler check passed with {env['CC']}", flush=True)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES"):
        if key in env:
            env["APPTAINERENV_" + key] = env[key]
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    with (attempt / "vllm.log").open("ab") as log:
        process = subprocess.Popen(args, env=env, stdout=log, stderr=log, start_new_session=True)
        atomic_json(attempt / "process.json", {"pid": process.pid, "raw_base_url": base, "job_id": job_id})
    try:
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(
                total=None, connect=10, sock_read=600)) as session:
            deadline = min(time.time() + request["warmup_seconds"], expires_at)
            while not stop.is_set():
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited with code {process.returncode}. See vllm.log.")
                if time.time() >= deadline:
                    raise TimeoutError("vLLM did not pass a real completion before the startup deadline")
                try:
                    await completion_check(session, base, timeout=min(60, max(1, deadline - time.time())))
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(5)
            if stop.is_set():
                return

            async def upstream():
                if process.poll() is not None or time.time() >= expires_at:
                    raise RuntimeError("Nemotron API allocation ended")
                return base, "", None

            token = (attempt / "token").read_text().strip()
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(attempt / "tls.pem", attempt / "tls.key")
            runner = web.AppRunner(proxy_app(session, token, upstream, allowed_address=login),
                                   access_log=None)
            await runner.setup()
            with bound_socket(address) as listener:
                await web.SockSite(runner, listener, ssl_context=tls).start()
                atomic_json(attempt / "ready.json", {
                    "attempt": attempt.name, "job_id": job_id, "hostname": socket.gethostname(),
                    "base_url": f"https://{address}:{listener.getsockname()[1]}/v1",
                    "model": MODEL, "checkpoint": CHECKPOINT,
                    "checkpoint_revision": request["checkpoint_revision"],
                    "image_digest": request["image_digest"], "expires_at": expires_at,
                    "ready_at": time.time(),
                })
                print(f"Nemotron ready in Slurm job {job_id}", flush=True)
                try:
                    while not stop.is_set() and process.poll() is None and time.time() < expires_at:
                        await asyncio.sleep(2)
                finally:
                    (attempt / "ready.json").unlink(missing_ok=True)
                    await runner.cleanup()
                if process.poll() is not None and not stop.is_set():
                    raise RuntimeError(f"vLLM exited with code {process.returncode}. See vllm.log.")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.to_thread(process.wait, 20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                await asyncio.to_thread(process.wait)


async def wait_ready(state: Path, wait_seconds: float) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        attempt = current_attempt(state)
        if (attempt / "failure.json").exists():
            raise RuntimeError(f"Nemotron job failed. See {attempt / 'vllm.log'} and its Slurm log.")
        job = await asyncio.to_thread(active_job, attempt)
        if job is None:
            raise RuntimeError("Nemotron API job is absent from Slurm. Check its submission and Slurm log.")
        if job[1] == "RUNNING" and (attempt / "ready.json").is_file():
            readiness(attempt, job[0])
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Nemotron job {job[0]} is {job[1]}. It remains in Slurm.")
        await asyncio.sleep(15)


def write_yaml(config: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=config.parent, delete=False) as stream:
        path = Path(stream.name)
        try:
            yaml.safe_dump(data, stream, sort_keys=False)
            stream.flush()
            os.replace(path, config)
        finally:
            path.unlink(missing_ok=True)


def nemotron_model(base: str, context: int) -> dict:
    return {"default": MODEL, "base_url": base, "api_key": "${" + TOKEN_ENV + "}",
            "provider": "custom", "api_mode": "chat_completions", "context_length": context}


def configure(config: Path) -> None:
    """Render inference settings in the copied campaign Hermes config."""
    data = yaml.safe_load(config.read_text())
    if os.environ.get(MODE_ENV) == "nemotron":
        data["model"] = nemotron_model(os.environ["LOAM_NEMOTRON_BASE_URL"],
                                       int(os.environ["LOAM_NEMOTRON_CONTEXT"]))
        for key in ("fallback_model", "fallback_providers", "custom_providers", "providers"):
            data.pop(key, None)
        # The /review path can pass its own credentials to delegate_task.
        review = data.setdefault("auxiliary", {}).setdefault("review", {})
        review.update({key: data["model"][key] for key in ("provider", "base_url", "api_key", "api_mode")})
        review.update(model=MODEL, fallback_chain=[])
    model = data.get("model") or {}
    if not isinstance(model, dict) or not all(model.get(key) for key in ("default", "base_url", "api_key")):
        raise ValueError("Campaign config requires an explicit model, base URL and API key")
    delegation = data.setdefault("delegation", {})
    delegation.update(model=model["default"], base_url=model["base_url"], api_key=model["api_key"],
                      api_mode=model.get("api_mode") or "chat_completions")
    delegation.pop("provider", None)
    delegation.pop("role_models", None)
    if os.environ.get(MODE_ENV) in ("mixed", "nemotron"):
        # Hermes's threaded timeout wrapper allows only one in-flight callback
        # per plugin/hook, so parallel campaign agents otherwise reject one
        # another. Loam's hook subprocesses already enforce their own deadlines.
        data.setdefault("plugins", {})["hook_callback_timeout"] = 0
        # Role policy deliberately forbids generic discovery tools. Expose the
        # real MCP/plugin schemas so agents can call their permitted tools.
        data.setdefault("tools", {})["tool_search"] = {"enabled": "off"}
        data.setdefault('agent', {}).setdefault('coding_context', 'off')
        compression = data.setdefault("compression", {})
        # Native role context plus MCP schemas can already occupy >32K tokens.
        # Compressing at that point repeatedly evicts skills without freeing
        # the fixed prompt. Leave 16K of the 64K window for output while allowing
        # useful work between summaries. Honor a deliberately smaller cap.
        existing = compression.get("threshold_tokens")
        cap = min(existing, 48_000) if isinstance(existing, int) and existing > 0 else 48_000
        # The stock host watchdog has a shared four-slot pool and refuses
        # concurrent agents' compression once it fills. Run on each caller's
        # thread; auxiliary requests retain their own bounded HTTP timeout.
        compression.update(enabled=True, threshold_tokens=cap, context_timeout_seconds=0)
        # The lean tail has a 10K-token floor, too large beside native tool
        # schemas in this 64K window. Keep a smaller supported legacy tail.
        for key, value in {"tail_mode": "legacy", "target_ratio": 0.15,
                           "protect_first_n": 1, "protect_last_n": 6,
                           "max_attempts": 6}.items():
            compression.setdefault(key, value)
    write_yaml(config, data)


def configure_tournament(source: Path, target: Path, base: str) -> None:
    data = yaml.safe_load(source.read_text())
    profile = data["profiles"][data["active_profile"]]
    tournament = profile.get("hypothesis_tournament") or {}
    if tournament.get("enabled") is True:
        route = {"model": MODEL, "provider": "openai", "inference_endpoint": base,
                 "api_key_env": TOKEN_ENV, "endpoint_is_chat_completions": False}
        tournament.update(route)
        for role in ("generation", "ranking"):
            options = tournament.get(role)
            tournament[role] = {**(options if isinstance(options, dict) else {}), **route}
        knowledge = tournament.get("knowledge")
        if isinstance(knowledge, dict) and knowledge.get("enabled"):
            knowledge["model"] = MODEL
    write_yaml(target, data)


def check_science_owner(campaign: Path) -> None:
    if os.environ.get("HPC_SYSTEM") != "jupiter" or os.environ.get("HPC_BASE_DIR") != str(campaign):
        raise ValueError("Source this campaign's Jupiter hpc_env.sh before running it")
    if os.environ.get("HPC_ISOLATED_MCP") == "1":
        sys.path.insert(0, str(REPO / "skills/hpc-deployment/scripts"))
        from hpc_control import Controller, _listening

        controller = Controller()
        if _listening(controller.url) and not controller.pids():
            raise RuntimeError("The isolated MCP port belongs to another process")
        return
    workers = subprocess.run(["pgrep", "-f", "hpc-deployment/scripts/mcp_server.py"],
                             capture_output=True, text=True, check=False).stdout.split()
    for pid in workers:
        try:
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if os.fsencode(f"HPC_BASE_DIR={campaign}") not in environment:
            raise RuntimeError("The active HPC MCP server belongs to another campaign")


async def run_campaign(args) -> int:
    campaign = args.campaign
    env = {**os.environ, "LOAM_REPO": str(REPO), MODE_ENV: args.mode, "ALCF_PROXY": "0",
           "LOAM_HPC_WAIT_SECONDS": str(args.hpc_wait_seconds)}
    env.setdefault('LOAM_HERMES_TOOLSETS', 'skills,delegation,loam-provenance,mcp-hpc')
    driver = [sys.executable, str(REPO / "scripts/drive_campaign.py"), "--campaign", str(campaign),
              "--repo", str(REPO), "--launcher", str(REPO / "run-campaign"),
              "--max-hours", str(args.max_hours), "--max-rounds", str(args.max_rounds),
              "--max-turns", str(args.max_turns), "--hpc-wait-seconds", str(args.hpc_wait_seconds),
              "--startup-seconds", str(args.hpc_wait_seconds + 900)]
    check_science_owner(campaign)
    config = next((path for path in (
        campaign / "config.yaml", campaign / ".hermes/config.yaml",
        REPO / ".hermes/.loam-home/config.yaml",
    ) if path.is_file()), None)
    if config is None:
        raise ValueError("No Hermes config.yaml found for this campaign")
    from campaign_preflight import _inference_roles

    if args.mode != "nemotron":
        roles = _inference_roles(config.parent, campaign / "loam.yaml")
        argo_url, _, argo_model = roles["reasoner"]
        if not argo_model.startswith("argo:"):
            raise ValueError("Select an Argo parent model in the campaign config.yaml")
        for role, (url, _, model) in roles.items():
            if args.mode == "mixed" and role.startswith("subagents"):
                continue  # Mixed mode renders the Argo default from the parent.
            if url != argo_url or not model.startswith("argo:"):
                raise ValueError(f"Set {role} to an Argo model at the parent endpoint before launching")
    run_state = campaign / "state"
    run_state.mkdir(exist_ok=True)
    with (run_state / "jupiter-launch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "argo":
            return subprocess.call(driver, cwd=campaign, env=env)
        state = state_dir(campaign)
        ready = False
        try:
            await wait_ready(state, args.wait_seconds)
            ready = True
            return await run_with_relay(args, state, env, driver, config)
        finally:
            failed = (current_attempt(state) / "failure.json").exists()
            if ready or failed:
                await cleanup_campaign(args.campaign)


async def cleanup_campaign(campaign: Path) -> None:
    """Release this campaign's inference job and isolated science controller."""
    result = {"at": time.time(), "campaign": str(campaign), "errors": []}
    if os.environ.get("HPC_ISOLATED_MCP") == "1":
        try:
            sys.path.insert(0, str(REPO / "skills/hpc-deployment/scripts"))
            from hpc_control import Controller

            controller = Controller()
            if Path(os.environ["HPC_BASE_DIR"]).resolve() != campaign:
                raise RuntimeError("Science cleanup refused a different campaign base directory")
            await asyncio.to_thread(controller._stop)
            result["science"] = "stopped"
        except Exception as exc:
            result["errors"].append(f"Science cleanup: {exc}")
    try:
        cancelled = []
        for attempt in owned_attempts(state_dir(campaign)):
            job = await asyncio.to_thread(active_job, attempt)
            if job:
                await asyncio.to_thread(scheduler, ["scancel", job[0]])
                cancelled.append(job[0])
        if cancelled:
            result["inference_job_cancelled"] = cancelled[0]
            result["inference_jobs_cancelled"] = cancelled
        else:
            result["inference"] = "already ended"
    except Exception as exc:
        result["errors"].append(f"Inference cleanup: {exc}")
    atomic_json(state_dir(campaign) / "cleanup.json", result)
    for error in result["errors"]:
        print(error, file=sys.stderr, flush=True)


async def nested_probe(state: Path, env: dict, config: Path) -> None:
    software = Path(env.get("LOAM_SOFTWARE_ROOT", REPO.parent / "software"))
    python = software / "hermes-agent/.venv/bin/python"
    if not python.is_file():
        raise RuntimeError("The nested probe requires the pinned Jupiter Hermes environment")
    with tempfile.TemporaryDirectory(prefix="hermes-probe-", dir=state) as directory:
        home = Path(directory)
        model = (nemotron_model(env["LOAM_NEMOTRON_BASE_URL"], int(env["LOAM_NEMOTRON_CONTEXT"]))
                 if env[MODE_ENV] == "nemotron" else yaml.safe_load(config.read_text())["model"])
        (home / "config.yaml").write_text(yaml.safe_dump({
            "model": model,
            "delegation": {"max_spawn_depth": 2, "max_concurrent_children": 4, "max_iterations": 8},
        }))
        probe_env = {**env, "HERMES_HOME": str(home), "HERMES_ENABLE_PROJECT_PLUGINS": "0"}
        with (state / "nested-probe.log").open("ab") as log:
            result = await asyncio.to_thread(subprocess.run, [
                str(python), str(REPO / "scripts/jupiter_nemotron_probe.py"), "--receipt",
                str(state / "nested-probe.json"),
            ], cwd=home, env=probe_env, stdout=log, stderr=log, timeout=900, check=False)
        if result.returncode:
            raise RuntimeError(f"Nested Nemotron delegation failed. See {state / 'nested-probe.log'}.")


async def run_with_relay(args, state: Path, env: dict, driver: list, config: Path) -> int:
    campaign = args.campaign
    relay_token = secrets.token_urlsafe(32)
    token_file = state / "relay-token"
    verified = None
    credentials = None
    switching = asyncio.Lock()
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(
            total=None, connect=10, sock_read=600)) as session:
        async def upstream():
            nonlocal verified, credentials
            async with switching:
                attempt = current_attempt(state)
                job_id = read_json(attempt / "submission.json")["job_id"]
                record = readiness(attempt, job_id)
                if verified != record:
                    credentials = (record["base_url"], (attempt / "token").read_text().strip(),
                                   pinned_tls(attempt / "tls.pem"))
                    await completion_check(session, credentials[0], credentials[1], tls=credentials[2])
                    with (state / "backends.jsonl").open("a") as log:
                        log.write(json.dumps(record) + "\n")
                    verified = record
                return credentials

        await upstream()
        runner = web.AppRunner(proxy_app(session, relay_token, upstream), access_log=None)
        await runner.setup()
        try:
            token_file.touch(mode=0o600)
            token_file.write_text(relay_token + "\n")
            with bound_socket("127.0.0.1") as listener:
                # SockSite owns its descriptor through runner.cleanup(), which
                # runs after this reservation socket's context has closed.
                await web.SockSite(runner, listener.dup()).start()
                base = f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
                env.update(LOAM_NEMOTRON_BASE_URL=base, LOAM_NEMOTRON_TOKEN_FILE=str(token_file),
                           LOAM_NEMOTRON_ROUTE_LOG=str(state / "delegations.jsonl"))
                if args.mode == "nemotron":
                    profile = state / "loam.yaml"
                    configure_tournament(campaign / "loam.yaml", profile, base)
                    request = read_json(current_attempt(state) / "request.json")
                    env.update(LOAM_NEMOTRON_PROFILE=str(profile),
                               LOAM_NEMOTRON_CONTEXT=str(request["max_model_len"]))
                    env[TOKEN_ENV] = relay_token
                for key in ("NO_PROXY", "no_proxy"):
                    env[key] = ",".join(filter(None, [env.get(key), "127.0.0.1", "localhost"]))
                if args.probe_tools:
                    await streaming_tool_check(session, base, relay_token)
                    if args.mode == "nemotron":
                        await structured_completion_check(session, base, relay_token)
                if args.probe_delegation:
                    await nested_probe(state, env, config)
                check_science_owner(campaign)
                primary = MODEL if args.mode == "nemotron" else "argo"
                atomic_json(state / "routing.json", {
                    "mode": args.mode, "parent": primary, "reasoner": primary, "evidence-reviewer": primary,
                    "generation": primary, "ranking": primary, "hypothesis-validator": MODEL,
                    "code-generator": MODEL, "relay": base, "streaming_tool_probe": args.probe_tools,
                    "structured_completion_probe": args.probe_tools and args.mode == "nemotron",
                    "nested_delegation_probe": args.probe_delegation,
                })
                process = await asyncio.create_subprocess_exec(*driver, cwd=campaign, env=env)
                loop = asyncio.get_running_loop()
                for signum in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(signum, process.send_signal, signum)
                try:
                    return await process.wait()
                finally:
                    for signum in (signal.SIGINT, signal.SIGTERM):
                        loop.remove_signal_handler(signum)
        finally:
            await runner.cleanup()
            token_file.unlink(missing_ok=True)


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("submit", "run", "status", "stop", "activate"):
        command = commands.add_parser(action)
        command.add_argument("--campaign", required=True, type=lambda value: Path(value).resolve())
        if action == "submit":
            command.add_argument("--model-dir", required=True, type=Path)
            command.add_argument("--model-revision", required=True)
            runtime = command.add_mutually_exclusive_group(required=True)
            runtime.add_argument("--image", type=Path, help="staged ARM64 vLLM Apptainer image")
            runtime.add_argument("--vllm", type=Path, help="vLLM executable in a dedicated environment")
            command.add_argument("--cc", type=Path, help="native C compiler, defaults to Jupiter GCC")
            command.add_argument("--cxx", type=Path, help="native C++ compiler, defaults to Jupiter GCC")
            command.add_argument("--image-digest")
            command.add_argument("--login-address", required=True)
            command.add_argument("--account", default="deepwing")
            command.add_argument("--reservation", help="Slurm reservation on the Booster partition")
            command.add_argument("--walltime", default="02:00:00")
            command.add_argument("--context", type=positive, default=65536)
            command.add_argument("--sequences", type=positive, default=8)
            command.add_argument("--warmup-seconds", type=positive, default=1800)
            command.add_argument("--replace", action="store_true")
            command.add_argument("--standby", action="store_true",
                                 help="warm a replacement without changing the running relay")
        elif action == "run":
            command.add_argument("--mode", choices=("argo", "mixed", "nemotron", "argo-only"), default="argo")
            command.add_argument("--wait-seconds", type=positive, default=6000)
            command.add_argument("--hpc-wait-seconds", type=positive, default=6000)
            command.add_argument("--max-hours", type=float, default=12)
            command.add_argument("--max-rounds", type=positive, default=20)
            command.add_argument("--max-turns", type=positive, default=100)
            command.add_argument("--probe-tools", action="store_true")
            command.add_argument("--probe-delegation", action="store_true")
    commands.add_parser("serve").add_argument("attempt", type=Path)
    commands.add_parser("configure").add_argument("config", type=Path)
    args = parser.parse_args()
    if args.action == "configure":
        configure(args.config)
        return 0
    if args.action == "serve":
        try:
            asyncio.run(serve(args.attempt))
        except Exception as exc:
            atomic_json(args.attempt / "failure.json", {"error": type(exc).__name__, "at": time.time()})
            raise
        return 0
    if not (args.campaign / "loam.yaml").is_file():
        parser.error("--campaign must be a prepared campaign with a loam.yaml")
    if args.action == "run":
        if not math.isfinite(args.max_hours) or args.max_hours <= 0:
            parser.error("--max-hours must be positive")
        if args.mode == "argo-only":
            args.mode = "argo"
        if args.mode == "argo" and (args.probe_tools or args.probe_delegation):
            parser.error("Nemotron probes require --mode mixed or nemotron")
        return asyncio.run(run_campaign(args))
    if args.action == "submit":
        if args.context < 64000:
            parser.error("Hermes 0.20.6 requires at least a 64000-token model context")
        print(json.dumps(submit(args)))
        return 0
    if args.action == "activate":
        print(json.dumps(asyncio.run(activate_standby(args.campaign))))
        return 0
    attempt = current_attempt(state_dir(args.campaign))
    job = active_job(attempt)
    if args.action == "stop" and job:
        scheduler(["scancel", job[0]])
        print(f"Cancelled campaign inference job {job[0]}")
    else:
        print(json.dumps({"attempt": attempt.name, "active_job": job,
                          "submission_confirmed": (attempt / "submission.json").exists(),
                          "ready_record": (attempt / "ready.json").exists()}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"jupiter_nemotron: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
