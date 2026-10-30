#!/usr/bin/env python3
"""Verify a hypothesis workspace's HPC runs where their files actually live.

A Loam hook cannot see HPC files: it is a short local subprocess with no MCP
access. This script can. It runs the same manifest program a remote child is
given (hypothesis_workspace.manifest_program) through the HPC MCP server's
run_code tool, and records what it finds in the LOCAL workspace record:

  fill     a completed run whose manifest is missing or invalid: hash its
           outputs on HPC and record them (manifest_status: recorded_by_loam,
           with verified_at). Until then other runs may not read its outputs.
  recheck  every completed run with a verified manifest: hash its outputs
           again and log anything changed, removed or added since as a
           violation (the local audit cannot re-hash remote bytes).
  batches  a run that used run_batch: ask the server for each batch's
           status and flag the ones still unfinished when the run closed
           (batches_unfinished, plus a violation each). The run stays completed.

The stop hook launches it detached (hypothesis_workspace._launch_verifier);
it can also be run by hand. It needs an interpreter with fastmcp, not Hermes's.
It never edits a result file and never re-scores anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

HOOKS = Path(__file__).resolve().parents[1] / "hooks"
sys.path.insert(0, str(HOOKS))
import hypothesis_workspace as workspace  # noqa: E402

# run_code(program, workdir) -> the program's stdout; raises on failure.
RunCode = Callable[[str, str], str]
# batch_status(batch_id) -> the server's status dict; raises on failure.
BatchStatus = Callable[[str], Mapping[str, Any]]


def fill(root: Path, run_id: str, remote_root: str, run_code: RunCode) -> str:
    """Record a verified manifest for one completed run that lacks one. Returns what happened."""

    run = workspace.read_json(root / "runs" / run_id / workspace.RUN_FILE)
    if run.get("state") != "completed" or run.get("manifest_status") in workspace.VERIFIED_MANIFESTS:
        return "skipped"
    try:
        printed = run_code(workspace.manifest_program(f"{remote_root}/outputs/{run_id}"), remote_root)
    except Exception as exc:
        workspace.log_violations(root, [f"verifier could not hash run {run_id} on HPC: {type(exc).__name__}: {exc}"], run_id)
        return "failed"
    entries, status = workspace.remote_manifest(printed, run_id)
    if status != "recorded":
        workspace.log_violations(root, [f"verifier got no usable manifest for run {run_id} ({status})"], run_id)
        return "failed"
    return "recorded" if workspace.record_verified_manifest(root, run_id, entries) else "skipped"


def recheck(root: Path, remote_root: str, run_code: RunCode) -> list[str]:
    """Re-hash every completed run on HPC; log and return what changed since it finished.

    A completed run whose manifest is not verified yet is filled first rather
    than skipped. The two verifier invocations are detached and independent, so
    a validator that finishes before a child's ``fill`` lands would otherwise
    leave exactly the run that needs checking unchecked (observed live: the
    "recheck" launch preceded r2's fill). Filling here makes recheck
    self-healing and order-independent.
    """

    experiment = workspace.read_json(root / workspace.EXPERIMENT_FILE)
    problems: list[str] = []
    for run_id in experiment.get("runs", []):
        run = workspace.read_json(root / "runs" / run_id / workspace.RUN_FILE)
        if run.get("state") != "completed":
            continue
        if run.get("manifest_status") not in workspace.VERIFIED_MANIFESTS:
            fill(root, run_id, remote_root, run_code)
            run = workspace.read_json(root / "runs" / run_id / workspace.RUN_FILE)
            if run.get("manifest_status") not in workspace.VERIFIED_MANIFESTS:
                continue  # fill failed and already logged why
        try:
            printed = run_code(workspace.manifest_program(f"{remote_root}/outputs/{run_id}"), remote_root)
        except Exception as exc:
            found = [f"verifier could not re-hash run {run_id} on HPC: {type(exc).__name__}: {exc}"]
        else:
            fresh, status = workspace.remote_manifest(printed, run_id)
            found = (
                workspace.manifest_changes(run.get("manifest", []), fresh, run_id)
                if status == "recorded"
                else [f"verifier got no usable manifest for run {run_id} ({status})"]
            )
        if found:
            workspace.log_violations(root, found, run_id)
            problems += found
    return problems


def batches(root: Path, run_id: str, status: BatchStatus) -> list[str]:
    """Flag the run's batches that were not complete when it closed. Returns their ids."""

    run = workspace.read_json(root / "runs" / run_id / workspace.RUN_FILE)
    unfinished: list[str] = []
    for batch_id in run.get("batches", []):
        try:
            complete = status(batch_id).get("complete") is True
        except Exception as exc:
            workspace.log_violations(root, [f"verifier could not read batch {batch_id}: {type(exc).__name__}: {exc}"], run_id)
            complete = False
        if not complete:
            unfinished.append(batch_id)
    workspace.record_batch_check(root, run_id, unfinished)
    return unfinished


def seed(root: Path, source: str, remote_root: str, cap: int, run_code: RunCode) -> str:
    """Copy a staged HPC inputs directory into this hypothesis's remote data/, and record it.

    The remote half of workspace.inputs_dir: a hook cannot reach HPC, so the
    stop/start hook launches this. The copy happens where the files are, and
    only the resulting hashes come back, which is what lets the LOCAL
    experiment.json state what the hypothesis was given.
    """

    program = workspace.seed_program(source, f"{remote_root}/data", cap)
    try:
        printed = run_code(program, f"{remote_root}/scratch")
    except Exception as exc:
        workspace.log_violations(root, [f"verifier could not seed inputs from {source}: {type(exc).__name__}: {exc}"])
        return "failed"
    entries, status = workspace.remote_manifest(printed, "")
    if status != "recorded":
        workspace.log_violations(root, [f"verifier got no usable input manifest from {source} ({status})"])
        return "failed"
    workspace.record_inputs(root, entries, source, "copied_on_hpc")
    return "recorded" if entries else "empty"


def mcp_call(url: str, tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """Call one tool on the HPC MCP server over HTTP and return its result as a dict."""

    from fastmcp import Client  # imported here so the module loads without fastmcp

    async def go() -> Any:
        async with Client(url) as client:
            return await client.call_tool(tool, dict(arguments))

    result = asyncio.run(go())
    data = result.data if isinstance(result.data, Mapping) else result.structured_content
    if not isinstance(data, Mapping):
        text = "".join(getattr(block, "text", "") for block in result.content)
        data = json.loads(text) if text.strip().startswith("{") else {"stdout": text}
    return data


def mcp_run_code(url: str, tool: str = "run_code", timeout: float = 300.0) -> RunCode:
    """A RunCode that calls the HPC MCP server's run_code tool."""

    def call(program: str, workdir: str) -> str:
        data = mcp_call(url, tool, {"payload": program, "workdir": workdir, "timeout": timeout})
        if str(data.get("returncode", "0")) not in {"0", "None"}:
            raise RuntimeError(f"returncode {data.get('returncode')}: {str(data.get('stderr', ''))[-500:]}")
        return str(data.get("stdout", ""))

    return call


def mcp_batch_status(url: str, tool: str = "batch_status") -> BatchStatus:
    """A BatchStatus that calls the HPC MCP server's batch_status tool."""

    return lambda batch_id: mcp_call(url, tool, {"batch_id": batch_id})


def traces(path: Path, db_path: Path, full: bool) -> dict[str, list[str]]:
    """Backfill missing per-agent traces under ``path`` from a LOCAL state.db.

    Unlike the other commands this needs no MCP: Hermes and its state.db are on
    the same machine as the workspace. ``path`` may be one hypothesis folder, the
    reasoner folder, or a whole <plan_id>/ (every hypothesis plus the reasoner).
    Covers agents whose subagent_stop never fired, so their trace was never
    captured live. Returns what was filled, per record.
    """

    out: dict[str, list[str]] = {}
    if (path / workspace.EXPERIMENT_FILE).is_file():
        out[path.name] = workspace.backfill_traces(path, db_path, full)
    elif (path / workspace.REASONER_FILE).is_file():
        out["reasoner"] = ["reasoner"] if workspace.backfill_reasoner_trace(path, db_path, full) else []
    else:
        for child in sorted(path.iterdir()) if path.is_dir() else []:
            if (child / workspace.EXPERIMENT_FILE).is_file():
                out[child.name] = workspace.backfill_traces(child, db_path, full)
            elif child.name == workspace.REASONER_DIR and (child / workspace.REASONER_FILE).is_file():
                out["reasoner"] = ["reasoner"] if workspace.backfill_reasoner_trace(child, db_path, full) else []
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("fill", "recheck", "batches", "seed", "traces"))
    parser.add_argument("--workspace", type=Path, required=True, help="local hypothesis or plan folder")
    # Not required for "batches", which only asks the server about batch ids.
    parser.add_argument("--remote-root", default="", help="the same hypothesis's folder on HPC")
    # Required only for the MCP commands; "traces" is local.
    parser.add_argument("--mcp-url", default="", help="HPC MCP server, e.g. http://127.0.0.1:8000/mcp")
    parser.add_argument("--tool", default="run_code", help="the server's run_code tool name")
    parser.add_argument("--run", help="run id, for fill and batches")
    parser.add_argument("--source", help="staged HPC inputs directory, for seed")
    parser.add_argument("--max-bytes", type=int, default=workspace.DEFAULT_MAX_INPUT_BYTES,
                        help="refuse a seed larger than this")
    parser.add_argument("--hermes-home", type=Path, help="Hermes home holding state.db, for traces")
    parser.add_argument("--trace-full", action="store_true", help="keep full trace bodies (traces)")
    args = parser.parse_args(argv)
    if args.command in {"fill", "batches"} and not args.run:
        parser.error(f"{args.command} needs --run")
    if args.command == "seed" and not args.source:
        parser.error("seed needs --source")
    if args.command in {"fill", "recheck", "seed"} and not args.remote_root:
        parser.error(f"{args.command} needs --remote-root")
    if args.command in {"fill", "recheck", "seed", "batches"} and not args.mcp_url:
        parser.error(f"{args.command} needs --mcp-url")
    if args.command == "traces":
        import os

        home = str(args.hermes_home) if args.hermes_home else os.environ.get("HERMES_HOME", "")
        db_path = Path(home) / "state.db" if home else None
        if db_path is None or not db_path.is_file():
            parser.error("traces needs --hermes-home (or $HERMES_HOME) with a state.db")
        print(json.dumps({"command": "traces", "workspace": str(args.workspace),
                          "filled": traces(args.workspace, db_path, args.trace_full)}))
        return 0
    if args.command == "batches":
        unfinished = batches(args.workspace, args.run, mcp_batch_status(args.mcp_url))
        print(json.dumps({"command": "batches", "workspace": str(args.workspace), "run": args.run, "unfinished": unfinished}))
        return 0
    run_code = mcp_run_code(args.mcp_url, args.tool)
    if args.command == "seed":
        print(json.dumps({"command": "seed", "workspace": str(args.workspace), "source": args.source,
                          "result": seed(args.workspace, args.source, args.remote_root, args.max_bytes, run_code)}))
    elif args.command == "fill":
        print(json.dumps({"command": "fill", "workspace": str(args.workspace), "run": args.run,
                          "result": fill(args.workspace, args.run, args.remote_root, run_code)}))
    else:
        problems = recheck(args.workspace, args.remote_root, run_code)
        print(json.dumps({"command": "recheck", "workspace": str(args.workspace), "problems": problems}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
