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
    """Re-hash every completed, verified run on HPC; log and return what changed since."""

    experiment = workspace.read_json(root / workspace.EXPERIMENT_FILE)
    problems: list[str] = []
    for run_id in experiment.get("runs", []):
        run = workspace.read_json(root / "runs" / run_id / workspace.RUN_FILE)
        if run.get("state") != "completed" or run.get("manifest_status") not in workspace.VERIFIED_MANIFESTS:
            continue
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


def mcp_run_code(url: str, tool: str = "run_code", timeout: float = 300.0) -> RunCode:
    """A RunCode that calls the HPC MCP server's run_code tool over HTTP."""

    from fastmcp import Client  # imported here so the module loads without fastmcp

    def call(program: str, workdir: str) -> str:
        async def go() -> Any:
            async with Client(url) as client:
                return await client.call_tool(tool, {"payload": program, "workdir": workdir, "timeout": timeout})

        result = asyncio.run(go())
        data = result.data if isinstance(result.data, Mapping) else result.structured_content
        if not isinstance(data, Mapping):
            text = "".join(getattr(block, "text", "") for block in result.content)
            data = json.loads(text) if text.strip().startswith("{") else {"stdout": text}
        if str(data.get("returncode", "0")) not in {"0", "None"}:
            raise RuntimeError(f"returncode {data.get('returncode')}: {str(data.get('stderr', ''))[-500:]}")
        return str(data.get("stdout", ""))

    return call


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("fill", "recheck"))
    parser.add_argument("--workspace", type=Path, required=True, help="local hypothesis folder")
    parser.add_argument("--remote-root", required=True, help="the same hypothesis's folder on HPC")
    parser.add_argument("--mcp-url", required=True, help="HPC MCP server, e.g. http://127.0.0.1:8000/mcp")
    parser.add_argument("--tool", default="run_code", help="the server's run_code tool name")
    parser.add_argument("--run", help="run id, for fill")
    args = parser.parse_args(argv)
    run_code = mcp_run_code(args.mcp_url, args.tool)
    if args.command == "fill":
        if not args.run:
            parser.error("fill needs --run")
        print(json.dumps({"command": "fill", "workspace": str(args.workspace), "run": args.run,
                          "result": fill(args.workspace, args.run, args.remote_root, run_code)}))
    else:
        problems = recheck(args.workspace, args.remote_root, run_code)
        print(json.dumps({"command": "recheck", "workspace": str(args.workspace), "problems": problems}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
