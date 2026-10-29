"""The subagent-stop hook queues Cairn ratings without printing or disturbing record_stop."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = ROOT / ".loam" / "hooks" / "subagent-stop.py"

REPORT = """### Tool Ratings
- tool://lammps | score: 0.80 | origin: none | note: stable
- mcp://hpc#run_code | score: 0.9 | origin: none
"""


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")


def _run(tmp_path: Path, cairn) -> subprocess.CompletedProcess[str]:
    payload = {
        "event": "subagent_stop", "repo_root": str(ROOT), "workflow_id": "wf", "loam_run_id": "run",
        "loam_hypothesis_orchestration": True, "loam_state_dir": str(tmp_path),
        "child_session_id": "s1", "child_subagent_id": "id-s1", "child_status": "completed",
        "child_summary": REPORT,
        "loam_execution": {"tool_name": "mcp_hpc_run_code", "function_name": "run_code", "tool_prefix": "mcp_hpc_"},
        "loam_cairn": cairn,
    }
    env = {**os.environ, "LOAM_LEDGER_PATH": str(tmp_path / "ledger.jsonl")}
    return subprocess.run([sys.executable, str(STOP)], cwd=str(ROOT), input=json.dumps(payload),
                          text=True, capture_output=True, env=env, check=False)


def _setup(tmp_path: Path) -> None:
    _append(tmp_path / "hypothesis-orchestration.jsonl", {
        "event": "agent_started", "role": "code-generator", "plan_id": "p", "hypothesis_id": "h",
        "child_subagent_id": "id-s1", "child_session_id": "s1",
    })
    _append(tmp_path / "tool-calls.jsonl", {
        "session_id": "s1", "tool_name": "mcp_hpc_run_code", "arguments": {"code": "lmp -in in.lj"},
    })


CFG = {
    "enabled": True, "rate_roles": ["code-generator"], "rate_agent_reports": True,
    "rate_hook_fallback": True, "auto_flush": False, "executor": "mcp://hpc#run_code",
    "entities": {
        "mcp://hpc#run_code": {"patterns": [], "task_tags": []},
        "tool://lammps": {"patterns": ["lmp"], "task_tags": ["md_simulation"]},
    },
}


def test_stop_hook_queues_ratings_and_stays_silent(tmp_path):
    _setup(tmp_path)
    proc = _run(tmp_path, CFG)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    queued = [r for r in _read(tmp_path / "cairn-ratings.jsonl") if r["status"] == "queued"]
    assert len(queued) == 2
    orchestration = _read(tmp_path / "hypothesis-orchestration.jsonl")
    assert any(r["event"] == "code_generator_completed" for r in orchestration)


def test_stop_hook_without_cairn_writes_no_queue(tmp_path):
    _setup(tmp_path)
    proc = _run(tmp_path, None)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert not (tmp_path / "cairn-ratings.jsonl").exists()
    assert any(r["event"] == "code_generator_completed" for r in _read(tmp_path / "hypothesis-orchestration.jsonl"))


def test_stop_hook_survives_cairn_import_failure(tmp_path):
    import shutil

    hooks = tmp_path / "hooks"
    shutil.copytree(ROOT / ".loam" / "hooks", hooks, ignore=shutil.ignore_patterns("__pycache__"))
    (hooks / "cairn_ratings.py").write_text("raise RuntimeError('boom')\n")
    state = tmp_path / "state"
    state.mkdir()
    _setup(state)
    payload = {
        "event": "subagent_stop", "repo_root": str(ROOT), "workflow_id": "wf", "loam_run_id": "run",
        "loam_hypothesis_orchestration": True, "loam_state_dir": str(state),
        "child_session_id": "s1", "child_subagent_id": "id-s1", "child_status": "completed",
        "child_summary": REPORT, "loam_cairn": CFG,
    }
    env = {**os.environ, "LOAM_LEDGER_PATH": str(tmp_path / "ledger.jsonl")}
    proc = subprocess.run([sys.executable, str(hooks / "subagent-stop.py")], cwd=str(ROOT),
                          input=json.dumps(payload), text=True, capture_output=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "cairn ratings unavailable" in proc.stderr
    assert any(r["event"] == "code_generator_completed" for r in _read(state / "hypothesis-orchestration.jsonl"))
