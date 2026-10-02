"""Read compact, validated hypothesis provenance memory for pre-LLM recall."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

DEFAULT_FILENAME = "hypothesis-memory.jsonl"


def _path(payload: Mapping[str, Any]) -> Path:
    state_dir = payload.get("loam_state_dir")
    runtime_state_dir = os.environ.get("LOAM_STATE_DIR", "").strip()
    if isinstance(state_dir, str) and state_dir.strip():
        return Path(state_dir).expanduser() / DEFAULT_FILENAME
    if runtime_state_dir:
        return Path(runtime_state_dir).expanduser() / DEFAULT_FILENAME
    root = payload.get("repo_root")
    return Path(root if isinstance(root, str) and root else ".") / ".loam" / "state" / DEFAULT_FILENAME


def _snapshot(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    path = _path(payload)
    if not path.is_file():
        return None
    workflow_id = str(payload.get("workflow_id") or "")
    session_id = str(payload.get("session_id") or payload.get("parent_session_id") or "")
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping) or row.get("event") != "hypothesis_memory_snapshot":
            continue
        if workflow_id and str(row.get("workflow_id") or "") == workflow_id:
            matches.append(row)
        elif not workflow_id and session_id and str(row.get("parent_session_id") or "") == session_id:
            matches.append(row)
    return matches[-1] if matches else None


def context_for(payload: Mapping[str, Any]) -> str:
    snapshot = _snapshot(payload)
    if snapshot is None:
        return ""
    hypotheses = snapshot.get("active_hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        return ""
    lines = ["Validated hypothesis state (evidence support, tournament ranking, and contribution credit are separate signals):"]
    for hypothesis in hypotheses[:6]:
        if not isinstance(hypothesis, Mapping):
            continue
        statement = " ".join(str(hypothesis.get("statement") or "").split())[:300]
        support = len(hypothesis.get("supporting_evidence") or [])
        contradict = len(hypothesis.get("contradicting_evidence") or [])
        lines.append(
            "- {identifier}: support={support}, contradiction={contradict}; {statement}".format(
                identifier=str(hypothesis.get("hypothesis_id") or "unknown"),
                support=support,
                contradict=contradict,
                statement=statement,
            )
        )
    lines.append("Use only source-backed evidence listed here as epistemic support; tournament Elo is an advisory prioritization signal.")
    return "\n".join(lines)
