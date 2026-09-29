"""Read the latest advisory hypothesis steering projection for pre-LLM context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

FILENAME = "hypothesis-steering.jsonl"


def context_for(payload: Mapping[str, Any]) -> str:
    state_dir = payload.get("loam_state_dir")
    if not isinstance(state_dir, str) or not state_dir.strip():
        return ""
    path = Path(state_dir).expanduser() / FILENAME
    if not path.is_file():
        return ""
    workflow_id = str(payload.get("workflow_id") or "")
    session_id = str(payload.get("session_id") or payload.get("parent_session_id") or "")
    snapshot: Mapping[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping) or row.get("event") != "hypothesis_steering_snapshot":
            continue
        if workflow_id and str(row.get("workflow_id") or "") == workflow_id:
            snapshot = row
        elif not workflow_id and session_id and str(row.get("parent_session_id") or "") == session_id:
            snapshot = row
    if snapshot is None or not isinstance(snapshot.get("hypotheses"), list):
        return ""
    lines = ["Hypothesis steering state (advisory; evidence acceptance, Elo ranking, and action selection remain distinct):"]
    for item in snapshot["hypotheses"][:4]:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "- {id}: Elo={elo:.1f}, evidence-rating={evidence:.1f}, priority={priority:.2f}; next={action}.".format(
                id=str(item.get("hypothesis_id") or "unknown"),
                elo=float(item.get("jnana_elo") or 0.0),
                evidence=float(item.get("evidence_rating") or 0.0),
                priority=float(item.get("steering_priority") or 0.0),
                action=str(item.get("recommended_action") or "review"),
            )
        )
    lines.append("Follow the recommendation only after checking its linked source-backed evidence and current task constraints.")
    return "\n".join(lines)
