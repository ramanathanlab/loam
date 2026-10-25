"""Append every observed tool call, with its arguments, to a local audit log.

Nothing else records what a child actually submits. The lifecycle ledger keeps a
child's goal but not its tool arguments, Hermes' agent.log records only the tool
name, duration and response size, and the MCP server log records only
`POST /mcp 200 OK`. So when a child reported "the HPC environment was
unavailable" there was no way to see that it had asked `inspect_filesystem` for
`/workspace` -- a path that does not exist on Aurora -- until someone
reconstructed it from timestamps across three logs.

That mattered because three rejected arguments in a row open Hermes'
process-global MCP circuit breaker, which then reports a healthy server as
unreachable to every agent in the process. The payload a child sends is
therefore load-bearing diagnostic data, not noise.

Observer only: every failure is swallowed, so this can never block or alter a
tool call.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

FILENAME = "tool-calls.jsonl"
# A delegated validator context is ~11 KB (its injected role definition alone is
# ~8.6 KB), so a 6 KB cap silently cut the record mid-payload and made a context
# that DID carry `selected_hypothesis` look as though it did not -- a wrong
# diagnosis drawn from this log on 2026-10-21. Keep the cap above a realistic
# delegation payload, and always record the true length so a truncated entry can
# never again be mistaken for a short one.
MAX_VALUE_CHARS = 24_000
MAX_TOTAL_CHARS = 64_000


def _audit_path(payload: Mapping[str, Any]) -> Path | None:
    state_dir = payload.get("loam_state_dir")
    if not isinstance(state_dir, str) or not state_dir.strip():
        state_dir = os.environ.get("LOAM_STATE_DIR", "").strip()
    if not state_dir:
        return None
    return Path(state_dir).expanduser() / FILENAME


def _clip(value: Any) -> Any:
    """Keep structure, bound size, and mark anything truncated."""

    if isinstance(value, str):
        if len(value) <= MAX_VALUE_CHARS:
            return value
        # State the full length, not just that something was cut: a reader must
        # be able to tell "this field was long" from "this field was short".
        return value[:MAX_VALUE_CHARS] + f"...<TRUNCATED by loam tool_audit: full length {len(value)} chars>"
    if isinstance(value, Mapping):
        return {str(k): _clip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v) for v in value[:20]]
    return value


def record(payload: Mapping[str, Any]) -> None:
    """Append one record for this tool call. Never raises."""

    try:
        path = _audit_path(payload)
        if path is None:
            return
        tool_name = str(payload.get("tool_name") or "")
        if not tool_name:
            return
        arguments = payload.get("arguments")
        record = {
            "record_version": "loam_tool_call_v1",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": str(payload.get("loam_run_id") or ""),
            "session_id": str(payload.get("session_id") or ""),
            "parent_turn_id": str(payload.get("parent_turn_id") or ""),
            "tool_name": tool_name,
            # The role tag travels in the child's goal, so a reader can tell a
            # code-generator's submission from the coordinator's own calls.
            "agent_role": str(payload.get("agent_role") or ""),
            "arguments": _clip(arguments) if isinstance(arguments, (Mapping, list)) else _clip(str(arguments or "")),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if len(line) > MAX_TOTAL_CHARS:
            record["arguments"] = {"truncated": True, "chars": len(line)}
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        os.chmod(path, 0o600)
    except Exception:  # noqa: BLE001 - an observer must never affect a tool call
        return
