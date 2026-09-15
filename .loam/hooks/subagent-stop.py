#!/usr/bin/env python3
"""Append a terminal observation record for one delegated Loam subagent.

A child's final response is retained as audit material only. This first slice
intentionally does not infer findings, evidence, success credit, or memory from
model-written summaries; those require a later workflow-specific validation
boundary.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ledger import write_stop_and_candidate


def _payload() -> dict[str, Any]:
    value = json.load(sys.stdin)
    if not isinstance(value, dict):
        raise ValueError("Loam hook payload must be a JSON object")
    if value.get("event") != "subagent_stop":
        raise ValueError("subagent-stop hook received the wrong event")
    if not isinstance(value.get("child_session_id"), str) or not value["child_session_id"]:
        raise ValueError("subagent_stop payload requires child_session_id")
    return value


def main() -> int:
    try:
        payload = _payload()
        write_stop_and_candidate(payload)
    except Exception as exc:  # observer failures must not change Hermes behavior
        print(f"loam subagent-stop observer failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
