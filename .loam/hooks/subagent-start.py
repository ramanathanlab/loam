#!/usr/bin/env python3
"""Append a pre-run allocation record for one delegated Loam subagent.

The script receives the canonical JSON payload from the Hermes adapter on stdin.
It is an observer: stdout is intentionally empty so it cannot change delegation
behavior. Failures are reported to stderr and a non-zero exit lets the adapter
log the hook failure while Hermes continues.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from hypothesis_orchestration import record_start
from ledger import build_start_record, write_record
from run_manifest import ensure as ensure_run_manifest


def _payload() -> dict[str, Any]:
    value = json.load(sys.stdin)
    if not isinstance(value, dict):
        raise ValueError("Loam hook payload must be a JSON object")
    if value.get("event") != "subagent_start":
        raise ValueError("subagent-start hook received the wrong event")
    if not isinstance(value.get("child_subagent_id"), str) or not value["child_subagent_id"]:
        raise ValueError("subagent_start payload requires child_subagent_id")
    return value


def main() -> int:
    try:
        payload = _payload()
        ensure_run_manifest(payload)
        write_record(payload, build_start_record(payload))
        record_start(payload)
    except Exception as exc:  # observer failures must not change Hermes behavior
        print(f"loam subagent-start observer failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
