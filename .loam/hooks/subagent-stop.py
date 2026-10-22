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

from hypothesis_orchestration import record_stop
from ledger import write_stop_and_candidate
from run_manifest import ensure as ensure_run_manifest


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
    except Exception as exc:
        print(f"loam subagent-stop: unparseable payload: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rc = 0
    # Terminal records FIRST, each isolated. write_stop_and_candidate carries
    # the validator's evidence score; record_stop appends the orchestration
    # completion row that lets later turns advance (a missing one leaves a
    # dangling "...started" that deadlocks the plan). A failure writing the
    # manifest must not drop either, and a failure in one must not drop the
    # other -- so they do not share a try with the manifest or with each other.
    for step in (write_stop_and_candidate, record_stop):
        try:
            step(payload)
        except Exception as exc:  # observer failures must not change Hermes behavior
            print(f"loam subagent-stop: {step.__name__} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1

    try:
        ensure_run_manifest(payload)
    except Exception as exc:
        print(f"loam subagent-stop: run_manifest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
