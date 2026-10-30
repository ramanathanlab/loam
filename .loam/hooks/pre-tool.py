#!/usr/bin/env python3
"""Observe explicit duplicate scientific delegation candidates without blocking.

Only `delegate_task` calls are inspected. Exact duplicates are written to a
separate audit stream and the script returns an empty response, so this first
policy slice cannot alter scheduling.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from falda_campaign import FALDA_TOOL_PREFIX
from hypothesis_orchestration import tool_policy, workspace_capture, workspace_policy
from memory_recall import audit_duplicates
from provenance import record as record_provenance
from run_manifest import ensure as ensure_run_manifest
from tool_audit import record as record_tool_call


def _payload() -> dict[str, Any]:
    value = json.load(sys.stdin)
    if not isinstance(value, dict) or value.get("event") != "pre_tool_call":
        raise ValueError("pre-tool hook requires a pre_tool_call JSON object")
    return value


def main() -> int:
    try:
        payload = _payload()
    except Exception as exc:
        # We cannot even parse the request, so we cannot tell whether it is a
        # campaign-tagged call. Blocking every tool in every session over a
        # parse bug would brick the harness, so we stay out of the way (the
        # harness default is allow) but report loudly.
        print(f"loam pre-tool: unparseable payload: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0

    # 1) ENFORCEMENT FIRST, in isolation. tool_policy is the role boundary; a
    #    failure here must NOT silently become "allow" (the adapter maps any
    #    nonzero hook exit to allow). For a tagged orchestration call we fail
    #    CLOSED by emitting a block; a non-campaign call we leave alone.
    #    workspace_policy is the second boundary: tool_policy decides WHAT a role
    #    may call, workspace_policy WHERE a code-generator may write. It shares
    #    this try, so its failure on a tagged call also blocks fail-closed.
    result: dict[str, Any] | None = None
    try:
        result = tool_policy(payload)
        if result is None:
            result = workspace_policy(payload)
        if result is None and payload.get("tool_name") == "delegate_task":
            result = audit_duplicates(payload)
    except Exception as exc:
        print(f"loam pre-tool: tool_policy failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if str(payload.get("tool_name") or "").startswith(FALDA_TOOL_PREFIX):
            # FALDA tools are allowed only when the guard can positively verify the call.
            result = {"action": "block", "message": "Loam could not verify this FALDA call; continue without FALDA tools."}
        elif payload.get("loam_hypothesis_orchestration") is True:
            result = {
                "action": "block",
                "message": (
                    "Loam could not evaluate the campaign role boundary for this "
                    "tagged call (orchestration state unreadable); blocked "
                    "fail-closed. Retry once the state is consistent."
                ),
            }

    # 2) Best-effort bookkeeping, each isolated so it can never void the policy
    #    verdict above. record runs unconditionally (a call about to be blocked
    #    is exactly the one worth auditing).
    try:
        record_tool_call(payload)
    except Exception as exc:
        print(f"loam pre-tool: record failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        # Layer 0 provenance (C4): observe-and-log the execution's input lineage.
        record_provenance(payload)
    except Exception as exc:
        print(f"loam pre-tool: provenance failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        ensure_run_manifest(payload)
    except Exception as exc:
        print(f"loam pre-tool: run_manifest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    if result is None:
        try:
            # Only for a call that will run: a blocked call's code never executed.
            workspace_capture(payload)
        except Exception as exc:
            print(f"loam pre-tool: workspace_capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    if result is not None:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
