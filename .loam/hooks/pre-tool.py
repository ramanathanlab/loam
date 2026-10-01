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

from hypothesis_orchestration import tool_policy
from memory_recall import audit_duplicates


def _payload() -> dict[str, Any]:
    value = json.load(sys.stdin)
    if not isinstance(value, dict) or value.get("event") != "pre_tool_call":
        raise ValueError("pre-tool hook requires a pre_tool_call JSON object")
    return value


def main() -> int:
    try:
        payload = _payload()
        result = tool_policy(payload)
        if result is None and payload.get("tool_name") == "delegate_task":
            result = audit_duplicates(payload)
        if result is not None:
            print(json.dumps(result))
    except Exception as exc:
        print(f"loam pre-tool audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
