#!/usr/bin/env python3
"""Inject validated Loam memory and optional tournament rankings before a model call.

All injected material is advisory. Validated parent memory is read-only; an
opt-in Jnana tournament may persist its separate workflow-scoped state and a
compact Loam audit report.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from hypothesis_tournament import context_for as tournament_context_for
from memory_recall import context_for as memory_context_for


def _payload() -> dict[str, Any]:
    value = json.load(sys.stdin)
    if not isinstance(value, dict) or value.get("event") != "pre_llm_call":
        raise ValueError("pre-llm hook requires a pre_llm_call JSON object")
    return value


def main() -> int:
    try:
        payload = _payload()
        contexts = [memory_context_for(payload), tournament_context_for(payload)]
        context = "\n\n".join(part for part in contexts if part)
        if context:
            print(json.dumps({"context": context}))
    except Exception as exc:
        print(f"loam pre-llm recall failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
