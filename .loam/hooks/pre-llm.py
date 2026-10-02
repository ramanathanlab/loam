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

from hypothesis_recall import context_for as hypothesis_context_for
from hypothesis_steering_recall import context_for as steering_context_for
from hypothesis_orchestration import context_for as orchestration_context_for
from hypothesis_orchestration import is_campaign_child, prepare_payload
from hypothesis_tournament import context_for as tournament_context_for
from memory_recall import context_for as memory_context_for
from run_manifest import ensure as ensure_run_manifest


def _payload() -> dict[str, Any]:
    value = json.load(sys.stdin)
    if not isinstance(value, dict) or value.get("event") != "pre_llm_call":
        raise ValueError("pre-llm hook requires a pre_llm_call JSON object")
    return value


def main() -> int:
    try:
        payload = _payload()
        ensure_run_manifest(payload)
        contexts = [
            memory_context_for(payload),
            hypothesis_context_for(payload),
            steering_context_for(payload),
        ]
        if payload.get("loam_hypothesis_orchestration") is True and not is_campaign_child(payload):
            prepare_payload(payload)
            contexts.append(tournament_context_for(payload))
            contexts.append(orchestration_context_for(payload))
        context = "\n\n".join(part for part in contexts if part)
        if context:
            print(json.dumps({"context": context}))
    except Exception as exc:
        print(f"loam pre-llm recall failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
