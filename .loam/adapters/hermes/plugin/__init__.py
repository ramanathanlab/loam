"""Hermes adapter for the repository-local Loam lifecycle contract.

This module is intentionally thin: all portable policy belongs in
`.loam/hooks/`, not in this Hermes-specific bridge.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# `.loam/adapters/hermes/plugin/__init__.py` -> repository root
REPO_ROOT = Path(__file__).resolve().parents[4]
LOAM_ROOT = REPO_ROOT / ".loam"
HOOKS_DIR = LOAM_ROOT / "hooks"
HOOK_TIMEOUT_SECONDS = 15
_DEFAULT_PRE_LLM_INTERVAL = 5
_PRE_LLM_TURNS: dict[str, int] = defaultdict(int)
_PRE_LLM_TURNS_LOCK = threading.Lock()


def _pre_llm_interval() -> int:
    """Read Loam's per-session pre-LLM hook interval from the environment."""
    raw_interval = os.environ.get("LOAM_PRE_LLM_EVERY", str(_DEFAULT_PRE_LLM_INTERVAL))
    try:
        return max(1, int(raw_interval))
    except ValueError:
        logger.warning(
            "Invalid LOAM_PRE_LLM_EVERY=%r; using %d",
            raw_interval,
            _DEFAULT_PRE_LLM_INTERVAL,
        )
        return _DEFAULT_PRE_LLM_INTERVAL


def _json_safe(value: Any) -> Any:
    """Convert adapter kwargs into a hook-safe JSON value."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _payload(event: str, **values: Any) -> dict[str, Any]:
    return {
        "event": event,
        "repo_root": str(REPO_ROOT),
        **{key: _json_safe(value) for key, value in values.items()},
    }


def _invoke(script_name: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Run one canonical Loam script and parse its optional JSON response."""
    script = HOOKS_DIR / script_name
    if not script.is_file():
        logger.warning("Loam hook missing: %s", script)
        return None

    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=HOOK_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Loam hook %s did not complete: %s", script_name, exc)
        return None

    if completed.returncode:
        logger.warning(
            "Loam hook %s exited %s: %s",
            script_name,
            completed.returncode,
            (completed.stderr or "").strip()[:500],
        )
        return None

    output = (completed.stdout or "").strip()
    if not output:
        return None
    try:
        response = json.loads(output)
    except json.JSONDecodeError:
        logger.warning("Loam hook %s emitted invalid JSON", script_name)
        return None
    if not isinstance(response, dict):
        logger.warning("Loam hook %s emitted a non-object JSON response", script_name)
        return None
    return response


def pre_tool_call(tool_name: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any] | None:
    return _invoke(
        "pre-tool.py",
        _payload("pre_tool_call", tool_name=tool_name, arguments=args, **kwargs),
    )


def post_tool_call(tool_name: str, args: dict[str, Any], result: str, **kwargs: Any) -> None:
    _invoke(
        "post-tool.py",
        _payload(
            "post_tool_call",
            tool_name=tool_name,
            arguments=args,
            result=result,
            **kwargs,
        ),
    )


def pre_llm_call(session_id: str = "", **kwargs: Any) -> dict[str, Any] | None:
    """Run Loam's pre-LLM hook every N turns, independently per session."""
    interval = _pre_llm_interval()
    counter_key = session_id or "__no_session__"

    with _PRE_LLM_TURNS_LOCK:
        _PRE_LLM_TURNS[counter_key] += 1
        turn = _PRE_LLM_TURNS[counter_key]

    # N=1 fires every turn; otherwise fire on N, 2N, 3N, ...
    if turn % interval:
        return None

    return _invoke(
        "pre-llm.py",
        _payload(
            "pre_llm_call",
            session_id=session_id,
            loam_turn=turn,
            loam_interval=interval,
            **kwargs,
        ),
    )


def subagent_start(**kwargs: Any) -> None:
    _invoke("subagent-start.py", _payload("subagent_start", **kwargs))


def subagent_stop(**kwargs: Any) -> None:
    _invoke("subagent-stop.py", _payload("subagent_stop", **kwargs))


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("subagent_start", subagent_start)
    ctx.register_hook("subagent_stop", subagent_stop)
