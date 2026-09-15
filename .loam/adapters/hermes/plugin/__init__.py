"""Hermes adapter for the repository-local Loam lifecycle contract.

This module is intentionally thin: all portable policy belongs in
`.loam/hooks/`, not in this Hermes-specific bridge.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

_ADAPTER_DIR = Path(__file__).resolve().parent
if str(_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_DIR))
from loam_config import effective_profile, payload_values

logger = logging.getLogger(__name__)

# `.loam/adapters/hermes/plugin/__init__.py` -> repository root
REPO_ROOT = Path(__file__).resolve().parents[4]
LOAM_ROOT = REPO_ROOT / ".loam"
HOOKS_DIR = LOAM_ROOT / "hooks"
HOOK_TIMEOUT_SECONDS = 15
_DEFAULT_PRE_LLM_INTERVAL = 5
_PRE_LLM_TURNS: dict[str, int] = defaultdict(int)
_PRE_LLM_TURNS_LOCK = threading.Lock()
_CHILD_SUBAGENT_IDS: dict[str, str] = {}
_CHILD_SUBAGENT_IDS_LOCK = threading.Lock()


def _profile():
    """Return the selected config profile, or None when Loam is inert."""

    return effective_profile()


def _pre_llm_interval(profile: Any) -> int:
    """Use profile cadence; legacy environment activation is handled by config."""

    return profile.pre_llm_every


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


def _invoke(
    script_name: str,
    payload: dict[str, Any],
    timeout_seconds: int = HOOK_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
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
            timeout=timeout_seconds,
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
    profile = _profile()
    if profile is None:
        return None
    values = dict(kwargs)
    values.update(payload_values(profile))
    session_id = values.get("session_id")
    if isinstance(session_id, str) and session_id:
        values.setdefault("parent_session_id", session_id)
    return _invoke(
        "pre-tool.py",
        _payload("pre_tool_call", tool_name=tool_name, arguments=args, **values),
    )


def post_tool_call(tool_name: str, args: dict[str, Any], result: str, **kwargs: Any) -> None:
    profile = _profile()
    if profile is None:
        return
    values = dict(kwargs)
    values.update(payload_values(profile))
    _invoke(
        "post-tool.py",
        _payload(
            "post_tool_call",
            tool_name=tool_name,
            arguments=args,
            result=result,
            **values,
        ),
    )


def pre_llm_call(session_id: str = "", **kwargs: Any) -> dict[str, Any] | None:
    """Run Loam's pre-LLM hook every N turns, independently per session."""
    profile = _profile()
    if profile is None:
        return None
    interval = _pre_llm_interval(profile)
    counter_key = session_id or "__no_session__"

    with _PRE_LLM_TURNS_LOCK:
        _PRE_LLM_TURNS[counter_key] += 1
        turn = _PRE_LLM_TURNS[counter_key]

    # N=1 fires every turn; otherwise fire on N, 2N, 3N, ...
    if turn % interval:
        return None

    timeout_seconds = HOOK_TIMEOUT_SECONDS
    tournament = profile.hypothesis_tournament
    if tournament is not None:
        timeout_seconds = max(timeout_seconds, tournament.timeout_seconds)
    return _invoke(
        "pre-llm.py",
        _payload(
            "pre_llm_call",
            session_id=session_id,
            parent_session_id=session_id,
            loam_turn=turn,
            loam_interval=interval,
            **payload_values(profile),
            **kwargs,
        ),
        timeout_seconds=timeout_seconds,
    )


def subagent_start(**kwargs: Any) -> None:
    profile = _profile()
    if profile is None:
        return
    values = dict(kwargs)
    values.update(payload_values(profile))
    child_session_id = values.get("child_session_id")
    child_subagent_id = values.get("child_subagent_id")
    if isinstance(child_session_id, str) and child_session_id and isinstance(child_subagent_id, str):
        with _CHILD_SUBAGENT_IDS_LOCK:
            _CHILD_SUBAGENT_IDS[child_session_id] = child_subagent_id
    _invoke("subagent-start.py", _payload("subagent_start", **values))


def subagent_stop(**kwargs: Any) -> None:
    profile = _profile()
    if profile is None:
        return
    # Hermes's current stop callback carries child_session_id but not the
    # stable child_subagent_id supplied at start. Recover the latter from this
    # process-local lifecycle map; use session ID only as a deterministic
    # fallback if the corresponding start observer failed or predates this map.
    values = dict(kwargs)
    values.update(payload_values(profile))
    child_session_id = values.get("child_session_id")
    if isinstance(child_session_id, str) and child_session_id:
        with _CHILD_SUBAGENT_IDS_LOCK:
            child_subagent_id = _CHILD_SUBAGENT_IDS.pop(child_session_id, "")
        values.setdefault("child_subagent_id", child_subagent_id or child_session_id)
    else:
        values.setdefault("child_subagent_id", "")
    _invoke("subagent-stop.py", _payload("subagent_stop", **values))


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("subagent_start", subagent_start)
    ctx.register_hook("subagent_stop", subagent_stop)
