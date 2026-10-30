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

_ADAPTER_DIR = Path(__file__).resolve().parent
if str(_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_DIR))
from loam_config import effective_profile, payload_values  # noqa: E402

logger = logging.getLogger(__name__)

# `.loam/adapters/hermes/plugin/__init__.py` -> repository root
REPO_ROOT = Path(__file__).resolve().parents[4]
LOAM_ROOT = REPO_ROOT / ".loam"
HOOKS_DIR = LOAM_ROOT / "hooks"
HOOK_TIMEOUT_SECONDS = 15
_DEFAULT_PRE_LLM_INTERVAL = 5
# Set once by `register`. The dispatch tool needs it to reach
# `ctx.dispatch_tool`, which is the only supported way for a plugin to
# invoke delegate_task with the parent agent resolved.
_PLUGIN_CONTEXT: Any | None = None
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

    # Hermes may run from a managed runtime that lacks hook dependencies (Jnana needs openai, PyYAML,
    # ...). LOAM_HOOK_PYTHON names an interpreter that has them. Hermes exports a PYTHONPATH for its own
    # interpreter; a different interpreter must not inherit it, or it imports incompatible packages.
    override = os.environ.get("LOAM_HOOK_PYTHON", "").strip()
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}} if override else None
    try:
        completed = subprocess.run(
            [override or sys.executable, str(script)],
            env=env,
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


def canonicalize_campaign_delegation(tool_name: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any] | None:
    """Replace model-reconstructed campaign delegation with persisted canonical work."""

    if tool_name != "delegate_task":
        return None
    profile = _profile()
    if profile is None or not profile.hypothesis_orchestration:
        return None
    values = dict(kwargs)
    values.update(payload_values(profile))
    values.setdefault("session_id", values.get("parent_session_id", ""))
    values.pop("arguments", None)
    try:
        if str(HOOKS_DIR) not in sys.path:
            sys.path.insert(0, str(HOOKS_DIR))
        from hypothesis_orchestration import canonical_delegation_args

        # The model's own args are passed so a generated-mode validator's
        # instruction to its code-generator survives canonicalization.
        payload = _payload("tool_request", arguments=args, **values)
        canonical = canonical_delegation_args(payload)
    except Exception as exc:
        logger.warning("Loam campaign delegation canonicalization failed: %s", exc)
        return None
    if canonical is None:
        return None
    try:
        from cairn_brief import decorate_delegation

        canonical = decorate_delegation(canonical, payload)
    except Exception as exc:
        logger.warning("Loam Cairn brief skipped: %s", exc)
    return {
        "args": canonical,
        "source": "loam-campaign",
        "reason": "canonical persisted campaign delegation",
    }


DISPATCH_TOOL_NAME = "loam_dispatch_round"
DISPATCH_TOOL_SCHEMA: dict[str, Any] = {
    "name": DISPATCH_TOOL_NAME,
    "description": (
        "Launch the campaign work Loam has already planned for this round: one "
        "hypothesis-validator per selected hypothesis, or the reasoner once every "
        "validator has reported. Takes no arguments -- the plan comes from persisted "
        "campaign state, not from you. Call this instead of composing a delegate_task "
        "batch yourself; you cannot see the hypotheses in full and transcribing them "
        "has failed before."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def dispatch_round(args: dict[str, Any], **kwargs: Any) -> str:
    """Dispatch the round's planned delegation without the model transcribing it.

    The hook computes each round's delegation from persisted state, so the
    authoritative arguments exist before the model writes anything. Until now
    they were printed into the coordinator's context with instructions to copy
    them verbatim, and the `tool_request` middleware repaired the copy
    afterwards -- a repair that does not cover every state (see
    `canonical_delegation_args`). Calling this tool removes the transcription
    step rather than correcting it.

    Dispatch goes through `ctx.dispatch_tool`, which resolves the parent agent
    and is the documented way for a plugin to invoke `delegate_task`. That path
    reaches the tool registry directly, so neither the `tool_request` middleware
    nor `pre_tool_call` fires again on the call this makes: no double rewrite,
    and `tool_policy` cannot veto Loam's own dispatch.
    """

    context = _PLUGIN_CONTEXT
    if context is None:
        return json.dumps({"error": "Loam plugin context unavailable; cannot dispatch."})
    profile = _profile()
    if profile is None or not profile.hypothesis_orchestration:
        return json.dumps({"error": "Loam hypothesis orchestration is not enabled for this profile."})
    values = dict(kwargs)
    values.update(payload_values(profile))
    values.setdefault("session_id", values.get("parent_session_id", ""))
    values.pop("arguments", None)
    try:
        if str(HOOKS_DIR) not in sys.path:
            sys.path.insert(0, str(HOOKS_DIR))
        from hypothesis_orchestration import canonical_delegation_args

        payload = _payload("tool_request", **values)
        canonical = canonical_delegation_args(payload)
    except Exception as exc:
        logger.warning("Loam round dispatch could not compute the plan: %s", exc)
        return json.dumps({"error": f"Loam could not compute this round's plan: {exc}"})
    if canonical is None:
        # Not an error: no round is pending. Said plainly so the coordinator
        # reports rather than inventing work to fill the turn.
        return json.dumps(
            {
                "status": "nothing_to_dispatch",
                "detail": (
                    "No campaign delegation is pending. Either this round's children are "
                    "already running, or the tournament has selected nothing. Do not "
                    "substitute work of your own; report and wait."
                ),
            }
        )
    # The tool_request middleware never sees this call, so Cairn's brief is added here.
    try:
        from cairn_brief import decorate_delegation

        canonical = decorate_delegation(canonical, payload)
    except Exception as exc:
        logger.warning("Loam Cairn brief skipped: %s", exc)
    return context.dispatch_tool("delegate_task", canonical)


def pre_llm_call(session_id: str = "", **kwargs: Any) -> dict[str, Any] | None:
    """Run Loam's pre-LLM hook every N turns, independently per session."""
    # Newer Hermes passes its own `parent_session_id` (a subagent's parent, empty for top-level
    # sessions). Loam's contract uses the name for the coordinator session itself, so keep Loam's
    # meaning and carry Hermes's value separately.
    hermes_parent = kwargs.pop("parent_session_id", None)
    if hermes_parent:
        kwargs["hermes_parent_session_id"] = hermes_parent
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
        # The knowledge pre-pass runs inside the same hook, before generation.
        knowledge = tournament.knowledge or {}
        timeout_seconds = max(timeout_seconds,
                              tournament.timeout_seconds + int(knowledge.get("timeout_seconds") or 0))
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
    global _PLUGIN_CONTEXT
    _PLUGIN_CONTEXT = ctx
    # Registered into the existing `delegation` toolset rather than a new `loam`
    # one. A plugin's own toolset is configurable and OFF until it is toggled on
    # per platform, so a `loam` toolset would leave this tool invisible to the
    # coordinator until someone edited config -- the same silent-no-op footgun
    # as a profile with no campaign block. `delegation` is already enabled
    # wherever orchestration runs, because that is where delegate_task lives.
    ctx.register_tool(
        name=DISPATCH_TOOL_NAME,
        toolset="delegation",
        schema=DISPATCH_TOOL_SCHEMA,
        handler=dispatch_round,
        description=DISPATCH_TOOL_SCHEMA["description"],
        emoji="🌱",
    )
    ctx.register_middleware("tool_request", canonicalize_campaign_delegation)
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("subagent_start", subagent_start)
    ctx.register_hook("subagent_stop", subagent_stop)
