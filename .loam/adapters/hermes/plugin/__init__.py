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
from loam_config import effective_profile, payload_values  # noqa: E402
import wandb_metrics  # noqa: E402

logger = logging.getLogger(__name__)

# Set in `register` once the metrics config is resolved. Gates the per-call
# token/tool observers so a profile without `metrics` pays nothing: the hooks
# are only registered, and the subagent role map only fed, when this is True.
_METRICS_ACTIVE = False

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
    if _METRICS_ACTIVE and isinstance(session_id, str) and session_id:
        # Stamp the tool's start so post_tool_call can measure the gap from the
        # triggering inference. Observer only; never affects the policy result.
        try:
            wandb_metrics.note_tool_start(session_id, str(kwargs.get("tool_call_id") or ""), str(kwargs.get("turn_id") or ""))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Loam metrics note_tool_start failed: %s", exc)
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
        canonical = canonical_delegation_args(_payload("tool_request", arguments=args, **values))
    except Exception as exc:
        logger.warning("Loam campaign delegation canonicalization failed: %s", exc)
        return None
    if canonical is None:
        return None
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
        "hypothesis-validator per selected hypothesis, then an independent "
        "evidence-reviewer per pending claim set, then the reasoner once those have "
        "reported. Takes no arguments -- the plan comes from persisted "
        "campaign state, not from you. Call this instead of composing a delegate_task "
        "batch yourself; you cannot see the hypotheses in full and transcribing them "
        "has failed before."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


RECORD_DECISION_TOOL_NAME = "loam_record_decision"
RECORD_DECISION_TOOL_SCHEMA: dict[str, Any] = {
    "name": RECORD_DECISION_TOOL_NAME,
    "description": (
        "Record ONE high-level decision in your workspace record, as you make it and "
        "BEFORE you act on it: the choice, why, the alternatives you rejected, the "
        "falsifiable outcome you expect, and your confidence. This is how the campaign "
        "keeps your reasoning -- your full transcript is not retained, and nothing else "
        "captures WHY you chose what you did. Call it when you commit to an approach, a "
        "tool or model, a tier size, or a go/stop; not for routine mechanical steps. "
        "It records into your own hypothesis workspace and returns once stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "what": {"type": "string", "description": "The decision, in one line."},
            "why": {"type": "string", "description": "The rationale for it."},
            "alternatives": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Options you considered and rejected, each with a brief reason.",
            },
            "expected": {
                "type": "string",
                "description": "The falsifiable result you predict if this decision is right.",
            },
            "confidence": {"type": "number", "description": "Your confidence, 0 to 1."},
        },
        "required": ["what", "why"],
    },
}


def record_decision(args: dict[str, Any], **kwargs: Any) -> str:
    """Write one declared decision into the calling agent's Pando workspace record.

    Recording happens here, in-process, rather than relying on pre_tool_call to
    fire for a plugin-registered tool: the handler already has the caller's
    session, and hypothesis_workspace.record_decision writes into the same
    flock-guarded run.json / experiment.json that captures every other call.
    """

    profile = _profile()
    if profile is None:
        return json.dumps({"error": "Loam is not active for this profile; decision not recorded."})
    what = str(args.get("what", "")).strip()
    if not what:
        return json.dumps({"error": "loam_record_decision needs a non-empty `what`."})
    values = dict(kwargs)
    values.update(payload_values(profile))
    values.setdefault("session_id", values.get("parent_session_id", ""))
    values.pop("arguments", None)
    try:
        if str(HOOKS_DIR) not in sys.path:
            sys.path.insert(0, str(HOOKS_DIR))
        from hypothesis_orchestration import workspace_decision

        stored = workspace_decision(_payload("tool_request", tool_name=RECORD_DECISION_TOOL_NAME,
                                             arguments=args, **values))
    except Exception as exc:
        logger.warning("Loam could not record a decision: %s", exc)
        return json.dumps({"error": f"Loam could not record the decision: {exc}"})
    if not stored:
        # Not an error the agent can fix: no open workspace run for this session
        # (e.g. the reasoner, whose record arrives in a later version). Say so
        # plainly rather than claim a record that was not written.
        return json.dumps({
            "status": "not_recorded",
            "detail": "No open workspace run is attributed to this session; the decision was not stored.",
        })
    return json.dumps({"status": "recorded", "what": what[:200]})


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

        canonical = canonical_delegation_args(_payload("tool_request", **values))
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
    return context.dispatch_tool("delegate_task", canonical)


def post_api_request(**kwargs: Any) -> None:
    """In-process token accounting for the W&B exporter. Observer; never raises.

    Fires after every LLM API call with the canonical `usage` buckets. Unlike
    the spawned lifecycle hooks this does no subprocess work: the metrics engine
    runs in-process because this event is on the hot path (see wandb_metrics).
    """

    if not _METRICS_ACTIVE:
        return
    try:
        wandb_metrics.record_api_request(kwargs)
    except Exception as exc:  # noqa: BLE001 - an observer must never affect a call
        logger.debug("Loam metrics post_api_request failed: %s", exc)


def post_tool_call(**kwargs: Any) -> None:
    """In-process per-tool-call wall-time accounting. Observer; never raises.

    Hermes supplies `duration_ms` (start of the call to its result), so the
    exporter records that rather than timing the call itself. The returned
    per-call record is also written into the tool call's Pando run workspace
    (code-generator children only), next to run.json.
    """

    if not _METRICS_ACTIVE:
        return
    try:
        record = wandb_metrics.record_tool_call(kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Loam metrics post_tool_call failed: %s", exc)
        return
    if record is not None:
        try:
            _store_tool_metrics_in_workspace(kwargs, record)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Loam metrics workspace store failed: %s", exc)


def _store_tool_metrics_in_workspace(kwargs: dict[str, Any], record: dict[str, Any]) -> None:
    """Append this tool call's metrics to its Pando run's metrics.jsonl.

    Resolves the call's run the same way pre-tool.py's workspace enforcement
    does -- its session's latest `agent_started` row -- then hands off to
    hypothesis_workspace, which no-ops for any role without a run folder. Only
    done when the per-hypothesis workspace is enabled.
    """

    profile = _profile()
    if profile is None or profile.workspace is None:
        return
    values = dict(kwargs)
    values.update(payload_values(profile))
    session_id = values.get("session_id")
    if isinstance(session_id, str) and session_id:
        values.setdefault("parent_session_id", session_id)
    payload = _payload("post_tool_call", **values)
    if str(HOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(HOOKS_DIR))
    from hypothesis_orchestration import _session_start
    import hypothesis_workspace

    start = _session_start(payload)
    if start is None:
        return
    hypothesis_workspace.record_tool_metrics(payload, start, record)


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
    if _METRICS_ACTIVE and isinstance(child_session_id, str) and child_session_id:
        # Attribute this child's later token/tool metrics to its Loam role.
        wandb_metrics.note_subagent_start(child_session_id, str(values.get("child_role") or ""))
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
    if _METRICS_ACTIVE and isinstance(child_session_id, str) and child_session_id:
        # The child's end-to-end wall time, recorded against its role.
        wandb_metrics.note_subagent_stop(
            child_session_id,
            str(values.get("child_role") or ""),
            values.get("duration_ms") or 0,
            str(values.get("child_status") or ""),
        )
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
    # Same toolset as the dispatch tool, for the same reason: `delegation` is on
    # wherever orchestration runs, so the tool is visible to every campaign agent
    # without a config toggle.
    ctx.register_tool(
        name=RECORD_DECISION_TOOL_NAME,
        toolset="delegation",
        schema=RECORD_DECISION_TOOL_SCHEMA,
        handler=record_decision,
        description=RECORD_DECISION_TOOL_SCHEMA["description"],
        emoji="📝",
    )
    ctx.register_middleware("tool_request", canonicalize_campaign_delegation)
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("subagent_start", subagent_start)
    ctx.register_hook("subagent_stop", subagent_stop)
    _register_metrics(ctx)


def _register_metrics(ctx: Any) -> None:
    """Turn on the in-process W&B exporter when the profile enables metrics.

    Kept apart from the lifecycle wiring above because it is observability, not
    campaign policy, and because its two hooks fire on the hot path: they are
    registered ONLY when `metrics` is on, so an unconfigured profile never pays
    the per-call cost of building their payloads. A malformed profile raises in
    loam_config; here that is swallowed so a metrics typo cannot fail plugin
    load (the loud failure still surfaces at preflight and in the other hooks).
    """

    global _METRICS_ACTIVE
    try:
        profile = _profile()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Loam metrics: profile load failed; exporter off (%s)", exc)
        return
    if profile is None or profile.metrics is None:
        return
    import os as _os

    wandb_metrics.configure(
        profile.metrics,
        state_dir=profile.state_dir,
        run_id=_os.environ.get("LOAM_RUN_ID", "") or profile.workflow_id,
        workflow_id=profile.workflow_id,
    )
    _METRICS_ACTIVE = True
    ctx.register_hook("post_api_request", post_api_request)
    ctx.register_hook("post_tool_call", post_tool_call)
