"""In-process Weights & Biases exporter for Loam campaign metrics.

This module is the engine behind the `metrics:` profile block. It is a Hermes
*in-process* integration, not a spawned lifecycle hook: `post_api_request` and
`post_tool_call` fire on every API call and every tool call, so routing each
through a subprocess (the pattern the four `.loam/hooks/` scripts use) would tax
the hot path unacceptably. The Hermes `observability/langfuse` plugin traces the
same events in-process for the same reason; this is that shape, aimed at W&B and
with Loam's agent-role attribution added.

What it records, into ONE W&B run per campaign (keyed by `LOAM_RUN_ID` so a
resumed process rejoins the same run):

  * tokens per agent -- from `post_api_request`'s canonical `usage` buckets,
    attributed to the Loam role (reasoner / hypothesis-validator /
    code-generator / coordinator) that owns the call's session, and
  * wall time per tool call -- from `post_tool_call`'s `duration_ms`.

Both land as per-call rows in an event table AND as cumulative scalar
aggregates (live charts), plus end-of-run summary tables for per-agent tokens
and per-tool time. Every event is also appended to a local JSONL under the
campaign state dir, so the record survives even when W&B is unreachable.

Runtime failure is always swallowed. A missing `wandb`, an absent API key, or
an unreachable host leaves the exporter inert; it never raises into the hook
caller and so can never slow or break a running campaign. (Configuration errors
are the opposite -- loud -- but those are raised earlier, in loam_config.py.)
"""

from __future__ import annotations

import atexit
import json
import logging
import netrc
import os
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Guards every piece of module state below: the hooks can fire from the
# coordinator thread and from delegated-child threads in the same process.
_LOCK = threading.RLock()

# Resolved MetricsConfig, or None when metrics are off. Set by `configure`.
_CONFIG: Any | None = None
_STATE_DIR: Path | None = None
_RUN_ID: str = ""
_WORKFLOW_ID: str = ""

# Sentinel mirroring langfuse's: "init was attempted and failed; stop trying".
_INIT_FAILED = object()
_RUN: Any = None  # the live wandb run, or _INIT_FAILED, or None (untried)
_WANDB: Any = None
_FINISHED = False
_EVENTS_FILE: Any = None  # append handle for the local JSONL fallback

# child_session_id -> Loam agent role. Populated at subagent_start. A session
# not in this map is the top-level coordinator.
_SESSION_ROLE: dict[str, str] = {}
COORDINATOR_ROLE = "coordinator"

# Cumulative aggregates.
_TOKEN_BUCKETS = ("input", "output", "cache_read", "cache_write", "reasoning")
_tokens_by_role: dict[str, dict[str, int]] = {}
_llm_calls_by_role: dict[str, int] = {}
_tool_by_role: dict[str, dict[str, int]] = {}
_agent_wall_ms: dict[str, int] = {}
# {count, time_ms, errors, inference_ms, gap_ms} per tool; the last two sum the
# matched-inference duration and inference->tool gap so means are derivable.
_tool_by_name: dict[str, dict[str, int]] = {}

# The inference that most recently finished in each session, kept so the tool
# call it triggers can be linked back to it (duration, tokens) and the gap
# between the inference ending and the tool starting can be measured. Also
# indexed by api_request_id for an exact match when the tool call carries one.
_last_inference_by_session: dict[str, dict[str, Any]] = {}
_inference_by_request: dict[str, dict[str, Any]] = {}
_MAX_INFERENCE_INDEX = 512
# Wall-clock start of a tool call, stamped at pre_tool_call and consumed at
# post_tool_call, keyed (session_id, tool_call_id). Absent -> post_tool_call
# approximates the start as its own arrival time minus Hermes' duration_ms.
_tool_start: dict[tuple[str, str], float] = {}

# Per-call event rows kept in memory to (re)build the W&B table. Bounded: the
# durable full log is the JSONL on disk, so dropping the oldest in-memory rows
# past the cap only trims the convenience table, never the record.
_EVENT_COLUMNS = (
    "ts",
    "kind",
    "agent_role",
    "session_id",
    "turn_id",
    "model",
    "tool_name",
    "status",
    "duration_ms",
    "inference_ms",
    "gap_ms",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "total_tokens",
)
_MAX_TABLE_ROWS = 50_000
_event_rows: list[list[Any]] = []
_event_count = 0
_step = 0
_rows_since_table_log = 0


def configure(config: Any, *, state_dir: Any, run_id: str = "", workflow_id: str = "") -> None:
    """Record the resolved metrics config for later lazy W&B init.

    Called once from the adapter's `register`. Passing a falsy/disabled config
    (or mode `disabled`) leaves the exporter inert. Does not touch wandb yet --
    init is deferred to the first real event so an idle campaign pays nothing.
    """

    global _CONFIG, _STATE_DIR, _RUN_ID, _WORKFLOW_ID
    with _LOCK:
        if config is None or not getattr(config, "enabled", False) or getattr(config, "mode", "") == "disabled":
            _CONFIG = None
            return
        _CONFIG = config
        _STATE_DIR = Path(str(state_dir)).expanduser() if state_dir else None
        _RUN_ID = str(run_id or "")
        _WORKFLOW_ID = str(workflow_id or "")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _role_for_session(session_id: str) -> str:
    if session_id and session_id in _SESSION_ROLE:
        return _SESSION_ROLE[session_id]
    return COORDINATOR_ROLE


def note_subagent_start(session_id: str, role: str) -> None:
    """Map a delegated child's session to its Loam role for later attribution."""

    if not session_id:
        return
    with _LOCK:
        _SESSION_ROLE[session_id] = role or "subagent"


def note_subagent_stop(session_id: str, role: str, duration_ms: int, status: str) -> None:
    """Record a child's end-to-end wall time against its role.

    Attribution prefers the role recorded at start (authoritative) over the one
    on the stop payload, then falls back to the stop payload, then "subagent".
    """

    with _LOCK:
        resolved = _SESSION_ROLE.get(session_id) or role or "subagent"
        try:
            ms = max(0, int(duration_ms or 0))
        except (TypeError, ValueError):
            ms = 0
        _agent_wall_ms[resolved] = _agent_wall_ms.get(resolved, 0) + ms
        run = _ensure_run()
        if run is None:
            return
        _log_scalars(
            {
                f"agent_wall_ms/by_agent/{resolved}": _agent_wall_ms[resolved],
                f"agent_wall_ms/last/{resolved}": ms,
            },
            _next_step(),
        )


# --- wandb lifecycle ------------------------------------------------------


def _has_online_credential(api_key_env: str, base_url: str = "") -> bool:
    """True when wandb can authenticate online without an interactive prompt.

    Online init hangs on a login prompt in a headless campaign when no
    credential is present, so the caller refuses rather than block. There are
    two credentials wandb will use on its own: a key in the configured env var
    (or the default WANDB_API_KEY), and an existing `wandb login`, which writes
    the host's key into ~/.netrc. Honour both -- otherwise a user who ran
    `wandb login` (the documented setup) would have to ALSO export the key just
    for Loam, and metrics would silently stay off until they did.
    """

    if os.environ.get(api_key_env, "").strip() or os.environ.get("WANDB_API_KEY", "").strip():
        return True
    host = "api.wandb.ai"
    if base_url:
        parsed = urllib.parse.urlparse(base_url if "//" in base_url else "//" + base_url)
        host = parsed.hostname or host
    try:
        auth = netrc.netrc().authenticators(host)
    except (OSError, netrc.NetrcParseError):
        return False
    # authenticators() -> (login, account, password); the key is the password.
    return bool(auth and auth[2])


def _ensure_run() -> Any:
    """Return the live wandb run, initializing it once. None when inert.

    Caller must hold _LOCK. Every failure path caches _INIT_FAILED so a broken
    or absent W&B never re-attempts init on each subsequent event.
    """

    global _RUN, _WANDB, _EVENTS_FILE

    if _RUN is _INIT_FAILED:
        return None
    if _RUN is not None:
        return _RUN
    if _CONFIG is None:
        _RUN = _INIT_FAILED
        return None

    try:
        _open_events_file()
    except Exception as exc:  # noqa: BLE001 - durability is best-effort too
        logger.debug("loam metrics: could not open local events log: %s", exc)

    try:
        import wandb  # type: ignore
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.info("loam metrics: wandb not importable; metrics export is inert (%s)", exc)
        _RUN = _INIT_FAILED
        return None

    mode = getattr(_CONFIG, "mode", "online")
    api_key_env = getattr(_CONFIG, "api_key_env", "WANDB_API_KEY")
    base_url = getattr(_CONFIG, "base_url", "")
    if mode == "online" and not _has_online_credential(api_key_env, base_url):
        # Online with no credential would hang on an interactive login prompt in
        # a non-interactive campaign. Refuse quietly rather than block the run.
        logger.warning(
            "loam metrics: mode=online but no W&B credential found; metrics export "
            "is inert. Set %s (or WANDB_API_KEY), run `wandb login`, or use "
            "metrics.mode: offline.",
            api_key_env,
        )
        _RUN = _INIT_FAILED
        return None

    if base_url:
        os.environ.setdefault("WANDB_BASE_URL", base_url)
    # Never let wandb spawn an interactive prompt from inside the harness.
    os.environ.setdefault("WANDB_SILENT", "true")

    try:
        init_kwargs: dict[str, Any] = {
            "project": getattr(_CONFIG, "project", "") or "loam",
            "mode": "offline" if mode == "offline" else "online",
            "job_type": "loam-campaign",
            "tags": list(getattr(_CONFIG, "tags", ()) or ()),
            "config": {
                "loam_run_id": _RUN_ID,
                "workflow_id": _WORKFLOW_ID,
            },
            "settings": wandb.Settings(silent=True) if hasattr(wandb, "Settings") else None,
        }
        # Keep W&B's local run files inside the campaign state dir rather than
        # littering wherever the harness was launched (the repo root, say). The
        # dir must exist; wandb writes a `wandb/` subfolder under it.
        if _STATE_DIR is not None:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            init_kwargs["dir"] = str(_STATE_DIR)
        if getattr(_CONFIG, "entity", ""):
            init_kwargs["entity"] = _CONFIG.entity
        if getattr(_CONFIG, "group", ""):
            init_kwargs["group"] = _CONFIG.group
        name = getattr(_CONFIG, "run_name", "") or _WORKFLOW_ID or _RUN_ID
        if name:
            init_kwargs["name"] = name
        # id + resume keep a restarted process (walltime resume is a Loam
        # reality) logging into the same run instead of forking a new one.
        if _RUN_ID:
            init_kwargs["id"] = _wandb_safe_id(_RUN_ID)
            init_kwargs["resume"] = "allow"
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        run = wandb.init(**init_kwargs)
    except Exception as exc:  # noqa: BLE001 - never break the campaign for a dashboard
        logger.warning("loam metrics: wandb.init failed; metrics export is inert (%s)", exc)
        _RUN = _INIT_FAILED
        return None

    _WANDB = wandb
    _RUN = run
    atexit.register(finish)
    logger.info("loam metrics: W&B run live (project=%s, mode=%s)", init_kwargs.get("project"), mode)
    return _RUN


def _wandb_safe_id(run_id: str) -> str:
    """W&B run ids may not contain '/'; keep them otherwise intact and bounded."""

    cleaned = run_id.replace("/", "_").strip()
    return cleaned[:128] or "loam-run"


def _open_events_file() -> None:
    global _EVENTS_FILE
    if _EVENTS_FILE is not None or _STATE_DIR is None:
        return
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _STATE_DIR / "metrics-events.jsonl"
    _EVENTS_FILE = path.open("a", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _next_step() -> int:
    global _step
    _step += 1
    return _step


def _log_scalars(data: Mapping[str, Any], step: int) -> None:
    """Thin wrapper so every wandb.log is wrapped in the same failure swallow."""

    if _RUN is None or _RUN is _INIT_FAILED:
        return
    try:
        _RUN.log(dict(data), step=step)
    except Exception as exc:  # noqa: BLE001
        logger.debug("loam metrics: wandb.log failed: %s", exc)


# --- event recording ------------------------------------------------------


def _append_jsonl(record: Mapping[str, Any]) -> None:
    # The local log is the durable record and must not depend on W&B init, so
    # the handle is opened here (idempotently) rather than only in _ensure_run.
    if _EVENTS_FILE is None:
        try:
            _open_events_file()
        except Exception as exc:  # noqa: BLE001
            logger.debug("loam metrics: could not open local events log: %s", exc)
    if _EVENTS_FILE is None:
        return
    try:
        _EVENTS_FILE.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        _EVENTS_FILE.flush()
    except Exception as exc:  # noqa: BLE001
        logger.debug("loam metrics: local events append failed: %s", exc)


def _push_row(row: Mapping[str, Any]) -> None:
    """Append one row to the in-memory table and (throttled) re-log the table."""

    global _event_count, _rows_since_table_log
    _event_rows.append([row.get(col) for col in _EVENT_COLUMNS])
    if len(_event_rows) > _MAX_TABLE_ROWS:
        del _event_rows[: len(_event_rows) - _MAX_TABLE_ROWS]
    _event_count += 1
    _rows_since_table_log += 1
    every = max(1, int(getattr(_CONFIG, "table_log_every", 50) or 50))
    if _rows_since_table_log >= every:
        _log_event_table()


def _log_event_table() -> None:
    global _rows_since_table_log
    if _RUN is None or _RUN is _INIT_FAILED or _WANDB is None:
        return
    try:
        table = _WANDB.Table(columns=list(_EVENT_COLUMNS), data=[list(r) for r in _event_rows])
        _RUN.log({"events": table}, step=_step)
        _rows_since_table_log = 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("loam metrics: event table log failed: %s", exc)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def record_api_request(payload: Mapping[str, Any]) -> None:
    """Account one LLM API call's token usage against the owning agent role.

    `payload['usage']` is Hermes' already-canonical bucket dict (input_tokens,
    output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,
    prompt_tokens, total_tokens). Missing or empty usage still counts the call.
    """

    with _LOCK:
        if _CONFIG is None:
            return
        session_id = str(payload.get("session_id") or "")
        role = _role_for_session(session_id)
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        buckets = {
            "input": _int(usage.get("input_tokens")),
            "output": _int(usage.get("output_tokens")),
            "cache_read": _int(usage.get("cache_read_tokens")),
            "cache_write": _int(usage.get("cache_write_tokens")),
            "reasoning": _int(usage.get("reasoning_tokens")),
        }
        total = _int(usage.get("total_tokens")) or (
            buckets["input"] + buckets["output"] + buckets["cache_read"] + buckets["cache_write"]
        )

        role_tokens = _tokens_by_role.setdefault(role, {b: 0 for b in _TOKEN_BUCKETS} | {"total": 0})
        for b in _TOKEN_BUCKETS:
            role_tokens[b] += buckets[b]
        role_tokens["total"] += total
        _llm_calls_by_role[role] = _llm_calls_by_role.get(role, 0) + 1

        api_request_id = str(payload.get("api_request_id") or "")
        turn_id = str(payload.get("turn_id") or "")
        inference_ms = round(float(payload.get("api_duration") or 0.0) * 1000.0, 3)
        # `ended_at` is the harness wall clock (time.time()) at which this
        # inference finished; the tool call it triggers measures its gap from
        # here. Fall back to now when the harness did not supply it.
        ended_at = payload.get("ended_at")
        try:
            ended_at = float(ended_at) if ended_at is not None else time.time()
        except (TypeError, ValueError):
            ended_at = time.time()
        inference = {
            "api_request_id": api_request_id,
            "turn_id": turn_id,
            "ended_at": ended_at,
            "duration_ms": inference_ms,
            "model": str(payload.get("model") or ""),
            "tokens": {
                "input": buckets["input"],
                "output": buckets["output"],
                "cache_read": buckets["cache_read"],
                "cache_write": buckets["cache_write"],
                "reasoning": buckets["reasoning"],
                "total": total,
            },
        }
        _last_inference_by_session[session_id] = inference
        if api_request_id:
            _inference_by_request[api_request_id] = inference
            if len(_inference_by_request) > _MAX_INFERENCE_INDEX:
                # Drop the oldest entries; dict preserves insertion order.
                for stale in list(_inference_by_request)[: len(_inference_by_request) - _MAX_INFERENCE_INDEX]:
                    _inference_by_request.pop(stale, None)

        record = {
            "record_version": "loam_metrics_event_v1",
            "kind": "api_request",
            "ts": _timestamp(),
            "loam_run_id": _RUN_ID,
            "workflow_id": _WORKFLOW_ID,
            "agent_role": role,
            "session_id": session_id,
            "turn_id": turn_id,
            "api_request_id": api_request_id,
            "model": str(payload.get("model") or ""),
            "provider": str(payload.get("provider") or ""),
            "inference_ms": inference_ms,
            "input_tokens": buckets["input"],
            "output_tokens": buckets["output"],
            "cache_read_tokens": buckets["cache_read"],
            "cache_write_tokens": buckets["cache_write"],
            "reasoning_tokens": buckets["reasoning"],
            "total_tokens": total,
        }
        _append_jsonl(record)

        if _ensure_run() is None:
            return
        step = _next_step()
        scalars: dict[str, Any] = {
            f"tokens/by_agent/{role}/total": role_tokens["total"],
            f"tokens/by_agent/{role}/input": role_tokens["input"],
            f"tokens/by_agent/{role}/output": role_tokens["output"],
            f"llm_calls/by_agent/{role}": _llm_calls_by_role[role],
            "tokens/all/total": sum(t["total"] for t in _tokens_by_role.values()),
            "tokens/all/input": sum(t["input"] for t in _tokens_by_role.values()),
            "tokens/all/output": sum(t["output"] for t in _tokens_by_role.values()),
            "tokens/all/reasoning": sum(t["reasoning"] for t in _tokens_by_role.values()),
            "llm_calls/all": sum(_llm_calls_by_role.values()),
            "inference_ms/last": inference_ms,
            f"inference_ms/by_agent/{role}/last": inference_ms,
        }
        _log_scalars(scalars, step)
        _push_row(
            {
                "ts": record["ts"],
                "kind": "api_request",
                "agent_role": role,
                "session_id": session_id,
                "turn_id": turn_id,
                "model": record["model"],
                "tool_name": "",
                "status": "",
                "duration_ms": None,
                "inference_ms": inference_ms,
                "gap_ms": None,
                "input_tokens": buckets["input"],
                "output_tokens": buckets["output"],
                "cache_read_tokens": buckets["cache_read"],
                "cache_write_tokens": buckets["cache_write"],
                "reasoning_tokens": buckets["reasoning"],
                "total_tokens": total,
            }
        )


def note_tool_start(session_id: str, tool_call_id: str, turn_id: str = "") -> None:
    """Stamp the wall-clock start of a tool call, read back at post_tool_call.

    Called from pre_tool_call, which fires immediately before the tool runs, so
    the inference->tool gap is measured from the real start rather than
    inferred from the post-call arrival time.
    """

    if _CONFIG is None or not session_id:
        return
    with _LOCK:
        _tool_start[(session_id, str(tool_call_id or ""))] = time.time()


def _matched_inference(session_id: str, api_request_id: str) -> dict[str, Any] | None:
    """The inference that triggered this tool call: exact by request id, else last."""

    if api_request_id and api_request_id in _inference_by_request:
        return _inference_by_request[api_request_id]
    return _last_inference_by_session.get(session_id)


def record_tool_call(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Account one tool call and return its per-call metrics record (or None).

    The wall time (start -> result) is Hermes' own `duration_ms`. The call is
    linked to the inference that triggered it (matched by api_request_id, else
    the session's last inference), carrying that inference's duration and token
    usage, and the gap between the inference ending and this tool starting. The
    returned record is what the caller writes into the Pando run workspace.
    """

    with _LOCK:
        if _CONFIG is None:
            return None
        session_id = str(payload.get("session_id") or "")
        role = _role_for_session(session_id)
        tool_name = str(payload.get("tool_name") or "unknown")
        status = str(payload.get("status") or "")
        tool_call_id = str(payload.get("tool_call_id") or "")
        turn_id = str(payload.get("turn_id") or "")
        api_request_id = str(payload.get("api_request_id") or "")
        ms = max(0, _int(payload.get("duration_ms")))
        is_error = bool(status) and status not in {"ok", "success", "completed", ""}

        # Resolve the tool's start time (stamped at pre_tool_call, else inferred
        # from arrival minus duration) and the gap since its inference ended.
        now = time.time()
        tool_start = _tool_start.pop((session_id, tool_call_id), None)
        if tool_start is None:
            tool_start = now - ms / 1000.0
        inference = _matched_inference(session_id, api_request_id)
        gap_ms: float | None = None
        inference_ms: float | None = None
        inf_tokens: dict[str, int] = {}
        if inference is not None:
            inference_ms = inference.get("duration_ms")
            inf_tokens = inference.get("tokens", {})
            ended_at = inference.get("ended_at")
            if isinstance(ended_at, (int, float)):
                # Clamp tiny negatives from clock granularity to 0.
                gap_ms = round(max(0.0, (tool_start - ended_at) * 1000.0), 3)

        by_name = _tool_by_name.setdefault(
            tool_name, {"count": 0, "time_ms": 0, "errors": 0, "gap_ms": 0, "inference_ms": 0}
        )
        by_name["count"] += 1
        by_name["time_ms"] += ms
        by_name["errors"] += 1 if is_error else 0
        by_name["gap_ms"] += int(gap_ms or 0)
        by_name["inference_ms"] += int(inference_ms or 0)
        by_role = _tool_by_role.setdefault(role, {"count": 0, "time_ms": 0})
        by_role["count"] += 1
        by_role["time_ms"] += ms

        record = {
            "record_version": "loam_metrics_event_v1",
            "kind": "tool_call",
            "ts": _timestamp(),
            "loam_run_id": _RUN_ID,
            "workflow_id": _WORKFLOW_ID,
            "agent_role": role,
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "status": status,
            "error_type": str(payload.get("error_type") or ""),
            # the five requested quantities
            "duration_ms": ms,
            "inference": {
                "api_request_id": api_request_id or (inference or {}).get("api_request_id", ""),
                "duration_ms": inference_ms,
                "tokens": dict(inf_tokens),
            },
            "gap_from_inference_ms": gap_ms,
        }
        _append_jsonl(record)

        if _ensure_run() is None:
            return record
        step = _next_step()
        mean_ms = by_name["time_ms"] / by_name["count"] if by_name["count"] else 0
        scalars: dict[str, Any] = {
            f"tool_time_ms/by_tool/{tool_name}/total": by_name["time_ms"],
            f"tool_time_ms/by_tool/{tool_name}/mean": mean_ms,
            f"tool_calls/by_tool/{tool_name}/count": by_name["count"],
            f"tool_calls/by_tool/{tool_name}/errors": by_name["errors"],
            f"tool_time_ms/by_agent/{role}/total": by_role["time_ms"],
            f"tool_calls/by_agent/{role}/count": by_role["count"],
            "tool_calls/all": sum(t["count"] for t in _tool_by_name.values()),
            "tool_time_ms/all": sum(t["time_ms"] for t in _tool_by_name.values()),
            "tool_time_ms/last": ms,
        }
        if gap_ms is not None:
            scalars["gap_ms/last"] = gap_ms
            scalars[f"gap_ms/by_tool/{tool_name}/last"] = gap_ms
        if inference_ms is not None:
            scalars["tool_inference_ms/last"] = inference_ms
        _log_scalars(scalars, step)
        _push_row(
            {
                "ts": record["ts"],
                "kind": "tool_call",
                "agent_role": role,
                "session_id": session_id,
                "turn_id": turn_id,
                "model": "",
                "tool_name": tool_name,
                "status": status,
                "duration_ms": ms,
                "inference_ms": inference_ms,
                "gap_ms": gap_ms,
                "input_tokens": inf_tokens.get("input"),
                "output_tokens": inf_tokens.get("output"),
                "cache_read_tokens": inf_tokens.get("cache_read"),
                "cache_write_tokens": inf_tokens.get("cache_write"),
                "reasoning_tokens": inf_tokens.get("reasoning"),
                "total_tokens": inf_tokens.get("total"),
            }
        )
        return record


# --- finalization ---------------------------------------------------------


def _summary_tables() -> None:
    if _RUN is None or _RUN is _INIT_FAILED or _WANDB is None:
        return
    try:
        agent_cols = [
            "agent_role",
            "llm_calls",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "total_tokens",
            "tool_calls",
            "tool_time_ms",
            "wall_ms",
        ]
        roles = set(_tokens_by_role) | set(_tool_by_role) | set(_agent_wall_ms) | set(_llm_calls_by_role)
        agent_rows = []
        for role in sorted(roles):
            tok = _tokens_by_role.get(role, {})
            trole = _tool_by_role.get(role, {})
            agent_rows.append(
                [
                    role,
                    _llm_calls_by_role.get(role, 0),
                    tok.get("input", 0),
                    tok.get("output", 0),
                    tok.get("cache_read", 0),
                    tok.get("cache_write", 0),
                    tok.get("reasoning", 0),
                    tok.get("total", 0),
                    trole.get("count", 0),
                    trole.get("time_ms", 0),
                    _agent_wall_ms.get(role, 0),
                ]
            )
        _RUN.log({"summary/agents": _WANDB.Table(columns=agent_cols, data=agent_rows)}, step=_step)

        tool_cols = ["tool_name", "count", "total_ms", "mean_ms", "errors", "mean_gap_ms", "mean_inference_ms"]
        tool_rows = [
            [
                name,
                v["count"],
                v["time_ms"],
                (v["time_ms"] / v["count"]) if v["count"] else 0,
                v["errors"],
                (v.get("gap_ms", 0) / v["count"]) if v["count"] else 0,
                (v.get("inference_ms", 0) / v["count"]) if v["count"] else 0,
            ]
            for name, v in sorted(_tool_by_name.items())
        ]
        _RUN.log({"summary/tools": _WANDB.Table(columns=tool_cols, data=tool_rows)}, step=_step)

        _RUN.summary["tokens_total"] = sum(t["total"] for t in _tokens_by_role.values())
        _RUN.summary["llm_calls_total"] = sum(_llm_calls_by_role.values())
        _RUN.summary["tool_calls_total"] = sum(t["count"] for t in _tool_by_name.values())
        _RUN.summary["tool_time_ms_total"] = sum(t["time_ms"] for t in _tool_by_name.values())
    except Exception as exc:  # noqa: BLE001
        logger.debug("loam metrics: summary tables failed: %s", exc)


def finish() -> None:
    """Flush final tables and close the W&B run. Idempotent; never raises."""

    global _FINISHED, _EVENTS_FILE
    with _LOCK:
        if _FINISHED:
            return
        _FINISHED = True
        if _RUN is not None and _RUN is not _INIT_FAILED:
            _log_event_table()
            _summary_tables()
            try:
                _RUN.finish()
            except Exception as exc:  # noqa: BLE001
                logger.debug("loam metrics: wandb.finish failed: %s", exc)
        if _EVENTS_FILE is not None:
            try:
                _EVENTS_FILE.close()
            except Exception:  # noqa: BLE001
                pass
            _EVENTS_FILE = None


def _reset_for_tests() -> None:
    """Restore module state between unit tests. Not used at runtime."""

    global _CONFIG, _STATE_DIR, _RUN_ID, _WORKFLOW_ID, _RUN, _WANDB, _FINISHED
    global _EVENTS_FILE, _event_count, _step, _rows_since_table_log
    with _LOCK:
        _CONFIG = None
        _STATE_DIR = None
        _RUN_ID = ""
        _WORKFLOW_ID = ""
        _RUN = None
        _WANDB = None
        _FINISHED = False
        if _EVENTS_FILE is not None:
            try:
                _EVENTS_FILE.close()
            except Exception:  # noqa: BLE001
                pass
        _EVENTS_FILE = None
        _SESSION_ROLE.clear()
        _tokens_by_role.clear()
        _llm_calls_by_role.clear()
        _tool_by_name.clear()
        _tool_by_role.clear()
        _agent_wall_ms.clear()
        _last_inference_by_session.clear()
        _inference_by_request.clear()
        _tool_start.clear()
        _event_rows.clear()
        _event_count = 0
        _step = 0
        _rows_since_table_log = 0
