"""Parse the ``### Tool Ratings`` block that agents end their reports with.

Follows the discipline of ``_evidence_score`` in hypothesis_orchestration: the
block must sit under a real heading line (a prose mention is ignored), the LAST
such heading wins, and reading is bounded to a capped window.
"""

from __future__ import annotations

import json
import math
import re
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cairn_common  # noqa: E402
from cairn_flush import spawn_flush  # noqa: E402
from hypothesis_orchestration import hpc_tool_names  # noqa: E402

_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*(?:\*\*|__|`)?Tool[ \t]+Ratings:?(?:\*\*|__|`)?:?[ \t]*$", re.IGNORECASE | re.MULTILINE)
_NEXT_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]", re.MULTILINE)
_ENTRY_RE = re.compile(r"^[ \t]*[-*][ \t]+(.+)$", re.MULTILINE)
_NOTE_RE = re.compile(r"\|[ \t]*note[ \t]*:", re.IGNORECASE)
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})", re.MULTILINE)

_WINDOW_CHARS = 4000
_MAX_ENTRY_LINES = 20
_MAX_MODES = 10
_MAX_MODE_CHARS = 48
_MAX_METRICS = 10
_ORIGINS = frozenset({"none", "agent", "software", "infrastructure"})


@dataclass(frozen=True)
class RatingEntry:
    entity: str
    score: float
    origin: str
    failure_modes: tuple[str, ...]
    metrics: dict[str, float]
    note: str


def _failure_modes(value: str) -> tuple[str, ...]:
    if value.strip().lower() in ("", "none"):
        return ()
    modes: list[str] = []
    for raw in value.split(","):
        mode = cairn_common.snake_tag(cairn_common.snake_tag(raw)[:_MAX_MODE_CHARS])
        if mode and mode != "none":
            modes.append(mode)
        if len(modes) >= _MAX_MODES:
            break
    return tuple(modes)


def _metrics(value: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw in value.split(","):
        name, sep, number = raw.partition("=")
        name = name.strip().lower()
        if not sep or not cairn_common.METRIC_KEY_RE.fullmatch(name):
            continue
        try:
            parsed = float(number.strip())
        except ValueError:
            continue
        if not math.isfinite(parsed):
            continue
        metrics[name] = parsed
        if len(metrics) >= _MAX_METRICS:
            break
    return metrics


def _parse_line(text: str) -> RatingEntry | str:
    note = ""
    match = _NOTE_RE.search(text)
    if match:
        note = text[match.end():].strip()
        text = text[:match.start()]
    fields = text.split("|")
    entity = fields[0].strip().strip("`*_").strip()
    if not entity:
        return "rejected: no tool id"
    values: dict[str, str] = {}
    for part in fields[1:]:
        key, sep, value = part.partition(":")
        if sep:
            values.setdefault(key.strip().lower(), value.strip())
    try:
        score = float(values.get("score", ""))
    except ValueError:
        return f"rejected {entity}: missing or invalid score"
    if not 0.0 <= score <= 1.0:  # also false for NaN
        return f"rejected {entity}: score out of range"
    origin = values.get("origin", "").lower()
    if origin not in _ORIGINS:
        return f"rejected {entity}: invalid origin"
    return RatingEntry(
        entity=entity,
        score=score,
        origin=origin,
        failure_modes=_failure_modes(values.get("failure_modes", "")),
        metrics=_metrics(values.get("metrics", "")),
        note=note,
    )


def _fences(text: str) -> list[tuple[int, int]]:
    """Character spans of fenced code blocks (``` or ~~~); an unclosed fence runs to the end."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        opening = _FENCE_RE.search(text, pos)
        if not opening:
            return spans
        marker = opening.group(1)
        line_end = text.find("\n", opening.end())
        if line_end < 0:
            spans.append((opening.start(), len(text)))
            return spans
        closing = re.compile(rf"^[ \t]{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$", re.MULTILINE)
        found = closing.search(text, line_end + 1)
        if not found:
            spans.append((opening.start(), len(text)))
            return spans
        spans.append((opening.start(), found.end()))
        pos = found.end()


def parse_rating_block(summary: str) -> tuple[list[RatingEntry], list[str]]:
    """Return (valid entries, one error per rejected line); last line wins per entity."""

    summary = summary.replace("\r\n", "\n").replace("\r", "\n")
    fences = _fences(summary)

    def fenced(pos: int) -> bool:
        return any(start <= pos < end for start, end in fences)

    headings = [m for m in _HEADING_RE.finditer(summary) if not fenced(m.start())]
    if not headings:
        return [], []
    begin = headings[-1].end()
    end = begin + _WINDOW_CHARS
    stop = next((m.start() for m in _NEXT_HEADING_RE.finditer(summary, begin) if not fenced(m.start())), end)
    fence = next((start for start, _ in fences if start >= begin), end)
    window = summary[begin:min(end, stop, fence)]
    entries: dict[str, RatingEntry] = {}
    errors: list[str] = []
    for line in _ENTRY_RE.findall(window)[:_MAX_ENTRY_LINES]:
        parsed = _parse_line(line)
        if isinstance(parsed, str):
            errors.append(parsed)
        else:
            entries[parsed.entity] = parsed
    return list(entries.values()), errors


_TAG_RE = re.compile(r"<[^<>]{0,200}>")
_SEQUENCE_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{30,}", re.IGNORECASE)
_SECRET_RE = re.compile(r"\.cairn|api_key|x-api-key|\{\"", re.IGNORECASE)
_NOTE_CHARS = 1500


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_note(note: str) -> tuple[str, bool]:
    """Return (clean note, dropped). Notes that look like sequences or secrets are dropped whole."""

    if _SEQUENCE_RE.search(note) or _SECRET_RE.search(note):
        return "", True
    note = " ".join(_TAG_RE.sub("", note).split())
    return note[:_NOTE_CHARS], False


def _tags(spec: Mapping[str, Any]) -> list[str]:
    """The entity's task_tags in Cairn's tag shape (config already normalises; this is defence)."""
    raw = spec.get("task_tags")
    tags = [cairn_common.snake_tag(t) for t in raw if isinstance(t, str)] if isinstance(raw, list) else []
    return [t for t in tags if t][:10]


def _backed(role: str, entity: str, executor: str, patterns: list[str], calls: list[Mapping[str, Any]], completed: list[dict[str, Any]], plan: str, hyp: str) -> bool:
    if role == "code-generator":
        if not calls:
            return False
        if entity == executor:
            return True
        if not patterns:
            return False
        for call in calls:
            args = call.get("arguments")
            if isinstance(args, Mapping) and args.get("truncated") is True:
                continue
            blob = json.dumps(args, default=str).lower()
            if any(pattern in blob for pattern in patterns):
                return True
        return False
    done = [r for r in completed if _text(r.get("plan_id")) == plan and _text(r.get("status")) == "completed"]
    if role == "hypothesis-validator":
        return any(_text(r.get("hypothesis_id")) == hyp for r in done)
    return bool(done)


def record_ratings(payload: Mapping[str, Any]) -> None:
    """subagent-stop step: turn a stopped child's own rating block into queue rows."""

    cfg = cairn_common.config(payload)
    state = cairn_common.state_dir(payload)
    if cfg is None or state is None:
        return
    orchestration = cairn_common.read_jsonl(state / "hypothesis-orchestration.jsonl")
    child_id = _text(payload.get("child_subagent_id"))
    session = _text(payload.get("child_session_id"))
    start = next(
        (
            row
            for row in reversed(orchestration)
            if row.get("event") == "agent_started"
            and (_text(row.get("child_subagent_id")) == child_id or _text(row.get("child_session_id")) == session)
        ),
        None,
    )
    if start is None:
        return
    role = _text(start.get("role"))
    plan = _text(start.get("plan_id"))
    hyp = _text(start.get("hypothesis_id"))
    status = _text(payload.get("child_status")) or "error"

    names = hpc_tool_names(payload.get("loam_execution"))
    # Only execution calls (run_code / run_batch) are evidence of a tool actually being used.
    hpc = {names[f] for f in ("run_code", "run_batch") if f in names}
    calls = [
        row
        for row in cairn_common.read_jsonl(state / "tool-calls.jsonl")
        if _text(row.get("session_id")) == session and _text(row.get("tool_name")) in hpc
    ]
    entities = cfg.get("entities") if isinstance(cfg.get("entities"), Mapping) else {}
    executor = _text(cfg.get("executor"))
    completed = [r for r in orchestration if r.get("event") == "code_generator_completed"]

    out: list[dict[str, Any]] = []

    def row(row_status: str, reason: str, source: str, event: dict[str, Any] | None, entity: str = "") -> None:
        out.append({
            "event_id": uuid.uuid4().hex,
            "queued_at": _now(),
            "status": row_status,
            "reason": reason,
            "role": role,
            "source": source,
            "plan_id": plan,
            "hypothesis_id": hyp,
            "run_id": _text(payload.get("loam_run_id")),
            "workflow_id": _text(payload.get("workflow_id")),
            "child_session_id": session,
            "event": event,
            **({"entity": entity[:200]} if entity else {}),
        })

    rates = role in (cfg.get("rate_roles") or []) and bool(cfg.get("rate_agent_reports"))
    entries: list[RatingEntry] = []
    configured = 0
    if rates:
        entries, errors = parse_rating_block(str(payload.get("child_summary") or ""))
        for error in errors:
            row("invalid", error, "agent", None)
        for entry in entries:
            known = entities.get(entry.entity)
            if not isinstance(known, Mapping):
                row("invalid", "unknown_entity", "agent", None, entry.entity)
                continue
            configured += 1
            modes = [cairn_common.snake_tag(f"{entry.origin}_{m}") for m in entry.failure_modes]
            if not modes and entry.origin != "none":
                modes = [f"{entry.origin}_unspecified"]
            note, dropped = _sanitize_note(entry.note)
            event: dict[str, Any] = {
                "reviewee": {"type": "capability", "external_id": entry.entity},
                "context": "general",
                "score": entry.score,
                "weight": cairn_common.ROLE_WEIGHTS[role],
                "rationale": f"[{role}, agent-authored] " + note,
                "task_tags": _tags(known),
                "failure_modes": modes[:10],
                "observed_at": _now(),
            }
            if entry.metrics:
                event["metrics"] = dict(entry.metrics)
            reason = "note_dropped" if dropped else ""
            if entry.origin == "agent":
                row("suppressed_agent_error", reason, "agent", event)
                continue
            patterns = [p for p in known.get("patterns", []) if isinstance(p, str)]
            ok = _backed(role, entry.entity, executor, patterns, calls, completed, plan, hyp)
            row("queued" if ok else "unbacked", reason, "agent", event)

    if role == "code-generator" and cfg.get("rate_hook_fallback") and not configured:
        # Mirrors ledger._observation_status: completed -> observed,
        # interrupted/cancelled -> excluded, anything else -> failed.
        if status == "completed":
            if rates:
                row("missing_block", "no_valid_ratings", "hook", None)
        elif status not in ("interrupted", "cancelled"):
            if not calls:
                row("no_tool_used", f"status_{status}", "hook", None)
            elif isinstance(entities.get(executor), Mapping):
                tags = _tags(entities[executor])
                row("queued", "", "hook", {
                    "reviewee": {"type": "capability", "external_id": executor},
                    "context": "general",
                    "score": 0.2,
                    "weight": cairn_common.FALLBACK_WEIGHT,
                    "rationale": f"[code-generator, loam-hook fallback] child ended with status {status} after {len(calls)} HPC call(s)",
                    "task_tags": tags,
                    "failure_modes": [cairn_common.snake_tag(f"lifecycle_{status}")],
                    "observed_at": _now(),
                })

    if not out:
        return
    queue = state / cairn_common.QUEUE
    for item in out:
        try:
            cairn_common.append_jsonl(queue, item)
        except OSError as exc:
            print(f"loam cairn: queue append failed: {exc}", file=sys.stderr)
    if any(item["status"] == "queued" for item in out):
        spawn_flush(payload)
