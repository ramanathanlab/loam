"""Append-only lifecycle ledger shared by Loam observer hooks.

The ledger is deliberately small and framework-neutral.  Hook scripts receive a
canonical JSON payload on stdin, translate it into a versioned record here, and
append one JSON object per line.  This is an observation boundary only: it does
not decide whether a subagent runs, interpret an LLM summary as evidence, or
modify Hermes scheduling.

The authoritative identity for a delegated child is Hermes's stable
``child_subagent_id``.  A deterministic Loam unit ID is derived from the parent
run/session, parent turn, and that child ID so records can later be joined without
using model-written task text as identity material.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hypothesis_provenance import (
    CONTRACT_VERSION as HYPOTHESIS_CONTRACT_VERSION,
    extract_contract_object as extract_hypothesis_contract,
    normalize_candidate as normalize_hypothesis_candidate,
    stable_id,
)

LEDGER_VERSION = "loam_subagent_ledger_v1"
DEFAULT_LEDGER_RELATIVE_PATH = ".loam/state/subagent-ledger.jsonl"


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Return JSON-compatible data without failing the observer hook."""

    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _stable_id(kind: str, **material: str) -> str:
    encoded = json.dumps(
        {"version": LEDGER_VERSION, "kind": kind, **material},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{kind}_{hashlib.sha256(encoded).hexdigest()}"


def ledger_path(payload: Mapping[str, Any]) -> Path:
    """Resolve state from the payload, runtime directory, then legacy overrides."""

    state_dir = _text(payload.get("loam_state_dir")) or os.environ.get("LOAM_STATE_DIR", "").strip()
    if state_dir:
        return Path(state_dir).expanduser() / "subagent-ledger.jsonl"
    configured = os.environ.get("LOAM_LEDGER_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    root = Path(_text(payload.get("repo_root")) or ".")
    return root / DEFAULT_LEDGER_RELATIVE_PATH


def build_start_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a pre-run allocation record from a ``subagent_start`` payload."""

    parent_session_id = _text(payload.get("parent_session_id"))
    parent_turn_id = _text(payload.get("parent_turn_id"))
    parent_subagent_id = _text(payload.get("parent_subagent_id"))
    child_session_id = _text(payload.get("child_session_id"))
    child_subagent_id = _text(payload.get("child_subagent_id"))
    unit_id = _stable_id(
        "unit",
        parent_session_id=parent_session_id,
        parent_turn_id=parent_turn_id,
        child_identity=child_subagent_id or child_session_id,
    )
    return {
        "record_version": LEDGER_VERSION,
        "event": "subagent_start",
        "observed_at": _timestamp(),
        "run_id": parent_session_id,
        "workflow_id": _text(payload.get("workflow_id")),
        "loam_profile": _text(payload.get("loam_profile")),
        "parent_session_id": parent_session_id,
        "parent_turn_id": parent_turn_id,
        "parent_subagent_id": parent_subagent_id,
        "child_session_id": child_session_id,
        "child_subagent_id": child_subagent_id,
        "unit_id": unit_id,
        "agent_role": _text(payload.get("child_role")),
        "task_goal": _text(payload.get("child_goal")),
        "status": "started",
    }


# A child whose own API calls failed still exits normally, so Hermes reports
# child_status "completed" and its summary is the provider's error. Trusting
# child_status alone therefore files an infrastructure failure as an observation
# that is eligible to carry findings. Observed 2026-10-31: five validators and
# five code-generators returned "API call failed after 3 retries: Connection
# error." and every one was recorded observed/completed, so a round advanced to
# its reasoner on validations that never ran.
#
# Matched only at the START of the summary, and only within a short one: a
# validator reporting that a provider failed mid-investigation is writing a real
# report that happens to quote the message, and must not be reclassified.
_CHILD_API_FAILURE = re.compile(
    r"\A\s*(?:API call failed after \d+ retries"
    r"|Upstream stream ended before completion"
    r"|Error code: 429\b)",
)
_CHILD_API_FAILURE_MAX_CHARS = 600


def _child_api_failed(child_summary: str) -> bool:
    """True when a summary is wholly a provider failure rather than a report."""

    summary = child_summary.strip()
    if len(summary) > _CHILD_API_FAILURE_MAX_CHARS:
        return False
    return bool(_CHILD_API_FAILURE.match(summary))


def _failure_class(child_status: str, child_summary: str = "") -> str:
    """Why a child produced no observation; empty when it genuinely completed."""

    if child_status == "completed":
        return "child_api_failure" if _child_api_failed(child_summary) else ""
    return child_status


def _observation_status(child_status: str, child_summary: str = "") -> str:
    if child_status == "completed":
        return "failed" if _child_api_failed(child_summary) else "observed"
    if child_status in {"interrupted", "cancelled"}:
        return "excluded"
    return "failed"


def build_stop_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a terminal observation record from a ``subagent_stop`` payload."""

    parent_session_id = _text(payload.get("parent_session_id"))
    parent_turn_id = _text(payload.get("parent_turn_id"))
    child_session_id = _text(payload.get("child_session_id"))
    child_subagent_id = _text(payload.get("child_subagent_id"))
    child_status = _text(payload.get("child_status")) or "error"
    unit_id = _stable_id(
        "unit",
        parent_session_id=parent_session_id,
        parent_turn_id=parent_turn_id,
        child_identity=child_subagent_id or child_session_id,
    )
    try:
        duration_ms = int(payload.get("duration_ms") or 0)
    except (TypeError, ValueError):
        duration_ms = 0
    return {
        "record_version": LEDGER_VERSION,
        "event": "subagent_stop",
        "observed_at": _timestamp(),
        "run_id": parent_session_id,
        "workflow_id": _text(payload.get("workflow_id")),
        "loam_profile": _text(payload.get("loam_profile")),
        "parent_session_id": parent_session_id,
        "parent_turn_id": parent_turn_id,
        "child_session_id": child_session_id,
        "child_subagent_id": child_subagent_id,
        "unit_id": unit_id,
        "agent_role": _text(payload.get("child_role")),
        "status": child_status,
        "observation_status": _observation_status(child_status, _text(payload.get("child_summary"))),
        # A child summary is unvalidated model output. Store it only as audit
        # material; v1 intentionally projects no findings, evidence, or credit.
        "child_summary": _text(payload.get("child_summary")),
        "finding_ids": [],
        "evidence_refs": [],
        "artifact_refs": [],
        "cost": {"elapsed_ms": max(0, duration_ms)},
        "failure_class": _failure_class(child_status, _text(payload.get("child_summary"))),
    }


def candidate_path(payload: Mapping[str, Any]) -> Path:
    """Store unvalidated contract candidates next to their lifecycle ledger."""

    return ledger_path(payload).with_name("candidate-results.jsonl")


def build_candidate_records(stop_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract opt-in pending candidates from a completed child without accepting them."""

    if stop_record.get("observation_status") != "observed":
        return []
    raw_summary = _text(stop_record.get("child_summary"))
    candidates = []
    # One contract, kept as a loop because the shape is the extension point: a
    # campaign that needs a second claim type adds an (version, extractor,
    # normalizer) triple rather than restructuring this function. The protein
    # metadata pilot was the first entry and was removed once it was clear its
    # claim vocabulary (a record's name, gene, sequence length) could not express
    # what a campaign measures.
    for version, extractor, normalizer in (
        (HYPOTHESIS_CONTRACT_VERSION, extract_hypothesis_contract, normalize_hypothesis_candidate),
    ):
        value, presence = extractor(raw_summary)
        if presence == "absent" or value is None:
            continue
        candidate, status = normalizer(value)
        candidates.append(
            {
                "record_version": version,
                "event": "candidate_result",
                "observed_at": _timestamp(),
                "unit_id": _text(stop_record.get("unit_id")),
                "run_id": _text(stop_record.get("run_id")),
                "workflow_id": _text(stop_record.get("workflow_id")),
                "loam_profile": _text(stop_record.get("loam_profile")),
                "parent_session_id": _text(stop_record.get("parent_session_id")),
                "parent_turn_id": _text(stop_record.get("parent_turn_id")),
                "child_session_id": _text(stop_record.get("child_session_id")),
                "child_subagent_id": _text(stop_record.get("child_subagent_id")),
                "candidate_status": status,
                "candidate_id": stable_id(
                    "candidate",
                    {"unit_id": _text(stop_record.get("unit_id")), "raw": value},
                ),
                "candidate": candidate,
                "raw_contract": value,
            }
        )
    return candidates


def write_stop_and_candidate(payload: Mapping[str, Any]) -> Path:
    """Persist the terminal lifecycle record and any opt-in unvalidated candidate."""

    stop_record = build_stop_record(payload)
    path = write_record(payload, stop_record)
    candidates = build_candidate_records(stop_record)
    for candidate in candidates:
        append_record(candidate_path(payload), candidate)
    return path
def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one complete JSONL record using one write under an advisory lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_safe(dict(record)), ensure_ascii=False, sort_keys=True) + "\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
    except Exception:
        # fdopen owns the descriptor only after entering its context.  Preserve
        # the original error for the hook caller to report on stderr.
        raise


def write_record(payload: Mapping[str, Any], record: Mapping[str, Any]) -> Path:
    """Append a record and return its location for a compact hook response."""

    path = ledger_path(payload)
    append_record(path, record)
    return path
