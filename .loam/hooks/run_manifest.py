"""Persist one deployment-scoped Loam run manifest alongside runtime artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import falda_memory
from hypothesis_orchestration import campaign_provenance

FILENAME = "run-manifest.json"
SCHEMA_VERSION = "loam_run_manifest_v1"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_dir(payload: Mapping[str, Any]) -> Path:
    configured = _text(payload.get("loam_state_dir")) or os.environ.get("LOAM_STATE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    ledger = os.environ.get("LOAM_LEDGER_PATH", "").strip()
    if ledger:
        return Path(ledger).expanduser().parent
    return Path(_text(payload.get("repo_root")) or ".") / ".loam" / "state"


def _run_id(payload: Mapping[str, Any]) -> str:
    return _text(payload.get("loam_run_id")) or os.environ.get("LOAM_RUN_ID", "").strip() or _text(
        payload.get("parent_session_id")
    ) or _text(payload.get("session_id")) or "direct-hook"


_FALDA_SWITCHES = ("recall_generation", "recall_reasoner", "write_validator_outcomes", "reasoner_tools")


def _falda_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The FALDA configuration this run started with, for evaluation filtering."""

    summary: dict[str, Any] = {"status": _text(payload.get("loam_falda_status")) or "disabled"}
    falda = payload.get("loam_falda")
    if isinstance(falda, Mapping):
        summary["pool"] = _text(falda.get("pool"))
        summary["workflow_tag"] = falda_memory.workflow_tag(payload)
        summary.update({name: falda.get(name) is True for name in _FALDA_SWITCHES})
    return summary


def _identity(payload: Mapping[str, Any], state_dir: Path) -> dict[str, Any]:
    return {
        "run_id": _run_id(payload),
        "state_dir": str(state_dir),
        "workflow_id": _text(payload.get("workflow_id")),
        "profile": _text(payload.get("loam_profile")),
        "repo_root": _text(payload.get("repo_root")),
        "execution": payload.get("loam_execution") if isinstance(payload.get("loam_execution"), Mapping) else None,
    }


def _deployment_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Identity minus the per-launch run_id.

    The manifest is deployment-scoped: the state dir, repo, profile, workflow and
    execution binding define the deployment. run_id is minted fresh on every
    launch (run-campaign calls uuidgen), so relaunching into an existing campaign
    directory must NOT be treated as an identity mismatch. We gate on the
    deployment fields and refresh run_id when a relaunch is detected.
    """

    return {k: v for k, v in identity.items() if k != "run_id"}


def _workspace_identity(workspace: Any) -> dict[str, Any]:
    """Whether the Pando-style per-hypothesis workspace was enabled, and how."""

    if not isinstance(workspace, Mapping) or workspace.get("enabled") is not True:
        return {"workspace_enabled": False}
    return {
        "workspace_enabled": True,
        "workspace_strict": workspace.get("strict") is not False,
        "workspace_root": str(workspace.get("root") or ""),
        "workspace_verifier": bool(workspace.get("verifier_mcp_url")),
    }


def _campaign(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Campaign flags plus the policy in force, for attributing results to it.

    Deliberately not part of `_identity`: editing a rule mid-campaign changes the
    digest, and that must be recorded rather than rejected as a deployment
    mismatch, which would stop a relaunch into an existing campaign directory.
    Refreshed on every update for the same reason -- a policy change that only
    appeared in a manifest created weeks earlier would say nothing about the
    results produced after it.
    """

    tournament = payload.get("loam_hypothesis_tournament")
    value = {
        "hypothesis_orchestration_enabled": payload.get("loam_hypothesis_orchestration") is True,
        "tournament_enabled": isinstance(tournament, Mapping) and tournament.get("enabled") is True,
        "falda": _falda_summary(payload),
    }
    if policy := campaign_provenance(payload):
        value["policy"] = policy
    # The per-hypothesis workspace governs every write a child makes, so
    # whether it was on (and strict) belongs here for the same reason the
    # policy does: it is what the results were produced under, and changing
    # it mid-campaign must be recorded rather than rejected.
    value.update(_workspace_identity(payload.get("loam_workspace")))
    return value


def _new_manifest(payload: Mapping[str, Any], state_dir: Path, event: str) -> dict[str, Any]:
    now = _timestamp()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "identity": _identity(payload, state_dir),
        "campaign": _campaign(payload),
        "lifecycle": {"first_event": event, "last_event": event, "event_counts": {event: 1}},
        "artifacts": {
            "subagent_ledger": "subagent-ledger.jsonl",
            "orchestration_ledger": "hypothesis-orchestration.jsonl",
            "campaign_ledger": "hypothesis-campaign.jsonl",
            "tournament_reports_dir": "tournament-reports",
            "falda_calls": "falda-calls.jsonl",
        },
    }


def ensure(payload: Mapping[str, Any]) -> Path:
    """Create or update the manifest while preserving its deployment identity."""

    state_dir = _state_dir(payload)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / FILENAME
    event = _text(payload.get("event")) or "unknown"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"invalid run manifest: {path}")
        current = _identity(payload, state_dir)
        stored = value.get("identity") if isinstance(value.get("identity"), Mapping) else {}
        if _deployment_identity(stored) != _deployment_identity(current):
            raise ValueError(f"run manifest identity mismatch: {path}")
        manifest = value
        # Same deployment, new launch: adopt the current run_id for provenance.
        manifest["identity"] = current
        manifest["campaign"] = _campaign(payload)
        lifecycle = manifest.setdefault("lifecycle", {})
        counts = lifecycle.setdefault("event_counts", {})
        counts[event] = int(counts.get(event, 0)) + 1
        lifecycle["last_event"] = event
        manifest["updated_at"] = _timestamp()
    else:
        manifest = _new_manifest(payload, state_dir, event)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=state_dir, delete=False) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    return path
