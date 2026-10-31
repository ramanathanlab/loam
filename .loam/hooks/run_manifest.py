"""Persist one deployment-scoped Loam run manifest alongside runtime artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from campaign import campaign_provenance

FILENAME = "run-manifest.json"
SCHEMA_VERSION = "loam_run_manifest_v1"
ARTIFACTS = {
    "subagent_ledger": "subagent-ledger.jsonl",
    "orchestration_ledger": "hypothesis-orchestration.jsonl",
    "campaign_ledger": "hypothesis-campaign.jsonl",
    "tournament_reports_dir": "tournament-reports",
}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _state_dir(payload: Mapping[str, Any]) -> Path:
    configured = _text(payload.get("loam_state_dir")) or os.environ.get("LOAM_STATE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    ledger = os.environ.get("LOAM_LEDGER_PATH", "").strip()
    if ledger:
        return Path(ledger).expanduser().parent
    return Path(_text(payload.get("repo_root")) or ".") / ".loam" / "state"


def _run_id(payload: Mapping[str, Any]) -> str:
    return (
        _text(payload.get("loam_run_id"))
        or os.environ.get("LOAM_RUN_ID", "").strip()
        or _text(payload.get("parent_session_id"))
        or _text(payload.get("session_id"))
        or "direct-hook"
    )


def _identity(payload: Mapping[str, Any], state_dir: Path) -> dict[str, Any]:
    return {
        "run_id": _run_id(payload),
        "state_dir": str(state_dir),
        "workflow_id": _text(payload.get("workflow_id")),
        "profile": _text(payload.get("loam_profile")),
        "repo_root": _text(payload.get("repo_root")),
        "execution": payload.get("loam_execution"),
    }


def _deployment(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Identity minus run_id, which run-campaign mints fresh on every launch.

    Relaunching into an existing campaign directory is the same deployment, so
    only these fields are gated; run_id is refreshed instead.
    """

    return {k: v for k, v in identity.items() if k != "run_id"}


def _campaign(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Campaign flags, policy, and workspace mode the results were produced under.

    Deliberately not part of `_identity` and refreshed on every event: editing a
    rule or toggling the workspace mid-campaign must be recorded, not rejected as
    a deployment mismatch that would block a relaunch.
    """

    tournament = payload.get("loam_hypothesis_tournament") or {}
    workspace = payload.get("loam_workspace") or {}
    value: dict[str, Any] = {
        "hypothesis_orchestration_enabled": payload.get("loam_hypothesis_orchestration") is True,
        "tournament_enabled": tournament.get("enabled") is True,
        "workspace_enabled": workspace.get("enabled") is True,
    }
    if policy := campaign_provenance(payload):
        value["policy"] = policy
    if value["workspace_enabled"]:
        value["workspace_strict"] = workspace.get("strict") is not False
        value["workspace_root"] = str(workspace.get("root") or "")
        value["workspace_verifier"] = bool(workspace.get("verifier_mcp_url"))
    return value


def ensure(payload: Mapping[str, Any]) -> Path:
    """Create or update the manifest while preserving its deployment identity."""

    state_dir = _state_dir(payload)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / FILENAME
    event = _text(payload.get("event")) or "unknown"
    identity = _identity(payload, state_dir)
    now = datetime.now(timezone.utc).isoformat()

    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"invalid run manifest: {path}")
        if _deployment(manifest["identity"]) != _deployment(identity):
            raise ValueError(f"run manifest identity mismatch: {path}")
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": now,
            "lifecycle": {"first_event": event, "event_counts": {}},
            "artifacts": dict(ARTIFACTS),
        }

    counts = manifest["lifecycle"]["event_counts"]
    counts[event] = counts.get(event, 0) + 1
    manifest["lifecycle"]["last_event"] = event
    manifest.update(identity=identity, campaign=_campaign(payload), updated_at=now)

    # Write-then-rename so concurrent hooks never read a torn file; the
    # temporary file is created 0600, so the manifest inherits that mode.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=state_dir, delete=False) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(handle.name, path)
    return path
