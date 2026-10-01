"""Load a named Loam profile from global configuration.

The global plugin stays inert when no usable configuration exists. Environment
variables remain explicit emergency overrides, not the normal operational API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - Hermes normally supplies PyYAML
    yaml = None

DEFAULT_CONFIG_PATH = Path.home() / ".hermes" / "loam.yaml"
DEFAULT_INTERVAL = 5
_ALLOWED_POLICIES = {"allow_and_audit", "block_exact_duplicate"}


@dataclass(frozen=True)
class HypothesisTournamentConfig:
    """Bounded Jnana tournament settings for an opt-in Loam profile."""

    jnana_root: Path
    state_file: Path
    reports_dir: Path
    initial_hypotheses: int
    matches_per_run: int
    top_k: int
    timeout_seconds: int
    provider: str
    inference_endpoint: str
    endpoint_is_chat_completions: bool
    model: str
    api_key_env: str | None
    omit_temperature: bool
    json_max_tokens: int


@dataclass(frozen=True)
class LoamProfile:
    name: str
    workflow_id: str
    state_dir: Path
    pre_llm_every: int
    duplicate_policy: str
    hypothesis_orchestration: bool = False
    hypothesis_tournament: HypothesisTournamentConfig | None = None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _interval(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _tournament_config(value: Any) -> HypothesisTournamentConfig | None:
    raw = _as_mapping(value)
    if raw is None or raw.get("enabled") is not True:
        return None
    required = {
        "jnana_root": _clean_text(raw.get("jnana_root")),
        "state_file": _clean_text(raw.get("state_file")),
        "reports_dir": _clean_text(raw.get("reports_dir")),
        "provider": _clean_text(raw.get("provider")) or "openai",
        "inference_endpoint": _clean_text(raw.get("inference_endpoint")) or _clean_text(raw.get("base_url")),
        "model": _clean_text(raw.get("model")),
        "api_key_env": _clean_text(raw.get("api_key_env")),
    }
    if any(not required[key] for key in ("jnana_root", "state_file", "reports_dir", "provider", "inference_endpoint", "model")):
        return None
    return HypothesisTournamentConfig(
        jnana_root=Path(required["jnana_root"]).expanduser(),
        state_file=Path(required["state_file"]).expanduser(),
        reports_dir=Path(required["reports_dir"]).expanduser(),
        initial_hypotheses=_positive_int(raw.get("initial_hypotheses"), 3),
        matches_per_run=_positive_int(raw.get("matches_per_run"), 1),
        top_k=_positive_int(raw.get("top_k"), 3),
        timeout_seconds=_positive_int(raw.get("timeout_seconds"), 30),
        provider=required["provider"],
        inference_endpoint=required["inference_endpoint"],
        endpoint_is_chat_completions=raw.get("endpoint_is_chat_completions") is True,
        model=required["model"],
        api_key_env=required["api_key_env"] or None,
        omit_temperature=raw.get("omit_temperature") is True,
        json_max_tokens=_positive_int(raw.get("json_max_tokens"), 1024),
    )


def config_path() -> Path:
    override = _clean_text(os.environ.get("LOAM_CONFIG_PATH"))
    return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH


def load_profile() -> LoamProfile | None:
    """Load the selected profile, returning None when Loam is intentionally inert."""

    path = config_path()
    if yaml is None or not path.is_file():
        return None
    try:
        root = _as_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError):
        return None
    if not root or root.get("enabled") is not True:
        return None
    requested = _clean_text(os.environ.get("LOAM_PROFILE")) or _clean_text(root.get("active_profile"))
    profiles = _as_mapping(root.get("profiles"))
    if not requested or not profiles:
        return None
    raw = _as_mapping(profiles.get(requested))
    if not raw or raw.get("enabled") is False:
        return None
    workflow_id = _clean_text(raw.get("workflow_id"))
    state_dir = _clean_text(raw.get("state_dir"))
    if not workflow_id or not state_dir:
        return None
    policy = _clean_text(raw.get("duplicate_policy")) or "allow_and_audit"
    if policy not in _ALLOWED_POLICIES:
        return None
    return LoamProfile(
        name=requested,
        workflow_id=workflow_id,
        state_dir=Path(state_dir).expanduser(),
        pre_llm_every=_interval(raw.get("pre_llm_every", DEFAULT_INTERVAL)),
        duplicate_policy=policy,
        hypothesis_orchestration=_as_mapping(raw.get("hypothesis_orchestration")) is not None
        and raw["hypothesis_orchestration"].get("enabled") is True,
        hypothesis_tournament=_tournament_config(raw.get("hypothesis_tournament")),
    )


def emergency_enabled() -> bool:
    return _clean_text(os.environ.get("LOAM_ENABLED")).casefold() in {"1", "true", "yes", "on"} or any(
        _clean_text(os.environ.get(name))
        for name in ("LOAM_LEDGER_PATH", "LOAM_PARENT_MEMORY_PATH", "LOAM_WORKFLOW_ID")
    )


def effective_profile() -> LoamProfile | None:
    """Config is normal activation; legacy env activation remains a temporary fallback."""

    profile = load_profile()
    if profile is not None:
        return profile
    if not emergency_enabled():
        return None
    state_dir = Path(_clean_text(os.environ.get("LOAM_LEDGER_PATH")) or ".loam/state").expanduser().parent
    return LoamProfile(
        name="environment-override",
        workflow_id=_clean_text(os.environ.get("LOAM_WORKFLOW_ID")) or "environment-workflow",
        state_dir=state_dir,
        pre_llm_every=_interval(os.environ.get("LOAM_PRE_LLM_EVERY", DEFAULT_INTERVAL)),
        duplicate_policy="allow_and_audit",
    )


def payload_values(profile: LoamProfile) -> dict[str, Any]:
    """Values passed to each hook, avoiding hook-local global-config reads."""

    values: dict[str, Any] = {
        "loam_profile": profile.name,
        "workflow_id": profile.workflow_id,
        "loam_state_dir": str(profile.state_dir),
        "loam_duplicate_policy": profile.duplicate_policy,
        "loam_hypothesis_orchestration": profile.hypothesis_orchestration,
    }
    tournament = profile.hypothesis_tournament
    if tournament is not None:
        # Deliberately omit any secret value: hooks receive only an environment
        # variable name from which they may resolve the API credential.
        values["loam_hypothesis_tournament"] = {
            "enabled": True,
            "jnana_root": str(tournament.jnana_root),
            "state_file": str(tournament.state_file),
            "reports_dir": str(tournament.reports_dir),
            "initial_hypotheses": tournament.initial_hypotheses,
            "matches_per_run": tournament.matches_per_run,
            "top_k": tournament.top_k,
            "timeout_seconds": tournament.timeout_seconds,
            "provider": tournament.provider,
            "inference_endpoint": tournament.inference_endpoint,
            "endpoint_is_chat_completions": tournament.endpoint_is_chat_completions,
            "model": tournament.model,
            "api_key_env": tournament.api_key_env or "",
            "omit_temperature": tournament.omit_temperature,
            "json_max_tokens": tournament.json_max_tokens,
        }
    return values
