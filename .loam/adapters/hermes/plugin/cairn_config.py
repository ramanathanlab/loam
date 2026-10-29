"""Parse and validate the optional ``cairn`` block of a Loam profile.

Standard library only. ``cairn_config`` returns a plain dict for hooks, or None
when the block is absent, not enabled, or invalid (Cairn then stays off and the
profile still loads).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

DEFAULT_BASE_URL = "https://api.cairnscore.ai"
DEFAULT_KEY_FILE = "~/.cairn/keys/agent-loam.key"
DEFAULT_IDENTITY = ""  # empty: generated at mint time as identity_prefix + random hex
DEFAULT_IDENTITY_PREFIX = "agent://anl-loam/loam-"
# .loam/ is three levels above this file (.loam/adapters/hermes/plugin/).
DEFAULT_FILE = Path(__file__).resolve().parents[3] / "cairn.json"
_OFF_VALUES = {"0", "off", "false", "no"}
DEFAULT_EXECUTOR = "mcp://hpc#run_code"
DEFAULT_TIMEOUT = 5.0
MAX_TIMEOUT = 10.0
MAX_LIST = 10
DEFAULT_RATE_ROLES = ["code-generator", "hypothesis-validator"]
ALLOWED_ROLES = {"code-generator", "hypothesis-validator", "reasoner"}
_ENTITY_PREFIXES = ("mcp://", "tool://")
_BOOL_KEYS = ("read_side", "rate_agent_reports", "rate_hook_fallback", "auto_flush", "auto_mint")
_NON_TAG_RE = re.compile(r"[^a-z0-9_]+")
# Mirrors cairn_client.PROJECT_RE. The project silo comes only from the launch environment or, by
# default, the campaign's workflow id, never from a profile or a model, and every Cairn request
# carries it as X-Cairn-Project.
PROJECT_ENV = "CAIRN_PROJECT"
_PROJECT_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,62}")
_warned: set[str] = set()
logger = logging.getLogger(__name__)


def default_project(workflow_id: str) -> str:
    """The silo a campaign gets without CAIRN_PROJECT: derived from its workflow id, so every runner of
    one campaign shares it with no setup (dated when there is no id)."""
    slug = re.sub(r"[^a-z0-9]+", "-", workflow_id.lower()).strip("-")
    if not slug:
        slug = datetime.now(timezone.utc).strftime("%Y%m%d")
    project = slug if slug.startswith("loam-") else f"loam-{slug}"
    return project[:63].rstrip("-")


def _project(workflow_id: str = "") -> tuple[str, str]:
    """(project, error): CAIRN_PROJECT when set and valid, else the campaign's default silo."""
    raw = os.environ.get(PROJECT_ENV, "").strip()
    if not raw:
        return default_project(workflow_id), ""
    if _PROJECT_RE.fullmatch(raw):
        return raw, ""
    project = default_project(workflow_id)
    error = f"{PROJECT_ENV}={raw!r} is invalid (^[a-z0-9][a-z0-9-]{{2,62}}$): using {project}"
    if error not in _warned:  # loud, but once per process
        _warned.add(error)
        logger.warning("Loam Cairn: %s", error)
        print(f"Loam Cairn: WARNING {error}", file=sys.stderr)
    return project, error


def _snake_tag(value: str) -> str:
    # Mirrors cairn_common.snake_tag (the adapter cannot import hooks): Cairn tags are ^[a-z][a-z0-9_]{0,63}$.
    tag = _NON_TAG_RE.sub("_", value.lower()).strip("_")[:64].rstrip("_")
    return tag if tag and "a" <= tag[0] <= "z" else ""


def _text(value: Any, default: str) -> str | None:
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_LIST:
        return None
    if not all(isinstance(item, str) and item.strip() for item in value):
        return None
    return [item.strip() for item in value]


def _entities(value: Any) -> dict[str, dict[str, list[str]]] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    result: dict[str, dict[str, list[str]]] = {}
    for entity_id, spec in value.items():
        if not isinstance(entity_id, str) or not entity_id.startswith(_ENTITY_PREFIXES):
            return None
        if spec is None:
            spec = {}
        if not isinstance(spec, Mapping):
            return None
        patterns = _string_list(spec.get("patterns"))
        tags = _string_list(spec.get("task_tags"))
        if patterns is None or tags is None:
            return None
        tags = [_snake_tag(t) for t in tags]
        if not all(tags):
            return None
        result[entity_id] = {"patterns": [p.lower() for p in patterns], "task_tags": tags}
    return result


def _default_block() -> Any:
    try:
        return json.loads(DEFAULT_FILE.read_text())
    except (OSError, ValueError):
        return None


def cairn_config(value: Any, workflow_id: str = "") -> dict[str, Any] | None:
    """Validated Cairn settings. ``value`` is the profile's ``cairn`` block; None means the repo default.
    ``workflow_id`` names the default project silo when CAIRN_PROJECT is not set."""
    if os.environ.get("LOAM_CAIRN", "").strip().lower() in _OFF_VALUES:
        return None
    if value is None:
        value = _default_block()
    return _parse(value, workflow_id)


def _parse(value: Any, workflow_id: str = "") -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or value.get("enabled") is not True:
        return None
    flags: dict[str, bool] = {}
    for key in _BOOL_KEYS:
        flag = value.get(key, True)
        if not isinstance(flag, bool):
            return None
        flags[key] = flag
    base_url = _text(value.get("base_url"), DEFAULT_BASE_URL)
    key_file = _text(value.get("key_file"), DEFAULT_KEY_FILE)
    identity = value.get("identity", DEFAULT_IDENTITY)
    if identity != "" and _text(identity, "") is None:
        return None
    identity = identity.strip() if isinstance(identity, str) else DEFAULT_IDENTITY
    prefix = _text(value.get("identity_prefix"), DEFAULT_IDENTITY_PREFIX)
    executor = _text(value.get("executor"), DEFAULT_EXECUTOR)
    if None in (base_url, key_file, prefix, executor) or "://" not in prefix or not executor.startswith(_ENTITY_PREFIXES):
        return None
    if not base_url.startswith(("http://", "https://")):
        return None
    timeout = value.get("timeout_seconds", DEFAULT_TIMEOUT)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0:
        return None
    roles = value.get("rate_roles")
    if roles is None:
        roles = list(DEFAULT_RATE_ROLES)
    elif not isinstance(roles, list) or not all(isinstance(r, str) and r in ALLOWED_ROLES for r in roles):
        return None
    entities = _entities(value.get("entities"))
    if entities is None:
        return None
    entities.setdefault(executor, {"patterns": [], "task_tags": ["code_exec"]})
    project, project_error = _project(workflow_id)
    return {
        "enabled": True,
        "base_url": base_url,
        "key_file": key_file,
        "identity": identity,
        "identity_prefix": prefix,
        **flags,
        "rate_roles": list(roles),
        "timeout_seconds": min(float(timeout), MAX_TIMEOUT),
        "executor": executor,
        "entities": entities,
        "project": project,
        "project_error": project_error,
    }
