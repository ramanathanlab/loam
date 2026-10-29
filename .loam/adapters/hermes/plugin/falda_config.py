"""Parse and validate the ``falda`` block of a Loam profile.

Standard library only. FALDA is on by default: with no profile block the repo's
``.loam/falda.json`` applies, a profile block replaces it entirely, and
``LOAM_FALDA=0`` turns it off. ``falda_config`` returns ``(block, status)``;
the block is a plain dict for hooks (secrets are referenced by env-var name
only), or None when FALDA is off or the block is invalid (the profile still loads).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

DEFAULT_BASE_URL = "https://falda.cairnscore.ai"
DEFAULT_TOKEN_ENV = "FALDA_TOKEN"
DEFAULT_KEY_DIR = "~/.falda/keys"
DEFAULT_RECALL_BUDGET_CHARS = 1500
# Recalls sit on the agent's path; writes and the key mint run in a detached flusher.
DEFAULT_RECALL_TIMEOUT = 2
DEFAULT_TIMEOUT = 10
# .loam/ is three levels above this file (.loam/adapters/hermes/plugin/).
DEFAULT_FILE = Path(__file__).resolve().parents[3] / "falda.json"
_OFF_VALUES = {"0", "off", "false", "no"}
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SWITCHES = (
    "recall_generation",
    "recall_reasoner",
    "write_validator_outcomes",
    "reasoner_tools",
    "auto_mint",
    "auto_flush",
)


def _default_block() -> Any:
    try:
        return json.loads(DEFAULT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _text(value: Any, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _positive(value: Any, default: float) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return value


def falda_config(value: Any) -> tuple[dict[str, Any] | None, str]:
    """Validated FALDA settings and a status: enabled, disabled, or rejected:<reason>."""

    if os.environ.get("LOAM_FALDA", "").strip().lower() in _OFF_VALUES:
        return None, "disabled"
    if value is None:
        value = _default_block()
    if not isinstance(value, Mapping) or value.get("enabled") is not True:
        return None, "disabled"
    pool = _text(value.get("pool"), "").lower()
    if not _UUID.match(pool):
        return None, "rejected:missing_pool"
    base_url = _text(value.get("base_url"), DEFAULT_BASE_URL).rstrip("/")
    if "base_url" in value and not _text(value.get("base_url"), ""):
        return None, "rejected:missing_base_url"
    if not base_url.startswith(("https://", "http://")):
        return None, "rejected:missing_base_url"
    switches: dict[str, bool] = {}
    for name in SWITCHES:
        flag = value.get(name, True)
        if not isinstance(flag, bool):
            return None, "rejected:bad_switch"
        switches[name] = flag
    recall_timeout = _positive(value.get("recall_timeout_seconds"), DEFAULT_RECALL_TIMEOUT)
    timeout = _positive(value.get("timeout_seconds"), DEFAULT_TIMEOUT)
    budget = _positive(value.get("recall_budget_chars"), DEFAULT_RECALL_BUDGET_CHARS)
    if recall_timeout is None or timeout is None or budget is None:
        return None, "rejected:bad_timeout"
    return (
        {
            "enabled": True,
            "base_url": base_url,
            "pool": pool,
            "token_env": _text(value.get("token_env"), DEFAULT_TOKEN_ENV),
            "key_dir": _text(value.get("key_dir"), DEFAULT_KEY_DIR),
            "recall_budget_chars": int(budget),
            "recall_timeout_seconds": recall_timeout,
            "timeout_seconds": timeout,
            **switches,
        },
        "enabled",
    )
