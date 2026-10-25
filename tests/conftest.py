"""Keep Loam tests independent of the invoking shell's campaign state."""

from __future__ import annotations

import pytest


_LOAM_ENV = (
    "HERMES_ACCEPT_HOOKS",
    "HERMES_HOME",
    "LOAM_CONFIG_PATH",
    "LOAM_ENABLED",
    "LOAM_LEDGER_PATH",
    "LOAM_PARENT_MEMORY_PATH",
    "LOAM_PRE_LLM_EVERY",
    "LOAM_PROFILE",
    "LOAM_REPO",
    "LOAM_RUN_ID",
    "LOAM_STATE_DIR",
    "LOAM_WORKFLOW_ID",
)


@pytest.fixture(autouse=True)
def isolate_loam_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent ambient campaign variables from redirecting test artifacts."""

    for name in _LOAM_ENV:
        monkeypatch.delenv(name, raising=False)
