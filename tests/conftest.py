"""Keep Loam tests independent of the invoking shell's campaign state."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HOOKS = Path(__file__).resolve().parents[1] / ".loam" / "hooks"
if str(_HOOKS) not in sys.path:
    sys.path.insert(0, str(_HOOKS))


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
    # Cairn is on by default and mints keys / calls the network; tests must never do either.
    # Tests that need Cairn on pass an explicit block or monkeypatch.delenv("LOAM_CAIRN").
    monkeypatch.setenv("LOAM_CAIRN", "off")


@pytest.fixture(autouse=True)
def no_cairn_network(monkeypatch: pytest.MonkeyPatch):
    """Fail any test that reaches the Cairn HTTP layer without patching it itself.

    Tests that exercise the client replace ``cairn_client._urlopen`` (or ``mint_key`` /
    ``submit_batch``) with a fake after this fixture runs; anything else that gets here is a bug.
    Subprocess tests are covered by ``LOAM_CAIRN=off`` / ``auto_flush: False`` instead.
    """

    import cairn_client

    attempts: list[str] = []

    def blocked(request, timeout=None):
        attempts.append(getattr(request, "full_url", str(request)))
        raise OSError("network access is blocked in tests")

    monkeypatch.setattr(cairn_client, "_urlopen", blocked)
    yield
    assert not attempts, f"test reached the Cairn network: {attempts}"
