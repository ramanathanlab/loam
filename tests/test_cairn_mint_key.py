"""Tests for the one-time Cairn reviewer key mint script."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
SCRIPT = ROOT / "scripts" / "cairn_mint_key.py"


def _load():
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location("cairn_mint_key", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load()


def _patch(monkeypatch, result):
    import cairn_client

    state = SimpleNamespace(identity=None, calls=0)

    def fake(base_url, identity, timeout, project=None):
        state.calls += 1
        state.identity = identity
        return result

    monkeypatch.setattr(cairn_client, "mint_key", fake)
    return state


@pytest.fixture
def fake_mint_ok(monkeypatch):
    import cairn_client

    return _patch(monkeypatch, cairn_client.CairnResult(True, 201, {"api_key": "ck_live_x"}, ""))


@pytest.fixture
def fake_mint_409(monkeypatch):
    import cairn_client

    return _patch(monkeypatch, cairn_client.CairnResult(False, 409, None, "http 409"))


def test_mint_writes_key_exclusively_with_private_modes(script, tmp_path, fake_mint_ok):
    key = tmp_path / "keys" / "agent-loam.key"
    assert script.main(["mint", "--key-file", str(key)]) == 0
    assert key.read_text().strip() == "ck_live_x"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert stat.S_IMODE(key.parent.stat().st_mode) == 0o700
    assert fake_mint_ok.identity.startswith("agent://anl-loam/loam-") and len(fake_mint_ok.identity) == len("agent://anl-loam/loam-") + 8


def test_mint_refuses_existing_key_and_never_calls_cairn(script, tmp_path, fake_mint_ok):
    key = tmp_path / "agent-loam.key"
    key.write_text("old")
    assert script.main(["mint", "--key-file", str(key)]) == 2
    assert key.read_text() == "old" and fake_mint_ok.calls == 0


def test_mint_failure_leaves_no_file(script, tmp_path, fake_mint_409):
    key = tmp_path / "agent-loam.key"
    assert script.main(["mint", "--key-file", str(key)]) == 1 and not key.exists()


def test_mint_response_without_api_key_is_failure(script, tmp_path, monkeypatch):
    import cairn_client

    _patch(monkeypatch, cairn_client.CairnResult(True, 201, {"created_at": "x"}, ""))
    key = tmp_path / "agent-loam.key"
    assert script.main(["mint", "--key-file", str(key)]) == 1 and not key.exists()


def test_check(script, tmp_path):
    key = tmp_path / "k" / "agent-loam.key"
    key.parent.mkdir(mode=0o700)
    key.write_text("x")
    key.chmod(0o600)
    assert script.main(["check", "--key-file", str(key)]) == 0
    key.chmod(0o644)
    assert script.main(["check", "--key-file", str(key)]) == 1
    key.chmod(0o600)
    key.parent.chmod(0o755)
    assert script.main(["check", "--key-file", str(key)]) == 1
    assert script.main(["check", "--key-file", str(tmp_path / "missing.key")]) == 1


def test_key_never_printed(script, tmp_path, fake_mint_ok, capsys):
    script.main(["mint", "--key-file", str(tmp_path / "a.key")])
    out = capsys.readouterr()
    assert "ck_live_x" not in out.out + out.err


def test_existing_open_parent_rejected_and_unchanged(script, tmp_path, fake_mint_ok):
    parent = tmp_path / "home"
    parent.mkdir()
    parent.chmod(0o755)
    assert script.main(["mint", "--key-file", str(parent / "agent.key")]) == 1
    assert fake_mint_ok.calls == 0
    assert stat.S_IMODE(parent.stat().st_mode) == 0o755
    assert not (parent / "agent.key").exists()


def test_existing_private_parent_unchanged(script, tmp_path, fake_mint_ok):
    parent = tmp_path / "priv"
    parent.mkdir()
    parent.chmod(0o700)
    assert script.main(["mint", "--key-file", str(parent / "agent.key")]) == 0
    assert stat.S_IMODE(parent.stat().st_mode) == 0o700


def test_parent_is_regular_file(script, tmp_path, fake_mint_ok):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    assert script.main(["mint", "--key-file", str(blocker / "agent.key")]) == 1
    assert fake_mint_ok.calls == 0


def test_nested_new_dirs_are_private(script, tmp_path, fake_mint_ok):
    key = tmp_path / "a" / "b" / "agent.key"
    assert script.main(["mint", "--key-file", str(key)]) == 0
    assert stat.S_IMODE(key.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(key.parent.parent.stat().st_mode) == 0o700


def test_write_failure_after_claim_reports_claimed(script, tmp_path, fake_mint_ok, monkeypatch, capsys):
    def boom(fd, data):
        raise OSError("disk full")

    monkeypatch.setattr(script.os, "write", boom)
    key = tmp_path / "agent.key"
    assert script.main(["mint", "--key-file", str(key), "--identity", "agent://anl-loam/loam"]) == 1
    err = capsys.readouterr().err
    assert "agent://anl-loam/loam" in err and "already been claimed" in err
    assert "ck_live_x" not in err


def test_close_failure_after_complete_write_keeps_key(script, tmp_path, fake_mint_ok, monkeypatch, capsys):
    real_close = script.os.close

    def failing_close(fd):
        real_close(fd)
        raise OSError("EIO on close")

    monkeypatch.setattr(script.os, "close", failing_close)
    key = tmp_path / "agent.key"
    assert script.main(["mint", "--key-file", str(key)]) == 1
    assert key.read_text().strip() == "ck_live_x"  # the claimed key is never thrown away
    err = capsys.readouterr().err
    assert "EIO on close" in err and str(key) in err and "ck_live_x" not in err


def test_no_response_warns_identity_may_be_claimed(script, tmp_path, monkeypatch, capsys):
    import cairn_client

    _patch(monkeypatch, cairn_client.CairnResult(False, 0, None, "TimeoutError: no response"))
    key = tmp_path / "agent.key"
    assert script.main(["mint", "--key-file", str(key)]) == 1
    assert not key.exists()
    err = capsys.readouterr().err.lower()
    assert "may already have been claimed" in err and "do not retry" in err


def test_http_refusal_does_not_claim_maybe(script, tmp_path, fake_mint_409, capsys):
    assert script.main(["mint", "--key-file", str(tmp_path / "agent.key")]) == 1
    assert "may already have been claimed" not in capsys.readouterr().err.lower()


def _patch_score(monkeypatch, result):
    import cairn_client

    seen = {}

    def fake(base_url, refs, timeout, project=None):
        seen["project"] = project
        return result

    monkeypatch.setattr(cairn_client, "score_batch", fake)
    return seen


def test_probe_ok_when_project_echoed(script, monkeypatch, capsys):
    import cairn_client

    seen = _patch_score(monkeypatch, cairn_client.CairnResult(
        True, 200, [{"composite_score": 0.5, "confidence": 0.0}], "", echo="ok"))
    assert script.main(["probe", "--project", "exp-silo-1"]) == 0
    assert seen["project"] == "exp-silo-1" and "project ok: exp-silo-1" in capsys.readouterr().out


def test_probe_fails_without_echo(script, monkeypatch):
    import cairn_client

    _patch_score(monkeypatch, cairn_client.CairnResult(True, 200, [], "", echo="missing"))
    assert script.main(["probe"]) == 1


def test_probe_refuses_without_project(script, monkeypatch):
    monkeypatch.delenv("CAIRN_PROJECT")
    assert script.main(["probe"]) == 1
