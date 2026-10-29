"""Tests for the shared reserve-then-claim key minting (no real network: mint_key is always patched)."""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HOOKS = Path(__file__).resolve().parents[1] / ".loam" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import cairn_client  # noqa: E402
import cairn_keys  # noqa: E402
from cairn_keys import KeyStatus, ensure_key  # noqa: E402

ID = "agent://anl-loam/loam-deadbeef"


def patch_mint(monkeypatch, result):
    state = SimpleNamespace(calls=0, identity=None)

    def fake(base_url, identity, timeout):
        state.calls += 1
        state.identity = identity
        return result

    monkeypatch.setattr(cairn_client, "mint_key", fake)
    return state


def test_present_key_makes_no_network_call(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, cairn_client.CairnResult(True, 201, {"api_key": "ck_x"}, ""))
    key = tmp_path / "k.key"
    key.write_text("old\n")
    st = ensure_key(key, "https://c", ID, 5)
    assert st.state == "present" and mint.calls == 0 and key.read_text() == "old\n"


def test_minted_writes_private_key_and_identity(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, cairn_client.CairnResult(True, 201, {"api_key": "ck_secret"}, ""))
    key = tmp_path / "keys" / "k.key"
    st = ensure_key(key, "https://c", ID, 5)
    assert st == KeyStatus("minted", ID, "") and mint.identity == ID
    assert key.read_text().strip() == "ck_secret"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600 and stat.S_IMODE(key.parent.stat().st_mode) == 0o700
    ident = Path(str(key) + ".identity")
    assert ident.read_text().strip() == ID and stat.S_IMODE(ident.stat().st_mode) == 0o600
    assert "ck_secret" not in repr(st)


def test_busy_when_reserved_by_someone_else(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, cairn_client.CairnResult(True, 201, {"api_key": "ck_x"}, ""))
    key = tmp_path / "k.key"
    key.write_text("")  # an empty reservation held by another minter
    st = ensure_key(key, "https://c", ID, 5)
    assert st.state == "busy" and mint.calls == 0 and key.exists()


def test_failed_removes_reservation(tmp_path, monkeypatch):
    patch_mint(monkeypatch, cairn_client.CairnResult(False, 409, None, "http 409"))
    key = tmp_path / "k.key"
    st = ensure_key(key, "https://c", ID, 5)
    assert st.state == "failed" and "409" in st.error and not key.exists()


def test_no_response_warns_may_be_claimed(tmp_path, monkeypatch):
    patch_mint(monkeypatch, cairn_client.CairnResult(False, 0, None, "TimeoutError: no response"))
    st = ensure_key(tmp_path / "k.key", "https://c", ID, 5)
    assert st.state == "failed" and "may already have been claimed" in st.error.lower()


def test_open_parent_dir_refused_without_network(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, cairn_client.CairnResult(True, 201, {"api_key": "ck_x"}, ""))
    parent = tmp_path / "open"
    parent.mkdir()
    parent.chmod(0o755)
    st = ensure_key(parent / "k.key", "https://c", ID, 5)
    assert st.state == "failed" and mint.calls == 0 and stat.S_IMODE(parent.stat().st_mode) == 0o755


def test_write_failure_after_claim_never_returns_key(tmp_path, monkeypatch):
    patch_mint(monkeypatch, cairn_client.CairnResult(True, 201, {"api_key": "ck_secret"}, ""))

    def boom(fd, data):
        raise OSError("disk full")

    monkeypatch.setattr(cairn_keys.os, "write", boom)
    st = ensure_key(tmp_path / "k.key", "https://c", ID, 5)
    assert st.state == "failed" and "already been claimed" in st.error and "ck_secret" not in st.error


def test_missing_api_key_in_response_is_failure(tmp_path, monkeypatch):
    patch_mint(monkeypatch, cairn_client.CairnResult(True, 201, {"created_at": "x"}, ""))
    key = tmp_path / "k.key"
    assert ensure_key(key, "https://c", ID, 5).state == "failed" and not key.exists()


# --- abandoned reservations, interrupts, mkdir race -------------------------------------------

import os  # noqa: E402
import time  # noqa: E402

OK_MINT = cairn_client.CairnResult(True, 201, {"api_key": "ck_secret"}, "")


def age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_stale_empty_reservation_is_renamed_aside_and_minted(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, OK_MINT)
    key = tmp_path / "k.key"
    key.write_text("")
    age(key, 700)
    st = ensure_key(key, "https://c", ID, 5)
    assert st.state == "minted" and mint.calls == 1 and key.read_text().strip() == "ck_secret"
    [aside] = list(tmp_path.glob("k.key.abandoned.*"))
    assert aside.read_text() == ""


def test_fresh_empty_reservation_is_busy(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, OK_MINT)
    key = tmp_path / "k.key"
    key.write_text("")
    age(key, 30)
    assert ensure_key(key, "https://c", ID, 5).state == "busy"
    assert mint.calls == 0 and not list(tmp_path.glob("*.abandoned.*"))


def test_stale_non_empty_key_never_touched(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, OK_MINT)
    key = tmp_path / "k.key"
    key.write_text("real\n")
    age(key, 10_000)
    assert ensure_key(key, "https://c", ID, 5).state == "present"
    assert key.read_text() == "real\n" and mint.calls == 0 and not list(tmp_path.glob("*.abandoned.*"))


def test_symlink_at_key_path_never_renamed(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, OK_MINT)
    target = tmp_path / "target"
    target.write_text("")
    age(target, 10_000)
    key = tmp_path / "k.key"
    key.symlink_to(target)
    assert ensure_key(key, "https://c", ID, 5).state == "busy"
    assert key.is_symlink() and mint.calls == 0 and not list(tmp_path.glob("*.abandoned.*"))


def test_stale_recovery_busy_when_lock_held(tmp_path, monkeypatch):
    import cairn_common

    mint = patch_mint(monkeypatch, OK_MINT)
    key = tmp_path / "k.key"
    key.write_text("")
    age(key, 700)
    with cairn_common.locked(Path(str(key) + ".lock"), blocking=False) as held:
        assert held
        assert ensure_key(key, "https://c", ID, 5).state == "busy"
    assert mint.calls == 0 and key.exists()


def test_keyboard_interrupt_during_mint_removes_reservation(tmp_path, monkeypatch):
    def boom(base_url, identity, timeout):
        raise KeyboardInterrupt

    monkeypatch.setattr(cairn_client, "mint_key", boom)
    key = tmp_path / "k.key"
    with pytest.raises(KeyboardInterrupt):
        ensure_key(key, "https://c", ID, 5)
    assert not key.exists()


def test_interrupt_after_complete_write_keeps_key(tmp_path, monkeypatch):
    patch_mint(monkeypatch, OK_MINT)
    real_fsync = cairn_keys.os.fsync

    def interrupted(fd):
        real_fsync(fd)
        raise KeyboardInterrupt

    monkeypatch.setattr(cairn_keys.os, "fsync", interrupted)
    key = tmp_path / "k.key"
    with pytest.raises(KeyboardInterrupt):
        ensure_key(key, "https://c", ID, 5)
    assert key.read_text().strip() == "ck_secret"


def test_mkdir_race_with_private_dir_continues(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, OK_MINT)
    real_mkdir = Path.mkdir
    raced = []

    def racing(self, mode=0o777, parents=False, exist_ok=False):
        if not raced:
            raced.append(self)
            real_mkdir(self, 0o700)  # another process won the race
            raise FileExistsError(str(self))
        return real_mkdir(self, mode, parents, exist_ok)

    monkeypatch.setattr(Path, "mkdir", racing)
    st = ensure_key(tmp_path / "keys" / "k.key", "https://c", ID, 5)
    assert st.state == "minted" and mint.calls == 1


def test_mkdir_race_with_open_dir_fails_without_network(tmp_path, monkeypatch):
    mint = patch_mint(monkeypatch, OK_MINT)
    real_mkdir = Path.mkdir

    def racing(self, mode=0o777, parents=False, exist_ok=False):
        real_mkdir(self, 0o755)
        raise FileExistsError(str(self))

    monkeypatch.setattr(Path, "mkdir", racing)
    st = ensure_key(tmp_path / "keys" / "k.key", "https://c", ID, 5)
    assert st.state == "failed" and "already exists" not in st.error and mint.calls == 0
