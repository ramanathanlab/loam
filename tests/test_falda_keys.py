"""FALDA machine keys: env, cached per pool under $HOME, minted once across processes."""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
import sys
import time
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / ".loam" / "hooks"
TESTS = Path(__file__).resolve().parent
for directory in (HOOKS, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import falda_keys  # noqa: E402
from falda_fake import POOL, falda_block  # noqa: E402

KEY = "falda_" + "k" * 40


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("FALDA_TOKEN", raising=False)
    return falda_block(key_dir=str(tmp_path / "keys"))


class Mint:
    def __init__(self, status=200, body=None, delay=0.0, counter: Path | None = None):
        self.status = status
        self.body = body if body is not None else {"api_key": KEY, "tenant": "machine-1", "pool": POOL,
                                                   "pool_created": False, "user": {}}
        self.delay = delay
        self.counter = counter
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "body": json.loads(body)})
        if self.counter is not None:
            with self.counter.open("a") as handle:
                handle.write("x")
        time.sleep(self.delay)
        return self.status, json.dumps(self.body).encode()


def _write_key(cfg, pool=POOL, key=KEY):
    path = falda_keys.key_path({**cfg, "pool": pool})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"api_key": key, "pool": pool}))
    return path


def test_env_wins(cfg, monkeypatch):
    _write_key(cfg)
    monkeypatch.setenv("FALDA_TOKEN", "falda_from_env")
    assert falda_keys.resolve_key(cfg) == "falda_from_env"
    assert falda_keys.ensure_key(cfg, 1) == ("falda_from_env", "env")


def test_cached_file_used_when_pool_matches(cfg):
    _write_key(cfg)
    assert falda_keys.resolve_key(cfg) == KEY
    assert falda_keys.ensure_key(cfg, 1) == (KEY, "cached")


def test_cached_file_for_other_pool_ignored(cfg):
    path = falda_keys.key_path(cfg)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"api_key": KEY, "pool": "00000000-0000-4000-8000-000000000000"}))
    assert falda_keys.resolve_key(cfg) == ""


def test_unreadable_key_file_is_empty(cfg):
    path = falda_keys.key_path(cfg)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert falda_keys.resolve_key(cfg) == ""


def test_key_path_is_per_pool(cfg):
    assert falda_keys.key_path(cfg).name == f"loam-{POOL}.json"


def test_mint_writes_0600_file_and_posts_pool_and_label(cfg, monkeypatch):
    mint = Mint()
    monkeypatch.setattr(falda_keys, "_transport", mint)
    assert falda_keys.ensure_key(cfg, 1) == (KEY, "minted")
    request = mint.requests[0]
    assert request["url"] == "https://falda.example/auth/machine"
    assert request["body"]["pool"] == POOL
    assert request["body"]["label"].startswith("loam-")
    assert "Authorization" not in request["headers"]
    path = falda_keys.key_path(cfg)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert falda_keys.resolve_key(cfg) == KEY
    assert not Path(str(path) + ".lock").exists()


def test_auto_mint_off_does_not_mint(cfg, monkeypatch):
    mint = Mint()
    monkeypatch.setattr(falda_keys, "_transport", mint)
    assert falda_keys.ensure_key({**cfg, "auto_mint": False}, 1) == ("", "disabled")
    assert mint.requests == []


def test_held_lock_skips_without_waiting(cfg, monkeypatch):
    mint = Mint()
    monkeypatch.setattr(falda_keys, "_transport", mint)
    lock = Path(str(falda_keys.key_path(cfg)) + ".lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    start = time.monotonic()
    assert falda_keys.ensure_key(cfg, 1) == ("", "minting_elsewhere")
    assert time.monotonic() - start < 0.5
    assert mint.requests == []


def test_stale_lock_replaced(cfg, monkeypatch):
    monkeypatch.setattr(falda_keys, "_transport", Mint())
    lock = Path(str(falda_keys.key_path(cfg)) + ".lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    old = time.time() - falda_keys.STALE_LOCK_SECONDS - 5
    os.utime(lock, (old, old))
    assert falda_keys.ensure_key(cfg, 1) == (KEY, "minted")


def test_failure_sets_cooldown(cfg, monkeypatch):
    monkeypatch.setattr(falda_keys, "_transport", Mint(status=500, body={"error": "boom"}))
    key, status = falda_keys.ensure_key(cfg, 1)
    assert key == "" and status.startswith("mint_failed:")
    mint = Mint()
    monkeypatch.setattr(falda_keys, "_transport", mint)
    assert falda_keys.ensure_key(cfg, 1) == ("", "cooling_down")
    assert mint.requests == []


def test_rate_limit_sets_long_cooldown(cfg, monkeypatch):
    monkeypatch.setattr(falda_keys, "_transport", Mint(status=429, body={"error": "rate_limited"}))
    assert falda_keys.ensure_key(cfg, 1) == ("", "mint_failed:rate_limited")
    marker = json.loads(Path(str(falda_keys.key_path(cfg)) + ".mint-failed").read_text())
    assert marker["reason"] == "rate_limited"
    # past the normal cool-down but inside the rate-limit one: still cooling down
    monkeypatch.setattr(falda_keys.time, "time", lambda: marker["at"] + falda_keys.MINT_COOLDOWN_SECONDS + 1)
    assert falda_keys.ensure_key(cfg, 1) == ("", "cooling_down")
    monkeypatch.setattr(falda_keys.time, "time", lambda: marker["at"] + falda_keys.RATE_LIMIT_COOLDOWN_SECONDS + 1)
    monkeypatch.setattr(falda_keys, "_transport", Mint())
    assert falda_keys.ensure_key(cfg, 1) == (KEY, "minted")


def test_transport_exception_is_a_failure_not_a_raise(cfg, monkeypatch):
    def _boom(*args):
        raise TimeoutError("slow")

    monkeypatch.setattr(falda_keys, "_transport", _boom)
    assert falda_keys.ensure_key(cfg, 1) == ("", "mint_failed:network:TimeoutError")


def test_no_key_material_in_cooldown_or_lock_files(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(falda_keys, "_transport", Mint(status=200, body={"error": "x", "api_key": ""}))
    falda_keys.ensure_key(cfg, 1)
    monkeypatch.setattr(falda_keys.time, "time", lambda: 10**12)
    monkeypatch.setattr(falda_keys, "_transport", Mint())
    falda_keys.ensure_key(cfg, 1)
    for path in (tmp_path / "keys").iterdir():
        if path != falda_keys.key_path(cfg):
            assert KEY not in path.read_text(), path


def _race(cfg, counter, results):
    falda_keys._transport = Mint(delay=0.3, counter=counter)
    results.put(falda_keys.ensure_key(cfg, 2)[1])


def test_concurrent_mints_one_request(cfg, tmp_path):
    context = multiprocessing.get_context("fork")
    counter = tmp_path / "count"
    results = context.Queue()
    workers = [context.Process(target=_race, args=(cfg, counter, results)) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
    statuses = sorted(results.get(timeout=1) for _ in workers)
    assert counter.read_text() == "x"
    assert statuses.count("minted") == 1
    assert set(statuses) <= {"minted", "minting_elsewhere", "cached"}
    assert falda_keys.resolve_key(cfg) == KEY


def test_launch_key_uses_cache_without_minting(cfg, monkeypatch):
    _write_key(cfg)
    mint = Mint()
    monkeypatch.setattr(falda_keys, "_transport", mint)
    assert falda_keys.launch_key(cfg, 3) == KEY
    assert mint.requests == []


def test_launch_key_mints_when_missing(cfg, monkeypatch):
    monkeypatch.setattr(falda_keys, "_transport", Mint())
    assert falda_keys.launch_key(cfg, 3) == KEY
    assert falda_keys.resolve_key(cfg) == KEY


def test_launch_key_waits_briefly_for_another_minter(cfg, monkeypatch):
    lock = Path(str(falda_keys.key_path(cfg)) + ".lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    import threading

    threading.Timer(0.4, lambda: _write_key(cfg)).start()
    start = time.monotonic()
    assert falda_keys.launch_key(cfg, 3) == KEY
    assert time.monotonic() - start < 2.0


def test_launch_key_is_bounded_when_falda_hangs(cfg, monkeypatch):
    monkeypatch.setattr(falda_keys, "_transport", Mint(delay=10))
    start = time.monotonic()
    assert falda_keys.launch_key(cfg, 1.0) == ""
    assert time.monotonic() - start < 2.0


def test_launch_key_respects_auto_mint_off(cfg, monkeypatch):
    mint = Mint()
    monkeypatch.setattr(falda_keys, "_transport", mint)
    assert falda_keys.launch_key({**cfg, "auto_mint": False}, 3) == ""
    assert mint.requests == []


def test_stale_lock_threshold_is_short():
    # a mint takes seconds; an abandoned launch-time mint must not block minting for long
    assert falda_keys.STALE_LOCK_SECONDS <= 60
