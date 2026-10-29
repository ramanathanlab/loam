"""FALDA write path: a local spool, drained by a detached flusher."""

from __future__ import annotations

import fcntl
import json
import sys
import time
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / ".loam" / "hooks"
TESTS = Path(__file__).resolve().parent
for directory in (HOOKS, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import falda_flush  # noqa: E402
import falda_memory  # noqa: E402
from falda_fake import POOL, FakeTransport, falda_block  # noqa: E402

KEY = "falda_" + "s" * 40


@pytest.fixture
def state(tmp_path):
    directory = tmp_path / "state"
    directory.mkdir()
    return directory


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("FALDA_TOKEN", KEY)
    return falda_block(key_dir=str(tmp_path / "keys"))


@pytest.fixture
def payload(state, cfg):
    return {"workflow_id": "wf-1", "loam_run_id": "run-1", "loam_state_dir": str(state), "loam_falda": cfg}


@pytest.fixture(autouse=True)
def _restore_transport():
    yield
    falda_memory.set_transport(None)


def _atom(n: int) -> dict:
    return {"type": "fact", "content": f"atom {n}", "confidence": "low", "tags": ["workflow:wf-1"]}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def test_enqueue_does_no_network(payload, state):
    fake = FakeTransport(error=AssertionError("network reached"))
    falda_memory.set_transport(fake)
    falda_flush.enqueue(payload, _atom(1), plan_id="p-1", hypothesis_id="h-1")
    [row] = _rows(state / falda_flush.SPOOL)
    assert row["atom"] == _atom(1)
    assert (row["plan_id"], row["hypothesis_id"], row["workflow_id"], row["run_id"]) == ("p-1", "h-1", "wf-1", "run-1")
    assert row["id"]
    assert fake.requests == []


def test_enqueue_never_raises(payload, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    falda_flush.enqueue({**payload, "loam_state_dir": str(blocker / "state")}, _atom(1), plan_id="", hypothesis_id="")


def test_flush_drains_in_batches_and_marks_sent(payload, state, cfg, monkeypatch):
    monkeypatch.setattr(falda_flush, "BATCH_SIZE", 2)
    for n in range(5):
        falda_flush.enqueue(payload, _atom(n), plan_id="p", hypothesis_id=f"h{n}")
    fake = FakeTransport()
    falda_memory.set_transport(fake)
    summary = falda_flush.flush(state, cfg)
    bodies = fake.bodies("/atoms/upsert")
    assert [len(body["atoms"]) for body in bodies] == [2, 2, 1]
    assert [atom["content"] for body in bodies for atom in body["atoms"]] == [f"atom {n}" for n in range(5)]
    assert all(body["pool"] == POOL for body in bodies)
    assert summary["sent"] == 5 and summary["pending"] == 0
    assert falda_flush.flush(state, cfg)["sent"] == 0
    assert len(fake.bodies("/atoms/upsert")) == 3
    calls = _rows(state / "falda-calls.jsonl")
    assert [row["op"] for row in calls] == ["upsert"] * 3


def test_failed_batch_leaves_rows_pending(payload, state, cfg):
    for n in range(3):
        falda_flush.enqueue(payload, _atom(n), plan_id="p", hypothesis_id="h")
    falda_memory.set_transport(FakeTransport(upsert=[(503, {"error": "overloaded"})]))
    summary = falda_flush.flush(state, cfg)
    assert summary["sent"] == 0 and summary["pending"] == 3 and summary["last_error"] == "overloaded"
    falda_memory.set_transport(FakeTransport())
    assert falda_flush.flush(state, cfg)["sent"] == 3


def test_overflow_drops_oldest_and_counts(payload, state, cfg, monkeypatch):
    monkeypatch.setattr(falda_flush, "MAX_PENDING", 3)
    for n in range(5):
        falda_flush.enqueue(payload, _atom(n), plan_id="p", hypothesis_id="h")
    fake = FakeTransport()
    falda_memory.set_transport(fake)
    summary = falda_flush.flush(state, cfg)
    sent = [atom["content"] for body in fake.bodies("/atoms/upsert") for atom in body["atoms"]]
    assert sent == ["atom 2", "atom 3", "atom 4"]
    assert summary["dropped"] == 2
    status = json.loads((state / falda_flush.STATUS).read_text())
    assert status["dropped_total"] == 2


def test_busy_lock_skips(payload, state, cfg):
    falda_flush.enqueue(payload, _atom(1), plan_id="p", hypothesis_id="h")
    fake = FakeTransport()
    falda_memory.set_transport(fake)
    with open(state / falda_flush.LOCK, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        start = time.monotonic()
        assert falda_flush.flush(state, cfg) == {"skipped": "locked"}
        assert time.monotonic() - start < 0.5
    assert fake.requests == []


def test_malformed_spool_rows_skipped(payload, state, cfg):
    falda_flush.enqueue(payload, _atom(1), plan_id="p", hypothesis_id="h")
    with (state / falda_flush.SPOOL).open("a") as handle:
        handle.write("{not json\n")
        handle.write(json.dumps({"id": "x", "atom": "not a dict"}) + "\n")
        handle.write("\xff\n")
    falda_flush.enqueue(payload, _atom(2), plan_id="p", hypothesis_id="h")
    fake = FakeTransport()
    falda_memory.set_transport(fake)
    assert falda_flush.flush(state, cfg)["sent"] == 2


def test_no_key_leaves_rows_pending(payload, state, cfg, monkeypatch):
    monkeypatch.delenv("FALDA_TOKEN")
    falda_flush.enqueue(payload, _atom(1), plan_id="p", hypothesis_id="h")
    fake = FakeTransport()
    falda_memory.set_transport(fake)
    summary = falda_flush.flush(state, {**cfg, "auto_mint": False})
    assert summary["pending"] == 1 and summary["key"] == "disabled"
    assert fake.requests == []


def test_status_file_has_no_key(payload, state, cfg):
    falda_flush.enqueue(payload, _atom(1), plan_id="p", hypothesis_id="h")
    falda_memory.set_transport(FakeTransport())
    falda_flush.flush(state, cfg)
    status = (state / falda_flush.STATUS).read_text()
    assert KEY not in status
    assert json.loads(status)["key"] == "env"


def test_spawn_flush_detaches_and_respects_auto_flush(payload, monkeypatch):
    launched = []
    monkeypatch.setattr(falda_flush.subprocess, "Popen", lambda argv, **kw: launched.append((argv, kw)))
    falda_flush.real_spawn_flush({**payload, "loam_falda": {**payload["loam_falda"], "auto_flush": False}})
    assert launched == []
    falda_flush.real_spawn_flush(payload)
    [(argv, kw)] = launched
    assert argv[1].endswith("falda_flush.py") and "--state-dir" in argv and "--config-json" in argv
    assert kw["start_new_session"] is True


def test_spawn_flush_never_raises(payload, monkeypatch):
    def _boom(*a, **k):
        raise OSError("no fork")

    monkeypatch.setattr(falda_flush.subprocess, "Popen", _boom)
    falda_flush.real_spawn_flush(payload)


def test_cli_report_counts(payload, state, cfg, capsys):
    falda_flush.enqueue(payload, _atom(1), plan_id="p", hypothesis_id="h")
    assert falda_flush.main(["--state-dir", str(state), "--config-json", json.dumps(cfg), "--report"]) == 0
    assert json.loads(capsys.readouterr().out)["pending"] == 1
