"""Tests for the at-most-once Cairn flusher."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import cairn_common  # noqa: E402
import cairn_client  # noqa: E402
import cairn_flush  # noqa: E402
from cairn_client import CairnResult  # noqa: E402
from cairn_flush import flush, main, spawn_flush, validate_event  # noqa: E402


def valid_event():
    return {
        "reviewee": {"type": "capability", "external_id": "tool://lammps"},
        "context": "general",
        "score": 0.8,
        "weight": 0.5,
        "rationale": "[code-generator, agent-authored] ok",
        "task_tags": [],
        "failure_modes": [],
        "observed_at": "2026-10-29T00:00:00+00:00",
    }


def queue(state, n, status="queued", event=None):
    event = valid_event() if event is None else event
    for i in range(n):
        cairn_common.append_jsonl(
            state / cairn_common.QUEUE,
            {"event_id": f"{status}-{i}-{len(cairn_common.read_jsonl(state / cairn_common.QUEUE))}",
             "queued_at": "t", "status": status, "reason": "", "role": "code-generator",
             "source": "agent", "event": event},
        )


class Fake:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def __call__(self, base_url, api_key, events, timeout):
        self.calls.append(events)
        assert api_key == "secret"
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


OK = CairnResult(True, 200, {}, "")


@pytest.fixture
def key(tmp_path):
    path = tmp_path / "k.key"
    path.write_text("secret\n")
    return path


def install(monkeypatch, *results):
    fake = Fake(results)
    monkeypatch.setattr(cairn_flush, "submit_batch", fake)
    return fake


@pytest.fixture
def fake_submit_ok(monkeypatch):
    return install(monkeypatch, OK)


def test_submits_queued_and_marks_submitted(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 3)
    assert flush(tmp_path, "https://c", key, 5)["submitted"] == 3
    assert len(fake_submit_ok.calls) == 1 and len(fake_submit_ok.calls[0]) == 3


def test_never_resubmits(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 2)
    flush(tmp_path, "https://c", key, 5)
    flush(tmp_path, "https://c", key, 5)
    assert len(fake_submit_ok.calls) == 1


def test_batches_of_100(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 205)
    flush(tmp_path, "https://c", key, 5)
    assert [len(c) for c in fake_submit_ok.calls] == [100, 100, 5]


def test_non_queued_rows_never_sent(tmp_path, key, fake_submit_ok):
    for s in ("unbacked", "suppressed_agent_error", "invalid", "missing_block", "no_tool_used"):
        queue(tmp_path, 1, status=s)
    flush(tmp_path, "https://c", key, 5)
    assert fake_submit_ok.calls == []


def test_invalid_event_excluded_so_batch_survives(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 1, event={**valid_event(), "score": 1.5})
    queue(tmp_path, 1)
    counts = flush(tmp_path, "https://c", key, 5)
    assert counts["invalid"] == 1 and counts["submitted"] == 1 and len(fake_submit_ok.calls[0]) == 1


def test_http_failure_marked_failed_and_resubmittable(tmp_path, key, monkeypatch):
    fake = install(monkeypatch, CairnResult(False, 422, None, "bad"), OK)
    queue(tmp_path, 2)
    assert flush(tmp_path, "https://c", key, 5)["failed"] == 2
    assert flush(tmp_path, "https://c", key, 5).get("submitted", 0) == 0
    assert len(fake.calls) == 1
    assert flush(tmp_path, "https://c", key, 5, resubmit_failed=True)["submitted"] == 2


def test_resubmit_failed_terminates_when_still_failing(tmp_path, key, monkeypatch):
    fake = install(monkeypatch, CairnResult(False, 500, None, "boom"))
    queue(tmp_path, 1)
    flush(tmp_path, "https://c", key, 5)
    assert flush(tmp_path, "https://c", key, 5, resubmit_failed=True)["failed"] == 1
    assert len(fake.calls) == 2


def test_network_failure_marked_unknown_and_never_resent(tmp_path, key, monkeypatch):
    fake = install(monkeypatch, CairnResult(False, 0, None, "timeout"))
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", key, 5)["unknown"] == 1
    flush(tmp_path, "https://c", key, 5, resubmit_failed=True)
    assert len(fake.calls) == 1


def test_in_flight_is_never_resent(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 1)
    event_id = cairn_common.read_jsonl(tmp_path / cairn_common.QUEUE)[0]["event_id"]
    cairn_common.append_jsonl(tmp_path / cairn_common.SUBMISSIONS,
                              {"event_id": event_id, "state": "in_flight", "at": "t", "error": ""})
    assert flush(tmp_path, "https://c", key, 5, resubmit_failed=True)["in_flight"] == 1
    assert fake_submit_ok.calls == []


def test_exception_in_submit_leaves_in_flight_and_releases_lock(tmp_path, key, monkeypatch):
    def boom(*a):
        raise RuntimeError("x")
    monkeypatch.setattr(cairn_flush, "submit_batch", boom)
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", key, 5)["in_flight"] == 1
    with cairn_common.locked(tmp_path / cairn_common.FLUSH_LOCK, blocking=False) as got:
        assert got


def test_concurrent_flusher_backs_off(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 1)
    with cairn_common.locked(tmp_path / "cairn-flush.lock"):
        assert flush(tmp_path, "https://c", key, 5)["locked"] == 1
    assert fake_submit_ok.calls == []


def test_missing_key_leaves_rows_pending(tmp_path, fake_submit_ok):
    queue(tmp_path, 1)
    flush(tmp_path, "https://c", tmp_path / "nope.key", 5)
    assert fake_submit_ok.calls == []
    assert flush(tmp_path, "https://c", tmp_path / "nope.key", 5)["pending"] == 1


def test_empty_key_leaves_rows_pending(tmp_path, fake_submit_ok):
    (tmp_path / "e.key").write_text("  \n")
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", tmp_path / "e.key", 5)["pending"] == 1
    assert fake_submit_ok.calls == []


def test_report_sends_nothing(tmp_path, key, fake_submit_ok, capsys):
    queue(tmp_path, 1)
    assert main(["--state-dir", str(tmp_path), "--base-url", "https://c", "--key-file", str(key), "--report"]) == 0
    assert fake_submit_ok.calls == [] and json.loads(capsys.readouterr().out)["pending"] == 1


def test_main_flushes(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 1)
    assert main(["--state-dir", str(tmp_path), "--base-url", "https://c", "--key-file", str(key)]) == 0
    assert len(fake_submit_ok.calls) == 1


def test_validate_event_accepts_builder_shape_and_rejects_bad():
    assert validate_event(valid_event()) == ""
    assert validate_event({**valid_event(), "metrics": {"wall_s": 3}}) == ""
    assert validate_event({**valid_event(), "metrics": {"wall_s": "x"}})
    assert validate_event({**valid_event(), "weight": 0})
    assert validate_event({**valid_event(), "rationale": "x" * 2001})
    assert validate_event({**valid_event(), "task_tags": ["a"] * 11})
    assert validate_event({**valid_event(), "reviewee": {"type": "capability", "external_id": "http://x"}})
    assert validate_event({**valid_event(), "context": "other"})
    assert validate_event({**valid_event(), "observed_at": 5})


def test_spawn_flush_respects_auto_flush(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cairn_flush.subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    block = {"enabled": True, "base_url": "https://c", "key_file": "~/k.key", "timeout_seconds": 5, "auto_flush": False}
    payload = {"loam_cairn": block, "loam_state_dir": str(tmp_path)}
    spawn_flush(payload)
    spawn_flush({"loam_cairn": {**block, "enabled": False, "auto_flush": True}, "loam_state_dir": str(tmp_path)})
    assert calls == []
    spawn_flush({"loam_cairn": {**block, "auto_flush": True}, "loam_state_dir": str(tmp_path)})
    assert len(calls) == 1
    argv, kwargs = calls[0][0][0], calls[0][1]
    assert "--state-dir" in argv and "~" not in "".join(argv[argv.index("--key-file") + 1])
    assert kwargs["start_new_session"] is True and kwargs["close_fds"] is True


def test_spawn_flush_never_raises(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("no")
    monkeypatch.setattr(cairn_flush.subprocess, "Popen", boom)
    spawn_flush({"loam_cairn": {"enabled": True, "auto_flush": True}, "loam_state_dir": str(tmp_path)})


def test_2xx_with_unparseable_body_is_submitted_and_never_resent(tmp_path, key, monkeypatch):
    fake = install(monkeypatch, CairnResult(False, 200, None, "ValueError: bad json"))
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", key, 5)["submitted"] == 1
    assert flush(tmp_path, "https://c", key, 5, resubmit_failed=True)["submitted"] == 1
    assert len(fake.calls) == 1


def test_gateway_error_is_unknown_and_never_resent(tmp_path, key, monkeypatch):
    fake = install(monkeypatch, CairnResult(False, 503, None, "unavailable"))
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", key, 5)["unknown"] == 1
    assert flush(tmp_path, "https://c", key, 5, resubmit_failed=True)["unknown"] == 1
    assert len(fake.calls) == 1


def test_422_is_failed(tmp_path, key, monkeypatch):
    install(monkeypatch, CairnResult(False, 422, None, "bad"))
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", key, 5)["failed"] == 1


def test_overlong_task_invalid_so_batch_survives(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 1, event={**valid_event(), "task": "x" * 501})
    queue(tmp_path, 1)
    counts = flush(tmp_path, "https://c", key, 5)
    assert counts["invalid"] == 1 and counts["submitted"] == 1 and len(fake_submit_ok.calls[0]) == 1


def test_validate_event_mirrors_cairn_tag_and_metric_rules():
    good = {**valid_event(), "task_tags": ["md_simulation"], "failure_modes": ["infrastructure_walltime"],
            "metrics": {"wall_s": 1.5, "b" * 32: 1}}
    assert validate_event(good) == ""
    for tags in (["md-simulation"], ["Md"], ["9x"], ["_x"], ["a" * 65], [""]):
        assert validate_event({**good, "task_tags": tags}), tags
        assert validate_event({**good, "failure_modes": tags}), tags
    assert validate_event({**good, "failure_modes": ["infrastructure:walltime"]})
    assert validate_event({**good, "task_tags": ["a" * 64]}) == ""
    for name in ("Wall", "9x", "_x", "a" * 33, "with-dash", ""):
        assert validate_event({**good, "metrics": {name: 1}}), name
    assert validate_event({**good, "metrics": {f"k{i}": i for i in range(21)}})
    assert validate_event({**good, "metrics": {f"k{i}": i for i in range(20)}}) == ""
    assert validate_event({**good, "metrics": {"x": float("nan")}})
    assert validate_event({**good, "metrics": {"x": float("inf")}})
    assert validate_event({**good, "metrics": {"x": True}})


def test_nonconforming_tag_is_invalid_so_batch_survives(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 1, event={**valid_event(), "failure_modes": ["lifecycle:error"]})
    queue(tmp_path, 1)
    counts = flush(tmp_path, "https://c", key, 5)
    assert counts["invalid"] == 1 and counts["submitted"] == 1 and len(fake_submit_ok.calls[0]) == 1


def _real_client(monkeypatch, exc):
    import cairn_client

    calls = []

    def fake(request, timeout=None):
        calls.append(request)
        raise exc

    monkeypatch.setattr(cairn_client, "_urlopen", fake)
    monkeypatch.setattr(cairn_flush, "submit_batch", cairn_client.submit_batch)
    return calls


def test_connection_refused_is_unsent_and_resent_next_flush(tmp_path, key, monkeypatch):
    import errno
    import urllib.error

    calls = _real_client(monkeypatch, urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused")))
    queue(tmp_path, 2)
    counts = flush(tmp_path, "https://c", key, 5)
    assert counts == {"unsent": 2} and len(calls) == 1  # not retried within the same flush
    fake = install(monkeypatch, OK)
    assert flush(tmp_path, "https://c", key, 5)["submitted"] == 2
    assert len(fake.calls) == 1 and len(fake.calls[0]) == 2


def test_read_timeout_is_unknown_and_never_resent(tmp_path, key, monkeypatch):
    calls = _real_client(monkeypatch, TimeoutError("The read operation timed out"))
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", key, 5) == {"unknown": 1}
    flush(tmp_path, "https://c", key, 5, resubmit_failed=True)
    assert len(calls) == 1


def test_counts_report_recorded_non_queued_rows(tmp_path, key, fake_submit_ok, capsys):
    for s, n in (("unbacked", 2), ("suppressed_agent_error", 1), ("invalid", 1), ("missing_block", 1), ("no_tool_used", 3)):
        queue(tmp_path, n, status=s)
    queue(tmp_path, 1)
    counts = flush(tmp_path, "https://c", key, 5)
    expected = {"unbacked": 2, "suppressed_agent_error": 1, "invalid": 1, "missing_block": 1, "no_tool_used": 3}
    assert counts == {"submitted": 1, "recorded": expected}
    assert main(["--state-dir", str(tmp_path), "--base-url", "https://c", "--key-file", str(key), "--report"]) == 0
    assert json.loads(capsys.readouterr().out) == {"submitted": 1, "recorded": expected}


def test_counts_without_recorded_rows_keep_their_shape(tmp_path, key, fake_submit_ok):
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", key, 5) == {"submitted": 1}


# --- auto-mint, status file, spawn argv -------------------------------------------------------


def fake_mint(monkeypatch, result=None):
    calls = []

    def fake(base_url, identity, timeout):
        calls.append(identity)
        return result or CairnResult(True, 201, {"api_key": "secret"}, "")

    monkeypatch.setattr(cairn_client, "mint_key", fake)
    return calls


def status_of(state):
    return json.loads((state / "cairn-status.json").read_text())


def test_flush_auto_mints_once_then_submits(tmp_path, monkeypatch):
    mints = fake_mint(monkeypatch)
    fake = install(monkeypatch, OK)
    queue(tmp_path, 2)
    key = tmp_path / "keys" / "k.key"
    counts = flush(tmp_path, "https://c", key, 5, auto_mint=True)
    assert counts == {"submitted": 2} and len(mints) == 1 and len(fake.calls) == 1
    assert mints[0].startswith("agent://anl-loam/loam-") and len(mints[0]) == len("agent://anl-loam/loam-") + 8
    assert status_of(tmp_path)["key"] == "minted" and status_of(tmp_path)["identity"] == mints[0]
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", key, 5, auto_mint=True)["submitted"] == 3
    assert len(mints) == 1 and status_of(tmp_path)["key"] == "present"


def test_flush_uses_explicit_identity(tmp_path, monkeypatch):
    mints = fake_mint(monkeypatch)
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    flush(tmp_path, "https://c", tmp_path / "k.key", 5, auto_mint=True, identity="agent://o/me")
    assert mints == ["agent://o/me"]


def test_flush_without_auto_mint_never_mints(tmp_path, monkeypatch):
    mints = fake_mint(monkeypatch)
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", tmp_path / "k.key", 5)["pending"] == 1
    assert mints == [] and status_of(tmp_path)["key"] == "missing"


def test_failed_mint_leaves_rows_pending_and_reports(tmp_path, monkeypatch):
    fake_mint(monkeypatch, CairnResult(False, 409, None, "http 409"))
    fake = install(monkeypatch, OK)
    queue(tmp_path, 1)
    assert flush(tmp_path, "https://c", tmp_path / "k.key", 5, auto_mint=True)["pending"] == 1
    st = status_of(tmp_path)
    assert st["key"] == "mint_failed" and "409" in st["error"] and fake.calls == []


def test_busy_mint_leaves_rows_pending(tmp_path, monkeypatch):
    mints = fake_mint(monkeypatch)
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    key = tmp_path / "k.key"
    key.write_text("")  # reserved by another process
    assert flush(tmp_path, "https://c", key, 5, auto_mint=True)["pending"] == 1
    assert mints == [] and status_of(tmp_path)["key"] == "mint_busy"


def test_report_never_mints_or_writes_status(tmp_path, monkeypatch, capsys):
    mints = fake_mint(monkeypatch)
    queue(tmp_path, 1)
    args = ["--state-dir", str(tmp_path), "--base-url", "https://c", "--key-file", str(tmp_path / "k.key"),
            "--auto-mint", "--report"]
    assert main(args) == 0
    assert mints == [] and not (tmp_path / "k.key").exists() and not (tmp_path / "cairn-status.json").exists()


def test_status_file_contents_and_no_key(tmp_path, monkeypatch):
    fake_mint(monkeypatch)
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    flush(tmp_path, "https://api.example", tmp_path / "k.key", 5, auto_mint=True)
    raw = (tmp_path / "cairn-status.json").read_text()
    st = json.loads(raw)
    assert set(st) == {"updated_at", "key_file", "key", "identity", "base_url", "counts", "error"}
    assert st["base_url"] == "https://api.example" and st["counts"] == {"submitted": 1} and st["error"] == ""
    assert st["key_file"] == str(tmp_path / "k.key")
    assert "secret" not in raw and not list(tmp_path.glob("*.tmp"))


def test_status_write_failure_is_swallowed(tmp_path, monkeypatch):
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    monkeypatch.setattr(cairn_flush.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("ro")))
    key = tmp_path / "k.key"
    key.write_text("secret\n")
    assert flush(tmp_path, "https://c", key, 5) == {"submitted": 1}


def test_main_passes_auto_mint_flags(tmp_path, monkeypatch):
    mints = fake_mint(monkeypatch)
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    main(["--state-dir", str(tmp_path), "--base-url", "https://c", "--key-file", str(tmp_path / "k.key"),
          "--auto-mint", "--identity-prefix", "agent://x/y-"])
    assert mints[0].startswith("agent://x/y-")


def test_spawn_flush_argv_includes_auto_mint(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cairn_flush.subprocess, "Popen", lambda *a, **k: calls.append(a[0]))
    block = {"enabled": True, "base_url": "https://c", "key_file": "~/k.key", "auto_flush": True,
             "auto_mint": True, "identity_prefix": "agent://p/q-"}
    spawn_flush({"loam_cairn": block, "loam_state_dir": str(tmp_path)})
    spawn_flush({"loam_cairn": {**block, "identity": "agent://o/me"}, "loam_state_dir": str(tmp_path)})
    spawn_flush({"loam_cairn": {**block, "auto_mint": False}, "loam_state_dir": str(tmp_path)})
    first, second, third = calls
    assert "--auto-mint" in first and first[first.index("--identity-prefix") + 1] == "agent://p/q-"
    assert "--identity" not in first and second[second.index("--identity") + 1] == "agent://o/me"
    assert "--auto-mint" not in third


def test_counts_report_unknown_entities(tmp_path, fake_submit_ok):
    for ent in ("tool://gromacs", "tool://gromacs", "tool://namd"):
        cairn_common.append_jsonl(tmp_path / cairn_common.QUEUE, {
            "event_id": f"u-{ent}-{len(cairn_common.read_jsonl(tmp_path / cairn_common.QUEUE))}",
            "status": "invalid", "reason": "unknown_entity", "entity": ent, "event": None})
    cairn_common.append_jsonl(tmp_path / cairn_common.QUEUE, {
        "event_id": "o", "status": "invalid", "reason": "bad_line", "event": None})
    counts = flush(tmp_path, "https://c", tmp_path / "none.key", 5)
    assert counts["unknown_entities"] == {"tool://gromacs": 2, "tool://namd": 1}
    assert counts["recorded"] == {"invalid": 4}


# --- fix round: backoff after ambiguous mint, locked-out flusher leaves status alone ---------


def test_recent_mint_failure_backs_off(tmp_path, monkeypatch):
    mints = fake_mint(monkeypatch, CairnResult(False, 0, None, "TimeoutError: no response"))
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    key = tmp_path / "k.key"
    assert flush(tmp_path, "https://c", key, 5, auto_mint=True)["pending"] == 1
    first = status_of(tmp_path)
    assert first["key"] == "mint_failed" and len(mints) == 1
    assert flush(tmp_path, "https://c", key, 5, auto_mint=True)["pending"] == 1
    assert len(mints) == 1 and status_of(tmp_path) == first  # skipped; status keeps the previous error


def test_old_mint_failure_is_retried(tmp_path, monkeypatch):
    mints = fake_mint(monkeypatch, CairnResult(False, 0, None, "TimeoutError: no response"))
    queue(tmp_path, 1)
    key = tmp_path / "k.key"
    flush(tmp_path, "https://c", key, 5, auto_mint=True)
    st = status_of(tmp_path)
    st["mint_failed_at"] = "2020-01-01T00:00:00+00:00"
    (tmp_path / "cairn-status.json").write_text(json.dumps(st))
    flush(tmp_path, "https://c", key, 5, auto_mint=True)
    assert len(mints) == 2


def test_backoff_survives_a_non_minting_flush(tmp_path, monkeypatch):
    mints = fake_mint(monkeypatch, CairnResult(False, 409, None, "HTTP 409"))
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    key = tmp_path / "k.key"
    flush(tmp_path, "https://c", key, 5, auto_mint=True)
    failed_at, failed_error = status_of(tmp_path)["mint_failed_at"], status_of(tmp_path)["error"]
    assert "409" in failed_error
    flush(tmp_path, "https://c", key, 5)  # the operator's manual flush
    st = status_of(tmp_path)
    assert st["key"] == "missing" and st["mint_failed_at"] == failed_at and st["error"] == failed_error
    flush(tmp_path, "https://c", key, 5, auto_mint=True)
    assert len(mints) == 1
    assert status_of(tmp_path)["mint_failed_at"] == failed_at


def test_successful_mint_drops_mint_failed_at(tmp_path, monkeypatch):
    fake_mint(monkeypatch)
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    (tmp_path / "cairn-status.json").write_text(
        json.dumps({"key": "missing", "error": "old", "mint_failed_at": "2020-01-01T00:00:00+00:00"})
    )
    flush(tmp_path, "https://c", tmp_path / "k.key", 5, auto_mint=True)
    st = status_of(tmp_path)
    assert st["key"] == "minted" and "mint_failed_at" not in st and st["error"] == ""


def test_present_key_drops_mint_failed_at(tmp_path, key, fake_submit_ok):
    (tmp_path / "cairn-status.json").write_text(json.dumps({"mint_failed_at": "2099-01-01T00:00:00+00:00"}))
    flush(tmp_path, "https://c", key, 5)
    assert "mint_failed_at" not in status_of(tmp_path)


def test_unparseable_or_naive_mint_failed_at_does_not_back_off(tmp_path, monkeypatch):
    mints = fake_mint(monkeypatch, CairnResult(False, 0, None, "x"))
    key = tmp_path / "k.key"
    for bad in ("garbage", "2099-01-01T00:00:00", "2099-01-01T00:00:00+00:00", 5):
        (tmp_path / "cairn-status.json").write_text(json.dumps({"mint_failed_at": bad}))
        flush(tmp_path, "https://c", key, 5, auto_mint=True)
    assert len(mints) == 4


def test_locked_out_flusher_does_not_overwrite_status(tmp_path, monkeypatch):
    install(monkeypatch, OK)
    queue(tmp_path, 1)
    (tmp_path / "cairn-status.json").write_text('{"sentinel": 1}')
    with cairn_common.locked(tmp_path / cairn_common.FLUSH_LOCK, blocking=False):
        assert flush(tmp_path, "https://c", tmp_path / "k.key", 5)["locked"] == 1
    assert json.loads((tmp_path / "cairn-status.json").read_text()) == {"sentinel": 1}
