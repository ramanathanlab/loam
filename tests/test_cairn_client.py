"""Tests for the shared Cairn helpers and the stdlib HTTP client."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / ".loam" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import cairn_client  # noqa: E402
import cairn_common  # noqa: E402


class _Response:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _Fake:
    def __init__(self, status=200, body=b"[]", raises=None) -> None:
        self.status, self.body, self.raises = status, body, raises
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout=None):
        self.request, self.timeout = request, timeout
        if self.raises is not None:
            raise self.raises
        return _Response(self.status, self.body)


@pytest.fixture
def fake_urlopen(monkeypatch):
    fake = _Fake()
    monkeypatch.setattr(cairn_client, "_urlopen", fake)
    return fake


def _install(monkeypatch, **kwargs) -> _Fake:
    fake = _Fake(**kwargs)
    monkeypatch.setattr(cairn_client, "_urlopen", fake)
    return fake


def test_score_batch_posts_refs(fake_urlopen):
    refs = [{"type": "capability", "external_id": "tool://lammps"}]
    r = cairn_client.score_batch("https://c.example", refs, 5)
    assert r.ok and r.status == 200 and isinstance(r.data, list)
    assert fake_urlopen.request.full_url == "https://c.example/v1/score/batch"
    assert json.loads(fake_urlopen.request.data) == {"refs": refs}
    assert fake_urlopen.request.get_method() == "POST"
    assert fake_urlopen.request.get_header("Content-type") == "application/json"
    assert fake_urlopen.request.get_header("Accept") == "application/json"
    assert not fake_urlopen.request.has_header("X-api-key")
    assert fake_urlopen.timeout == 5


def test_submit_batch_sends_key_header(fake_urlopen):
    cairn_client.submit_batch("https://c.example/", "k-123", [{"score": 1}], 5)
    assert fake_urlopen.request.get_header("X-api-key") == "k-123"
    assert fake_urlopen.request.full_url == "https://c.example/v1/scores/batch"
    assert json.loads(fake_urlopen.request.data) == {"events": [{"score": 1}]}


def test_mint_key_body(fake_urlopen):
    cairn_client.mint_key("https://c.example", "agent://x/y", 5)
    assert fake_urlopen.request.full_url == "https://c.example/v1/keys"
    assert json.loads(fake_urlopen.request.data) == {"reviewer_external_id": "agent://x/y", "label": "loam"}


def test_http_error_is_a_result_not_an_exception(monkeypatch):
    err = urllib.error.HTTPError("https://c.example/v1/x?secret=1", 422, "bad", {}, io.BytesIO(b"body"))
    _install(monkeypatch, raises=err)
    r = cairn_client.submit_batch("https://c.example", "k", [], 5)
    assert (r.ok, r.status) == (False, 422) and "422" in r.error and "HTTPError" in r.error
    assert "secret" not in r.error


def test_network_error_has_status_zero(monkeypatch):
    _install(monkeypatch, raises=urllib.error.URLError("boom"))
    r = cairn_client.score_batch("https://c.example", [], 5)
    assert (r.ok, r.status) == (False, 0)


def test_timeout_and_oserror_have_status_zero(monkeypatch):
    for exc in (TimeoutError("t"), OSError("o")):
        _install(monkeypatch, raises=exc)
        r = cairn_client.score_batch("https://c.example", [], 5)
        assert (r.ok, r.status) == (False, 0) and type(exc).__name__ in r.error


def test_error_text_never_contains_the_key(monkeypatch):
    _install(monkeypatch, raises=urllib.error.URLError("k-secret"))
    assert "k-secret" not in cairn_client.submit_batch("https://c.example", "k-secret", [], 5).error


def test_bad_json_keeps_status(monkeypatch):
    _install(monkeypatch, status=200, body=b"<html>")
    r = cairn_client.score_batch("https://c.example", [], 5)
    assert (r.ok, r.status) == (False, 200) and r.error


def test_empty_body_is_ok_with_none(monkeypatch):
    _install(monkeypatch, status=204, body=b"")
    r = cairn_client.submit_batch("https://c.example", "k", [], 5)
    assert r.ok and r.status == 204 and r.data is None


def test_locked_nonblocking_reports_contention(tmp_path):
    with cairn_common.locked(tmp_path / "sub" / "x.lock") as first:
        with cairn_common.locked(tmp_path / "sub" / "x.lock", blocking=False) as second:
            assert first is True and second is False
    with cairn_common.locked(tmp_path / "sub" / "x.lock", blocking=False) as again:
        assert again is True


def test_read_jsonl_skips_garbage(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"a": 1}\nnot json\n\n[1]\n{"b": 2}\n')
    assert cairn_common.read_jsonl(p) == [{"a": 1}, {"b": 2}]
    assert cairn_common.read_jsonl(tmp_path / "missing.jsonl") == []


def test_append_jsonl_creates_parents_and_appends(tmp_path):
    p = tmp_path / "a" / "b.jsonl"
    cairn_common.append_jsonl(p, {"x": 1})
    cairn_common.append_jsonl(p, {"y": 2})
    assert cairn_common.read_jsonl(p) == [{"x": 1}, {"y": 2}]


def test_config_requires_enabled_true():
    assert cairn_common.config({"loam_cairn": {"enabled": True, "a": 1}}) == {"enabled": True, "a": 1}
    assert cairn_common.config({"loam_cairn": {"enabled": False}}) is None
    assert cairn_common.config({"loam_cairn": {"enabled": "yes"}}) is None
    assert cairn_common.config({"loam_cairn": "x"}) is None
    assert cairn_common.config({}) is None


def test_state_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("LOAM_STATE_DIR", raising=False)
    assert cairn_common.state_dir({}) is None
    monkeypatch.setenv("LOAM_STATE_DIR", str(tmp_path / "env"))
    assert cairn_common.state_dir({}) == tmp_path / "env"
    assert cairn_common.state_dir({"loam_state_dir": str(tmp_path / "p")}) == tmp_path / "p"


def test_marker_semantics():
    assert cairn_common.marker("x [loam-plan=abc] y", "loam-plan") == "abc"
    assert cairn_common.marker("nothing", "loam-plan") == ""
    assert cairn_common.marker("[loam-plan=abc", "loam-plan") == ""


def test_constants():
    assert cairn_common.ROLE_WEIGHTS == {"code-generator": 1.0, "hypothesis-validator": 0.5, "reasoner": 0.2}
    assert cairn_common.FALLBACK_WEIGHT == 0.2
    assert cairn_common.REPUTATION == "cairn-reputation.jsonl"


def test_protocol_error_is_a_result(monkeypatch):
    import http.client

    _install(monkeypatch, raises=http.client.IncompleteRead(b""))
    r = cairn_client.score_batch("https://c.example", [], 5)
    assert (r.ok, r.status) == (False, 0) and "IncompleteRead" in r.error


def test_unserializable_body_is_a_result(fake_urlopen):
    r = cairn_client.submit_batch("https://c.example", "k", [{"x": object()}], 5)
    assert (r.ok, r.status) == (False, 0) and "TypeError" in r.error
    assert fake_urlopen.request is None


def test_append_jsonl_large_row_intact(tmp_path):
    p = tmp_path / "big.jsonl"
    big = "x" * 20000
    cairn_common.append_jsonl(p, {"v": big})
    cairn_common.append_jsonl(p, {"w": 1})
    assert cairn_common.read_jsonl(p) == [{"v": big}, {"w": 1}]


def test_read_jsonl_survives_invalid_utf8(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_bytes(b'{"a": 1}\n\xff\xfe\x80\n{"b": 2}\n')
    assert cairn_common.read_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_failures_before_any_byte_left_are_marked_unsent(monkeypatch):
    import errno
    import socket

    for exc in (
        urllib.error.URLError(socket.gaierror(8, "nodename nor servname provided")),
        urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused")),
        ConnectionRefusedError(errno.ECONNREFUSED, "refused"),
        urllib.error.URLError(OSError(errno.EHOSTUNREACH, "no route")),
        OSError(errno.ENETUNREACH, "network unreachable"),
    ):
        _install(monkeypatch, raises=exc)
        r = cairn_client.submit_batch("https://c.example", "k", [], 5)
        assert (r.ok, r.status, r.sent) == (False, 0, False), exc


def test_ambiguous_failures_stay_sent(monkeypatch):
    import http.client

    for exc in (
        TimeoutError("read timed out"),
        urllib.error.URLError(TimeoutError("timed out")),
        urllib.error.URLError("boom"),
        ConnectionResetError(54, "reset"),
        http.client.IncompleteRead(b""),
        OSError("o"),
    ):
        _install(monkeypatch, raises=exc)
        r = cairn_client.submit_batch("https://c.example", "k", [], 5)
        assert (r.ok, r.status, r.sent) == (False, 0, True), exc
    err = urllib.error.HTTPError("https://c.example", 503, "x", {}, io.BytesIO(b""))
    _install(monkeypatch, raises=err)
    assert cairn_client.submit_batch("https://c.example", "k", [], 5).sent is True
    _install(monkeypatch, status=200, body=b"<html>")
    assert cairn_client.submit_batch("https://c.example", "k", [], 5).sent is True


def test_request_construction_errors_are_unsent(fake_urlopen):
    r = cairn_client.submit_batch("https://c.example", "k", [{"x": object()}], 5)
    assert (r.status, r.sent) == (0, False)
    r = cairn_client.submit_batch("c.example", "k", [], 5)  # no scheme: Request() raises ValueError
    assert (r.ok, r.status, r.sent) == (False, 0, False) and "ValueError" in r.error
    assert fake_urlopen.request is None


def test_result_sent_defaults_to_true():
    assert cairn_client.CairnResult(False, 0, None, "x").sent is True
