"""Tests for the shared HTTP helper used by the FALDA and Cairn clients."""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

HOOKS = Path(__file__).resolve().parents[1] / ".loam" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import loam_http  # noqa: E402

COMPUTE = "x4102c5s3b0n0"


def test_compute_and_login_hostnames() -> None:
    assert loam_http.is_aurora_compute(COMPUTE)
    assert loam_http.is_aurora_compute(COMPUTE + ".hostmgmt2000.cm.aurora.alcf.anl.gov")
    assert not loam_http.is_aurora_compute("aurora-uan-0010")
    assert not loam_http.is_aurora_compute("ryans-macbook")
    assert not loam_http.is_aurora_compute("")


def test_env_proxy_wins() -> None:
    assert loam_http.proxy_for({"https_proxy": "http://p:1"}, COMPUTE) is None
    assert loam_http.proxy_for({"HTTPS_PROXY": "http://p:1"}, COMPUTE) is None


def test_compute_node_without_env_uses_alcf_proxy() -> None:
    assert loam_http.proxy_for({}, COMPUTE) == loam_http.ALCF_PROXY
    assert loam_http.proxy_for({"https_proxy": " "}, COMPUTE) == loam_http.ALCF_PROXY


def test_login_and_laptop_go_direct() -> None:
    assert loam_http.proxy_for({}, "aurora-uan-0010") is None
    assert loam_http.proxy_for({}, "laptop") is None


def test_urlopen_uses_alcf_opener_without_touching_environ(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Opener:
        def open(self, request, timeout=None):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return "response"

    def _build_opener(*handlers):
        for handler in handlers:
            if isinstance(handler, urllib.request.ProxyHandler):
                captured["proxies"] = dict(handler.proxies)
        return _Opener()

    for name in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(loam_http.socket, "gethostname", lambda: COMPUTE)
    monkeypatch.setattr(loam_http.urllib.request, "build_opener", _build_opener)
    before = dict(os.environ)
    result = loam_http.urlopen(urllib.request.Request("https://example.invalid/x"), 1.5)
    assert result == "response"
    assert captured["proxies"] == {"https": loam_http.ALCF_PROXY, "http": loam_http.ALCF_PROXY}
    assert captured["timeout"] == 1.5
    assert dict(os.environ) == before


def test_urlopen_direct_uses_stdlib(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(loam_http.socket, "gethostname", lambda: "laptop")
    monkeypatch.setattr(loam_http.urllib.request, "urlopen", lambda request, timeout=None: seen.setdefault("t", timeout))
    loam_http.urlopen(urllib.request.Request("https://example.invalid/x"), 3)
    assert seen == {"t": 3}


def test_bounded_returns_result() -> None:
    assert loam_http.bounded(lambda: 7, 2, None) == 7


def test_bounded_falls_back_on_raise() -> None:
    assert loam_http.bounded(lambda: 1 / 0, 2, "fb") == "fb"


def test_bounded_is_wall_clock() -> None:
    start = time.monotonic()
    assert loam_http.bounded(lambda: time.sleep(5) or 1, 0.3, "fb") == "fb"
    assert time.monotonic() - start < 1.0
