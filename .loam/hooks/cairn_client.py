"""Stdlib HTTP client for the Cairn API. Never raises; errors become results."""

from __future__ import annotations

import errno
import http.client
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

_urlopen = urllib.request.urlopen


@dataclass(frozen=True)
class CairnResult:
    ok: bool
    status: int  # 0 means no HTTP response was received
    data: Any
    error: str
    # False only when the request provably never reached the server (safe to resend).
    sent: bool = True


_UNSENT_ERRNOS = frozenset({errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ECONNREFUSED})


def _never_left(exc: BaseException) -> bool:
    """True for DNS failure, refused connection or unreachable network: no byte reached the server."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, (socket.gaierror, ConnectionRefusedError)):
        return True
    return isinstance(reason, OSError) and reason.errno in _UNSENT_ERRNOS


def _post(url: str, body: Any, timeout: float, api_key: str = "") -> CairnResult:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    status = 0
    try:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
    except (TypeError, ValueError) as exc:  # body not JSON-serializable, or a malformed URL
        return CairnResult(False, 0, None, f"{type(exc).__name__}: request not built", sent=False)
    try:
        with _urlopen(request, timeout=timeout) as response:
            status = int(response.status if response.status is not None else response.getcode())
            raw = response.read()
        data = json.loads(raw.decode("utf-8")) if raw.strip() else None
    except urllib.error.HTTPError as exc:
        return CairnResult(False, int(exc.code), None, f"HTTPError: HTTP {exc.code}")
    except ValueError as exc:  # JSON or unicode decoding
        return CairnResult(False, status, None, f"{type(exc).__name__}: invalid response body (HTTP {status})")
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        return CairnResult(False, 0, None, f"{type(exc).__name__}: no response", sent=not _never_left(exc))
    return CairnResult(True, status, data, "")


def _url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def score_batch(base_url: str, refs: list[dict[str, str]], timeout: float) -> CairnResult:
    return _post(_url(base_url, "/v1/score/batch"), {"refs": refs}, timeout)


def submit_batch(base_url: str, api_key: str, events: list[dict[str, Any]], timeout: float) -> CairnResult:
    return _post(_url(base_url, "/v1/scores/batch"), {"events": events}, timeout, api_key)


def mint_key(base_url: str, identity: str, timeout: float) -> CairnResult:
    body = {"reviewer_external_id": identity, "label": "loam"}
    return _post(_url(base_url, "/v1/keys"), body, timeout)
