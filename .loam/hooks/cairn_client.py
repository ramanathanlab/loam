"""Stdlib HTTP client for the Cairn API. Never raises; errors become results."""

from __future__ import annotations

import errno
import http.client
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import loam_http

# Forces the ALCF proxy on Aurora compute nodes without exporting it (loam_http).
_urlopen = loam_http.urlopen

# Every request carries X-Cairn-Project so it reads and writes this experiment's private silo.
# A request without the header would hit production, so an unset or invalid project is never sent.
PROJECT_ENV = "CAIRN_PROJECT"
PROJECT_HEADER = "X-Cairn-Project"
PROJECT_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,62}")  # use fullmatch


def project_error(project: str | None) -> str:
    """"" when ``project`` is a valid Cairn project name, otherwise why it is not."""
    if not project:
        return f"{PROJECT_ENV} is not set; refusing to call Cairn without {PROJECT_HEADER}"
    if not PROJECT_RE.fullmatch(project):
        return f"{PROJECT_ENV}={project!r} is not a valid Cairn project name (^[a-z0-9][a-z0-9-]{{2,62}}$)"
    return ""


def env_project() -> str:
    return os.environ.get(PROJECT_ENV, "").strip()


@dataclass(frozen=True)
class CairnResult:
    ok: bool
    status: int  # 0 means no HTTP response was received
    data: Any
    error: str
    # False only when the request provably never reached the server (safe to resend).
    sent: bool = True
    # "ok" when the response echoed our project, "missing"/"mismatch" otherwise, "" with no response.
    echo: str = ""


_UNSENT_ERRNOS = frozenset({errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ECONNREFUSED})


def _never_left(exc: BaseException) -> bool:
    """True for DNS failure, refused connection or unreachable network: no byte reached the server."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, (socket.gaierror, ConnectionRefusedError)):
        return True
    return isinstance(reason, OSError) and reason.errno in _UNSENT_ERRNOS


def _echo(headers: Any, project: str) -> str:
    try:
        got = headers.get(PROJECT_HEADER) if headers is not None else None
    except Exception:  # noqa: BLE001
        got = None
    if got is None:
        state = "missing"
    else:
        state = "ok" if str(got).strip() == project else "mismatch"
    if state != "ok":
        print(f"loam cairn: WARNING response {PROJECT_HEADER} echo {state} (sent {project!r}, got {got!r})", file=sys.stderr)
    return state


def _post(url: str, body: Any, timeout: float, project: str | None, api_key: str = "") -> CairnResult:
    project = env_project() if project is None else project.strip()
    refused = project_error(project)
    if refused:
        print(f"loam cairn: ERROR {refused}", file=sys.stderr)
        return CairnResult(False, 0, None, refused, sent=False)
    headers = {"Content-Type": "application/json", "Accept": "application/json", PROJECT_HEADER: project}
    if api_key:
        headers["X-Api-Key"] = api_key
    status = 0
    try:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
    except (TypeError, ValueError) as exc:  # body not JSON-serializable, or a malformed URL
        return CairnResult(False, 0, None, f"{type(exc).__name__}: request not built", sent=False)
    echo = ""
    try:
        with _urlopen(request, timeout=timeout) as response:
            status = int(response.status if response.status is not None else response.getcode())
            echo = _echo(getattr(response, "headers", None), project)
            raw = response.read()
        data = json.loads(raw.decode("utf-8")) if raw.strip() else None
    except urllib.error.HTTPError as exc:
        return CairnResult(False, int(exc.code), None, f"HTTPError: HTTP {exc.code}", echo=_echo(exc.headers, project))
    except ValueError as exc:  # JSON or unicode decoding
        return CairnResult(False, status, None, f"{type(exc).__name__}: invalid response body (HTTP {status})", echo=echo)
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        return CairnResult(False, 0, None, f"{type(exc).__name__}: no response", sent=not _never_left(exc))
    return CairnResult(True, status, data, "", echo=echo)


def _url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


# ``project=None`` reads CAIRN_PROJECT; callers that bound the project at launch pass it explicitly.
def score_batch(base_url: str, refs: list[dict[str, str]], timeout: float, project: str | None = None) -> CairnResult:
    return _post(_url(base_url, "/v1/score/batch"), {"refs": refs}, timeout, project)


def submit_batch(
    base_url: str, api_key: str, events: list[dict[str, Any]], timeout: float, project: str | None = None
) -> CairnResult:
    return _post(_url(base_url, "/v1/scores/batch"), {"events": events}, timeout, project, api_key)


def mint_key(base_url: str, identity: str, timeout: float, project: str | None = None) -> CairnResult:
    body = {"reviewer_external_id": identity, "label": "loam"}
    return _post(_url(base_url, "/v1/keys"), body, timeout, project)
