"""Shared HTTP plumbing for Loam's best-effort services (FALDA, Cairn).

Two things the stdlib does not give us. Proxy selection: on an Aurora compute node the only way out is
the ALCF proxy; when none is exported it is applied to our own opener only, never to os.environ.
(run-campaign separately exports https_proxy for Hermes's FALDA MCP connection, with local hosts in
no_proxy.) A wall-clock cap: urllib's ``timeout`` bounds each socket operation, not the request, so a
slow-dripping response can outlast it.
"""

from __future__ import annotations

import os
import re
import socket
import threading
import urllib.request
from typing import Any, Callable, Mapping, TypeVar

T = TypeVar("T")

ALCF_PROXY = "http://proxy.alcf.anl.gov:3128"
# Aurora compute nodes are named like x4102c5s3b0n0; login nodes are aurora-uan-*.
_COMPUTE_RE = re.compile(r"^x\d{4}c\d+s\d+b\d+n\d+(\.|$)")
_PROXY_ENV = ("https_proxy", "HTTPS_PROXY")


def is_aurora_compute(hostname: str) -> bool:
    return bool(_COMPUTE_RE.match(hostname or ""))


def proxy_for(environ: Mapping[str, str], hostname: str) -> str | None:
    """The proxy to force, or None to leave urllib's defaults (an env proxy, or direct)."""

    if any(str(environ.get(name) or "").strip() for name in _PROXY_ENV):
        return None
    return ALCF_PROXY if is_aurora_compute(hostname) else None


def urlopen(request: urllib.request.Request, timeout: float) -> Any:
    """``urllib.request.urlopen`` with the ALCF proxy forced on compute nodes; never touches os.environ."""

    proxy = proxy_for(os.environ, socket.gethostname())
    if proxy is None:
        return urllib.request.urlopen(request, timeout=timeout)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
    return opener.open(request, timeout=timeout)


def bounded(fn: Callable[[], T], seconds: float, fallback: T) -> T:
    """Run ``fn`` for at most ``seconds`` of wall clock; ``fallback`` if it raises or is still running.

    The worker is a daemon thread: an abandoned call never keeps the process alive.
    """

    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException:  # noqa: BLE001 - best effort: any failure means fallback
            pass

    worker = threading.Thread(target=_run, name="loam-bounded", daemon=True)
    worker.start()
    worker.join(max(0.0, float(seconds)))
    return box["value"] if "value" in box else fallback
