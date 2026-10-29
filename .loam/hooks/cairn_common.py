"""Shared helpers for the Cairn reputation hooks (standard library only)."""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPUTATION = "cairn-reputation.jsonl"
QUEUE = "cairn-ratings.jsonl"
SUBMISSIONS = "cairn-submissions.jsonl"
REPUTATION_LOCK = "cairn-reputation.lock"
FLUSH_LOCK = "cairn-flush.lock"

ROLE_WEIGHTS = {"code-generator": 1.0, "hypothesis-validator": 0.5, "reasoner": 0.2}
FALLBACK_WEIGHT = 0.2

# Cairn's server rules (one non-conforming event 422s its whole atomic batch).
TAG_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")  # task_tags and failure_modes; use fullmatch
METRIC_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")  # metric names; use fullmatch
_NON_TAG_RE = re.compile(r"[^a-z0-9_]+")


def snake_tag(value: str) -> str:
    """Normalise text to a Cairn tag, or "" when nothing usable is left."""
    tag = _NON_TAG_RE.sub("_", str(value).lower()).strip("_")[:64].rstrip("_")
    return tag if tag and "a" <= tag[0] <= "z" else ""


def config(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    block = payload.get("loam_cairn")
    if isinstance(block, Mapping) and block.get("enabled") is True:
        return dict(block)
    return None


def state_dir(payload: Mapping[str, Any]) -> Path | None:
    raw = payload.get("loam_state_dir")
    text = raw.strip() if isinstance(raw, str) else ""
    text = text or os.environ.get("LOAM_STATE_DIR", "").strip()
    return Path(text).expanduser() if text else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for chunk in raw.split(b"\n"):
        line = chunk.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
    # Exactly one unbuffered write() on an O_APPEND descriptor keeps concurrent appends whole.
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


@contextmanager
def locked(path: Path, blocking: bool = True) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A fresh descriptor per call, so two calls contend even within one process.
    handle = open(path, "a+")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            acquired = True
        except OSError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def marker(text: str, key: str) -> str:
    prefix = f"[{key}="
    start = text.find(prefix)
    if start < 0:
        return ""
    end = text.find("]", start + len(prefix))
    return text[start + len(prefix) : end] if end >= 0 else ""
