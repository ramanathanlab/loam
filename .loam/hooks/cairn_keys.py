"""Reserve-then-claim minting of Loam's Cairn reviewer key (standard library only).

A Cairn reviewer identity can be claimed once, ever, so the key file is reserved (O_EXCL, 0600)
before the identity is claimed, an existing key is never overwritten, and a key that was fully
written is never removed. The key is never printed, logged or returned.
"""

from __future__ import annotations

import os
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cairn_client  # noqa: E402
import cairn_common  # noqa: E402

STALE_SECONDS = 600  # an empty reservation older than this was abandoned (a mint takes at most ~10 s)


@dataclass(frozen=True)
class KeyStatus:
    state: str  # present | minted | busy | failed
    identity: str
    error: str


def _prepare_dir(directory: Path) -> str:
    """Create missing directories (0700) or validate an existing one; never chmod existing. Return an error or ""."""
    if directory.exists():
        if not directory.is_dir():
            return f"key directory path is not a directory: {directory}"
        mode = stat.S_IMODE(directory.stat().st_mode)
        if mode & 0o077:
            return (
                f"key directory {directory} has mode {mode:04o}; it must have no "
                "group or other permissions (fix it yourself, it is not changed here)"
            )
        return ""
    missing = []
    probe = directory
    while not probe.exists():
        missing.append(probe)
        probe = probe.parent
    for created in reversed(missing):
        try:
            created.mkdir(mode=0o700)
        except FileExistsError:
            if not created.is_dir():
                raise
            continue  # another process just created it; validated below
        created.chmod(0o700)
    return _prepare_dir(directory) if missing else ""


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _non_empty(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def ensure_key(key_file: Path, base_url: str, identity: str, timeout: float, project: str | None = None) -> KeyStatus:
    path = Path(key_file).expanduser()
    if _non_empty(path):
        return KeyStatus("present", identity, "")
    refused = cairn_client.project_error(cairn_client.env_project() if project is None else project)
    if refused:  # before reserving: a refused request must not look like an ambiguous claim
        return KeyStatus("failed", identity, refused)
    try:
        error = _prepare_dir(path.parent)
    except OSError as exc:
        return KeyStatus("failed", identity, f"cannot prepare key file {path}: {exc}")
    if error:
        return KeyStatus("failed", identity, error)
    fd = -1
    try:
        try:
            # Reserve the file before claiming the identity (claimable once, ever).
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if not _set_aside_abandoned(path):
                return KeyStatus("busy", identity, f"key file already exists or appeared: {path}")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # exactly one retry
    except FileExistsError:
        return KeyStatus("busy", identity, f"key file already exists or appeared: {path}")
    except OSError as exc:
        return KeyStatus("failed", identity, f"cannot prepare key file {path}: {exc}")

    def discard() -> None:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            pass

    progress = {'written': False}
    try:
        return _claim_and_write(path, fd, discard, progress, base_url, identity, timeout, project)
    except BaseException:
        # Ctrl-C / SIGTERM / SystemExit must not leave an empty reservation behind,
        # but a completely written key is never discarded.
        if not progress['written']:
            discard()
        raise


def _is_stale_empty(info: os.stat_result) -> bool:
    return stat.S_ISREG(info.st_mode) and info.st_size == 0 and time.time() - info.st_mtime > STALE_SECONDS


def _set_aside_abandoned(path: Path) -> bool:
    """Rename an abandoned empty reservation aside (never unlink). True when the path was freed."""
    try:
        before = os.lstat(path)
    except OSError:
        return False
    if not _is_stale_empty(before):
        return False
    with cairn_common.locked(Path(str(path) + ".lock"), blocking=False) as acquired:
        if not acquired:
            return False
        try:
            now = os.lstat(path)
            if (now.st_dev, now.st_ino) != (before.st_dev, before.st_ino) or not _is_stale_empty(now):
                return False
            os.rename(path, f"{path}.abandoned.{os.getpid()}.{int(time.time())}")
        except OSError:
            return False
    return True


def _claim_and_write(
    path: Path, fd: int, discard, progress: dict, base_url: str, identity: str, timeout: float, project: str | None
) -> KeyStatus:
    # Keys are not bound to projects, but every Cairn request still carries the header.
    result = cairn_client.mint_key(base_url, identity, timeout, project=project)
    api_key = result.data.get("api_key") if result.ok and isinstance(result.data, dict) else None
    if not isinstance(api_key, str) or not api_key:
        discard()
        message = f"mint failed for {identity} (status {result.status}): {result.error or 'response carried no api_key'}"
        if result.status == 0:
            message += (
                f"; no response was received, so identity {identity} MAY already have been "
                "claimed; do not retry blindly — ask Cairn whether it exists before minting again"
            )
        return KeyStatus("failed", identity, message)
    try:
        _write_all(fd, (api_key + "\n").encode())
    except OSError as exc:
        discard()
        return KeyStatus(
            "failed",
            identity,
            f"identity {identity} has already been claimed but the key could not "
            f"be written to {path} ({exc}); contact Cairn to recover it",
        )
    progress['written'] = True
    # The key is fully written: from here on a failure must never remove it.
    problems = []
    try:
        os.fsync(fd)
    except OSError as exc:
        problems.append(f"fsync failed: {exc}")
    try:
        os.close(fd)
    except OSError as exc:
        problems.append(f"close failed: {exc}")
    _write_identity(path, identity)
    if problems:
        return KeyStatus(
            "failed",
            identity,
            f"minted key for {identity} was written to {path}, but {'; '.join(problems)}. "
            "The file was kept: run `check`, verify its contents and back it up now.",
        )
    return KeyStatus("minted", identity, "")


def _write_identity(path: Path, identity: str) -> None:
    """Record the (non-secret) identity next to the key so status can report it. Best effort."""
    try:
        fd = os.open(str(path) + ".identity", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            _write_all(fd, (identity + "\n").encode())
        finally:
            os.close(fd)
    except OSError:
        pass
