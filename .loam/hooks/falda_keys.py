"""FALDA machine keys for Loam (standard library only).

A key reaches only the pool it enrolled in, so keys are cached per pool:
``<key_dir>/loam-<pool>.json`` (0600, directory 0700). ``$HOME`` is shared across
Aurora nodes and every compute node leaves through the one proxy IP (FALDA allows
50 mints/hour per IP), so one process mints for everyone: the mint is reserved by
atomic create of a lock file (O_EXCL, not flock, which parallel filesystems do not
reliably honour), and a process that loses the race simply goes on without a key.
Only the detached flusher mints; hooks call ``resolve_key``, which never touches
the network. The key is never logged or written anywhere but the key file.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loam_http  # noqa: E402

STALE_LOCK_SECONDS = 600  # a mint takes seconds; an older reservation was abandoned
MINT_COOLDOWN_SECONDS = 600
RATE_LIMIT_COOLDOWN_SECONDS = 3600

Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


def _urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with loam_http.urlopen(request, timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read() or b""


_transport: Transport = _urllib_transport


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def key_path(cfg: Mapping[str, Any]) -> Path:
    directory = Path(_text(cfg.get("key_dir")) or "~/.falda/keys").expanduser()
    return directory / f"loam-{_text(cfg.get('pool'))}.json"


def _sidecar(cfg: Mapping[str, Any], suffix: str) -> Path:
    return Path(str(key_path(cfg)) + suffix)


def resolve_key(cfg: Mapping[str, Any]) -> str:
    """The key from the environment, else from the cached file for this pool, else "". No network."""

    try:
        env_key = os.environ.get(_text(cfg.get("token_env")) or "FALDA_TOKEN", "").strip()
        if env_key:
            return env_key
        value = json.loads(key_path(cfg).read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and _text(value.get("pool")) == _text(cfg.get("pool")):
            return _text(value.get("api_key"))
    except Exception:  # noqa: BLE001 - a missing or unreadable key file means "no key"
        pass
    return ""


def _cooling_down(cfg: Mapping[str, Any]) -> bool:
    try:
        marker = json.loads(_sidecar(cfg, ".mint-failed").read_text(encoding="utf-8"))
        wait = RATE_LIMIT_COOLDOWN_SECONDS if marker.get("reason") == "rate_limited" else MINT_COOLDOWN_SECONDS
        return time.time() - float(marker.get("at", 0)) < wait
    except Exception:  # noqa: BLE001
        return False


def _record_failure(cfg: Mapping[str, Any], reason: str) -> None:
    try:
        _sidecar(cfg, ".mint-failed").write_text(json.dumps({"at": time.time(), "reason": reason}), encoding="utf-8")
    except OSError:
        pass


def _reserve(lock: Path) -> bool:
    """Take the mint reservation; replace one abandoned for longer than STALE_LOCK_SECONDS."""

    for _ in range(2):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
            return True
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime <= STALE_LOCK_SECONDS:
                    return False
                lock.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                return False
    return False


def _label() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = "unknown"
    return f"loam-{socket.gethostname().split('.')[0]}-{user}"[:120]


def _mint(cfg: Mapping[str, Any], timeout: float) -> tuple[str, str]:
    """(key, "") on success, ("", reason) on failure."""

    url = _text(cfg.get("base_url")).rstrip("/") + "/auth/machine"
    body = json.dumps({"pool": _text(cfg.get("pool")), "label": _label()}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        status, raw = _transport(url, headers, body, timeout)
    except Exception as exc:  # noqa: BLE001 - log the type only
        return "", f"network:{type(exc).__name__}"
    try:
        value = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, ValueError):
        value = {}
    if not isinstance(value, Mapping):
        value = {}
    if status >= 400:
        return "", _text(value.get("error")) or f"http_{status}"
    key = _text(value.get("api_key"))
    return (key, "") if key.startswith("falda_") else ("", "no_api_key")


def _write_key(cfg: Mapping[str, Any], key: str) -> None:
    path = key_path(cfg)
    record = {"api_key": key, "pool": _text(cfg.get("pool")), "label": _label(), "minted_at": time.time()}
    fd, temporary = tempfile.mkstemp(prefix=".loam-key-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)  # readers never see a partial key
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def ensure_key(cfg: Mapping[str, Any], timeout: float) -> tuple[str, str]:
    """(key, status): env | cached | minted | minting_elsewhere | cooling_down | mint_failed:<reason> | disabled."""

    try:
        if os.environ.get(_text(cfg.get("token_env")) or "FALDA_TOKEN", "").strip():
            return resolve_key(cfg), "env"
        key = resolve_key(cfg)
        if key:
            return key, "cached"
        if cfg.get("auto_mint") is not True:
            return "", "disabled"
        if _cooling_down(cfg):
            return "", "cooling_down"
        path = key_path(cfg)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = _sidecar(cfg, ".lock")
        if not _reserve(lock):
            return "", "minting_elsewhere"
        try:
            key = resolve_key(cfg)  # another process may have finished between our check and the lock
            if key:
                return key, "cached"
            key, reason = _mint(cfg, timeout)
            if not key:
                _record_failure(cfg, reason)
                return "", f"mint_failed:{reason}"
            _write_key(cfg, key)
            try:
                _sidecar(cfg, ".mint-failed").unlink()
            except OSError:
                pass
            return key, "minted"
        finally:
            try:
                lock.unlink()
            except OSError:
                pass
    except Exception as exc:  # noqa: BLE001 - never raise into a flusher
        return "", f"mint_failed:client:{type(exc).__name__}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Print the cached FALDA key for a Loam falda block (no network).")
    parser.add_argument("--print-key", action="store_true", required=True)
    parser.add_argument("--config-json", required=True, help="the falda block as JSON")
    args = parser.parse_args(argv)
    try:
        cfg = json.loads(args.config_json)
    except ValueError:
        return 2
    print(resolve_key(cfg) if isinstance(cfg, Mapping) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
