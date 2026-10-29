#!/usr/bin/env python3
"""One-time operator tool: mint Loam's Cairn reviewer key, or check it.

A Cairn reviewer identity can be claimed only once, so `mint` refuses to run
when the key file already exists and never overwrites. The key is never printed.

Exit codes: 0 success, 1 failure, 2 refused because the key file exists.
"""

from __future__ import annotations

import argparse
import os  # noqa: F401  (tests patch os.write/os.close through this module)
import secrets
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".loam" / "hooks"))

import cairn_keys  # noqa: E402

DEFAULT_IDENTITY_PREFIX = "agent://anl-loam/loam-"
DEFAULT_KEY_FILE = "~/.cairn/keys/agent-loam.key"
DEFAULT_BASE_URL = "https://api.cairnscore.ai"


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _mint(args: argparse.Namespace) -> int:
    path = Path(args.key_file).expanduser()
    status = cairn_keys.ensure_key(path, args.base_url, args.identity, args.timeout)
    if status.state == "present":
        _err(f"refusing to mint: key file already exists: {path}")
        return 2
    if status.state == "busy":
        _err(f"refusing to mint: key file appeared: {path}")
        return 2
    if status.state == "failed":
        _err(status.error)
        return 1
    print(
        f"minted key for {args.identity} at {path}; back it up now, "
        "it cannot be re-minted."
    )
    return 0


def _check(args: argparse.Namespace) -> int:
    path = Path(args.key_file).expanduser()
    try:
        file_stat = path.stat()
        dir_stat = path.parent.stat()
    except OSError:
        _err(f"key file missing: {path}")
        return 1
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size == 0:
        _err(f"key file is empty or not a regular file: {path}")
        return 1
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
        _err(f"key file is accessible to group or others: {path}")
        return 1
    if stat.S_IMODE(dir_stat.st_mode) & 0o077:
        _err(f"key directory is accessible to group or others: {path.parent}")
        return 1
    print(f"key file ok: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    mint = sub.add_parser("mint", help="mint the reviewer key (once, ever)")
    mint.add_argument("--identity", default=None, help="default: agent://anl-loam/loam-<8 random hex>")
    mint.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    mint.add_argument("--base-url", default=DEFAULT_BASE_URL)
    mint.add_argument("--timeout", type=float, default=10.0)
    mint.set_defaults(func=_mint)
    check = sub.add_parser("check", help="verify the key file and its permissions")
    check.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    check.set_defaults(func=_check)
    args = parser.parse_args(argv)
    if getattr(args, "identity", "x") is None:
        args.identity = DEFAULT_IDENTITY_PREFIX + secrets.token_hex(4)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
