#!/usr/bin/env python3
"""Loam pre-tool policy hook.

Read one canonical Loam JSON payload from stdin. Return `{}` to allow the tool,
or `{"action": "block", "message": "..."}` to veto it.
"""

from __future__ import annotations

import json
import sys


def evaluate_code():
    # do some basic check on the code snippet
    # ensure the path to optionally supplied additional_paths exists
    pass

def register(ctx):
    ctx.register_hook('pre_tool_call', evaluate_code)

def main() -> int:
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
