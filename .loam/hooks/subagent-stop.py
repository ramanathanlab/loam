#!/usr/bin/env python3
"""Loam subagent-stop observer hook starter."""

from __future__ import annotations

import json
import sys

def dump_memory():
    # dump memory to hierarchical memory store
    pass

def register(ctx):
    ctx.register_hook('subagent_stop', dump_memory)

def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
