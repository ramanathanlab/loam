#!/usr/bin/env python3
"""Backfill unvalidated Loam candidates from completed lifecycle records.

Use after a parser upgrade when a prior completed child record contains an opt-in
contract that was not extracted at hook time. The command only appends missing
candidate IDs; it never accepts claims or rewrites the ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

HOOKS = Path(__file__).resolve().parents[1] / "hooks"
sys.path.insert(0, str(HOOKS))
from ledger import append_record, build_candidate_record  # noqa: E402


def _read(path: Path) -> Iterable[dict[str, Any]]:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        yield record


def backfill(ledger: Path, candidates: Path) -> int:
    existing = {
        str(row.get("candidate_id") or "")
        for row in _read(candidates)
    } if candidates.is_file() else set()
    added = 0
    for record in _read(ledger):
        if record.get("event") != "subagent_stop":
            continue
        candidate = build_candidate_record(record)
        if candidate is None or candidate["candidate_id"] in existing:
            continue
        append_record(candidates, candidate)
        existing.add(candidate["candidate_id"])
        added += 1
    return added


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    if not args.ledger.is_file():
        parser.error(f"ledger file does not exist: {args.ledger}")
    added = backfill(args.ledger, args.candidates)
    print(json.dumps({"candidates_added": added, "candidates": str(args.candidates)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
