#!/usr/bin/env python3
"""Project accepted scientific findings into compact parent-scoped Loam memory.

The input is the append-only validation-decision stream.  This command accepts
only decisions whose validation status is ``accepted`` and whose finding ID is
non-empty.  It never reads child prose or pending candidates, so unvalidated
agent output cannot enter parent memory by this route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

MEMORY_VERSION = "loam_parent_memory_v1"


def _stable_id(material: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"version": MEMORY_VERSION, **dict(material)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"memory_{hashlib.sha256(encoded).hexdigest()}"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        yield value


def project(decisions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one deterministic, deduplicated memory snapshot per parent scope."""

    scopes: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "accepted": {},
            "attempted_unit_ids": set(),
            "source_accessions": set(),
            "validation_failures": set(),
        }
    )
    for decision in decisions:
        parent_session_id = str(decision.get("parent_session_id") or "")
        parent_turn_id = str(decision.get("parent_turn_id") or "")
        workflow_id = str(decision.get("workflow_id") or "")
        scope = scopes[(workflow_id, parent_session_id, parent_turn_id)]
        unit_id = str(decision.get("unit_id") or "")
        if unit_id:
            scope["attempted_unit_ids"].add(unit_id)
        status = str(decision.get("validation_status") or "")
        finding_id = str(decision.get("finding_id") or "")
        if status != "accepted" or not finding_id:
            if status:
                scope["validation_failures"].add(status)
            continue
        source_kind = str(decision.get("source_kind") or "")
        source_accession = str(decision.get("source_accession") or "")
        if source_accession:
            scope["source_accessions"].add(source_accession)
        scope["accepted"].setdefault(
            finding_id,
            {
                "finding_id": finding_id,
                "source_kind": source_kind,
                "source_accession": source_accession,
                "source_url": str(decision.get("source_url") or ""),
                "field": str(decision.get("field") or ""),
                "value": decision.get("authoritative_value"),
                "unit_ids": set(),
            },
        )["unit_ids"].add(unit_id)

    snapshots = []
    for (workflow_id, parent_session_id, parent_turn_id), scope in sorted(scopes.items()):
        findings = []
        for finding in sorted(scope["accepted"].values(), key=lambda row: row["finding_id"]):
            findings.append({**finding, "unit_ids": sorted(unit for unit in finding["unit_ids"] if unit)})
        payload = {
            "record_version": MEMORY_VERSION,
            "event": "parent_memory_snapshot",
            "workflow_id": workflow_id,
            "parent_session_id": parent_session_id,
            "parent_turn_id": parent_turn_id,
            "accepted_findings": findings,
            "accepted_finding_ids": [finding["finding_id"] for finding in findings],
            "attempted_unit_ids": sorted(scope["attempted_unit_ids"]),
            "source_accessions_seen": sorted(scope["source_accessions"]),
            "validation_failure_statuses": sorted(scope["validation_failures"]),
        }
        payload["memory_id"] = _stable_id(
            {
                "parent_session_id": parent_session_id,
                "parent_turn_id": parent_turn_id,
                "accepted_finding_ids": payload["accepted_finding_ids"],
            }
        )
        snapshots.append(payload)
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.decisions.is_file():
        parser.error(f"decision file does not exist: {args.decisions}")
    snapshots = project(_read_jsonl(args.decisions))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n" for snapshot in snapshots),
        encoding="utf-8",
    )
    print(json.dumps({"snapshots_written": len(snapshots), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
