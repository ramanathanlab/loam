#!/usr/bin/env python3
"""Project accepted hypothesis evidence and explicit credit into parent context.

This projector reads only hypothesis acceptance, accepted evidence, and explicit
credit records. It never reads raw subagent summaries, pending candidates, or
Jnana tournament outcomes; tournament ranking remains a separate advisory signal.

It is the SINGLE authoritative writer of `hypothesis-memory.jsonl` (the file the
pre-LLM recall hook reads). No agent may hand-write or append to that file: it is
a derived projection and is regenerated here from the ledgers (H10). The write is
atomic (tmp + os.replace) so a concurrent recall read never sees a torn file.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        result.append(value)
    return result


def project(
    hypotheses: Iterable[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]], credit: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Build workflow/session-scoped compact hypothesis snapshots."""

    scopes: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"hypotheses": {}, "credit": defaultdict(float)})
    hypothesis_scope: dict[str, tuple[str, str]] = {}
    for row in hypotheses:
        if row.get("event") != "hypothesis_accepted" or row.get("status") != "active":
            continue
        hypothesis_id = str(row.get("hypothesis_id") or "")
        if not hypothesis_id:
            continue
        scope_key = (str(row.get("workflow_id") or ""), str(row.get("parent_session_id") or ""))
        scopes[scope_key]["hypotheses"].setdefault(
            hypothesis_id,
            {
                "hypothesis_id": hypothesis_id,
                "statement": str(row.get("statement") or ""),
                "parent_hypothesis_keys": list(row.get("parent_hypothesis_keys") or []),
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "context_evidence": [],
                "origin_unit_id": str(row.get("unit_id") or ""),
            },
        )
        hypothesis_scope[hypothesis_id] = scope_key

    seen_evidence: set[str] = set()
    for row in evidence:
        if row.get("event") != "evidence_accepted" or row.get("validation_status") != "accepted":
            continue
        hypothesis_id = str(row.get("hypothesis_id") or "")
        scope_key = hypothesis_scope.get(hypothesis_id)
        evidence_id = str(row.get("evidence_id") or "")
        if scope_key is None or not evidence_id or evidence_id in seen_evidence:
            continue
        seen_evidence.add(evidence_id)
        hypothesis = scopes[scope_key]["hypotheses"][hypothesis_id]
        stance = str(row.get("stance") or "context")
        target = {"supports": "supporting_evidence", "contradicts": "contradicting_evidence"}.get(stance, "context_evidence")
        hypothesis[target].append(
            {
                "evidence_id": evidence_id,
                "claim": str(row.get("claim") or ""),
                "source_url": str(row.get("source_url") or ""),
                "source_ref": str(row.get("source_ref") or ""),
            }
        )

    seen_credit: set[str] = set()
    for row in credit:
        if row.get("event") != "credit_assigned" or not str(row.get("basis") or "").startswith("accepted_evidence"):
            continue
        credit_id = str(row.get("credit_id") or "")
        if not credit_id or credit_id in seen_credit:
            continue
        seen_credit.add(credit_id)
        hypothesis_id = str(row.get("hypothesis_id") or "")
        scope_key = hypothesis_scope.get(hypothesis_id)
        if scope_key is None:
            continue
        try:
            scopes[scope_key]["credit"][hypothesis_id] += float(row.get("amount") or 0)
        except (TypeError, ValueError):
            continue

    snapshots = []
    for (workflow_id, parent_session_id), scope in sorted(scopes.items()):
        active = []
        for hypothesis in sorted(scope["hypotheses"].values(), key=lambda item: item["hypothesis_id"]):
            hypothesis["evidence_score"] = len(hypothesis["supporting_evidence"]) - len(hypothesis["contradicting_evidence"])
            hypothesis["contribution_credit"] = scope["credit"].get(hypothesis["hypothesis_id"], 0.0)
            for key in ("supporting_evidence", "contradicting_evidence", "context_evidence"):
                hypothesis[key].sort(key=lambda item: item["evidence_id"])
            active.append(hypothesis)
        snapshots.append(
            {
                "record_version": "loam_hypothesis_memory_v1",
                "event": "hypothesis_memory_snapshot",
                "workflow_id": workflow_id,
                "parent_session_id": parent_session_id,
                "active_hypotheses": active,
            }
        )
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--credit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snapshots = project(_rows(args.hypotheses), _rows(args.evidence), _rows(args.credit))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in snapshots)
    tmp = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, args.out)  # atomic: recall never reads a half-written projection
    print(json.dumps({"snapshots_written": len(snapshots), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
