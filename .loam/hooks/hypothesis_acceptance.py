"""Run reconciliation for reviews an independent reviewer has already written.

This is the step a person used to run by hand between rounds:

    reconcile/hypothesis_provenance.py --candidates ... --reviews ...
    reconcile/project_hypothesis_memory.py ...
    reconcile/project_hypothesis_steering.py ...

Nothing here decides whether a claim is true. Every acceptance decision was
made, per claim, by an `evidence-reviewer` child that did not produce the claim
and never saw the producing validator's prose; `hypothesis_review.py` turned its
contract into a review row and set the weights. This module only applies the
reconciler to the review rows that exist and regenerates the two projections the
recall hook reads, so an unattended campaign's next round sees the evidence its
last round produced.

It is idempotent: each candidate is reconciled once, recorded in
`hypothesis-acceptance.jsonl`, and skipped afterwards. The reconciler's own
output is content-addressed anyway, but the log means a review that accepts
nothing is not retried on every turn.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ledger import append_record
from parser import Parser

ACCEPTANCE_LOG = "hypothesis-acceptance.jsonl"
REVIEWS_FILE = "hypothesis-reviews.jsonl"
CANDIDATES_FILE = "candidate-results.jsonl"
HYPOTHESIS_LEDGER = "hypothesis-ledger.jsonl"
EVIDENCE_LEDGER = "evidence-ledger.jsonl"
CREDIT_LEDGER = "credit-ledger.jsonl"
MEMORY_FILE = "hypothesis-memory.jsonl"
STEERING_FILE = "hypothesis-steering.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(payload: Mapping[str, Any], module_name: str) -> Any:
    """Import a `.loam/reconcile/` module under a private name.

    `reconcile/hypothesis_provenance.py` and `hooks/hypothesis_provenance.py`
    share a module name and are different modules; importing the reconciler
    normally would either collide in `sys.modules` or shadow the hook one for
    the rest of the process. Load it by path under a prefixed name instead.
    """

    path = Path(payload.get("repo_root") or ".") / ".loam" / "reconcile" / f"{module_name}.py"
    alias = f"_loam_reconcile_{module_name}"
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reconciler {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _reconciled(parser: Parser, payload: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get("candidate_id") or "")
        for row in parser.rows(payload, ACCEPTANCE_LOG)
        # A refused review is settled too: it will be refused identically next
        # turn, and an unattended campaign that retries it every round spends
        # the allocation on the same error.
        if row.get("event") in ("candidate_reconciled", "candidate_reconcile_failed")
    }


def reconcile(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reconcile every unreconciled review, then regenerate both projections.

    Returns a count summary. Raises only on a programming error: a malformed
    review is reported per candidate so one bad row cannot stall the campaign's
    acceptance for every other.
    """

    parser = Parser()
    reviews = [row for row in parser.rows(payload, REVIEWS_FILE) if str(row.get("candidate_id") or "")]
    if not reviews:
        return {"reconciled": 0, "hypotheses": 0, "evidence": 0, "credit": 0, "failed": 0}
    done = _reconciled(parser, payload)
    pending = [row for row in reviews if str(row["candidate_id"]) not in done]
    if not pending:
        return {"reconciled": 0, "hypotheses": 0, "evidence": 0, "credit": 0, "failed": 0}

    candidates = {
        str(row.get("candidate_id") or ""): row
        for row in parser.rows(payload, CANDIDATES_FILE)
        if row.get("event") == "candidate_result"
    }
    provenance = _load(payload, "hypothesis_provenance")
    counts = {"reconciled": 0, "hypotheses": 0, "evidence": 0, "credit": 0, "failed": 0}
    outputs = (
        (HYPOTHESIS_LEDGER, 0, "hypotheses"),
        (EVIDENCE_LEDGER, 1, "evidence"),
        (CREDIT_LEDGER, 2, "credit"),
    )
    for review in pending:
        candidate_id = str(review["candidate_id"])
        candidate = candidates.get(candidate_id)
        if candidate is None:
            _log(parser, payload, candidate_id, "candidate_missing", review)
            counts["failed"] += 1
            continue
        try:
            produced = provenance.reconcile(candidate, review)
        except Exception as exc:  # a malformed review must not stall the campaign
            _log(parser, payload, candidate_id, f"{type(exc).__name__}: {exc}", review)
            counts["failed"] += 1
            continue
        for filename, index, name in outputs:
            for row in produced[index]:
                append_record(parser.format_path(payload, filename), row)
                counts[name] += 1
        counts["reconciled"] += 1
        _log(parser, payload, candidate_id, "", review, produced)
    if counts["reconciled"]:
        project(payload)
    return counts


def _log(
    parser: Parser,
    payload: Mapping[str, Any],
    candidate_id: str,
    error: str,
    review: Mapping[str, Any],
    produced: tuple[list, list, list] | None = None,
) -> None:
    """Record that this candidate was reconciled, or why it could not be.

    Either way the candidate is not retried: a review the reconciler refuses
    will be refused identically next turn, and an unattended campaign that
    retries it every round spends the allocation on the same error.
    """

    parser.append(
        payload,
        ACCEPTANCE_LOG,
        {
            "event": "candidate_reconciled" if not error else "candidate_reconcile_failed",
            "observed_at": _now(),
            "run_id": payload.get("loam_run_id", ""),
            "workflow_id": payload.get("workflow_id", ""),
            "candidate_id": candidate_id,
            "reviewer_unit_id": review.get("reviewer_unit_id", ""),
            "plan_id": review.get("plan_id", ""),
            "accepted_hypotheses": len(produced[0]) if produced else 0,
            "accepted_evidence": len(produced[1]) if produced else 0,
            "verdicts": review.get("review_verdicts", {}),
            "error": error,
        },
    )


def project(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Regenerate `hypothesis-memory.jsonl` and `hypothesis-steering.jsonl`.

    Both are derived projections with a single authoritative writer, so they are
    rebuilt from the ledgers rather than appended to.
    """

    parser = Parser()
    memory = _load(payload, "project_hypothesis_memory")
    steering = _load(payload, "project_hypothesis_steering")
    hypotheses = parser.rows(payload, HYPOTHESIS_LEDGER)
    evidence = parser.rows(payload, EVIDENCE_LEDGER)
    credit = parser.rows(payload, CREDIT_LEDGER)
    _replace(
        parser.format_path(payload, MEMORY_FILE),
        memory.project(hypotheses, evidence, credit),
    )
    _replace(
        parser.format_path(payload, STEERING_FILE),
        steering.project(hypotheses, evidence, _reports(payload)),
    )
    return {"hypotheses": len(hypotheses), "evidence": len(evidence)}


def _reports(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every tournament report, so steering can read each hypothesis's Elo."""

    configured = (payload.get("loam_hypothesis_tournament") or {}).get("reports_dir") or ""
    if not configured:
        return []
    directory = Path(configured).expanduser()
    if not directory.is_dir():
        return []
    reports = []
    for path in sorted(directory.glob("tournament-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict):
            reports.append(value)
    return reports


def _replace(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write a projection atomically, so a concurrent recall never reads a torn file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)
