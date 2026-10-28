#!/usr/bin/env python3
"""Explicit reconciliation for hypothesis provenance candidates.

This is deliberately not an automated truth classifier. An operator supplies a
review JSONL record for each candidate they choose to accept. Only source-backed
review evidence with ``validation_status: accepted`` enters the evidence ledger
or creates a contribution-credit record. Tournament results are not read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

HOOKS = Path(__file__).resolve().parents[1] / "hooks"
sys.path.insert(0, str(HOOKS))
from hypothesis_provenance import CONTRACT_VERSION  # noqa: E402
from ledger import append_record  # noqa: E402

HYPOTHESIS_VERSION = "loam_hypothesis_v1"
EVIDENCE_VERSION = "loam_evidence_v1"
CREDIT_VERSION = "loam_credit_v2"
EVIDENCE_ELO_SCALE = 400.0 / 2.302585092994046  # 400 / ln(10)


def _weight(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"accepted evidence {name} must be a number in [0, 1]")
    return float(value)


def _id(kind: str, material: Mapping[str, Any]) -> str:
    encoded = json.dumps({"kind": kind, **dict(material)}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"{kind}_{hashlib.sha256(encoded).hexdigest()}"


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        yield value


def reconcile(candidate_record: Mapping[str, Any], review: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce hypothesis, accepted-evidence, and credit records for one review."""

    if candidate_record.get("record_version") != CONTRACT_VERSION or candidate_record.get("candidate_status") != "pending":
        return [], [], []
    candidate = candidate_record.get("candidate")
    if not isinstance(candidate, Mapping) or review.get("candidate_id") != candidate_record.get("candidate_id"):
        return [], [], []
    approved = review.get("accept_hypothesis_keys", [])
    reviews = review.get("evidence", [])
    if not isinstance(approved, list) or not all(isinstance(key, str) for key in approved):
        raise ValueError("review accept_hypothesis_keys must be a string list")
    if not isinstance(reviews, list):
        raise ValueError("review evidence must be a list")

    scope = {
        "workflow_id": str(candidate_record.get("workflow_id") or ""),
        "parent_session_id": str(candidate_record.get("parent_session_id") or ""),
        "parent_turn_id": str(candidate_record.get("parent_turn_id") or ""),
        "candidate_id": str(candidate_record.get("candidate_id") or ""),
        "unit_id": str(candidate_record.get("unit_id") or ""),
    }
    hypotheses_by_key = {row["hypothesis_key"]: row for row in candidate.get("hypotheses", []) if isinstance(row, Mapping)}
    unknown = set(approved) - set(hypotheses_by_key)
    if unknown:
        raise ValueError(f"review references unknown hypothesis keys: {sorted(unknown)}")

    ranking_ids = review.get("ranking_hypothesis_ids", {})
    if not isinstance(ranking_ids, Mapping) or not all(isinstance(key, str) and isinstance(value, str) and value.strip() for key, value in ranking_ids.items()):
        raise ValueError("review ranking_hypothesis_ids must map hypothesis keys to non-empty Jnana IDs")
    if set(ranking_ids) - set(hypotheses_by_key):
        raise ValueError("review ranking_hypothesis_ids references unknown hypothesis keys")
    hypothesis_ids: dict[str, str] = {}
    hypothesis_rows = []
    for key in sorted(set(approved)):
        proposal = hypotheses_by_key[key]
        hypothesis_id = _id("hypothesis", {**scope, "hypothesis_key": key, "statement": proposal["statement"]})
        hypothesis_ids[key] = hypothesis_id
        hypothesis_rows.append(
            {
                "record_version": HYPOTHESIS_VERSION,
                "event": "hypothesis_accepted",
                "hypothesis_id": hypothesis_id,
                **scope,
                "hypothesis_key": key,
                "statement": proposal["statement"],
                "parent_hypothesis_keys": proposal.get("parent_hypothesis_keys", []),
                "status": "active",
                "jnana_hypothesis_id": str(ranking_ids.get(key) or hypothesis_id),
                "acceptance_basis": "explicit_operator_review",
                "scoring_prior_elo": 1200.0,
            }
        )

    evidence_rows, credit_rows = [], []
    allowed = {(row.get("hypothesis_key"), row.get("claim"), row.get("source_url"), row.get("stance")) for row in candidate.get("evidence_candidates", []) if isinstance(row, Mapping)}
    seen_evidence: set[str] = set()
    for item in reviews:
        if not isinstance(item, Mapping):
            raise ValueError("review evidence items must be objects")
        key = item.get("hypothesis_key")
        claim = item.get("claim")
        source_url = item.get("source_url")
        stance = item.get("stance")
        if (key, claim, source_url, stance) not in allowed:
            raise ValueError("review evidence must exactly match a candidate evidence item")
        if key not in hypothesis_ids:
            raise ValueError("accepted evidence must attach to an accepted hypothesis")
        if item.get("validation_status") != "accepted":
            continue
        source_ref = item.get("source_ref")
        if not isinstance(source_ref, str) or not source_ref.strip():
            raise ValueError("accepted evidence requires a non-empty source_ref")
        strength = _weight(item.get("strength"), "strength", 0.7)
        reliability = _weight(item.get("reliability"), "reliability", 0.7)
        relevance = _weight(item.get("relevance"), "relevance", 0.8)
        independence_group_id = item.get("independence_group_id")
        if independence_group_id is None:
            independence_group_id = source_ref.strip()
        if not isinstance(independence_group_id, str) or not independence_group_id.strip():
            raise ValueError("accepted evidence requires a non-empty independence_group_id")
        polarity = {"supports": 1.0, "contradicts": -1.0, "context": 0.0}[str(stance)]
        evidence_log_odds_delta = polarity * strength * reliability * relevance
        evidence_rating_delta = EVIDENCE_ELO_SCALE * evidence_log_odds_delta
        evidence_id = _id(
            "evidence",
            {
                "hypothesis_id": hypothesis_ids[key],
                "claim": claim,
                "source_url": source_url,
                "source_ref": source_ref.strip(),
                "stance": stance,
                "independence_group_id": independence_group_id.strip(),
            },
        )
        if evidence_id in seen_evidence:
            continue
        seen_evidence.add(evidence_id)
        evidence_rows.append(
            {
                "record_version": EVIDENCE_VERSION,
                "event": "evidence_accepted",
                "evidence_id": evidence_id,
                "hypothesis_id": hypothesis_ids[key],
                **scope,
                "claim": claim,
                "source_url": source_url,
                "source_ref": source_ref.strip(),
                "stance": stance,
                "strength": strength,
                "reliability": reliability,
                "relevance": relevance,
                "independence_group_id": independence_group_id.strip(),
                "evidence_log_odds_delta": evidence_log_odds_delta,
                "evidence_rating_delta": evidence_rating_delta,
                "validation_status": "accepted",
                "validation_basis": "explicit_operator_review",
            }
        )
        credit_rows.append(
            {
                "record_version": CREDIT_VERSION,
                "event": "credit_assigned",
                "credit_id": _id("credit", {"evidence_id": evidence_id, "unit_id": scope["unit_id"]}),
                "hypothesis_id": hypothesis_ids[key],
                "evidence_id": evidence_id,
                "recipient_kind": "subagent_unit",
                "recipient_id": scope["unit_id"],
                "credit_type": "marginal_validated_evidence_rating_delta",
                "amount": abs(evidence_rating_delta),
                "signed_evidence_rating_delta": evidence_rating_delta,
                "basis": "accepted_evidence_rating_delta",
                "independence_group_id": independence_group_id.strip(),
            }
        )
    return hypothesis_rows, evidence_rows, credit_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True, help="Operator-authored JSONL acceptance reviews")
    parser.add_argument("--hypotheses-out", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    parser.add_argument("--credit-out", type=Path, required=True)
    args = parser.parse_args()
    reviews = {str(row.get("candidate_id") or ""): row for row in _rows(args.reviews)}
    counts = {"hypotheses": 0, "evidence": 0, "credit": 0}
    for candidate in _rows(args.candidates):
        review = reviews.get(str(candidate.get("candidate_id") or ""))
        if review is None:
            continue
        hypotheses, evidence, credit = reconcile(candidate, review)
        for row in hypotheses:
            append_record(args.hypotheses_out, row)
            counts["hypotheses"] += 1
        for row in evidence:
            append_record(args.evidence_out, row)
            counts["evidence"] += 1
        for row in credit:
            append_record(args.credit_out, row)
            counts["credit"] += 1
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
