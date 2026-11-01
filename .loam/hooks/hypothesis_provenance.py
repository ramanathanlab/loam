"""Opt-in, unvalidated hypothesis/evidence provenance candidate contract.

This contract records proposed hypotheses and source-located evidence candidates
from a completed child.  It deliberately does not accept claims, alter a parent
hypothesis, or assign credit; those occur only in explicit reconciliation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

CONTRACT_VERSION = "hypothesis_provenance_v1"
_FENCED_JSON = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n?\s*(\{.*?\})\s*```", re.DOTALL)
_KEY = re.compile(r"[a-z][a-z0-9_-]{0,79}")
_STANCES = {"supports", "contradicts", "context"}


def _stripped(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def stable_id(kind: str, material: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"contract_version": CONTRACT_VERSION, "kind": kind, **dict(material)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{kind}_{hashlib.sha256(encoded).hexdigest()}"


def extract_fenced_contract(summary: str, version: str) -> tuple[dict[str, Any] | None, str]:
    """The last fenced JSON object in a child's summary declaring ``version``.

    Shared by every contract an agent emits this way, so a second contract type
    does not fork the extraction rules -- which fences count, which JSON parses
    -- into a near-copy that then drifts from this one.
    """

    for match in _FENCED_JSON.finditer(summary or ""):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("contract_version") == version:
            return value, "present"
    return None, "absent"


def extract_contract_object(summary: str) -> tuple[dict[str, Any] | None, str]:
    return extract_fenced_contract(summary, CONTRACT_VERSION)


def normalize_candidate(value: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if value.get("contract_version") != CONTRACT_VERSION:
        return None, "unsupported_contract"
    hypotheses = value.get("hypotheses")
    evidence = value.get("evidence_candidates", [])
    if not isinstance(hypotheses, list) or not hypotheses:
        return None, "invalid_hypotheses"
    if not isinstance(evidence, list):
        return None, "invalid_evidence_candidates"

    normalized_hypotheses: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for raw in hypotheses:
        if not isinstance(raw, Mapping):
            return None, "invalid_hypothesis"
        key = raw.get("hypothesis_key")
        statement = _stripped(raw.get("statement"))
        parents = raw.get("parent_hypothesis_keys", [])
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            return None, "invalid_hypothesis_key"
        if key in seen_keys:
            return None, "duplicate_hypothesis_key"
        if not statement or len(statement) > 4000:
            return None, "invalid_hypothesis_statement"
        if not isinstance(parents, list) or not all(isinstance(parent, str) and _KEY.fullmatch(parent) for parent in parents):
            return None, "invalid_parent_hypothesis_keys"
        seen_keys.add(key)
        normalized_hypotheses.append(
            {"hypothesis_key": key, "statement": statement, "parent_hypothesis_keys": sorted(set(parents))}
        )

    normalized_evidence: list[dict[str, str]] = []
    for raw in evidence:
        if not isinstance(raw, Mapping):
            return None, "invalid_evidence_candidate"
        key, stance = raw.get("hypothesis_key"), raw.get("stance")
        claim, source_url = _stripped(raw.get("claim")), _stripped(raw.get("source_url"))
        if key not in seen_keys:
            return None, "unknown_evidence_hypothesis"
        if not claim or len(claim) > 4000:
            return None, "invalid_evidence_claim"
        if not source_url.startswith(("https://", "http://")):
            return None, "invalid_evidence_source_url"
        if stance not in _STANCES:
            return None, "invalid_evidence_stance"
        normalized_evidence.append({"hypothesis_key": key, "claim": claim, "source_url": source_url, "stance": stance})
    return {
        "contract_version": CONTRACT_VERSION,
        "hypotheses": normalized_hypotheses,
        "evidence_candidates": normalized_evidence,
    }, "pending"
