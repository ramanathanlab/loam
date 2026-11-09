"""Opt-in, unvalidated hypothesis/evidence provenance candidate contract.

This contract records proposed hypotheses and source-located evidence candidates
from a completed child.  It deliberately does not accept claims, alter a parent
hypothesis, or assign credit; those occur only in explicit reconciliation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

CONTRACT_VERSION = "hypothesis_provenance_v1"
_FENCED_JSON = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n?\s*(\{.*?\})\s*```", re.DOTALL)
_KEY = re.compile(r"[a-z][a-z0-9_-]{0,79}")
_STANCES = {"supports", "contradicts", "context"}
_MAX_SPILL_BYTES = 4 * 1024 * 1024


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


def _fenced_contract(text: str, version: str) -> tuple[dict[str, Any] | None, str]:
    for match in _FENCED_JSON.finditer(text or ""):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("contract_version") == version:
            return value, "present"
    return None, "absent"


def _spilled_summary(full_path: str | None) -> str | None:
    """The child's complete summary from the file the harness spilled it to.

    Best effort by design: every failure here means the truncated summary was
    all there ever was, which is exactly what the caller already assumed.
    """

    if not full_path:
        return None
    try:
        path = Path(full_path)
        # A summary is prose; anything of this size is not one, and reading it
        # would cost a hook more than the contract it is trying to recover.
        if not path.is_file() or path.stat().st_size > _MAX_SPILL_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def extract_fenced_contract(
    summary: str, version: str, full_path: str | None = None
) -> tuple[dict[str, Any] | None, str]:
    """The last fenced JSON object in a child's summary declaring ``version``.

    Shared by every contract an agent emits this way, so a second contract type
    does not fork the extraction rules -- which fences count, which JSON parses
    -- into a near-copy that then drifts from this one.

    ``full_path`` is the harness's spill file for this child, carried by the
    `subagent_stop` payload as `child_summary_full_path`. It matters because
    `summary` is not always what the child wrote: Hermes trims an over-budget
    summary to a head+tail window, and the budget is sized against the PARENT's
    remaining context, so it tightens as a round proceeds. A contract spanning
    the cut then parses as nothing at all.

    That failure is not uniform across roles -- it lands hardest on whoever
    runs last. On 2026-11-06 the validators (17:39) passed whole at 9.6-12.4 kB
    while both evidence-reviewers (18:03) were cut to ~2.2 kB and logged
    `contract_absent`, discarding two complete reviews whose only defect was
    being thorough. The reviewer is also the one role whose entire output IS
    the contract, so it has the most to lose and the least margin.

    The spilled file is the same text, untrimmed, so re-reading it recovers the
    contract exactly. Tried only after the summary itself yields nothing, so
    the common (untruncated) case does no I/O.
    """

    value, presence = _fenced_contract(summary, version)
    if presence == "present":
        return value, presence
    spilled = _spilled_summary(full_path)
    if spilled is None or spilled == summary:
        return None, "absent"
    return _fenced_contract(spilled, version)


def extract_contract_object(
    summary: str, full_path: str | None = None
) -> tuple[dict[str, Any] | None, str]:
    return extract_fenced_contract(summary, CONTRACT_VERSION, full_path)


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
        # Type only. The pattern is NOT re-checked here: every key in
        # `seen_keys` already matched it, so the membership pass below rejects a
        # malformed parent anyway, and one diagnosis for "this parent names
        # nothing" beats two that depend on which character it starts with.
        if not isinstance(parents, list) or not all(isinstance(parent, str) for parent in parents):
            return None, "invalid_parent_hypothesis_keys"
        seen_keys.add(key)
        normalized_hypotheses.append(
            {"hypothesis_key": key, "statement": statement, "parent_hypothesis_keys": sorted(set(parents))}
        )

    # A parent names another hypothesis IN THIS CONTRACT, so it must resolve
    # like one. Checked here rather than in the loop above because a contract
    # may legitimately declare a child before its parent, and `seen_keys` is
    # only complete once every hypothesis has been read.
    #
    # Without this, the shape check alone decided the matter -- and it let a
    # dangling reference through on a coin flip. Validators write the global
    # hypothesis UUID they were ASSIGNED into this field (that id already
    # travels in the ledger row, so it is never needed here), and a UUID is all
    # `[0-9a-f-]`, so `_KEY` fullmatches it whenever its first hex digit happens
    # to be a-f -- 6/16 of the time. On 2026-11-06 the identical mistake was
    # accepted for `b89b1a61-...` and rejected for `4d1c2b46-...` in the same
    # round. The acceptance was the worse outcome: the row was stored with a
    # parent matching no key in the contract, and because both
    # `reconcile/hypothesis_provenance.py` and `project_hypothesis_memory.py`
    # copy this list through verbatim -- neither resolves it against the
    # key->hypothesis_id map -- nothing downstream could ever catch it.
    for proposal in normalized_hypotheses:
        for parent in proposal["parent_hypothesis_keys"]:
            if parent not in seen_keys:
                return None, "unknown_parent_hypothesis_key"

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
