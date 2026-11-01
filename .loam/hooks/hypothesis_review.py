"""The independent-review contract that turns a pending candidate into a review.

A validator's provenance candidate is untrusted model output and lands
`pending`. Acceptance has until now required a person to author the review JSONL
that `reconcile/hypothesis_provenance.py --reviews` reads. What that boundary
actually protects is **independence of the acceptor from the producer**, not
humanity as such: a claim carrying a resolvable `source_url`, a stance, and a
named artifact is checkable by anyone who did not make it.

So this module defines a second contract, emitted by a different agent --
`.loam/agents/evidence-reviewer.md` -- which never sees the producing
validator's prose. Everything the review decides is a verdict per claim; every
number that gives an accepted claim its *force* is computed here, from the
declared verification mode, and is deliberately not readable from the reviewer's
output. A reviewer can therefore accept a claim, but cannot weight its own
acceptance.

Two verification modes are distinguished because "verify" does two different
jobs (see `docs/` and the campaign's AUTOMATION notes):

``source_confirmed``   an independent record supports the claim; the reviewer
                       re-fetched `source_url` and read it.
``artifact_attested``  the claimed number genuinely came from a named artifact
                       of the campaign's own workspace, whose hash Loam
                       recorded. This establishes exactly what the boundary
                       exists for -- that a summary's number is not invented --
                       and says nothing about whether the metric is *true*.

Anything else is ``unverifiable``, which is a first-class verdict and not a
rejection: most of what a computational campaign measures has no authoritative
record to re-fetch.
"""

from __future__ import annotations

from typing import Any, Mapping

from hypothesis_provenance import extract_fenced_contract

CONTRACT_VERSION = "hypothesis_review_v1"

VERDICTS = ("accepted", "rejected", "unverifiable")
VERIFICATIONS = ("source_confirmed", "artifact_attested")

# Weight policy: the force an accepted claim carries, by how it was verified.
# Set here rather than taken from the reviewer's output on purpose -- a reviewer
# that could choose its own strength/reliability/relevance could inflate a weak
# attestation into a strong one, and nothing downstream would see the
# difference. Keyed by verification mode, so the only lever the reviewer has is
# the mode it can defend.
WEIGHT_POLICY: dict[str, dict[str, float]] = {
    # An independent record was read. Reliable, and the claim is about that record.
    "source_confirmed": {"strength": 0.8, "reliability": 0.9, "relevance": 0.8},
    # The number provably came from a hashed artifact; its truth is a separate,
    # often unanswerable question, so this carries less than a confirmed source.
    "artifact_attested": {"strength": 0.7, "reliability": 0.7, "relevance": 0.8},
    # Same check with the hash missing, because the campaign runs without the
    # per-hypothesis workspace and Loam recorded no manifest. The reviewer read
    # a file under the plan's own HPC workdir and found the number there, which
    # is worth something -- it still catches a number no output file contains --
    # but nothing ties that file to a declared run, so it cannot carry the
    # weight of an attested one. Selected by the hook, never by the reviewer.
    "artifact_attested_unhashed": {"strength": 0.6, "reliability": 0.5, "relevance": 0.8},
}

MAX_NOTE_CHARS = 2_000


def _stripped(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def extract_contract_object(summary: str) -> tuple[dict[str, Any] | None, str]:
    return extract_fenced_contract(summary, CONTRACT_VERSION)


def _at_index(items: list[Mapping[str, Any]], value: Any) -> Mapping[str, Any] | None:
    """The candidate evidence item a review's `index` refers to, if it refers to one.

    Accepts the integer or its string form, because a model emitting JSON
    sometimes quotes a number. A bool is not an index.
    """

    if isinstance(value, bool) or value is None:
        return None
    try:
        position = int(value)
    except (TypeError, ValueError):
        return None
    return items[position] if 0 <= position < len(items) else None


def independence_group(verification: str, source_url: str, unit_id: str) -> str:
    """Which claims count as one source rather than several.

    `project_hypothesis_steering.py` discounts repeats within a group by
    1.0 / 0.25 / 0.10, so this is what stops one validator's five failed gates
    from reading as five independent contradictions: every artifact-attested
    claim out of one unit shares that unit's group. A source-confirmed claim
    groups by the record it was read from instead, so two validators citing the
    same PDB entry do not double-count either.
    """

    if verification == "artifact_attested" and unit_id:
        return f"unit:{unit_id}"
    return f"source:{source_url}"


def normalize_review(
    value: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    *,
    ranking_hypothesis_ids: Mapping[str, str] | None = None,
    manifest_available: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """Turn one reviewer's contract into the review row the reconciler accepts.

    Returns (row, status). A row is produced whenever the contract parses
    against this candidate, even when it accepts nothing: a review that rejects
    every claim is a result, and recording it is what keeps the candidate from
    being re-reviewed every round.

    The reconciler raises on a review item that does not exactly match a
    candidate evidence item, so unmatched items are dropped here and counted in
    `dropped_unmatched` rather than being allowed to fail the whole acceptance.
    """

    if value.get("contract_version") != CONTRACT_VERSION:
        return None, "unsupported_contract"
    candidate = candidate_record.get("candidate")
    if not isinstance(candidate, Mapping):
        return None, "candidate_not_pending"
    candidate_id = str(candidate_record.get("candidate_id") or "")
    declared = _stripped(value.get("candidate_id"))
    if declared and declared != candidate_id:
        # The reviewer reviewed something else, or transcribed the id. Either
        # way this is not a review of the candidate it was dispatched for.
        return None, "candidate_id_mismatch"
    items = value.get("reviews")
    if not isinstance(items, list) or not items:
        return None, "invalid_reviews"

    candidate_items = [row for row in candidate.get("evidence_candidates", []) if isinstance(row, Mapping)]
    allowed = {
        (row.get("hypothesis_key"), row.get("claim"), row.get("source_url"), row.get("stance")): row
        for row in candidate_items
    }
    keys = {row.get("hypothesis_key") for row in candidate.get("hypotheses", []) if isinstance(row, Mapping)}
    unit_id = str(candidate_record.get("unit_id") or "")

    evidence: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    counts = {verdict: 0 for verdict in VERDICTS}
    dropped = 0
    for item in items:
        if not isinstance(item, Mapping):
            dropped += 1
            continue
        # `index` is the reliable path and the one the reviewer is told to use:
        # the fields below must otherwise match a 300-character claim byte for
        # byte, and a model asked to copy that back is one normalized dash away
        # from having its review silently dropped. An index resolves to the
        # candidate's own item, so the identity cannot drift at all. Verbatim
        # matching stays for an operator-authored review, which has no index.
        source = _at_index(candidate_items, item.get("index"))
        if source is None:
            identity = (
                _stripped(item.get("hypothesis_key")),
                _stripped(item.get("claim")),
                _stripped(item.get("source_url")),
                _stripped(item.get("stance")),
            )
            if identity not in allowed:
                dropped += 1
                continue
            source = allowed[identity]
        key = source.get("hypothesis_key")
        claim = source.get("claim")
        source_url = source.get("source_url")
        stance = source.get("stance")
        verdict = _stripped(item.get("verdict"))
        verification = _stripped(item.get("verification"))
        source_ref = _stripped(item.get("source_ref"))
        checked = _stripped(item.get("checked"))[:MAX_NOTE_CHARS]
        if verdict not in VERDICTS:
            dropped += 1
            continue
        # An acceptance has to name what was checked and how. Without either, it
        # is an assertion that the claim is fine, which is the thing the
        # boundary refuses to take on trust -- so it degrades to unverifiable
        # rather than being dropped, and the audit row says why.
        if verdict == "accepted" and (verification not in VERIFICATIONS or not source_ref):
            verdict = "unverifiable"
            checked = (checked + " [Loam: downgraded from accepted -- no verification mode or no source_ref]").strip()
        counts[verdict] += 1
        audit.append(
            {
                "hypothesis_key": key,
                "claim": claim,
                "source_url": source_url,
                "stance": stance,
                "verdict": verdict,
                "verification": verification if verification in VERIFICATIONS else "none",
                "source_ref": source_ref,
                "checked": checked,
            }
        )
        if verdict != "accepted":
            continue
        policy = verification
        if verification == "artifact_attested" and not manifest_available:
            policy = "artifact_attested_unhashed"
            audit[-1]["attestation"] = "unhashed"
        weights = WEIGHT_POLICY[policy]
        evidence.append(
            {
                "hypothesis_key": key,
                "claim": claim,
                "source_url": source_url,
                "stance": stance,
                "validation_status": "accepted",
                "source_ref": source_ref,
                "independence_group_id": independence_group(verification, source_url, unit_id),
                **weights,
            }
        )

    accepted_keys = {row["hypothesis_key"] for row in evidence}
    approved = value.get("accept_hypothesis_keys")
    approved = {_stripped(key) for key in approved} if isinstance(approved, list) else set()
    # A hypothesis enters the ledger only if the reviewer approved its statement
    # AND at least one of its claims survived review: approval alone would admit
    # a statement nothing backs, and accepted evidence alone would admit a
    # statement the reviewer declined to stand behind. The reconciler also
    # refuses accepted evidence attached to an unaccepted hypothesis, so the
    # evidence of a key that fails this test is dropped with it.
    accept = sorted((approved & accepted_keys) & keys)
    evidence = [row for row in evidence if row["hypothesis_key"] in set(accept)]

    row = {
        "candidate_id": candidate_id,
        "accept_hypothesis_keys": accept,
        "ranking_hypothesis_ids": {key: str(ranking_hypothesis_ids.get(key) or "") for key in accept}
        if ranking_hypothesis_ids
        else {},
        "evidence": evidence,
        "review_verdicts": counts,
        "review_audit": audit,
        "dropped_unmatched": dropped,
        "declined_hypothesis_keys": sorted(((approved | accepted_keys) & keys) - set(accept)),
    }
    row["ranking_hypothesis_ids"] = {key: value for key, value in row["ranking_hypothesis_ids"].items() if value}
    return row, "reviewed"
