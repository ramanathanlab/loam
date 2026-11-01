#!/usr/bin/env python3
"""Audit what independent review has been accepting, for calibration.

    python3 scripts/review_audit.py --state <campaign>/state [--sample 5]

An `evidence-reviewer` child can accept a claim it did not produce, which is
what makes unattended acceptance sound. What no amount of structure inside that
loop can detect is a reviewer that accepts everything: a 100% accept rate looks
identical, from the inside, to a campaign whose claims all happen to be good.
So the calibration check is deliberately outside the loop and belongs to a
person -- this script only lays the numbers out and pulls a sample to read.

It is read-only. It accepts nothing, rejects nothing, and writes nothing.

What to look for
----------------

* **Accept rate near 100%.** Rubber-stamping. The reviewers are being handed
  claims they cannot actually check and are approving them anyway, or the role
  definition's adversarial framing is not landing.
* **Accept rate near 0%.** Either the validators are emitting claims with no
  artifact or source to check -- look at the `unverifiable` share -- or the
  reviewers cannot reach the files, which is a plumbing fault, not a verdict.
* **`unverifiable` dominating.** Expected for a campaign whose metrics are
  computed rather than retrieved, but if it is near-total, check whether the
  artifact manifest is reaching the reviewers at all (`artifact_attested`
  acceptances require either a workspace manifest or a readable workdir).
* **A reviewer with no `checked` notes**, or notes that describe no specific
  file or value. An acceptance that cannot say what was read is the failure the
  boundary exists to catch, one level up.
* **`dropped_unmatched` above zero.** Claims that reached no verdict because the
  review item matched no candidate item. A few mean a reviewer retyped a claim
  instead of using its index; many mean the contract is being misread.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _rate(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, required=True, help="the campaign's state directory")
    parser.add_argument("--sample", type=int, default=5, help="accepted claims to print in full (default 5)")
    parser.add_argument("--seed", type=int, default=None, help="fix the sample, for a reproducible audit")
    args = parser.parse_args()

    reviews = _rows(args.state / "hypothesis-reviews.jsonl")
    candidates = _rows(args.state / "candidate-results.jsonl")
    evidence = _rows(args.state / "evidence-ledger.jsonl")
    acceptance = _rows(args.state / "hypothesis-acceptance.jsonl")

    pending = [row for row in candidates if row.get("candidate_status") == "pending"]
    reviewed_ids = {str(row.get("candidate_id") or "") for row in reviews}
    print(f"candidates:  {len(candidates)} recorded, {len(pending)} pending, {len(reviewed_ids)} reviewed")
    print(f"reviews:     {len(reviews)}")
    print(f"accepted:    {len(evidence)} evidence rows in the ledger")
    failures = [row for row in acceptance if row.get("event") == "candidate_reconcile_failed"]
    if failures:
        print(f"reconcile:   {len(failures)} review(s) the reconciler refused")
        for row in failures[:5]:
            print(f"             {row.get('candidate_id', '')[:28]}  {row.get('error', '')[:90]}")
    if not reviews:
        print("\nNo reviews yet -- nothing to calibrate.")
        return 0

    verdicts: Counter[str] = Counter()
    by_reviewer: dict[str, Counter[str]] = defaultdict(Counter)
    modes: Counter[str] = Counter()
    dropped = 0
    silent = 0
    accepted_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for review in reviews:
        reviewer = str(review.get("reviewer_unit_id") or review.get("reviewer_session_id") or "unknown")
        dropped += int(review.get("dropped_unmatched") or 0)
        for item in review.get("review_audit") or []:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict") or "")
            verdicts[verdict] += 1
            by_reviewer[reviewer][verdict] += 1
            if verdict != "accepted":
                continue
            mode = str(item.get("verification") or "none")
            if item.get("attestation") == "unhashed":
                mode += " (unhashed)"
            modes[mode] += 1
            if not str(item.get("checked") or "").strip():
                silent += 1
            accepted_items.append((review, item))

    total = sum(verdicts.values())
    print(f"\nclaims reviewed: {total}")
    for verdict in ("accepted", "rejected", "unverifiable"):
        print(f"  {verdict:<13} {verdicts.get(verdict, 0):>4}  {_rate(verdicts.get(verdict, 0), total)}")
    if dropped:
        print(f"  {'dropped':<13} {dropped:>4}  (matched no candidate claim, so never reached a verdict)")
    print("\nhow the acceptances were verified:")
    for mode, count in modes.most_common():
        print(f"  {mode:<28} {count:>4}")
    if silent:
        print(f"\n{silent} acceptance(s) carry no `checked` note -- read those first.")

    print("\nper reviewer:")
    for reviewer, counts in sorted(by_reviewer.items()):
        seen = sum(counts.values())
        flag = ""
        if seen >= 3 and counts.get("accepted", 0) == seen:
            flag = "   <-- accepted everything it saw"
        print(f"  {reviewer[:24]:<24} {seen:>3} claims, accept rate {_rate(counts.get('accepted', 0), seen)}{flag}")

    if args.sample and accepted_items:
        if args.seed is not None:
            random.seed(args.seed)
        print(f"\n--- {min(args.sample, len(accepted_items))} accepted claim(s) to check by hand ---")
        for review, item in random.sample(accepted_items, min(args.sample, len(accepted_items))):
            print(f"\nhypothesis {str(review.get('hypothesis_id') or '')[:8]}  plan {review.get('plan_id', '')}")
            print(f"  stance:     {item.get('stance')}  via {item.get('verification')}")
            print(f"  claim:      {str(item.get('claim') or '')[:300]}")
            print(f"  source_url: {item.get('source_url')}")
            print(f"  source_ref: {item.get('source_ref')}")
            print(f"  checked:    {str(item.get('checked') or '')[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
