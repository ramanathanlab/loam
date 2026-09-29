# Hypothesis provenance and evidence credit

This optional Loam path records how a hypothesis moves through delegated work
without imposing a fixed acquisition loop or treating a child completion as
scientific success.

## Boundary

```text
child summary -> pending candidate -> explicit operator reconciliation
             -> accepted source-backed evidence -> explicit credit record
```

- Raw child summaries remain lifecycle audit material.
- A candidate is not evidence and receives no credit.
- Only a review item marked `validation_status: accepted` with a non-empty
  `source_ref` creates accepted evidence and one credit record for its origin
  subagent unit.
- Jnana Elo remains a separate advisory prioritization signal; this path never
  reads tournament output to accept evidence or assign credit.

## Child opt-in contract

A child may return a fenced JSON object:

```json
{
  "contract_version": "hypothesis_provenance_v1",
  "hypotheses": [
    {
      "hypothesis_key": "intervention-x",
      "statement": "Intervention X improves outcome Y through mechanism Z.",
      "parent_hypothesis_keys": []
    }
  ],
  "evidence_candidates": [
    {
      "hypothesis_key": "intervention-x",
      "claim": "Study reports an improved Y after X.",
      "source_url": "https://example.org/study",
      "stance": "supports"
    }
  ]
}
```

On normal completed child observation, Loam appends the shape-valid object to
`candidate-results.jsonl` with `candidate_status: pending`. It does not create
a hypothesis, evidence, or credit record at hook time.

## Explicit reconciliation

Prepare an operator-authored review JSONL file. Each review names a pending
`candidate_id`, identifies accepted hypothesis keys, and must repeat candidate
evidence exactly before accepting it:

```json
{
  "candidate_id": "candidate_...",
  "accept_hypothesis_keys": ["intervention-x"],
  "evidence": [
    {
      "hypothesis_key": "intervention-x",
      "claim": "Study reports an improved Y after X.",
      "source_url": "https://example.org/study",
      "stance": "supports",
      "validation_status": "accepted",
      "source_ref": "DOI:10.1000/example"
    }
  ]
}
```

Run:

```bash
python3 .loam/reconcile/hypothesis_provenance.py \
  --candidates .loam/state/candidate-results.jsonl \
  --reviews .loam/state/hypothesis-reviews.jsonl \
  --hypotheses-out .loam/state/hypothesis-ledger.jsonl \
  --evidence-out .loam/state/evidence-ledger.jsonl \
  --credit-out .loam/state/credit-ledger.jsonl
```

Project compact read-only pre-LLM context afterward:

```bash
python3 .loam/reconcile/project_hypothesis_memory.py \
  --hypotheses .loam/state/hypothesis-ledger.jsonl \
  --evidence .loam/state/evidence-ledger.jsonl \
  --credit .loam/state/credit-ledger.jsonl \
  --out .loam/state/hypothesis-memory.jsonl
```

The pre-LLM hook injects the projected hypothesis state only for the current
workflow/session. It reports source-backed support and contradiction counts,
and explicitly reminds the model that evidence, Elo, and credit are different
signals.

## Comparable evidence rating and steering

Accepted review evidence may optionally include `strength`, `reliability`, and
`relevance` in `[0, 1]`, plus an `independence_group_id`. Defaults are `0.7`,
`0.7`, `0.8`, and the `source_ref` respectively. Reconciliation records a
signed log-odds delta and its Elo-shaped equivalent; the originating subagent
receives the absolute marginal evidence-rating delta as contribution credit.
Repeated evidence from an independence group is discounted by `1.0`, `0.25`,
and then `0.10`.

Build advisory steering state after a Jnana tournament report:

```bash
python3 .loam/reconcile/project_hypothesis_steering.py \
  --hypotheses .loam/state/hypothesis-ledger.jsonl \
  --evidence .loam/state/evidence-ledger.jsonl \
  --tournament-reports .loam/state/hypothesis-research/tournament-reports \
  --out .loam/state/hypothesis-steering.jsonl
```

The projection keeps Jnana Elo and evidence rating distinct, then records a
versioned advisory priority using weights `elo=0.35`, `evidence=0.50`, and
`uncertainty=0.15`. It also emits an action recommendation, especially when
rank and evidence ratings disagree by at least 100 Elo-shaped points. It does
not schedule work or accept evidence.
