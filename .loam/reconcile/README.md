# Reconciliation

Reconciliation is the acceptance boundary. A child's summary is untrusted: the
lifecycle hooks extract shape-valid structured claims out of it and append them
to `$LOAM_STATE_DIR/candidate-results.jsonl` with `candidate_status: pending`.
Nothing in that file has been accepted. A candidate receives no finding,
evidence, memory, or credit until a reconciler is run explicitly against a
review.

## The candidate contract

One contract: `hypothesis_provenance_v1` — proposed hypotheses plus
source-located evidence candidates. It is documented in full, with the
reconciler's inputs and outputs, in
[`HYPOTHESIS_PROVENANCE.md`](HYPOTHESIS_PROVENANCE.md). A validator is told to
emit it in `.loam/agents/hypothesis-validator.md`.

It is deliberately domain-neutral: a hypothesis key, a statement, and evidence
carrying a claim, a resolvable `source_url`, and a `supports` / `contradicts` /
`context` stance. That shape fits any field.

## Adding a second contract

`build_candidate_records` in `.loam/hooks/ledger.py` iterates a tuple of
`(version, extractor, normalizer)`. A campaign needing another claim type adds
an entry and a module exposing those three names.

Before adding one, check that the claim vocabulary can express what the campaign
actually measures. The first contract here was
`protein_structure_metadata_v1`, which verified claims about one RCSB PDB or
UniProt record — a protein's recommended name, gene symbol, sequence length —
against an automatic re-fetch of that record. It was removed (2026-10-28)
because no agent definition ever instructed a child to emit it, it accumulated
four unreconciled records in 2026-09 and none after, and its vocabulary had no
shape for a measurement: no interface score, no stability, no activity. A
contract that cannot carry a campaign's evidence will not be used by it.

`project_parent_memory.py` went with it. It projected that reconciler's
per-field decisions (`validation_status`, `authoritative_value`) into
`parent-memory.jsonl`, and the surviving reconciler emits no such rows — its
accepted evidence carries a claim, a `source_url` and a stance.

**`parent-memory.jsonl` therefore has no writer at all.** Removing the projector
was described at the time as leaving the file with exactly one writer, the
reasoner, "as its one durable findings cache". That was wrong: `tool_policy`
grants the reasoner no tool capability whatsoever (`reasoner has no tool
capability; return the synthesis to the harness`), so it cannot open a file.
Measured 2026-11-01, after four dispatched rounds: the file does not exist.
`reasoner.md` has been corrected to stop promising a write target the role
cannot reach, and the durable findings path is now accepted evidence →
`hypothesis-memory.jsonl`. `parser.Parser` still reads `parent-memory.jsonl`, so
an operator may hand-author it; nothing in the system produces it.

## Who authors the review

Acceptance requires a review, and what the review protects is **independence of
the acceptor from the producer** — not humanity as such. A claim carrying a
resolvable `source_url`, a stance, and a named artifact is checkable by anyone
who did not make it.

So there are two authors, and the reconciler cannot tell them apart:

- **an operator**, writing review rows by hand (the original path, unchanged);
- **an `evidence-reviewer` child**, dispatched per pending candidate by the
  orchestration hook when a profile sets `hypothesis_orchestration.evidence_review:
  true`. It never sees the producing validator's prose, cannot execute code, and
  cannot weight its own acceptance — `hypothesis_review.py` sets `strength`,
  `reliability` and `relevance` from the declared verification mode
  (`source_confirmed` or `artifact_attested`), and groups one unit's attested
  claims under one independence group so a single run's five failed gates do not
  read as five independent contradictions. `hypothesis_acceptance.py` then runs
  this reconciler and both projectors.

`unverifiable` is a first-class verdict there, distinct from rejection: most of
what a computational campaign measures has no authoritative record to re-fetch,
and `artifact_attested` establishes only that the number genuinely came from a
hashed artifact of a declared run — the failure this boundary exists to catch —
not that the metric is true.

What stays operator-owned is **calibration**: sampling the auto-accepted claims
and watching the accept rate. A reviewer accepting ~100% is not reviewing, and
no amount of structure inside the loop detects that from the inside — so the
check sits outside it, in `scripts/review_audit.py`:

```bash
python3 scripts/review_audit.py --state <campaign>/state --sample 5
```

Read-only. It prints the verdict split, how each acceptance was verified, the
per-reviewer accept rate (flagging a reviewer that accepted everything it saw),
any acceptance carrying no note of what was checked, and a random sample of
accepted claims to read. It is not per-round work; run it when a campaign has
accumulated a few rounds of reviews.

Note what went with the retired contract: its reconciler fetched the
authoritative source and compared field by field, which needed a per-source
adapter and is why it was campaign-specific. `hypothesis_provenance_v1`
reconciles against a review instead, which needs no schema knowledge of any
source — the reviewer reads the record or the artifact and judges — so the
review path above stays domain-neutral.
