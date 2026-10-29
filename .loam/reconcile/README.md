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

Note what went with it: that contract's reconciler fetched the authoritative
source and compared field by field. `hypothesis_provenance_v1` reconciles
against an explicit **review** instead, so acceptance is a human decision about
a sourced claim rather than an automatic equality check. If a campaign wants
automatic re-fetch verification, that is a new contract plus a source adapter,
and it belongs to the campaign that needs it.
