---
name: egfr-binders
---

# PLACEHOLDER — not a real campaign objective

This file exists so the campaign-context loader has something to load and the
tests have a fixture shaped like the real thing. Replace the whole body before
running a campaign against it. The goal text below is illustrative only and has
not been reviewed by anyone who knows the target.

## Goal

Design de novo protein binders against human EGFR (UniProt P00533).

## Success criteria

- A shortlist of candidate binder sequences with co-fold interface confidence
  above the campaign's agreed threshold against the EGFR extracellular domain.
- Each shortlisted candidate is developable: no disqualifying chemical
  liabilities, acceptable charge and solubility proxies.
- Every reported metric traces to an executed run, not to a design that was
  never folded.

## Constraints

- Binders are de novo designs. A hit recovered from an existing database is a
  baseline, not a result.
- The target sequence is fetched from UniProt at run time by accession, never
  reproduced from memory.
- Compute is bounded per turn; prefer one decisive co-fold over a panel of
  speculative ones.

## Decision this informs

Which candidate designs, if any, justify the cost of molecular-dynamics
confirmation and eventual experimental validation.
