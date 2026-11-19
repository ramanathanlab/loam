# Designed protein binders against human EGFR

## Objective

Identify designed protein binders against human EGFR (UniProt P00533), ranked by
computed binding free energy, good enough to justify committing wet-lab
expression and binding assays to a shortlist. The epitope, the receptor variant,
and how widely paralog selectivity is tested are deliberately open: the campaign
explores them, and a hypothesis's choice among them is ranked alongside the
binder it proposes.

## Success criteria

- At least 300 candidates complete the funnel end to end -- generated backbone,
  designed sequence, co-folded complex, and an MD-based free-energy estimate --
  with an actual output file inspected at each stage rather than a summary
  number taken on trust. Reaching this needs ONE batch of hundreds of rows, not
  hundreds of batches. (300 because this is a 48-device test; scale it up with
  the allocation.)
- Every surviving candidate's record states the epitope it targets, the receptor
  variant assumed, and which paralogs it was counter-screened against. A result
  missing these is not comparable to the others and does not count toward the
  bar above.
- The top-ranked candidates are stable across two independent rounds. A ranking
  that reshuffles completely between rounds is not yet a result.
- At least one epitope or design-strategy hypothesis is refuted on evidence and
  recorded as refuted. A campaign that only accumulates positives has not
  explored.

## Constraints

- The compute budget is 4 Aurora nodes for ONE HOUR = 48 XPU devices at once.
  The hour is hard: the queue reclaims the nodes at the wall, mid-batch, so an
  experiment that cannot finish inside it is out of scope. Prefer many short
  parallel jobs over any long one.
- WIDTH IS STILL THE BINDING CONSTRAINT. The question at every stage is "how
  many rows can I submit at once", never "how do I get one through". One
  `run_batch` of 48 rows costs about the same wall clock as one of 8 rows, so
  a stage sized to a handful of candidates idles most of the machine.
- Idle time is spent budget: one minute of the block is ~0.8 GPU-hours. Never
  leave the machine on a serial step.
- Every claim is computational. No wet-lab validation is available in this
  campaign, so no result may be reported as a measured affinity.
- Execution is remote on Aurora. A design that cannot be evaluated with the
  tools available there is out of scope regardless of how promising it looks.
- Selectivity is a property of a comparison, not of a single number. Do not
  report an EGFR binding estimate as evidence of selectivity unless a
  counter-screen was actually run.

## The decision this informs

Whether to commit wet-lab expression and binding assays to a shortlist of
designed binders, and which epitope a follow-on campaign should concentrate on.
A result is decisive only if it would change one of those two choices; a binder
that is merely interesting changes neither.

## Background a validator needs

- Human EGFR is UniProt P00533. Retrieve its sequence and any structure at run
  time from UniProt or RCSB; both answer from Aurora. AlphaFold DB does NOT
  (403 via the proxy, 404 at the worker), so do not depend on it. Copies are
  also staged under $HPC_TARGETS_DIR if you would rather not fetch.
- Only the extracellular domain is reachable by a designed binder; the kinase
  domain is intracellular. A proposal to bind it is out of scope on delivery
  grounds, not affinity grounds, and saying so is the correct response.
- The ERBB family -- ERBB2/HER2, ERBB3, ERBB4 -- shares substantial
  extracellular similarity with EGFR, so apparent EGFR affinity is not
  evidence of selectivity.
- EGFR is a validated oncology target with approved biologics, so prior art is
  a baseline to beat, not a finding to reproduce. A design on a known epitope
  must say which one and justify itself against what already binds there.
