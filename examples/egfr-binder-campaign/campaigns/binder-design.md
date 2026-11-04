---
name: binder-design
---

# Designed protein binders against human EGFR

## Objective

Identify designed protein binders against human EGFR (UniProt P00533), ranked by
computed binding free energy, good enough to justify committing wet-lab
expression and binding assays to a shortlist. The epitope, the receptor variant,
and how widely paralog selectivity is tested are deliberately open: the campaign
explores them, and a hypothesis's choice among them is ranked alongside the
binder it proposes.

## Success criteria

- At least 1,000 candidates complete the funnel end to end -- generated backbone,
  designed sequence, co-folded complex, and an MD-based free-energy estimate --
  with an actual output file inspected at each stage rather than a summary
  number taken on trust.
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

- The compute budget is a single wide, short allocation: 1024 nodes for one
  hour, not 1024 node-hours to spend at leisure. Every experiment must complete
  inside that one-hour wall clock, and an evaluation that cannot is out of scope
  however rigorous it would be. Prefer many short parallel jobs over any long
  one.
- The allocation is held whether or not work is running on it, so idle time is
  spent budget. One minute of the whole block is roughly seventeen node-hours.
  Do not leave the machine waiting on a serial step that could have been
  batched.
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

- Human EGFR is UniProt P00533. Retrieve its sequence and any structure from
  their authoritative source at run time. Do not reproduce either from
  recollection.
- Only the extracellular domain is reachable by a designed protein binder; the
  kinase domain is intracellular. A proposal to bind the kinase domain is out of
  scope on delivery grounds, not on affinity grounds, and saying so is the
  correct response to one.
- The ERBB family -- ERBB2/HER2, ERBB3, ERBB4 -- shares substantial
  extracellular similarity with EGFR. Apparent EGFR affinity is therefore not
  evidence of EGFR selectivity.
- EGFR is a validated oncology target with approved biologics against it, so
  prior art is a baseline to beat rather than a finding to reproduce. A design
  that lands on a known epitope must say which one, and justify itself against
  what already binds there.
