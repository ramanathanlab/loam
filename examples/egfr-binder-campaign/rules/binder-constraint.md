---
name: binder-constraint
scope: hypothesis-validator,reasoner
---

- Hypotheses about the harness's own internals -- hook ordering, delegation
  mechanics, ledger schemas -- are out of bounds for this campaign, even when
  they look promising, because the harness is the instrument here and not the
  subject.
- Do not spend a later round re-running a binder a prior round already
  executed. Steer the next designs toward an iteration on a prior design or 
  on generating totally new designs.
- Do not hallucinate or make up sequences of binders yourself, all binder sequences
  should be generated using tool calls or come from literature. This does not mean
  scraping a section of a protein from RCSB either.
