---
name: evidence-integrity
applies: always
scope: all
enforced_by: partially -- an outcome bearing on the hypothesis, a refutation included, travels as a hypothesis_provenance_v1 candidate that is accepted only through explicit review -- by an operator, or by an independent evidence-reviewer when `evidence_review` is on; the rest is advisory
---

- A datum you did not produce in this campaign has an authoritative source.
  Retrieve it from that source at run time by its identifier in that source, and
  verify what came back: the identifier matches, the size and shape are what the
  identifier implies, the feature you are relying on is present.
- Never reproduce such a datum from memory, and never accept one that was. A
  hardcoded or unverifiable input is a blocked result, not a validation,
  regardless of how plausible it looks.
- Report the result you actually obtained, including when it refutes the
  hypothesis or fails outright. A refutation is a finding; a plausible number is
  not a measurement.
- A failed, unfinished, or unexecuted run is not evidence. Say so, and never
  substitute unexecuted code, a plan, or reasoning for an observation.
