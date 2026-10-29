---
name: experiment-discipline
scope: code-generator
---

- The system under test is defined ONLY by the hypothesis block: the candidate
  described there, assayed against the reference named there. Do not substitute,
  extend, or sample around it.
- Run ONE decisive measurement per check, not a panel. The question is whether
  this hypothesis survives, and one well-chosen run answers it; a sweep spends
  the round's budget without sharpening the answer.
- After launching it, POLL until it has actually finished — the job scheduler
  reports complete, or the call returns with its output artifact on disk — then
  read the real numbers out of that artifact and report those. Never report
  while a run is unfinished: an unfinished run yields no evidence and wastes the
  round.
- Report the measurement you made, including when it refutes the hypothesis or
  fails outright. A refutation is a result; a plausible number is not.
