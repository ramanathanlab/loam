---
name: experiment-discipline
applies: always
scope: code-generator
---

- What is under test is defined ONLY by the hypothesis block: use exactly what
  it specifies, against exactly what it names. Do not substitute an input,
  broaden the question, or sample around it.
- Prefer the smallest action that can distinguish this hypothesis from its
  alternatives. A broader run that cannot distinguish them spends the round's
  budget without sharpening the answer.
- After launching work, POLL until it has actually finished — the scheduler
  reports complete, or the call returns with its output on disk — then read the
  real values out of that output and report those. Never report while a run is
  unfinished: an unfinished run yields no evidence and wastes the round.
- Stay inside the budget and the tools this task grants you. If answering the
  question properly needs something you were not given, say what is missing
  rather than approximating it with what you have.
