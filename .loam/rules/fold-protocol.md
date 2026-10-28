---
name: fold-protocol
scope: code-generator
---

- The molecules under test are defined ONLY by the hypothesis block: the
  candidate design described there, folded against the target named there.
- The decisive experiment is a complex co-fold of candidate binder + target
  (both chains in one input), reporting interface confidence (ipTM / ipSAE /
  pDockQ) and per-residue pLDDT, per the ENVIRONMENT SETUP recipe.
- Run EXACTLY ONE fold (one candidate against the target) as the decisive check,
  not a panel of many. A single Chai fold takes roughly 2–3 minutes.
- After launching it, POLL until it has actually finished — `batch_status` shows
  complete, or the `run_code` call returns with the `.cif` on disk — then read
  the real metrics and report those. Never report while a run is unfinished: an
  unfinished run yields no evidence and wastes the round. The budget is one fold
  per check, so make the one fold count and wait for it rather than firing many
  and reporting early.
