---
name: binder-cofold
scope: code-generator
---

- The decisive experiment for a binder hypothesis is a complex co-fold of
  candidate binder + target, both chains in one input.
- Report interface confidence and per-residue pLDDT. The headline metric is
  ipSAE; fall back to ipTM, then pDockQ, and say which one you are quoting.
- Run EXACTLY ONE co-fold as the decisive check — one candidate against the
  target, not a panel. Follow the recipe in the ENVIRONMENT SETUP block rather
  than reconstructing one. A single Chai-1 co-fold takes roughly 2–3 minutes, so
  the one fold this check gets is affordable to wait for.
- A co-fold scores an interface. It does not measure affinity, specificity, or
  developability — do not report it as though it did.
