---
name: aurora-boltz2
description: >-
  Predict structure and binding affinity with Boltz-2 on an Aurora XPU. Use as a
  second, independent predictor when a Chai-1 interface score needs
  corroboration, or when you want an affinity estimate alongside a structure.
---

# Boltz-2 on Aurora

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_boltz-2026-venv.sh
boltz predict <DATA> --out_dir out --accelerator xpu --cache $BOLTZ_CACHE
```

`boltz predict [OPTIONS] DATA` (verified rc=0): `DATA` is a positional input —
a YAML spec (or a directory of them) describing the chains and any ligands.
`--accelerator xpu` is required on Aurora; `--cache $BOLTZ_CACHE` is set by the
activate script and avoids re-downloading weights per run. Run
`boltz predict --help` for the full option list before composing a tier.

Use it for a genuinely independent check: agreement between Chai-1 and Boltz-2
on an interface means more than a high score from either alone. Boltz-2 also
predicts affinity, which Chai-1 does not.

Expect torch XPU kernel-override warnings and a `requests` warning on stderr;
check the return code and the output files, not stderr cleanliness.
