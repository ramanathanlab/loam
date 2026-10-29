---
name: aurora-bindcraft
description: >-
  Design binders with BindCraft2 or FreeBindCraft/ColabDesign on Aurora — the
  two JAX-based (AlphaFold2 + ProteinMPNN) pipelines. Use as an alternative
  generator to the torch stack, and note neither venv can import torch.
---

# BindCraft on Aurora

Two separate venvs, both JAX:

## bindcraft2-2026-venv

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_bindcraft2-2026-venv.sh
bindcraft <args>
python $REPO/aurora/smoke_xpu.py     # run this first on a new allocation
```

BindCraft2 = AlphaFold2 + ProteinMPNN in JAX.

## freebindcraft-2026-venv

```bash
source $R/envs/_scripts/activate_freebindcraft-2026-venv.sh
python $FREEBINDCRAFT_REPO/bindcraft.py <args>
python $FREEBINDCRAFT_REPO/aurora/validate_aurora.py
```

FreeBindCraft and ColabDesign. This venv also carries **`pdbfixer` and OpenMM
8.6.1**, which matters because the MD venv has no pdbfixer — see
`aurora-openmm-ph`.

## Pitfalls

- **Neither venv can import torch**, by design. Do not mix a torch step into the
  same process or shell; run them in separate `bash -lc` invocations.
- Run the bundled smoke/validate script before a tier: these are JAX-on-XPU
  ports and a broken allocation shows up there in seconds rather than after a
  long design run.

Source: `envs/VENVS.txt`. Neither venv was individually probed; confirm the entry
point with `--help` first.
