---
name: aurora-rfdiffusion-idp
description: >-
  Generate backbones for intrinsically disordered proteins and flexible regions
  with RFdiffusion-IDP on Aurora (inference only). Use when the design target is
  a disordered tail, an IDR or a linear motif rather than a folded pocket.
---

# RFdiffusion-IDP on Aurora

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_rfdiffusion-idp-2026-venv.sh
python $RFDIFFUSION_IDP_REPO/scripts/run_inference.py \
  inference.model_directory_path=$RFDIFFUSION_MODELS \
  inference.schedule_directory_path=$RFDIFFUSION_IDP_SCHEDULES \
  <further hydra overrides>
```

**Both directory paths are required** — the activate script exports
`$RFDIFFUSION_MODELS` and `$RFDIFFUSION_IDP_SCHEDULES`, and inference fails
without them on the command line. Inference only: no training in this venv.

Hydra overrides configure the rest (contigs, number of designs, output). Inspect
the repo's example configs under `$RFDIFFUSION_IDP_REPO` before composing a run.

Disordered targets need care downstream, not just in generation: a naive co-fold
collapses a disordered region onto the ordered core and the binder relocates
there, so a global ipTM will rank core-binders rather than your intended
epitope. Use distance constraints (`--constraint-path` in `aurora-chai-fold`)
and score the epitope contacts you actually want.

Source: `envs/VENVS.txt`. This venv was NOT individually probed — run
`run_inference.py --help` and confirm the env vars are set before a tier.
