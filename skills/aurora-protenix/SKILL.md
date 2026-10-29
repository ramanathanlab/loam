---
name: aurora-protenix
description: >-
  Run Protenix (AlphaFold3-style) structure prediction on Aurora, including the
  two kernel flags that are mandatory on XPU. Use as a third independent
  predictor, or when a campaign names AF3-class prediction.
---

# Protenix on Aurora

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_protenix-2026-venv.sh
protenix pred --input x.json --out_dir out \
  --triatt_kernel torch --trimul_kernel torch
```

**Both kernel flags are required** on this system — the fused CUDA kernels are
unavailable, so `--triatt_kernel torch --trimul_kernel torch` selects the portable
path. Omitting them fails.

`protenix pred [OPTIONS]` (verified rc=0) takes: `input` (JSON file or
directory), `out_dir`, `seeds` (comma-separated), `cycle` (number of cycles),
`step` (diffusion steps), and more — run `protenix pred --help` for the current
list. Multiple `seeds` is the cheap way to get replicate predictions.

Also available: `process_components_cif`, `setup_pubchem_library` for ligand
component preparation.
