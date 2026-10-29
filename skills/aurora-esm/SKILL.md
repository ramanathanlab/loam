---
name: aurora-esm
description: >-
  Fold with ESMFold2 and embed sequences with ESMC/ESM3 on Aurora via the
  project's ready-made scripts. Use for fast single-chain folds, for PLM
  embeddings to cluster or diversify a library, and for cheap pre-filtering
  before spending co-folds.
---

# ESM on Aurora

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_esm-2026-venv.sh
echo $ESM_SCRIPTS
```

`$ESM_SCRIPTS` resolves to (verified):

```
<root>/software/esm/scripts/aurora
```

and contains:

| script | purpose |
|---|---|
| `fold_sequence.py` | ESMFold2 single-sequence fold |
| `embed_sequences.py` | sequence embeddings |
| `embed_results_csv.py` | embeddings for a results CSV |
| `binder_embeddings/` | binder-specific embedding helpers |
| `esmfold2_smoke.py`, `esmc_smoke.py` | smoke tests — run one first |
| `readme_example.py` | worked example |
| `run_esmc_smoke.pbs` | PBS example |
| `setup_uv_venv.sh` | venv setup (already done) |

Read the script you intend to use (`--help` or its source) before invoking it;
these are project scripts, not a stable CLI. `pydssp` is available for secondary
structure, and `import esm` works in-venv.

ESMFold2 folds a single chain. It does NOT give you a complex interface score —
use `aurora-chai-fold`, `aurora-boltz2` or `aurora-protenix` for that.
