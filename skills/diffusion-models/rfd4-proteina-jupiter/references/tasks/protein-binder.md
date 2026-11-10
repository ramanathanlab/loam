# Protein-binder generation on Jupiter

This is a packaged Jupiter adaptation of the installed RFD4 checkout's `skills/rfd4-proteina-generate/references/tasks/protein-binder.md`. The verified source revision is recorded in Loam's `ports/jupiter/rfd4-runtime-validation.json`; consult the installed checkout for later API changes. See [annotated-CIF construction](../annotated-cif.md) before generation.

## Required input

Use a real folded target with authoritative provenance, verified chain identity and sequence mapping, and a justified retained region. Default PPI conditioning is target backbone-only internal coordinates (`C_CRDGRP`), full target sequence (`C_SEQ`), and selected hotspot atoms (`C_HOT`). This preserves target internal geometry while leaving global pose and sidechains free. Only the binder chain is generated. Preserve explicit campaign conditions.

The installed upstream example `docs/how-to/input_prep/examples/condition_spec_ppi.yaml` illustrates these annotations; it is a PD-L1 example, not an EGFR input. Never substitute it as target-specific campaign evidence.

## Run one design

Execute through MCP in the assigned workspace. Set `INPUT` to the validated annotated CIF and `OUT` to a fresh output directory:

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/RFD4-Proteina"
source "$REPO/.venv/bin/activate"
export CLUSTER=local WANDB_MODE=disabled OMP_NUM_THREADS=4
cd "$REPO"
"$REPO/.venv/bin/rfd4-generate" \
  inputs="$INPUT" out_dir="$OUT" \
  recycle_strategy=medium/front_loaded \
  num_replicates=1 batch_size=1 compile_lengths=null num_workers=0 \
  save_chain_of_interest=true seed=42
```

`recycle_strategy` is a Hydra configuration group, not a `ppi` subfield. `ppi.recycle_strategy=medium` fails and `+ppi.recycle_strategy=medium` does not set the model schedule. The selected medium/front_loaded group defines 60 sampling steps and 180 total trunk passes. Retain the campaign's sampling budget; do not switch presets just to force success.

Use a separate task per design with a 6000-second payload limit, or independent `run_batch(gpus=1, timeout_each=6000)` rows. Stream output to a log file immediately. Keep the task's GPU assignment; avoid oversubscribing a GPU.

## Outputs and metrics

Inspect the actual CIF under `OUT/inference_samples/` before reporting success: identify target and generated chain, generated residues, finite coordinates, and conditioning/provenance annotations. A command starting or an input CIF being written is not a generated binder. Save input/output checksums, source and checkpoint revisions, seed, effective overrides, and failure logs.

`save_chain_of_interest=true` saves the generated chain for downstream use. Do not equate a generated structure with validated binding. ProteinMPNN sequence design, the campaign's selected co-folding tool, structural metrics, and energy evaluation remain separate stages.

Upstream `ppi_refolding` metrics require the configured AF3 oracle and target templating; do not enable them assuming Chai supplies that oracle. Hotspot metrics assess input hotspot obedience. Consult the installed `docs/how-to/evaluate-with-metrics.md` and effective metric configuration before selecting metric tags.

## Installation failures

A missing `libXrender.so.1` means the native runtime is absent or `jupiter_env.sh` was not sourced. An upstream shell preflight that cannot import AtomWorks while absolute RFD4 Python can is an environment-selection problem: explicitly activate the RFD4 venv and set `CLUSTER=local` before invoking the wrapper. Do not reinstall a functioning model environment based only on that wrapper error.

On Jupiter use `num_workers=0` for single-design inference: PyTorch loads data in the main process rather than spawning four loader subprocesses. This matches the successful manual installation test and avoids oversubscription under one-core task affinity. The runtime installer configures this default via `scripts/configure_jupiter_rfd4.py`; keep the explicit CLI override for auditable runs. It changes data-loading parallelism, not the intended model recycle schedule.
