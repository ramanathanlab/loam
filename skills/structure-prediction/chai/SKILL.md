---
name: chai
description: Use chai on Julich Jupiter ARM64 GH200 nodes with its dedicated software/chai-lab/.venv environment.
---

# Chai-1 on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed under `$LOAM_SOFTWARE_ROOT`, with a dedicated `.venv` per tool.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/chai-lab"
PY="$REPO/.venv/bin/python"
"$REPO/.venv/bin/chai-lab" fold input.fasta output --num-trunk-recycles 1 --num-diffn-timesteps 50 --num-diffn-samples 1 --seed 42
```

Use entity FASTA headers such as >protein|name=target. First inference downloads model assets to CHAI_DOWNLOADS_DIR. Use an empty output directory. These reduced settings test plumbing; use production defaults for scientific conclusions.

For a campaign complex, write two FASTA records from the verified target sequence and the actual designed binder sequence, then use the installed production defaults explicitly:

```bash
"$REPO/.venv/bin/chai-lab" fold complex.fasta fresh_output \
  --num-trunk-recycles 3 --num-diffn-timesteps 200 \
  --num-diffn-samples 5 --seed 42 --no-use-msa-server
```

This uses single-sequence input with no external MSA server. Record that limitation. Inspect each `pred.model_idx_*.cif` and corresponding `scores.model_idx_*.npz`: verify both complete input sequences, finite coordinates, and residue counts. The installed score files include `ptm`, `iptm`, and `aggregate_score`; read their actual values rather than inventing a confidence metric. A valid prediction with low interface confidence is evidence against a reliable designed interface, not a runtime failure or proof of binding. These settings completed on a real 205-residue EGFR target plus 60-residue binder on Jupiter.

Run campaign scientific workloads through `run_batch(gpus=1)` after staging with `write_files`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.


## Executable reference

Read `references/smoke-workload.py` with `read_skill(name="chai", file="references/smoke-workload.py")` for a concrete Jupiter workflow with output assertions. Stage its contents into the assigned workspace before execution, using this tool's Python and `--out <absolute-output-directory> --software $LOAM_SOFTWARE_ROOT`. Reduced smoke settings validate plumbing; choose campaign settings for scientific evidence.

## Campaign execution

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` in a bash subprocess before activating the tool environment. Preserve the assigned `CUDA_VISIBLE_DEVICES`; use `cuda:0` within a one-GPU row. Stage scripts and inputs with `write_files`, execute real sweeps with `run_batch(gpus=1)`, poll completion, then inspect actual output files. Batch floors are configured by the deployment server; consult the campaign setup. Never pad with duplicate jobs. Keep each row within the remaining Slurm walltime.
