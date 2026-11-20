---
name: esmfold2
description: Use esmfold2 on Julich Jupiter ARM64 GH200 nodes with its dedicated software/esmfold2/.venv environment.
---

# esmfold2 on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed under `$LOAM_SOFTWARE_ROOT`, with a dedicated `.venv` per tool.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/esmfold2"
PY="$REPO/.venv/bin/python"
"$PY" -c "from esm.models.esmfold2 import EsmFold2Model"
```

The installed esm 3.4.2 provides EsmFold2Model directly; the Aurora transformers fork is no longer required. Download biohub/ESMFold2 and biohub/ESMC-6B weights before offline use. ARM64 uses native PyTorch attention when the upstream x86-only flash-attention wheel is unavailable. The manual native CUDA fold passed on a real 20-residue fixture; larger complexes and production settings still require validation. Do not substitute ESMFold v1.

Run campaign scientific workloads through `run_batch(gpus=1)` after staging with `write_files`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.


## Executable reference

Read `references/smoke-workload.py` with `read_skill(name="esmfold2", file="references/smoke-workload.py")` for a concrete Jupiter workflow with output assertions. Stage its contents into the assigned workspace before execution, using this tool's Python and `--out <absolute-output-directory> --software $LOAM_SOFTWARE_ROOT`. Reduced smoke settings validate plumbing; choose campaign settings for scientific evidence.

## Campaign execution

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` in a bash subprocess before activating the tool environment. Preserve the assigned `CUDA_VISIBLE_DEVICES`; use `cuda:0` within a one-GPU row. Stage scripts and inputs with `write_files`, execute real sweeps with `run_batch(gpus=1)`, poll completion, then inspect actual output files. Batch floors are configured by the deployment server; consult the campaign setup. Never pad with duplicate jobs. Keep each row within the remaining Slurm walltime.
