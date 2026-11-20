---
name: openmm
description: Use openmm on Julich Jupiter ARM64 GH200 nodes with its dedicated software/openmm/.venv environment.
---

# openmm on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed under `$LOAM_SOFTWARE_ROOT`, with a dedicated `.venv` per tool.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/openmm"
PY="$REPO/.venv/bin/python"
"$PY" -m openmm.testInstallation
```

Select platform CUDA for GPU simulation. Preserve scheduler CUDA_VISIBLE_DEVICES. Test platform availability on a compute node; a login-node import is insufficient.

Run campaign scientific workloads through `run_batch(gpus=1)` after staging with `write_files`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.


## Executable reference

Read `references/smoke-workload.py` with `read_skill(name="openmm", file="references/smoke-workload.py")` for a concrete Jupiter workflow with output assertions. Stage its contents into the assigned workspace before execution, using this tool's Python and `--out <absolute-output-directory> --software $LOAM_SOFTWARE_ROOT`. Reduced smoke settings validate plumbing; choose campaign settings for scientific evidence.

## Campaign execution

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` in a bash subprocess before activating the tool environment. Preserve the assigned `CUDA_VISIBLE_DEVICES`; use `cuda:0` within a one-GPU row. Stage scripts and inputs with `write_files`, execute real sweeps with `run_batch(gpus=1)`, poll completion, then inspect actual output files. Batch floors are configured by the deployment server; consult the campaign setup. Never pad with duplicate jobs. Keep each row within the remaining Slurm walltime.
