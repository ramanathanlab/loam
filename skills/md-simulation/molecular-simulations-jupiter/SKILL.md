---
name: molecular-simulations-jupiter
description: Use molecular-simulations on Julich Jupiter ARM64 GH200 nodes with its dedicated software/molecular-simulations/.venv environment.
---

# molecular-simulations on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/molecular-simulations"
PY="$REPO/.venv/bin/python"
"$PY" -c "import molecular_simulations"
```

Select platform CUDA. ARM64 AmberTools 26 is installed in software/ambertools. The shared environment script sets AMBERHOME and adds its binaries to PATH. Importing the Python package alone does not verify builders.

Run GPU workloads through `mcp__hpc__run_code` or `run_batch(gpus=1)`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

The original [molecular-simulations](../molecular-simulations/SKILL.md) skill describes input/output semantics. Use this Jupiter environment and native CUDA invocation; Aurora installation paths and measured timings do not apply. Check `JUPITER.md` and `ports/jupiter/status.json` for verified readiness. CLI/import examples alone do not establish successful inference.
