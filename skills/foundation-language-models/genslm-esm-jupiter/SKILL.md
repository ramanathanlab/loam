---
name: genslm-esm-jupiter
description: Use genslm-esm on Julich Jupiter ARM64 GH200 nodes with its dedicated software/genslm-esm/.venv environment.
---

# genslm-esm on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/genslm-esm"
PY="$REPO/.venv/bin/python"
"$PY" -c "import genslm_esm"
```

The current upstream genslm_esm package differs from the Aurora genslm_esm_sae fork. SAE inference requires that exact fork and checkpoint access; do not claim SAE support from the base package.

Run GPU workloads through `mcp__hpc__run_code` or `run_batch(gpus=1)`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

The original [genslm-esm](../genslm-esm/SKILL.md) skill describes input/output semantics. Use this Jupiter environment and native CUDA invocation; Aurora installation paths and measured timings do not apply. Check `JUPITER.md` and `ports/jupiter/status.json` for verified readiness. CLI/import examples alone do not establish successful inference.
