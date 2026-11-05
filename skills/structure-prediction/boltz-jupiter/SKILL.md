---
name: boltz-jupiter
description: Use boltz on Julich Jupiter ARM64 GH200 nodes with its dedicated software/boltz/.venv environment.
---

# boltz on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/boltz"
PY="$REPO/.venv/bin/python"
"$REPO/.venv/bin/boltz" predict input.yaml --out_dir output --cache "$BOLTZ_CACHE" --accelerator gpu --devices 1
```

Model weights download on first use. Specify MSA inputs or explicitly select the MSA service. Use the GPU accelerator, with one visible device.

Run GPU workloads through `mcp__hpc__run_code` or `run_batch(gpus=1)`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

The original [boltz](../boltz/SKILL.md) skill describes input/output semantics. Use this Jupiter environment and native CUDA invocation; Aurora installation paths and measured timings do not apply. Check `JUPITER.md` and `ports/jupiter/status.json` for verified readiness. CLI/import examples alone do not establish successful inference.
