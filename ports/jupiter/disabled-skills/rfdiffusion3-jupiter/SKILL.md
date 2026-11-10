---
name: rfdiffusion3-jupiter
description: Use rfdiffusion3 on Julich Jupiter ARM64 GH200 nodes with its dedicated software/foundry/.venv environment.
---

# rfdiffusion3 on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/foundry"
PY="$REPO/.venv/bin/python"
"$REPO/.venv/bin/rfd3" design --help
```

Use the Foundry JSON design specification; obtain RFD3 checkpoints using the installed foundry CLI. Read the checkout model docs for the installed revision.

Run GPU workloads through `mcp__hpc__run_code` or `run_batch(gpus=1)`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

The original [rfdiffusion3](../rfdiffusion3/SKILL.md) skill describes input/output semantics. Use this Jupiter environment and native CUDA invocation; Aurora installation paths and measured timings do not apply. Check `JUPITER.md` and `ports/jupiter/status.json` for verified readiness. CLI/import examples alone do not establish successful inference.
