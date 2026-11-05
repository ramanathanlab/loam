---
name: protenix-jupiter
description: Use protenix on Julich Jupiter ARM64 GH200 nodes with its dedicated software/protenix/.venv environment.
---

# protenix on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/protenix"
PY="$REPO/.venv/bin/python"
"$REPO/.venv/bin/protenix" pred --help
```

The shared environment sets LAYERNORM_TYPE=torch to avoid eager compilation of the optional fused layernorm. The environment uses torch 2.7.1+cu128 and the explicit ARM64 Triton override in ports/jupiter/protenix-overrides.txt. Inspect the installed CLI before choosing kernel options. Use native torch triangle kernels when fused kernels are unavailable. Download model assets before offline execution.

Run GPU workloads through `mcp__hpc__run_code` or `run_batch(gpus=1)`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

The original [protenix](../protenix/SKILL.md) skill describes input/output semantics. Use this Jupiter environment and native CUDA invocation; Aurora installation paths and measured timings do not apply. Check `JUPITER.md` and `ports/jupiter/status.json` for verified readiness. CLI/import examples alone do not establish successful inference.
