---
name: esmfold2-jupiter
description: Use esmfold2 on Julich Jupiter ARM64 GH200 nodes with its dedicated software/esmfold2/.venv environment.
---

# esmfold2 on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/esmfold2"
PY="$REPO/.venv/bin/python"
"$PY" -c "from esm.models.esmfold2 import EsmFold2Model"
```

The installed esm 3.4.2 provides EsmFold2Model directly; the Aurora transformers fork is no longer required. Download biohub/ESMFold2 and biohub/ESMC-6B weights before offline use. ARM64 uses native PyTorch attention when the upstream x86-only flash-attention wheel is unavailable. Inference still requires GPU verification. Do not substitute ESMFold v1.

Run GPU workloads through `mcp__hpc__run_code` or `run_batch(gpus=1)`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

The original [esmfold2](../esmfold2/SKILL.md) skill describes input/output semantics. Use this Jupiter environment and native CUDA invocation; Aurora installation paths and measured timings do not apply. Check `JUPITER.md` and `ports/jupiter/status.json` for verified readiness. CLI/import examples alone do not establish successful inference.
