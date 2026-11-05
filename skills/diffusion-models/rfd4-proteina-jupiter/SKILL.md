---
name: rfd4-proteina-jupiter
description: Use rfd4-proteina on Julich Jupiter ARM64 GH200 nodes with its dedicated software/RFD4-Proteina/.venv environment.
---

# rfd4-proteina on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/RFD4-Proteina"
PY="$REPO/.venv/bin/python"
cd "$REPO"
"$REPO/.venv/bin/rfd4-generate" --help
```

Read the checkout AGENTS.md and skills/rfd4-proteina-setup/SKILL.md, then its generate skill. Hydra uses key=value arguments. Checkpoint access and generation must pass upstream preflight before use; no Aurora lightning shim is needed.

Run GPU workloads through `mcp__hpc__run_code` or `run_batch(gpus=1)`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

The original [rfd4-proteina](../rfd4-proteina/SKILL.md) skill describes input/output semantics. Use this Jupiter environment and native CUDA invocation; Aurora installation paths and measured timings do not apply. Check `JUPITER.md` and `ports/jupiter/status.json` for verified readiness. CLI/import examples alone do not establish successful inference.
