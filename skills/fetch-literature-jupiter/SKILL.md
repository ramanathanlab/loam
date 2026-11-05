---
name: fetch-literature-jupiter
description: Use fetch-literature on Julich Jupiter ARM64 GH200 nodes with its dedicated software/fetch-literature/.venv environment.
---

# fetch-literature on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/fetch-literature"
PY="$REPO/.venv/bin/python"
"$PY" -c "import requests"
```

Use the PubMed/Europe PMC API recipes in the original skill. Internet access is required; preserve identifiers, source URLs, and retrieval dates.

Run GPU workloads through `mcp__hpc__run_code` or `run_batch(gpus=1)`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

The original [fetch-literature](../fetch-literature/SKILL.md) skill describes input/output semantics. Use this Jupiter environment and native CUDA invocation; Aurora installation paths and measured timings do not apply. Check `JUPITER.md` and `ports/jupiter/status.json` for verified readiness. CLI/import examples alone do not establish successful inference.
