---
name: rag-build-index
description: Use rag-build-index on Julich Jupiter ARM64 GH200 nodes with its dedicated software/rag-build-index/.venv environment.
---

# rag-build-index on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed under `$LOAM_SOFTWARE_ROOT`, with a dedicated `.venv` per tool.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/rag-build-index"
PY="$REPO/.venv/bin/python"
"$PY" -c "import faiss, datasets, sentence_transformers"
```

Use the original index-building API with this environment. Download the selected embedding model before offline execution; retain model identity and document metadata.


## Executable reference

Read `references/smoke-workload.py` with `read_skill(name="rag-build-index", file="references/smoke-workload.py")` for a concrete Jupiter workflow with output assertions. Stage its contents into the assigned workspace before execution, using this tool's Python and `--out <absolute-output-directory> --software $LOAM_SOFTWARE_ROOT`. Reduced smoke settings validate plumbing; choose campaign settings for scientific evidence.

## Campaign execution

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` in a bash subprocess before activating the tool environment. Preserve the assigned `CUDA_VISIBLE_DEVICES`; use `cuda:0` within a one-GPU row. Stage scripts and inputs with `write_files`, execute real sweeps with `run_batch(gpus=1)`, poll completion, then inspect actual output files. Batch floors are configured by the deployment server; consult the campaign setup. Never pad with duplicate jobs. Keep each row within the remaining Slurm walltime.
