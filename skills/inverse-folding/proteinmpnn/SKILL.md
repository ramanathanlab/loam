---
name: proteinmpnn
description: Use proteinmpnn on Julich Jupiter ARM64 GH200 nodes with its dedicated software/proteinmpnn/.venv environment.
---

# proteinmpnn on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed under `$LOAM_SOFTWARE_ROOT`, with a dedicated `.venv` per tool.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/proteinmpnn"
PY="$REPO/.venv/bin/python"
"$PY" "$REPO/protein_mpnn_run.py" --pdb_path input.pdb --out_folder output --num_seq_per_target 2 --sampling_temp 0.1 --seed 42
```

Upstream ProteinMPNN automatically selects CUDA. Its pretrained weights are in the checkout. No IPEX shim is needed.

For mmCIF inputs from RFD4 or Chai, use Gemmi from this tool's own venv for conversion. Gemmi installed in another tool environment is not visible here. See `references/workflow.md` for conversion and chain-preservation checks.

Run campaign scientific workloads through `run_batch(gpus=1)` after staging with `write_files`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.


## Executable reference

Read `references/smoke-workload.py` with `read_skill(name="proteinmpnn", file="references/smoke-workload.py")` for a concrete Jupiter workflow with output assertions. Stage its contents into the assigned workspace before execution, using this tool's Python and `--out <absolute-output-directory> --software $LOAM_SOFTWARE_ROOT`. Reduced smoke settings validate plumbing; choose campaign settings for scientific evidence.

## Campaign execution

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` in a bash subprocess before activating the tool environment. Preserve the assigned `CUDA_VISIBLE_DEVICES`; use `cuda:0` within a one-GPU row. Stage scripts and inputs with `write_files`, execute real sweeps with `run_batch(gpus=1)`, poll completion, then inspect actual output files. Batch floors are configured by the deployment server; consult the campaign setup. Never pad with duplicate jobs. Keep each row within the remaining Slurm walltime.
