# ESMFold2 on Jupiter (ARM64 GH200, CUDA)

Use the tool's own venv, where the installed `esm` 3.4.2 provides
`esm.models.esmfold2.EsmFold2Model` directly. The Aurora transformers fork is not used here, and the
Aurora example script is not a Jupiter recipe.

## Environment

The Site profile's `worker_init` runs on every worker before your code and exports `HF_HOME`
(default `$LOAM_CACHE_ROOT`, else `$LOAM_SOFTWARE_ROOT/.hf-cache`) and the internet proxy if one is
set up. The redacted example is `examples/site-profiles/jupiter.yaml`.

```bash
REPO="$LOAM_SOFTWARE_ROOT/esmfold2"
PY="$REPO/.venv/bin/python"
"$PY" -c 'from esm.models.esmfold2 import EsmFold2Model'   # import diagnostic only
```

Download the `biohub/ESMFold2` and `biohub/ESMC-6B` weights before offline use, and confirm
checkpoint permissions and cache completeness. ARM64 uses native PyTorch attention because the
upstream x86-only flash-attention wheel is unavailable; that alone does not prove GPU inference or
equivalent numerical performance.

## Example (validated native inference API)

```python
from esm.models.esmfold2 import (EsmFold2Model, ESMFold2InputBuilder,
                                ProteinInput, StructurePredictionInput)
model = EsmFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
result = ESMFold2InputBuilder().fold(
    model, StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)]),
    num_loops=1, num_sampling_steps=20, num_diffusion_samples=1, seed=42)
output_cif.write_text(result.complex.to_mmcif())
```

Here `sequence` is an authoritative protein sequence and `output_cif` a fresh `Path` in the assigned
workspace. These are reduced settings; the Aurora Python/CLI signatures are not guaranteed to match
this installation, so inspect the installed API before building a task. First-use file fetching and
native-PyTorch fallback warnings are expected; an FP8-on-CPU loading warning alone does not mean
the subsequent `.cuda()` inference is running on CPU.

## Execution contract on Jupiter

- Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Do
  not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims.
  Do not use Aurora modules, `ZE_AFFINITY_MASK`, XPU launchers, ALCF proxies or PBS submissions.
- Jupiter has four GH200 GPUs per node. Keep the `CUDA_VISIBLE_DEVICES` assigned by the launcher;
  inside a one-GPU task use `cuda:0`.
- Stage the script and inputs with `write_files` (only under the campaign's `HPC_BASE_DIR`), then
  run the science through `run_batch(gpus=1, timeout_each=6000)`, one design per row, with a
  subprocess deadline no larger than 6000 seconds. Activate this tool's own venv inside each row
  so the Globus worker's Python is not inherited. Poll `batch_status` until `unfinished == 0`, then
  inspect the actual output files before reporting success.
- `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis only.
- Never pad a batch with duplicate jobs. Keep each row within the remaining Slurm walltime; queue
  waits are separate, so do not cancel queued work just because it waits.
- Stream logs to workspace files and preserve failed attempts.

## Smoke workload

`scripts/jupiter/smoke-workload.py` runs the API above on the 20-residue Trp-cage (PDB 1L2Y) and
asserts on the outputs. Stage it into a fresh run directory and run it with this tool's Python
through `run_batch`:

```bash
"$PY" smoke-workload.py --out <fresh-absolute-directory> --software "$LOAM_SOFTWARE_ROOT"
```

Then inspect `result.json` and the scientific outputs. It verifies installation plumbing and
artifact generation only; it does not establish binding efficacy or count as campaign evidence.

Latest recorded manual validation (on the `moeen` branch's Jupiter port, not a fresh run of this
checkout): **passed** — native ESMFold2 trained-model folding, 153 atoms with finite coordinates.
Evidence directory: `$LOAM_SOFTWARE_ROOT/verification/jupiter-skills-smoke-20261110/esmfold2`.
Larger complexes and production settings still need validation; revalidate after environment or
checkpoint changes.

## Resource index

- `read_skill(name="esmfold2", file="SKILL.md")` — the System-neutral method
- `read_skill(name="esmfold2", file="references/jupiter.md")` — this file
- `read_skill(name="esmfold2", file="scripts/jupiter/smoke-workload.py")` — the smoke workload
