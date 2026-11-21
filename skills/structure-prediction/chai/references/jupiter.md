# Chai-1 on Jupiter (ARM64 GH200, CUDA)

Use the native `chai-lab` CLI in the tool's own venv, not the Aurora `chai_xpu.py` launcher.

## Environment

The Site profile's `worker_init` runs on every worker before your code and exports the tool caches,
including `CHAI_DOWNLOADS_DIR` (default `$LOAM_SOFTWARE_ROOT/chai-lab/downloads`) and the internet
proxy if one is set up. The redacted example is `examples/site-profiles/jupiter.yaml`.

```bash
REPO="$LOAM_SOFTWARE_ROOT/chai-lab"
PY="$REPO/.venv/bin/python"
```

The first inference downloads model assets to `$CHAI_DOWNLOADS_DIR`; asset downloads or cold
filesystem reads can dominate startup.

## Example

Plumbing check (reduced settings):

```bash
"$REPO/.venv/bin/chai-lab" fold input.fasta output --num-trunk-recycles 1 \
  --num-diffn-timesteps 50 --num-diffn-samples 1 --seed 42
```

Campaign complex, production defaults stated explicitly:

```bash
"$REPO/.venv/bin/chai-lab" fold complex.fasta fresh_output \
  --num-trunk-recycles 3 --num-diffn-timesteps 200 \
  --num-diffn-samples 5 --seed 42 --no-use-msa-server
```

This uses single-sequence input with no external MSA server; record that limitation. Select MSA and
template use explicitly according to the experiment; network access uses the inherited proxy
settings. These production settings completed on a real 205-residue EGFR target plus a 60-residue
binder on Jupiter.

## Results

Inspect each `pred.model_idx_*.cif` and its `scores.model_idx_*.npz`: verify both complete input
sequences, finite coordinates and residue counts. The installed score files include `ptm`, `iptm`
and `aggregate_score`; read their actual values rather than inventing a confidence metric. Extract
binder pLDDT from its actual atoms. Historical XPU timings, patches, defaults and cache claims in
the Aurora reference are not Jupiter validation.

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

`scripts/jupiter/smoke-workload.py` folds the 20-residue Trp-cage (PDB 1L2Y) with reduced settings
and asserts on the outputs. Stage it into a fresh run directory and run it with this tool's Python
through `run_batch`:

```bash
"$PY" smoke-workload.py --out <fresh-absolute-directory> --software "$LOAM_SOFTWARE_ROOT"
```

Then inspect `result.json` and the scientific outputs. It verifies installation plumbing and
artifact generation only; it does not establish binding efficacy or count as campaign evidence.

Latest recorded manual validation (on the `moeen` branch's Jupiter port, not a fresh run of this
checkout): **passed** — single-sequence trained-model folding and confidence output. Evidence
directory: `$LOAM_SOFTWARE_ROOT/verification/jupiter-skills-smoke-20261110/chai`. This describes
only the tested path and cache state at audit time; revalidate after environment or checkpoint
changes.

## Resource index

- `read_skill(name="chai", file="SKILL.md")` — the System-neutral method
- `read_skill(name="chai", file="references/jupiter.md")` — this file
- `read_skill(name="chai", file="scripts/jupiter/smoke-workload.py")` — the smoke workload
- `examples/ubq.fasta`, `examples/ubq_lig.fasta` — example inputs (any System)
