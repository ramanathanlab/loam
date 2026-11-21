# Protenix on Jupiter (ARM64 GH200, CUDA)

Use `protenix` from the tool's own venv with the native CUDA installation.

## Environment

The Site profile's `worker_init` runs on every worker before your code and exports
`PROTENIX_ROOT_DIR` (default `$LOAM_SOFTWARE_ROOT/protenix/data`), `LAYERNORM_TYPE=torch` and the
internet proxy if one is set up. The redacted example is `examples/site-profiles/jupiter.yaml`.
`LAYERNORM_TYPE=torch` avoids eager compilation of the optional fused layernorm.

```bash
REPO="$LOAM_SOFTWARE_ROOT/protenix"
PY="$REPO/.venv/bin/python"
"$REPO/.venv/bin/protenix" pred --help      # inspect before choosing kernels or model names
```

The environment uses torch 2.7.1+cu128 with a pinned ARM64 Triton override (recorded on the `moeen`
branch in `ports/jupiter/protenix-overrides.txt`). Download model assets before offline execution,
and confirm the selected checkpoint and chemical-component assets are present.

## Example (validated small CUDA recipe)

```bash
"$REPO/.venv/bin/protenix" pred -i "$INPUT_JSON" -o "$OUT" \
  -n protenix_mini_esm_v0.5.0 -s 42 -c 1 -p 20 -e 1 \
  --use_msa false --trimul_kernel torch --triatt_kernel torch
```

These are reduced settings; set model, seed, sample count, recycles, diffusion steps and MSA/template
use explicitly for campaign work. Use native torch triangle kernels when the fused kernels are
unavailable, and check the installed option names rather than copying obsolete flags.

The mini ESM model still needs the large `esm2_t36_3B_UR50D.pt` checkpoint, its own folding
checkpoint, and the common CCD/cluster assets. The first cold run including downloads took about
14 minutes; do not diagnose a quiet log as a GPU or DataLoader stall — inspect download progress.
After an interruption, verify or replace incomplete cache files before retrying: the installed
downloader checks that common assets exist, not that they are complete. Keep successful caches.

## Results

Inspect the actual mmCIF and confidence JSON under the output tree, including the expected entities
and pLDDT/pTM/ipTM where available. The Aurora fork's kernel requirements, oneAPI workaround,
checkpoints, fixes and timings say nothing about Jupiter behaviour.

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

`scripts/jupiter/smoke-workload.py` runs the recipe above on the 20-residue Trp-cage (PDB 1L2Y)
and asserts on the outputs. Stage it into a fresh run directory and run it with this tool's Python
through `run_batch`:

```bash
"$PY" smoke-workload.py --out <fresh-absolute-directory> --software "$LOAM_SOFTWARE_ROOT"
```

Then inspect `result.json` and the scientific outputs. It verifies installation plumbing and
artifact generation only; it does not establish binding efficacy or count as campaign evidence.

Latest recorded manual validation (on the `moeen` branch's Jupiter port, not a fresh run of this
checkout): **passed** — mini ESM CUDA folding on the real checkpoint/embedding path, producing a CIF
with exactly 20 residues, 154 atoms and finite coordinates. Evidence directory:
`$LOAM_SOFTWARE_ROOT/verification/jupiter-skills-smoke-20261110/protenix`. This validates only that
reduced single-sequence path at audit time; revalidate after environment or checkpoint changes.

## Resource index

- `read_skill(name="protenix", file="SKILL.md")` — the System-neutral method
- `read_skill(name="protenix", file="references/jupiter.md")` — this file
- `read_skill(name="protenix", file="scripts/jupiter/smoke-workload.py")` — the smoke workload
- `read_skill(name="protenix", file="examples/example_gb1.json")` — example input (any System)
