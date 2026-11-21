# Boltz-2 on Jupiter (ARM64 GH200, CUDA)

Use the native `boltz` CLI in the tool's own venv with CUDA, not the Aurora XPU runner.

## Environment

The Site profile's `worker_init` runs on every worker before your code and exports the tool caches,
including `BOLTZ_CACHE` (default `$LOAM_SOFTWARE_ROOT/boltz/cache`) and the internet proxy if one is
set up. The redacted example is `examples/site-profiles/jupiter.yaml`.

```bash
REPO="$LOAM_SOFTWARE_ROOT/boltz"
PY="$REPO/.venv/bin/python"
```

Model weights download on first use. Verify the required checkpoints and assets are in
`$BOLTZ_CACHE`; Aurora's cache locations say nothing about what is present here.

## Example

```bash
"$REPO/.venv/bin/boltz" predict "$INPUT_YAML" \
  --out_dir "$OUT" --cache "$BOLTZ_CACHE" \
  --accelerator gpu --devices 1 --num_workers 0 --no_kernels
```

Set recycling and sampling parameters explicitly from the approved campaign budget. Network MSA or
model-asset retrieval uses the inherited proxy settings; keep TLS verification on.

## Jupiter-specific arguments

| flag | value | why |
|---|---|---|
| `--accelerator gpu --devices 1` | required | one visible CUDA device |
| `--cache` | `"$BOLTZ_CACHE"` | set by the Site profile's `worker_init` |
| `--no_kernels` | required | see below |
| `--num_workers 0` | required | single-case loader |

Jupiter's installed Boltz lacks the optional `cuequivariance_torch` triangle kernels. Its default
prediction path fails at inference with `ModuleNotFoundError`, despite successful input preparation
and CUDA detection. `--no_kernels` runs the native PyTorch implementation on CUDA; it keeps the
trained model and does not replace inference with a mock.

## Results

Inspect the actual predicted structure and confidence/NPZ files under
`boltz_results_*/predictions/`. Compare confidence scales and field names in the installed version
rather than assuming Chai's scale.

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

`scripts/jupiter/smoke-workload.py` folds the 20-residue Trp-cage (PDB 1L2Y) single-sequence on CUDA
with native PyTorch kernels and asserts on the outputs. Stage it into a fresh run directory and run
it with this tool's Python through `run_batch`:

```bash
"$PY" smoke-workload.py --out <fresh-absolute-directory> --software "$LOAM_SOFTWARE_ROOT"
```

Then inspect `result.json` and the scientific outputs. It verifies installation plumbing and
artifact generation only; it does not establish binding efficacy or count as campaign evidence.

Latest recorded manual validation (on the `moeen` branch's Jupiter port, not a fresh run of this
checkout): **passed** — single-sequence CUDA folding with native PyTorch kernels. Evidence
directory: `$LOAM_SOFTWARE_ROOT/verification/jupiter-skills-boltz-final-20261110/boltz`. This
describes only the tested path and cache state at audit time; revalidate after environment or
checkpoint changes.

## Resource index

- `read_skill(name="boltz", file="SKILL.md")` — the System-neutral method
- `read_skill(name="boltz", file="references/jupiter.md")` — this file
- `read_skill(name="boltz", file="scripts/jupiter/smoke-workload.py")` — the smoke workload
- `examples/prot_no_msa.yaml`, `examples/lig_aff.yaml`, `examples/ubq_msa.fasta` — example inputs
  (any System; not campaign results)
