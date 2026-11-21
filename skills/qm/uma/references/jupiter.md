# UMA on Jupiter (ARM64 GH200, CUDA)

Use the native CUDA fairchem install in its own venv. The Aurora Nitro/XPU fast path, IPEX
assumptions, checkpoint locations and timings are not Jupiter recipes; do not copy them.

**Currently blocked: the real model request returned a Hugging Face gated-repository 401.** The
native CUDA fairchem calculator is installed and imports, but energy/force inference has **not**
passed. Obtain checkpoint access first (below). Do not claim UMA ready or substitute an unrelated
model.

## Environment

The Site profile's `worker_init` runs on every worker before your code and sets `FAIRCHEM_CACHE_DIR`
(default `$LOAM_SOFTWARE_ROOT/fairchem/cache`) and `HF_HOME`. The redacted example is
`examples/site-profiles/jupiter.yaml`.

```bash
REPO="$LOAM_SOFTWARE_ROOT/fairchem"
PY="$REPO/.venv/bin/python"
"$PY" -c 'import fairchem.core'      # an import does not verify model evaluation
```

The device is `"cuda"`. Preserve the one-GPU task's `CUDA_VISIBLE_DEVICES`; inside it the GPU is
`cuda:0`. Before building a calculator, read the installed fairchem API at the installed version; the
Aurora notes' API examples are not verified against it.

## Checkpoint access

The smoke run requested `facebook/UMA`'s `checkpoints/uma-s-1p2p1.pt` and got `GatedRepoError: 401`.
That is checkpoint authentication/authorization, not evidence that reinstalling PyTorch will help.
The user must obtain access to the model on Hugging Face, then log in interactively on the login node
with the same `HF_HOME` the workers use (the one the Site profile's `worker_init` sets):

```bash
export HF_HOME="${HF_HOME:-${LOAM_CACHE_ROOT:-$LOAM_SOFTWARE_ROOT/.hf-cache}}"   # as worker_init sets it
umask 077
"$LOAM_SOFTWARE_ROOT/fairchem/.venv/bin/hf" auth login
```

Keep tokens private; never put them in skill files, campaign prompts, command arguments, logs or
Git. Rerun the smoke workload after access is granted.

## Execution contract on Jupiter

- Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir, with
  absolute campaign paths. Do not submit scheduler jobs, use SSH, replace the scientific software,
  or apply Aurora XPU shims. Do not use Aurora modules, `ZE_AFFINITY_MASK`, XPU launchers, ALCF
  proxies or PBS submissions.
- Jupiter has four GH200 GPUs per node. Keep the `CUDA_VISIBLE_DEVICES` assigned by the launcher;
  inside a one-GPU task use `cuda:0`.
- Stage the script and inputs with `write_files` (only under the campaign's `HPC_BASE_DIR`;
  read-only inspection roots do not grant write access to installed software), then run the science
  through `run_batch(gpus=1, timeout_each=6000)`, one run per row, with a subprocess deadline no
  larger than 6000 seconds (or the campaign YAML's timeouts; allocation walltime still limits
  completion). Activate this tool's own venv inside each row so the Globus worker's Python is not
  inherited. Poll `batch_status` until `unfinished == 0`, then inspect the actual output files
  before reporting success.
- `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis only.
- Never pad a batch with duplicate jobs. Keep each row within the remaining Slurm walltime; queue
  waits are separate, so do not cancel queued work just because it waits.
- Stream logs to workspace files and preserve failed attempts. Retain input/checkpoint identity,
  precision, device, settings, trajectories and failure evidence.

## Smoke workload

`scripts/jupiter/smoke-workload.py` runs a neutral singlet water single point with
`pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda", inference_settings="batch")` and
`FAIRChemCalculator(predictor, task_name="omol")`, and asserts finite energy and forces. Batch mode
avoids assuming the optional Nitro fast path is available. Stage it into a fresh run directory and
run it with this tool's Python through `run_batch`:

```bash
"$PY" smoke-workload.py --out <fresh-absolute-directory>
```

Then inspect `result.json` and `energy-forces.json`. It verifies installation plumbing and
checkpoint access only; it is not evidence of production accuracy and does not count as campaign
evidence.

Latest recorded manual validation (on the `moeen` branch's Jupiter port, not a fresh run of this
checkout): **blocked** — checkpoint access denied by Hugging Face. Evidence directory:
`$LOAM_SOFTWARE_ROOT/verification/jupiter-skills-smoke-20261110/uma` (the evidence root's
`summary.json` records the run). This describes only the tested path and cache/auth state at audit
time; revalidate after environment or checkpoint changes.

## Resource index

- `read_skill(name="uma", file="SKILL.md")` — the System-neutral method
- `read_skill(name="uma", file="references/jupiter.md")` — this file
- `read_skill(name="uma", file="scripts/jupiter/smoke-workload.py")` — the smoke workload
