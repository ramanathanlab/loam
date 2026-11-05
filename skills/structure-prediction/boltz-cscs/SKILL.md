---
name: boltz-cscs
description: Predict biomolecular structures (proteins, complexes, protein-ligand, DNA/RNA) and binding affinity with Boltz-2 on CSCS Alps/Daint (NVIDIA GH200, aarch64, CUDA). CSCS/CUDA port of the Aurora `boltz` skill, using upstream Boltz on GPU. Use to fold a sequence/complex, dock a SMILES/CCD ligand, or estimate affinity on an Alps GH200 node via SLURM.
---

# Boltz-2 on CSCS Alps/Daint (GH200, CUDA)

CSCS port of the Aurora `boltz` skill. This is the **stock upstream `boltz` PyPI package**
(not the Aurora XPU fork): standard CUDA inference with `--accelerator gpu`.

## Install locations (this machine)
- venv (python 3.12, torch 2.15.0+cu132): `/ritom/scratch/cscs/avasan/skill-venvs/boltz`
  (CLI at `.../boltz/bin/boltz`)
- weights cache (downloaded once, on shared scratch): `/ritom/scratch/cscs/avasan/skill-venvs/.boltz_cache`
  (`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols/`, `mols.tar`)

```bash
BOLTZ=/ritom/scratch/cscs/avasan/skill-venvs/boltz/bin/boltz
export BOLTZ_CACHE=/ritom/scratch/cscs/avasan/skill-venvs/.boltz_cache   # or pass --cache
```
First run auto-downloads the weights + CCD (compute nodes have internet). After that it is
offline. No `module load`, no `ZE_AFFINITY_MASK`, no proxy.

## IMPORTANT: run with `--no_kernels` on aarch64

Boltz's accelerated triangular-mult kernel needs NVIDIA's `cuequivariance_ops_torch`, whose
compiled `libcue_ops.so` has **no working aarch64 build** (install succeeds but import fails
with `libcue_ops.so: cannot open shared object file`). Without the flag, inference dies with
`ModuleNotFoundError: No module named 'cuequivariance_torch'` (or the .so error). **Pass
`--no_kernels`** to use the pure-PyTorch path — correct, slightly slower. (This is the same
flag the Aurora skill documents for the CUDA-only kernels.)

## Running on a compute node (SLURM)

Use `--accelerator gpu` (NOT `xpu`). Whole-node GH200; one process uses `cuda:0`.

```bash
#!/bin/bash
#SBATCH --job-name=boltz
#SBATCH --partition=debug          # low/normal for bigger jobs; weight DL + fold ~ a few min on debug
#SBATCH --account=g245
#SBATCH --nodes=1
#SBATCH --time=00:25:00
#SBATCH --output=/ritom/scratch/cscs/avasan/skill-verify/logs/boltz-%j.out   # shared scratch, not /tmp
BOLTZ=/ritom/scratch/cscs/avasan/skill-venvs/boltz/bin/boltz
export BOLTZ_CACHE=/ritom/scratch/cscs/avasan/skill-venvs/.boltz_cache
$BOLTZ predict input.yaml --out_dir ./boltz_out --cache $BOLTZ_CACHE \
  --accelerator gpu --devices 1 --recycling_steps 1 --sampling_steps 50 \
  --diffusion_samples 1 --output_format pdb --override --num_workers 0 --no_kernels
```

Input YAML / FASTA formats, affinity, multimers, DNA/RNA, `--use_msa_server`, all options
(`--recycling_steps`, `--sampling_steps`, `--diffusion_samples`, `--step_scale`,
`--use_potentials`, `--write_full_pae`, …) and the output layout
(`out/boltz_results_<name>/predictions/<name>/…` + `confidence_*.json`, `affinity_*.json`)
are **identical to the Aurora skill** — see its `SKILL.md`. For production use the defaults
(`--recycling_steps 3 --sampling_steps 200 --diffusion_samples 5`). Keep `--no_kernels`.

## Verified on Alps (2026-11-04, GH200, debug, job 5257716)
Ubiquitin (76 aa), single-sequence (`msa: empty`), `--recycling_steps 1 --sampling_steps 50
--diffusion_samples 1 --no_kernels`: **complex_plddt 0.929, chains_ptm 0.921** (matches the
Aurora 0.93). Weight download ~once; inference < 1 min after cache warm. PASS.

## Differences from the Aurora skill
- `--accelerator gpu` instead of `xpu`; drop `ZE_AFFINITY_MASK` and the ALCF proxy env vars.
- Weights live in `BOLTZ_CACHE` on `/ritom/scratch` and auto-download on first run.
- **`--no_kernels` is required** here (cuequivariance ops lib has no aarch64 build).

## Troubleshooting (CSCS-specific)
- `ModuleNotFoundError: cuequivariance_torch` or `libcue_ops.so ... No such file`: you
  omitted `--no_kernels`. Add it.
- Tries to re-download weights: `--cache`/`BOLTZ_CACHE` not pointing at the shared cache.
- OOM on large complexes: lower `--diffusion_samples` / `--max_parallel_samples` (one GH200 = ~97 GB).
