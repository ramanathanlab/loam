---
name: protenix-cscs
description: Predict biomolecular structures (proteins, DNA/RNA, ligands, ions, complexes) with Protenix (ByteDance AlphaFold3-style diffusion model) on CSCS Alps/Daint (NVIDIA GH200, aarch64, CUDA), writing mmCIF + pLDDT/pTM/PAE confidence JSON. CSCS/CUDA port of the Aurora `protenix` skill using upstream protenix on GPU. Use for AF3-like complex/single-chain prediction on an Alps GH200 node via SLURM.
---

# Protenix on CSCS Alps/Daint (GH200, CUDA)

CSCS port of the Aurora `protenix` skill. Stock upstream `protenix` PyPI package on CUDA.

## Two CSCS-specific gotchas (both resolved in this install)

1. **torch pin → use torch 2.5.1.** protenix pins `torch==2.3.1`, which has **no aarch64
   CUDA wheel** (aarch64 CUDA wheels start ~2.5). Bumping to 2.6.0 then broke checkpoint
   loading: torch 2.6 defaults `torch.load(weights_only=True)`, and the ESM2 checkpoint
   contains an `argparse.Namespace` → `UnpicklingError: Unsupported global ... argparse.Namespace`.
   **torch 2.5.1+cu124 (aarch64) threads the needle** — CUDA works *and* `weights_only`
   defaults to the legacy behavior, so the ESM2 load succeeds. It is already installed here.
2. **CLI is `predict`, not `pred`.** Upstream renamed it, and uses long flags
   (`--input/--out_dir/--seeds/--cycle/--step/--sample/--model_name/--use_msa`). There is
   **no `--trimul_kernel/--triatt_kernel`** — upstream does not use cuequivariance at all
   (that was an Aurora-fork addition), so no kernel flags are needed; the torch kernels run
   by default.

## Install locations (this machine)
- venv (python 3.12, **torch 2.5.1** CUDA aarch64): `/ritom/scratch/cscs/avasan/skill-venvs/protenix`
  (CLI `.../protenix/bin/protenix`)
- data/weights root: `PROTENIX_ROOT_DIR=/ritom/scratch/cscs/avasan/skill-venvs/.protenix_data`
  (CCD cache + ESM2-3B + model checkpoints auto-download on first run; compute nodes have internet)

```bash
PROTENIX=/ritom/scratch/cscs/avasan/skill-venvs/protenix/bin/protenix
export PROTENIX_ROOT_DIR=/ritom/scratch/cscs/avasan/skill-venvs/.protenix_data
export PYTHONUNBUFFERED=1
```
No `module load`, no `ZE_AFFINITY_MASK`, no `LD_LIBRARY_PATH` MKL hack, no proxy. Device is
CUDA auto (`protenix/utils/device.py` picks cuda on the GH200).

## Running on a compute node (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=protenix
#SBATCH --partition=debug          # first run downloads ESM2-3B; give ~28 min
#SBATCH --account=g245
#SBATCH --nodes=1
#SBATCH --time=00:28:00
#SBATCH --output=/ritom/scratch/cscs/avasan/skill-verify/logs/protenix-%j.out   # shared scratch
export PROTENIX_ROOT_DIR=/ritom/scratch/cscs/avasan/skill-venvs/.protenix_data
export PYTHONUNBUFFERED=1
cd "$(mktemp -d ./protenix_run.XXXX)"      # ESM embeddings cache to CWD; never run inside the venv dir
/ritom/scratch/cscs/avasan/skill-venvs/protenix/bin/protenix predict \
  --input /path/to/input.json --out_dir ./out --seeds 101 \
  --model_name protenix_mini_esm_v0.5.0 --cycle 4 --step 20 --sample 1 --use_msa false
```

Input JSON format (`proteinChain`/`dnaSequence`/`rnaSequence`/`ligand`/`ion`, `count`,
optional MSA/template paths), the three offline model names (`protenix_mini_esm_v0.5.0`,
`protenix_base_default_v1.0.0`, `protenix-v2`), and the output layout
(`<out>/<name>/seed_<seed>/predictions/<name>_sample_<k>.cif` +
`<name>_summary_confidence_sample_<k>.json` with plddt/ptm/iptm/ranking_score) are
**identical to the Aurora skill** — see its `SKILL.md`. For production use `--step 200`.
`--use_msa false` for fast offline runs; `--use_msa true` needs network (login node, or the
compute node which also has internet).

## Verified on Alps (2026-11-04, GH200, debug, job 5257973)
GB1 (56 aa), `protenix_mini_esm_v0.5.0`, 1 seed, `--cycle 4 --step 20 --sample 1 --use_msa
false`: **pLDDT 94.1, pTM 0.924**, has_clash 0, ranking_score 0.185. 5m25s incl. first-run
ESM2-3B download. Matches the Aurora mini_esm result (93.5 / 0.92). PASS.

## Differences from the Aurora skill
- torch 2.5.1 (not 2.3.1); CLI `predict` (not `pred`) with long flags; no `--trimul/triatt_kernel`.
- Drop `ZE_AFFINITY_MASK`, the oneAPI `LD_LIBRARY_PATH` MKL workaround, and the ALCF proxy.
- Weights under `PROTENIX_ROOT_DIR` on `/ritom/scratch`, auto-downloaded.

## Troubleshooting (CSCS-specific)
- `UnpicklingError: ... argparse.Namespace` / `Weights only load failed`: the venv is on
  torch ≥2.6. Reinstall 2.5.1: `uv pip install --python .../protenix/bin/python
  --reinstall-package torch --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1`.
- `No such command 'pred'`: use `predict`.
- `torch.cuda.is_available()` False: you are on the login node / no allocation, or torch is
  the +cpu wheel (reinstall from the cu124 index as above).
