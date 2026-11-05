---
name: chai-cscs
description: Predict all-atom structures of proteins, complexes, protein-ligand (SMILES), DNA/RNA and glycans with Chai-1 on CSCS Alps/Daint (NVIDIA GH200, aarch64, CUDA), optionally with ColabFold MSAs and restraints. CSCS/CUDA port of the Aurora `chai` skill using upstream chai_lab on GPU. Use to fold or co-fold molecules on an Alps GH200 node via SLURM.
---

# Chai-1 on CSCS Alps/Daint (GH200, CUDA)

CSCS port of the Aurora `chai` skill. This is the **stock upstream `chai_lab` PyPI
package** on CUDA.

## IMPORTANT: no launcher, no monkeypatches here

The Aurora skill insists on the `scripts/chai_xpu.py` launcher to patch four XPU breakages
(IPEX stub, antipickle adapter, pandas-3 MSA, RDKit ETKDG). **None of that applies on
CUDA** — the stock CLI runs clean. Do **not** use the XPU launcher here.

Also note the **entry point is `chai-lab`**, not `chai`:
```bash
CHAI=/ritom/scratch/cscs/avasan/skill-venvs/chai/bin/chai-lab          # `chai_lab.main:cli`
export CHAI_DOWNLOADS_DIR=/ritom/scratch/cscs/avasan/skill-venvs/.chai_downloads   # weights cache
```

## Install locations (this machine)
- venv (python 3.12, **torch 2.6.0+cu126** — chai pins torch 2.6.0; the aarch64-CUDA build
  lives on the PyTorch `cu126` index, so it was installed from there, not plain PyPI which
  only has the +cpu aarch64 wheel): `/ritom/scratch/cscs/avasan/skill-venvs/chai`
- weights cache (auto-downloaded, ~7 GB incl. ESM2-3B): `.../skill-venvs/.chai_downloads`

First run downloads `conformers_v1.apkl`, `esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt`, and
`models_v2/*.pt` from chaiassets.com (compute nodes have internet). Offline afterward.
No `module load`, no `ZE_AFFINITY_MASK`, no proxy. GPU is `cuda` automatically.

## Running on a compute node (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=chai
#SBATCH --partition=debug          # first run needs ~28 min budget for the 7 GB download
#SBATCH --account=g245
#SBATCH --nodes=1
#SBATCH --time=00:28:00
#SBATCH --output=/ritom/scratch/cscs/avasan/skill-verify/logs/chai-%j.out   # shared scratch
export CHAI_DOWNLOADS_DIR=/ritom/scratch/cscs/avasan/skill-venvs/.chai_downloads
/ritom/scratch/cscs/avasan/skill-venvs/chai/bin/chai-lab fold input.fasta ./chai_out \
  --num-trunk-recycles 1 --num-diffn-timesteps 50 --num-diffn-samples 1 --seed 42
```

Input FASTA with entity headers (`>protein|name=...`, `>ligand|name=...`, dna/rna/glycan),
all `fold` options (`--use-msa-server`, `--msa-directory`, `--constraint-path`,
`--num-trunk-recycles` (3), `--num-diffn-timesteps` (200), `--num-diffn-samples` (5),
`--seed`, …), and the output layout (`pred.model_idx_{i}.cif` with per-atom pLDDT in the
B-factor column, `scores.model_idx_{i}.npz` with aggregate_score/ptm/iptm) are **identical
to the Aurora skill** — see its `SKILL.md`. For production use the defaults with
`--use-msa-server`. `--device` is unnecessary (CUDA auto); OUT_DIR must be empty/nonexistent.

## Verified on Alps (2026-11-04, GH200, debug, job 5257724)
Ubiquitin (76 aa), single-sequence + ESM embeddings, `--num-trunk-recycles 1
--num-diffn-timesteps 50 --num-diffn-samples 1`: weights auto-downloaded, trunk recycle
1.9 s, 49 diffusion steps @ ~27 it/s, wrote `pred.model_idx_0.cif` + `scores.model_idx_0.npz`.
Score 0.1871 — expected for a monomer (aggregate_score ≈ 0.2*ptm + 0.8*iptm, iptm=0 for one
chain; judge monomers by ptm / pLDDT). PASS.

## Differences from the Aurora skill
- Entry point `chai-lab` (not `chai`); torch is 2.6.0+**cu126**.
- **No** `chai_xpu.py` launcher, no IPEX/antipickle/pandas/RDKit patches, no
  `ZE_AFFINITY_MASK`, no ALCF proxy. Stock CUDA path.
- Weights cache in `CHAI_DOWNLOADS_DIR` on `/ritom/scratch`.

## Troubleshooting (CSCS-specific)
- `chai: No such file or directory`: the script is `chai-lab`, not `chai`.
- `torch ... +cpu` / `cuda False`: reinstall torch from the cu126 index
  (`uv pip install --python .../chai/bin/python --reinstall-package torch --index-url https://download.pytorch.org/whl/cu126 torch==2.6.0+cu126`).
- Ligand missing: grep the log for `Failed to tokenize` (invalid SMILES) — the RDKit issue
  is XPU-only and does not occur here.
