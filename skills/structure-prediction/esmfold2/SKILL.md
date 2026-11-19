---
name: esmfold2
description: Predict protein structure from a single sequence (no MSA) with ESMFold2 (ESMC-6B + folding trunk + diffusion module) on Aurora Intel XPU, writing mmCIF with pLDDT/pTM. Use for fast single-sequence folds, or protein/DNA/ligand complexes via the Python API.
---

# ESMFold2 on Aurora (Intel XPU)

Install: `/flare/FRAME-IDP/msinclair/software/esm` — the `esm` package plus a **forked
`transformers`** that provides `transformers.models.esmfold2`. Do not edit the install directory.
Triton / flash-attn / cuequivariance paths are CUDA-only and fall back to eager PyTorch + SDPA
on XPU (correct, slower).

## Environment

```bash
REPO=/flare/FRAME-IDP/msinclair/software/esm
PY=$REPO/.venv/bin/python
module load frameworks                      # frameworks/2026.1.0; usually already loaded
export ZE_AFFINITY_MASK=4                   # ONE plain tile index; "4.0" makes XPU vanish
export PYTHONPATH="$REPO/transformers/src:$REPO:${PYTHONPATH:-}"   # the fork MUST come first
export HF_HOME=$REPO/.hf_cache              # weights already cached
export HF_HUB_OFFLINE=1
```

Cached under `$HF_HOME/hub/`: `models--biohub--ESMFold2`, `models--biohub--ESMC-6B` (~25 GB,
auto-loaded), plus ESMC 300M/600M. Nothing downloads.

## Example

```bash
# Minimal runner: scripts/run_esmfold2_example.sh <out_dir> [tile]
bash scripts/run_esmfold2_example.sh ./work 4

# Equivalent direct call (production settings)
$PY $REPO/scripts/aurora/fold_sequence.py \
  --seq MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE \
  --loops 20 --steps 100 --out gb1_pred.cif
```

Python API, for complexes and multiple samples:

```python
import torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from esm.models.esmfold2 import (ESMFold2InputBuilder, ProteinInput, DNAInput,
                                 LigandInput, StructurePredictionInput)
from esm.utils.misc import get_default_device

model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(get_default_device()).eval()
model.set_kernel_backend(None)              # REQUIRED off-CUDA: Triton kernels are CUDA-only
spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="MTYK...")])
res = ESMFold2InputBuilder().fold(model, spi, num_loops=20, num_sampling_steps=100,
                                 num_diffusion_samples=1)
open("out.cif", "w").write(res.complex.to_mmcif())
print(float(res.plddt.mean()), res.ptm, res.iptm)
```

A multi-entity example (protein + modified-base DNA + SAH ligand) is in
`$REPO/scripts/aurora/readme_example.py`.

## Arguments (`fold_sequence.py`)

| flag | default | meaning |
|---|---|---|
| `--seq S` / `--fasta F` / `--length N` | random 48-mer | input (FASTA: **first record only**) |
| `--loops` | 20 | trunk recycling loops |
| `--steps` | 100 | diffusion sampling steps |
| `--seed` | 0 | sampling seed (not honoured on XPU — see Must-know) |
| `--chunk-size` | none | set 32/16 to cut O(L²) trunk memory on long chains |
| `--pdb` | off | `infer_protein_as_pdb` fast path; PDB out, no pTM |
| `--out` | pred.cif | output path |

## Outputs

- mmCIF via `result.complex.to_mmcif()`, all-atom coordinates.
- `result.plddt` per-residue on a **0-1** scale (not 0-100); >0.7 is confident.
- `result.ptm` global; `result.iptm` for interfaces (None/meaningless for a single chain).
- A random/synthetic sequence yields pLDDT ≈ 0.2 — a useful sanity contrast.

## Must-know

- The forked `$REPO/transformers/src` **must** be first on `PYTHONPATH`; stock transformers lacks
  `models.esmfold2` and you get `ImportError … models.esmfold2`.
- Call `model.set_kernel_backend(None)` on XPU (the helper script does it).
- `ZE_AFFINITY_MASK` must be a plain integer in FLAT mode. One tile = 64 GB.
- **Cold start is expensive** (~90 s model load plus first-call XPU kernel JIT). Fold many
  sequences in ONE process to amortize it; for a single one-off fold, boltz or protenix start faster.
- Seeding uses `torch.cuda.*` RNG only, so diffusion sampling is effectively unseeded on XPU and
  results vary run to run.
- The CLI folds only the first FASTA record — use the Python API for complexes.
- Memory: weights ~14 GB; L ≲ 1000 fits one tile, L ≳ 1400 may OOM → `--chunk-size 32` or 16.
- Do not `uv pip install` into this venv; it relies on the module's torch and a plain install pulls
  a CUDA wheel.
- Harmless noise: missing transformer_engine / xformers / flash-attn warnings, and
  `🚨 No checkpoint found for ESMCFor...` lines.
