# ESMFold2 on Aurora (Intel XPU)

Install: `$LOAM_SOFTWARE_ROOT/esm` — the `esm` package plus a **forked `transformers`** that
provides `transformers.models.esmfold2`. Do not edit the install directory. Triton / flash-attn /
cuequivariance paths are CUDA-only and fall back to eager PyTorch + SDPA on XPU (correct, slower).

## Environment

```bash
REPO=$LOAM_SOFTWARE_ROOT/esm
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
# Minimal runner: scripts/aurora/run_esmfold2_example.sh <out_dir> [tile]
bash scripts/aurora/run_esmfold2_example.sh ./work 4

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

## Must-know

- The forked `$REPO/transformers/src` **must** be first on `PYTHONPATH`; stock transformers lacks
  `models.esmfold2` and you get `ImportError … models.esmfold2`.
- Call `model.set_kernel_backend(None)` on XPU (the helper script does it).
- `ZE_AFFINITY_MASK` must be a plain integer in FLAT mode. One tile = 64 GB.
- Cold start is ~90 s model load plus first-call XPU kernel JIT.
- Seeding uses `torch.cuda.*` RNG only, so diffusion sampling is effectively unseeded on XPU and
  results vary run to run.
- The CLI folds only the first FASTA record — use the Python API for complexes.
- Memory: L ≲ 1000 fits one tile, L ≳ 1400 may OOM → `--chunk-size 32` or 16.
- Do not `uv pip install` into this venv; it relies on the module's torch and a plain install pulls
  a CUDA wheel.
- Harmless noise: missing transformer_engine / xformers / flash-attn warnings, and
  `🚨 No checkpoint found for ESMCFor...` lines.

## Resource index

- `read_skill(name="esmfold2", file="scripts/aurora/run_esmfold2_example.sh")` — minimal one-tile runner
