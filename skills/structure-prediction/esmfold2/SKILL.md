---
name: esmfold2
description: Predict protein 3D structure from a single amino-acid sequence (no MSA) with ESMFold2 (biohub/ESMFold2 + ESMC-6B language model) on Aurora Intel XPU, writing mmCIF with pLDDT/pTM. Use when you need a fast single-sequence fold of a protein chain (or, via the Python API, protein/DNA/ligand complexes) on an ALCF Aurora compute node.
---

# ESMFold2 on Aurora (Intel Max 1550 / XPU)

ESMFold2 = ESMC-6B protein language model + folding trunk + diffusion structure module.
It runs on XPU through an Aurora-ported checkout: the `esm` package (device-agnostic
`get_default_device()`, bf16 autocast) plus a **forked `transformers`** that provides
`transformers.models.esmfold2`. Fused Triton / flash-attn / cuequivariance paths are
CUDA-only and fall back to eager PyTorch + SDPA on XPU (correct, slower).

## Environment

```bash
REPO=/flare/FRAME-IDP/msinclair/software/esm
# venv is built on the DEFAULT frameworks module (frameworks/2026.1.0, /opt/aurora/26.181.0);
# it uses include-system-site-packages=true, so torch-xpu comes from the module.
module load frameworks                      # usually already loaded on a compute node
export ZE_AFFINITY_MASK=4                   # ONE plain tile index in FLAT mode (NOT "4.0")
export PYTHONPATH="$REPO/transformers/src:$REPO:${PYTHONPATH:-}"   # fork MUST come first
export HF_HOME=$REPO/.hf_cache              # weights already cached here
export HF_HUB_OFFLINE=1                     # avoid hub lookups (compute nodes have no internet w/o proxy)
PY=$REPO/.venv/bin/python
```

If you ever need the hub: `export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128` and drop `HF_HUB_OFFLINE`.

## Weights (already downloaded — do not re-download)

`$REPO/.hf_cache/hub/`:
- `models--biohub--ESMFold2` (folding model)
- `models--biohub--ESMC-6B` (~25 GB LM, auto-loaded by ESMFold2; distinct from `biohub/esmc-6b-2024-12`)
- also `esmc-300m-2024-12`, `ESMC-600M-2024-12` (ESMC embedding models)

## Verified minimal command (single chain -> mmCIF)

Helper script in the repo: `$REPO/scripts/aurora/fold_sequence.py` (a copy of the tested
wrapper is in `scripts/run_esmfold2_example.sh` next to this file).

```bash
$PY $REPO/scripts/aurora/fold_sequence.py \
  --seq MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE \
  --loops 4 --steps 20 --out gb1_pred.cif
```

Verified 2026-10-29 on tile 4 (GB1, L=56): model load 92.6 s, fold 253.5 s,
**pLDDT 0.843, pTM 0.769**, peak 13.9 GB; total wall 6 m 27 s.

### fold_sequence.py options
| flag | default | meaning |
|---|---|---|
| `--seq S` / `--fasta F` / `--length N` | random 48-mer | input (FASTA: first record only) |
| `--loops` | 20 | trunk recycling loops |
| `--steps` | 100 | diffusion sampling steps |
| `--seed` | 0 | sampling seed (note: not honored on XPU, see gotchas) |
| `--chunk-size` | None | lower (32/16) to cut O(L^2) trunk memory for long chains |
| `--pdb` | off | `infer_protein_as_pdb` fast path, PDB out, no pTM (unverified this session) |
| `--out` | pred.cif | output path |

Use the defaults (`--loops 20 --steps 100`) for production quality; 4/20 is a smoke test.

## Python API (complexes, multiple samples)

```python
import torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, DNAInput, LigandInput, StructurePredictionInput
from esm.utils.misc import get_default_device
dev = get_default_device()                        # -> xpu
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(dev).eval()
model.set_kernel_backend(None)                    # REQUIRED off-CUDA: Triton kernels are CUDA-only
spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="MTYK...")])
res = ESMFold2InputBuilder().fold(model, spi, num_loops=20, num_sampling_steps=100, num_diffusion_samples=1)
open("out.cif", "w").write(res.complex.to_mmcif())
print(float(res.plddt.mean()), res.ptm, res.iptm)
```
Multi-entity example (protein + DNA w/ modified base + SAH ligand, 1MHT):
`$REPO/scripts/aurora/readme_example.py` (prior output `$REPO/1mht_pred.cif`; not re-run this session).

## Output & confidence
- mmCIF (`result.complex.to_mmcif()`), coordinates for all atoms.
- `result.plddt` per-residue in **0-1** (not 0-100); script prints the mean. >0.7 = confident.
- `result.ptm` (global), `result.iptm` (interfaces; None/meaningless for a single chain).
- A synthetic/random sequence gives pLDDT ~0.2 — a useful sanity contrast.

## Aurora / XPU gotchas
- `ZE_AFFINITY_MASK` must be a plain integer in FLAT mode (`4`), `4.0` makes XPU vanish. One tile = 64 GB.
- `ImportError ... models.esmfold2`: the fork `$REPO/transformers/src` is not first on PYTHONPATH (stock transformers lacks it).
- Call `model.set_kernel_backend(None)` on XPU (the script does it).
- First fold in a fresh process is slow (253 s for L=56 here vs. ~5 s reported in earlier runs; heavy sys CPU time = XPU kernel JIT/first-call overhead and node contention). Fold many sequences in ONE process to amortize load (~90 s) + warm-up.
- Harmless noise: warnings about transformer_engine / xformers / flash-attn missing, and `🚨 No checkpoint found for ESMCFor...` lines.
- Seeding uses `torch.cuda.*` RNG only, so diffusion sampling is unseeded on XPU (results vary run-to-run).
- Memory: weights ~14 GB; L ≲ 1000 fits one tile; L ≳ 1400 at default chunk may OOM -> `--chunk-size 32` or 16.
- Do not `uv pip install` into this venv (would pull CUDA torch); it relies on the module's torch.

## Troubleshooting
- `torch.xpu.is_available()==False`: check `ZE_AFFINITY_MASK`, that you are on a compute node, and `module load frameworks`.
- Hub/network errors: set `HF_HOME` to the cache above and `HF_HUB_OFFLINE=1`.
- OOM: lower `--chunk-size`, shorten input, or use a different tile that's free.
