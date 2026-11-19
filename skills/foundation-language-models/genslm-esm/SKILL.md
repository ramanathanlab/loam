---
name: genslm-esm
description: Embed codon (DNA) or amino-acid sequences with GenSLM-ESM (ESMC-architecture 600M codon/protein language model) on Aurora Intel XPU, extract features from its trained Top-K sparse autoencoders, and train or validate new SAEs. Use for GenSLM-ESM embeddings, codon language model representations, or sparse-autoencoder interpretability on Aurora.
---

# GenSLM-ESM + SAE on Aurora (Intel XPU)

Install: `/flare/FRAME-IDP/msinclair/software/genslm-esm-sae` (package `genslm_esm_sae` v0.2.0,
branch `feat/multi-layer-trainer`). Do not modify it. It wraps the private `genslm-esm` v1.1.0
package (`genslm_esm.modeling.GenslmEsmcModel`, `genslm_esm.data.GenslmEsmcDataCollator`), already
installed in the venv. The sibling `genslm-esm-sae-nanfix` differs only in `train.py` (an eval-NaN
fix); use it only if you need that.

## Environment

**Two venvs exist — use `venvs/aurora`.** It is built on the Aurora `frameworks` module
(`include-system-site-packages = true`) and gets XPU PyTorch from it. `venvs/ci` is isolated with
its own CUDA torch build (`torch.xpu.is_available()` is False) and is for CI/linting only.

```bash
module use /soft/modulefiles
module load frameworks                      # frameworks/2026.1.0, torch 2.13.0a0 (XPU)
source /flare/FRAME-IDP/msinclair/software/genslm-esm-sae/venvs/aurora/bin/activate
export ZE_AFFINITY_MASK=11                  # ONE tile
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_HOME=/flare/FRAME-IDP/msinclair/hf_cache
export HF_HUB_OFFLINE=1                     # weights are pre-cached; no network needed
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp
```

| What | Path |
|---|---|
| Backbone `genslm-test/genslm-esmc-600M-contrastive-2026-10-step-200000` (36 layers, d_model 1152, 2.2 GB fp32) | `$HF_HOME/hub/models--genslm-test--genslm-esmc-600M-contrastive-2026-10-step-200000/snapshots/d1b2513f…/` |
| Trained SAEs, best by eval loss (layers 18-36, codon, codebook 16384, k=64) | `<install>/campaigns/600m-codon/runs/l18-36-lr2e-4/best/layerNN/` |
| Periodic checkpoints | `.../runs/l18-36-lr2e-4/checkpoint-{21000,22000}/layerNN/` |

## Example

```bash
# Minimal runner: scripts/embed_and_sae.py
python scripts/embed_and_sae.py --out OUTDIR [--layer 32 --sae-dir .../best/layer32]
```

```python
import torch
from genslm_esm_sae.backbone import Backbone
from genslm_esm_sae.sae import SaeModel

dev = torch.device('xpu')
bb = Backbone('genslm-test/genslm-esmc-600M-contrastive-2026-10-step-200000',
              layer=[1, 32],              # 1-BASED; int -> (n_tok, d), list -> (n_layers, n_tok, d)
              modality='codon',           # or 'aminoacid'
              max_length=2048, dtype=torch.bfloat16)
bb.model.to(dev)                          # Backbone does NOT move itself to the device
batch = bb.collator([{'codon': s} for s in seqs])
batch = {k: v.to(dev) for k, v in batch.items()}
acts = bb(batch)                          # float32; real tokens only, flattened across the batch

counts  = (batch['codon_labels'] != -100).sum(1).tolist()
per_seq = torch.split(acts[-1], counts)   # split back per sequence; .mean(0) = mean-pooled embedding

sae = SaeModel.from_pretrained('.../runs/l18-36-lr2e-4/best/layer32').eval().to(dev)
with torch.inference_mode():
    out = sae(acts[-1])                   # MUST be the layer + modality the SAE was trained on
out.feature_magnitudes                    # (n_tokens, 16384), exactly k=64 nonzero per token
out.reconstruction_loss                   # (n_tokens,) MSE in z-scored space
```

Train / validate CLIs (single tile):

```bash
genslm-esm-sae-train    --config my_train.yaml          # add --resume-from-checkpoint to resume
genslm-esm-sae-validate --config my_train.yaml --checkpoint-dir <ckpt>/layerNN
```

## Arguments

- **Input formats.** Codon: an uppercase string of **space-separated codons**
  (`'ATG GCT AAA … TAA'`); each codon is one token and the stop codon counts. Amino acid:
  space-separated one-letter residues (`'M A K G E L'`). Training data: HDF5 shards with
  equal-length string datasets `codon` and `aminoacid`. Real example:
  `/flare/GenomeScaleLM/azton/refseq_h5/refseq-validation-10k-faafna.h5`.
- **`Backbone`**: `model_id`, `layer` (int or list, 1-based, 1-36), `modality`, `max_length`, `dtype`.
- **Train config YAML**: `model_id`, `layer`, `modality`, `max_length`,
  `train_data`/`validation_data` (globs — use absolute paths), `codebook_dim`, `k`, `output_dir`,
  and `training_args` (passed to `transformers.TrainingArguments`). See
  `campaigns/600m-codon/configs*/`.
- For amino acids, pass `modality='aminoacid'` to `Backbone` (or build the collator with
  `GenslmEsmcDataCollator(return_codon=False, return_aminoacid=True, …)`) and use the
  `aminoacid_labels` key. **All existing SAEs are codon-trained.**

## Outputs

- `bb(batch)` → float32 activations, special/pad tokens already dropped and sequences concatenated.
- SAE → `feature_magnitudes` (n_tokens × 16384, k=64 active), `reconstruction_loss` (per token).
- Training writes `output_dir/checkpoint-N/` plus `train_config.yaml`; validation writes
  `validation.json` **into `--checkpoint-dir`**.

## Must-know

- **Only `venvs/aurora` sees XPU torch.** `venvs/ci` is a CUDA build.
- `Backbone` loads on CPU — call `bb.model.to('xpu')` yourself (the Trainer does it inside
  `SaeTrainer`).
- `bb(batch)` returns **real tokens only, flattened across the batch**. Use the `*_labels != -100`
  mask to split per sequence.
- Layers are **1-based** (`layers must be between 1 and 36`).
- Garbage SAE features mean a layer or modality mismatch — check `sae.config.layer` and
  `sae.config.modality`. Feed the SAE raw float32 hidden states; it z-scores internally.
- **Use the `-2026-10-step-200000` model ID.** `configs/first_sweep.yaml` names
  `genslm-test/genslm-esmc-600M-contrastive`, which is **not** in the local cache and fails offline.
- **`validate` writes into `--checkpoint-dir`**, so copy a production checkpoint to your own dir first.
- Validation runs single-process — don't use torchrun/mpiexec.
- **Under an MPI/PMIx launcher** (Loam `run_batch`/`run_code`, EL, Parsl) the task inherits a live
  PALS/PMIx context and the HF `Trainer` may attempt a distributed init with no peers. Scrub it:
  ```bash
  unset PMIX_RANK PMIX_NAMESPACE PMI_RANK PMI_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE \
        PALS_RANKID PALS_NODEID PARSL_WORKER_RANK
  export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
  ```
- **Give `run_batch` a generous per-row timeout** (`timeout_each >= 1800` for ~12 rows/node). One
  run loads the 2.2 GB backbone in ~160 s solo; many rows per node contend on Lustre and the 600 s
  default times out every row mid-load. This is I/O contention, not a bug.
- `HF_HUB_OFFLINE=1` plus `HF_HOME` are required; the model repo is private, so a real hub fetch
  needs an `HF_TOKEN` and the proxy.
- PBS: request whole nodes (`select=N`); `ngpus=12` never schedules.
- A large `aten::topk` kernel-override warning prints on every import. Harmless.
