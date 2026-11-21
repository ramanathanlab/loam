---
name: genslm-esm
description: Embed codon (DNA) or amino-acid sequences with GenSLM-ESM (ESMC-architecture 600M codon/protein language model), extract features from its trained Top-K sparse autoencoders, and train or validate new SAEs. Use for GenSLM-ESM embeddings, codon language model representations, or sparse-autoencoder interpretability.
systems: [aurora]
---

# GenSLM-ESM + SAE

## Before you run it: read the reference file for this System

The install, the venv, the device selection, the model cache and the trained SAE checkpoints differ
per System. Read `references/<HPC_SYSTEM>.md` (for example
`read_skill(name="genslm-esm", file="references/aurora.md")`) and follow it; it is authoritative
over this file for anything System-specific. This skill has a reference file for **aurora** only. It
is **unsupported** on any other System: report the step as blocked rather than adapting Aurora's
instructions.

On Jupiter in particular only the upstream `genslm_esm` base package is installed — not the
`genslm_esm_sae` fork this skill uses — and the documented trained checkpoints are inaccessible
there (Hugging Face answered 401/`RepositoryNotFoundError` for
`genslm-test/genslm-esmc-300M-contrastive-2026-10-step-200000` and the two older
`genslm-esmc-300M-{contrastive,joint}` repositories). Report that as a software and checkpoint
blocker; never substitute random weights, an unrelated ESM model, or a base-package import as
evidence.

Software, caches and model weights are found only through Site profile variables: the install is
under `$LOAM_SOFTWARE_ROOT`. Never modify the install; write outputs to the campaign workspace.

## Core API

```python
import torch
from genslm_esm_sae.backbone import Backbone
from genslm_esm_sae.sae import SaeModel

dev = torch.device(DEVICE)                # the System's accelerator: see the reference file
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

sae = SaeModel.from_pretrained(SAE_CHECKPOINT).eval().to(dev)
with torch.inference_mode():
    out = sae(acts[-1])                   # MUST be the layer + modality the SAE was trained on
out.feature_magnitudes                    # (n_tokens, 16384), exactly k=64 nonzero per token
out.reconstruction_loss                   # (n_tokens,) MSE in z-scored space
```

Train / validate CLIs (one accelerator, single process):

```bash
genslm-esm-sae-train    --config my_train.yaml          # add --resume-from-checkpoint to resume
genslm-esm-sae-validate --config my_train.yaml --checkpoint-dir <ckpt>/layerNN
```

A minimal runner is bundled per System: `scripts/<system>/embed_and_sae.py --out OUTDIR
[--layer 32 --sae-dir <ckpt>/layerNN]`.

## Inputs

- **Codon**: an uppercase string of **space-separated codons** (`'ATG GCT AAA … TAA'`); each codon
  is one token and the stop codon counts. **Amino acid**: space-separated one-letter residues
  (`'M A K G E L'`). Tokenization must match the checkpoint — never silently translate or
  substitute formats.
- **Training data**: HDF5 shards with equal-length string datasets `codon` and `aminoacid`.
- **`Backbone`**: `model_id`, `layer` (int or list, 1-based, 1–36), `modality`, `max_length`,
  `dtype`.
- **Train config YAML**: `model_id`, `layer`, `modality`, `max_length`,
  `train_data`/`validation_data` (globs — use absolute paths), `codebook_dim`, `k`, `output_dir`,
  and `training_args` (passed to `transformers.TrainingArguments`). The install's
  `campaigns/600m-codon/configs*/` hold worked examples.
- For amino acids, pass `modality='aminoacid'` to `Backbone` (or build the collator with
  `GenslmEsmcDataCollator(return_codon=False, return_aminoacid=True, …)`) and use the
  `aminoacid_labels` key. **All existing SAEs are codon-trained.**

## Outputs

- `bb(batch)` → float32 activations, special/pad tokens already dropped and sequences concatenated.
- SAE → `feature_magnitudes` (n_tokens × 16384, k=64 active), `reconstruction_loss` (per token).
- Training writes `output_dir/checkpoint-N/` plus `train_config.yaml`; validation writes
  `validation.json` **into `--checkpoint-dir`**.

Record the input data and tokenization, the model and checkpoint IDs, the selected layers, tensor
shapes, the device and the outputs, and check the features are finite before interpreting them.

## Must-know

- `Backbone` loads on CPU — move it to the accelerator yourself (the Trainer does it inside
  `SaeTrainer`).
- `bb(batch)` returns **real tokens only, flattened across the batch**. Use the `*_labels != -100`
  mask to split per sequence.
- Layers are **1-based** (`layers must be between 1 and 36`).
- Garbage SAE features mean a layer or modality mismatch — check `sae.config.layer` and
  `sae.config.modality`. Feed the SAE raw float32 hidden states; it z-scores internally.
- **Use a model ID that is actually in the local cache.** `configs/first_sweep.yaml` names
  `genslm-test/genslm-esmc-600M-contrastive`, which is not cached and fails offline; the cached
  backbone is the `-2026-10-step-200000` revision. The repository is private, so a real hub fetch
  needs a token and a proxy.
- **`validate` writes into `--checkpoint-dir`**, so copy a production checkpoint to your own
  directory first.
- Validation runs single-process — don't use torchrun/mpiexec.
- **Under an MPI/PMIx launcher** (Loam `run_batch`/`run_code`, EL, Parsl) the task inherits a live
  PALS/PMIx context and the HF `Trainer` may attempt a distributed init with no peers. Scrub it:
  ```bash
  unset PMIX_RANK PMIX_NAMESPACE PMI_RANK PMI_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE \
        PALS_RANKID PALS_NODEID PARSL_WORKER_RANK
  export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
  ```
- **Give `run_batch` a generous per-row timeout** (`timeout_each >= 1800` for ~12 rows/node). One
  run loads the 2.2 GB backbone in ~160 s solo; many rows per node contend on the parallel
  filesystem and the 600 s default times out every row mid-load. This is I/O contention, not a bug.
