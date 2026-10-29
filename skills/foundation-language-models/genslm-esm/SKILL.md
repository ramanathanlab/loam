---
name: genslm-esm
description: Run GenSLM-ESM (ESMC-architecture codon/protein language model, 600M contrastive checkpoint) and its Top-K sparse autoencoders (genslm-esm-sae) on ALCF Aurora XPUs. You can embed codon (DNA, as space-separated codons) or amino-acid sequences to get per-token or mean-pooled hidden states from any of the 36 layers, extract SAE features from trained per-layer SAEs (layers 18-36, 16384 features, k=64), and train or validate SAEs with the genslm-esm-sae-train and genslm-esm-sae-validate CLIs. Use it for GenSLM / GenSLM-ESM embeddings, codon language model representations, sparse autoencoder features or interpretability on GenSLM-ESM, or SAE training and validation on Aurora.
---

# GenSLM-ESM + SAE on Aurora (XPU)

Software: `/flare/FRAME-IDP/msinclair/software/genslm-esm-sae` (package `genslm_esm_sae` v0.2.0, git branch `feat/multi-layer-trainer`). Do not modify that directory.
It wraps the private `genslm-esm` v1.1.0 package (`genslm_esm.modeling.GenslmEsmcModel`, `genslm_esm.data.GenslmEsmcDataCollator`), which is already installed in the venv.

Sibling directory `genslm-esm-sae-nanfix` is a branch checkout with one extra commit, "fix(train): keep non-finite eval tokens from poisoning eval and best SAEs". Only `train.py` differs. Don't use it unless you need that eval NaN fix. Everything below was tested against the main directory.

## Environment (verified)

Two venvs exist. Use **`venvs/aurora`**. It is built on the Aurora `frameworks` module (`include-system-site-packages = true`) and gets the XPU PyTorch from that module.
`venvs/ci` is isolated (`include-system-site-packages = false`) and has its own `torch 2.14.1+cu130` (CUDA build; `torch.xpu.is_available()` is False, and the import is slow, over 60 s on Lustre). It is for CI/linting, not XPU work.

```bash
module use /soft/modulefiles
module load frameworks                      # aurora_frameworks-2026.1.0, torch 2.13.0a0 (XPU)
source /flare/FRAME-IDP/msinclair/software/genslm-esm-sae/venvs/aurora/bin/activate
export ZE_AFFINITY_MASK=11                  # pin to ONE tile (pick your own tile id)
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_HOME=/flare/FRAME-IDP/msinclair/hf_cache
export HF_HUB_OFFLINE=1                     # weights are pre-cached; no network needed
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp
```
Versions seen: torch 2.13.0a0+gitcf30153, transformers 4.50.3, genslm_esm_sae 0.2.0.
`torch.device('xpu')` and the HF Trainer (`TrainingArguments.device` = `xpu:0`) both pick up the pinned tile automatically.

## Weights (local)

| What | Path |
|---|---|
| Backbone `genslm-test/genslm-esmc-600M-contrastive-2026-10-step-200000` (36 layers, d_model 1152, 18 heads, 2.2 GB fp32) | `$HF_HOME/hub/models--genslm-test--genslm-esmc-600M-contrastive-2026-10-step-200000/snapshots/d1b2513fe2a4533a85f75f3ae22062883d698a62/` |
| Trained SAEs, best by eval loss (layers 18-36, codon, codebook 16384, k=64, step ~22000) | `.../genslm-esm-sae/campaigns/600m-codon/runs/l18-36-lr2e-4/best/layerNN/` (`config.json`, `model.safetensors`, `metadata.json` with eval_loss/dead_fraction) |
| Periodic checkpoints | `.../runs/l18-36-lr2e-4/checkpoint-21000/`, `checkpoint-22000/` (each has `layerNN/` subdirs) |
| Early debug SAE | `.../campaigns/600m-codon/debug/layer18/` |

Note: `configs/first_sweep.yaml` names `genslm-test/genslm-esmc-600M-contrastive`. That model is **not** in the local cache. Use the `-2026-10-step-200000` ID (as the campaign configs do). The cache also holds Biohub ESMC 300M/600M/6B and ESMFold2, which this tool doesn't use.

## Input formats

- **Codon**: an uppercase string of space-separated codons, e.g. `'ATG GCT AAA ... TAA'`. Each codon is one token, and the stop codon counts as a token.
- **Amino acid**: space-separated one-letter residues, e.g. `'M A K G E L'`.
- Training/validation data: HDF5 shards with equal-length string datasets `codon` and `aminoacid` (same space-separated format). Real example: `/flare/GenomeScaleLM/azton/refseq_h5/refseq-validation-10k-faafna.h5` (10k seqs).

## Verified recipe 1: embeddings + SAE features (Python)

The tested script is `scripts/embed_and_sae.py`. Run it with `python scripts/embed_and_sae.py --out OUTDIR [--layer 32 --sae-dir .../best/layer32]`.
Core API:
```python
import torch
from genslm_esm_sae.backbone import Backbone
from genslm_esm_sae.sae import SaeModel

dev = torch.device('xpu')
bb = Backbone('genslm-test/genslm-esmc-600M-contrastive-2026-10-step-200000',
              layer=[1, 32],            # 1-based; int -> (n_tok, d), list -> (n_layers, n_tok, d)
              modality='codon',          # or 'aminoacid'
              max_length=2048, dtype=torch.bfloat16)
bb.model.to(dev)                         # Backbone does NOT move itself to the device
batch = bb.collator([{'codon': s} for s in seqs])
batch = {k: v.to(dev) for k, v in batch.items()}
acts = bb(batch)                         # float32; special/pad tokens already dropped and sequences concatenated
counts = (batch['codon_labels'] != -100).sum(1).tolist()
per_seq = torch.split(acts[-1], counts)  # split back into sequences; .mean(0) gives a mean-pooled embedding

sae = SaeModel.from_pretrained('.../runs/l18-36-lr2e-4/best/layer32').eval().to(dev)
with torch.inference_mode():
    out = sae(acts[-1])                  # must be the SAME layer + modality the SAE was trained on (sae.config.layer)
out.feature_magnitudes                   # (n_tokens, 16384), exactly k=64 nonzero per token
out.reconstruction_loss                  # (n_tokens,) MSE in z-scored space
```
For amino acids, build the collator with `GenslmEsmcDataCollator(return_codon=False, return_aminoacid=True, max_length=..., tokenizer=...)` (or pass `modality='aminoacid'` to `Backbone`) and use the `aminoacid_labels` key. All existing SAEs are **codon**-trained.

Test result (tile 11): 2 codon seqs (21+20 codons) gave acts `[2, 41, 1152]`, all values finite, and a pooled cosine similarity of 0.995. 2 aa seqs gave `[2, 39, 1152]`. SAE layer 32 gave features `[41, 16384]`, max active per token 64, and mean recon loss 0.037. Backbone load took 1.3 s and peak XPU memory was 2.6 GB. The whole script took about 14 s warm (about 55 s the first time, from cold Lustre and imports).

## Verified recipe 2: CLIs (train / validate), single tile

```bash
genslm-esm-sae-train    --config tiny_train.yaml          # add --resume-from-checkpoint to resume
genslm-esm-sae-validate --config tiny_train.yaml --checkpoint-dir <ckpt>/layerNN   # or a single-layer ckpt dir
```
The config is a YAML file with these keys: `model_id`, `layer` (int or list), `modality`, `max_length`, `train_data`/`validation_data` (globs), `codebook_dim`, `k`, `output_dir`, and `training_args` (a dict passed to `transformers.TrainingArguments`). See `configs/first_sweep.yaml` and `campaigns/600m-codon/configs*/`.
The tested tiny config (8 seqs truncated to 64 codons, codebook 1024, k 16, 4 steps, bf16, `dataloader_num_workers: 0`) is at `/lus/flare/projects/FRAME-IDP/msinclair/testing_grounds/env_tests/test_runs/genslm-esm/tiny_train.yaml`.
- Training ran for 4 steps on xpu (loss about 1.09) in 40 s wall. It writes `output_dir/checkpoint-4/` and `train_config.yaml`.
- Validation runs a single process (don't use torchrun/mpiexec) and writes `validation.json` into the checkpoint dir. It takes about 13 s.
- Validating the real `best/layer32` SAE on 512 tokens gave recon loss 0.157 with 64 active features per token. The validate CLI writes into `--checkpoint-dir`, so **copy a production checkpoint to your own dir first** (the test copied `best/layer32` to `test_runs/genslm-esm/best_layer32_copy/`).

Multi-tile/multi-node training (not re-tested here) uses `campaigns/600m-codon/train_layers.pbs` / `train_stack.pbs`: `mpiexec -n 12 --ppn 12 ... rank_env.sh python -m genslm_esm_sae.train`. `rank_env.sh` maps `PALS_RANKID` to `RANK`/`LOCAL_RANK`, and a launch needs `WORLD_SIZE` and `MASTER_ADDR`/`MASTER_PORT`. oneCCL requires an mpiexec launch, not torchrun.

## Aurora/XPU gotchas

- A huge "Overriding a previously registered kernel ... aten::topk ... dispatch key: XPU" warning appears on every import. It is harmless and comes from the frameworks module.
- `Backbone` loads on CPU. Call `bb.model.to('xpu')` yourself (the Trainer does this inside `SaeTrainer`).
- `bb(batch)` returns activations of **real tokens only, flattened across the batch**. Use the `*_labels != -100` mask to split them per sequence.
- The SAE z-scores each token internally, so feed raw float32 hidden states.
- Only `venvs/aurora` sees XPU torch. `venvs/ci` doesn't.
- `HF_HUB_OFFLINE=1` plus `HF_HOME=/flare/FRAME-IDP/msinclair/hf_cache` are needed; otherwise HF tries the network. If you really need the hub, set the proxy: `export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128`. The model repo is private (`genslm-test/...`), so an `HF_TOKEN` is needed.
- PBS: request whole nodes (`select=N`). `ngpus=12` never schedules. `ZE_FLAT_DEVICE_HIERARCHY=FLAT` exposes 12 tiles.
- Every run's config sets `bf16: true`. The backbone runs under bf16 autocast while the SAE stays in fp32.

## Troubleshooting

- `FileNotFoundError: no HDF5 shards match` means the glob in `train_data`/`validation_data` is wrong. Use absolute paths.
- `OSError ... genslm-esmc-600M-contrastive` (offline) means you used the model ID from `first_sweep.yaml`. Use `...-2026-10-step-200000`.
- `layers must be between 1 and 36` means layers are 1-based.
- Garbage SAE features come from a layer or modality mismatch. Check `sae.config.layer` and `sae.config.modality`.
- NaN eval losses in long training runs are what the `genslm-esm-sae-nanfix` branch addresses.

## Test artifacts

`/lus/flare/projects/FRAME-IDP/msinclair/testing_grounds/env_tests/test_runs/genslm-esm/` contains:
- `run.sh` + `run.log` (recipe 1)
- `out/results.json`, `*_meanpool.pt`, `sae_layer32_features.pt`
- `run_cli.sh` + `run_cli.log` (recipe 2)
- `tiny_*.h5`, `tiny_run/`, `best_layer32_copy/validation.json`
