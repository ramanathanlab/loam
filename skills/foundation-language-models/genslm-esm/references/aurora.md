# GenSLM-ESM + SAE on Aurora (Intel XPU)

Install: `$LOAM_SOFTWARE_ROOT/genslm-esm-sae` (package `genslm_esm_sae` v0.2.0, branch
`feat/multi-layer-trainer`). Do not modify it. It wraps the private `genslm-esm` v1.1.0 package
(`genslm_esm.modeling.GenslmEsmcModel`, `genslm_esm.data.GenslmEsmcDataCollator`), already installed
in the venv. The sibling `genslm-esm-sae-nanfix` differs only in `train.py` (an eval-NaN fix); use
it only if you need that.

## Environment

**Two venvs exist — use `venvs/aurora`.** It is built on the Aurora `frameworks` module
(`include-system-site-packages = true`) and gets XPU PyTorch from it. `venvs/ci` is isolated with
its own CUDA torch build (`torch.xpu.is_available()` is False) and is for CI/linting only.

```bash
module use /soft/modulefiles
module load frameworks                      # frameworks/2026.1.0, torch 2.13.0a0 (XPU)
source $LOAM_SOFTWARE_ROOT/genslm-esm-sae/venvs/aurora/bin/activate
export ZE_AFFINITY_MASK=11                  # ONE tile; it then appears as xpu:0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_HOME=${LOAM_CACHE_ROOT:?set cache_root in your Site profile}
export HF_HUB_OFFLINE=1                     # weights are pre-cached; no network needed
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp
```

The device is `torch.device('xpu')`; with the mask set it is the only tile visible.

| What | Path |
|---|---|
| Backbone `genslm-test/genslm-esmc-600M-contrastive-2026-10-step-200000` (36 layers, d_model 1152, 2.2 GB fp32) | `$HF_HOME/hub/models--genslm-test--genslm-esmc-600M-contrastive-2026-10-step-200000/snapshots/<hash>/` |
| Trained SAEs, best by eval loss (layers 18–36, codon, codebook 16384, k=64) | `$LOAM_SOFTWARE_ROOT/genslm-esm-sae/campaigns/600m-codon/runs/l18-36-lr2e-4/best/layerNN/` |
| Periodic checkpoints | `.../runs/l18-36-lr2e-4/checkpoint-{21000,22000}/layerNN/` |

## Example

```bash
SKILL=$HPC_LOAM_HOME/skills/foundation-language-models/genslm-esm
python $SKILL/scripts/aurora/embed_and_sae.py --out OUTDIR \
  [--layer 32 --sae-dir $LOAM_SOFTWARE_ROOT/genslm-esm-sae/campaigns/600m-codon/runs/l18-36-lr2e-4/best/layer32]
```

Training data is a glob of HDF5 shards with `codon` and `aminoacid` string datasets that you can
read — the sweeps were run against a RefSeq validation shard (`refseq-validation-10k-faafna.h5`) in a
genome-scale LM project's own directory, not under your `$LOAM_DATA_ROOT`. Name the absolute path of
a shard set you actually have access to.

## Must-know

- **Only `venvs/aurora` sees XPU torch.** `venvs/ci` is a CUDA build, and its torch import is slow
  (over 60 s on the parallel filesystem).
- `HF_HUB_OFFLINE=1` plus `HF_HOME` are required: the model repository is private, so a real hub
  fetch needs an `HF_TOKEN` and the ALCF proxy.
- PBS: request whole nodes (`select=N`); `ngpus=12` never schedules.
- A large `aten::topk` kernel-override warning prints on every import. Harmless.
- Under Loam `run_batch`, reference the runner by its worker-visible copy (as `$SKILL` above) — the
  Aurora worker cannot see a local skill directory.

## Resource index

- `read_skill(name="genslm-esm", file="scripts/aurora/embed_and_sae.py")` — embed codon and
  amino-acid sequences on one tile and extract SAE features, writing shapes and a feature summary
