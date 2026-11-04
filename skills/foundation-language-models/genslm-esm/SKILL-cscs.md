---
name: genslm-esm-cscs
description: Run GenSLM-ESMC (ramanathanlab genslm_esm) protein/codon language-model inference on CSCS Alps/Daint (NVIDIA GH200, aarch64, CUDA). CSCS/CUDA port of the Aurora `genslm-esm` skill, built as a native .venv from the ramanathanlab/genslm-esm clone. Covers the env and the HF-gated checkpoint caveat. Use for GenslmEsmcModel embeddings/MLM on an Alps GH200 node.
---

# GenSLM-ESMC on CSCS Alps/Daint (GH200, CUDA)

CSCS port of the Aurora `genslm-esm` skill. Unlike the Aurora install (two venvs on the XPU
`frameworks` module), this is a **native CUDA `.venv`** built from the GitHub clone. The
package is light (`torch`, `einops`, `transformers<4.51`) and installs cleanly on aarch64.

## Install locations (this machine)
- Repo clone (private, GitHub): `/ritom/scratch/cscs/avasan/skill-venvs/software/genslm-esm`
  (`git@github.com:ramanathanlab/genslm-esm.git` — needs GitHub SSH access)
- venv (python 3.12, **torch 2.15.0+cu132**, CUDA; `genslm_esm` editable-installed):
  `/ritom/scratch/cscs/avasan/skill-venvs/genslm-esm-venv`

```bash
PY=/ritom/scratch/cscs/avasan/skill-venvs/genslm-esm-venv/bin/python
```
No `module load`, no `ZE_AFFINITY_MASK`, no proxy. CUDA auto on the GH200. No separate `esm`
package is needed — ESMC is implemented on top of `transformers` inside `genslm_esm`
(`genslm_esm.modeling.ESMC` / `GenslmEsmcModel`).

### How it was installed (reproduce after re-clone)
```bash
cd /ritom/scratch/cscs/avasan/skill-venvs
uv venv --python 3.12 genslm-esm-venv
UV_LINK_MODE=copy uv pip install --python genslm-esm-venv/bin/python -e software/genslm-esm
```
(The default-PyPI aarch64 torch wheel is already CUDA 13.2 — no special index needed, unlike
chai/protenix.)

## Checkpoints are HF-gated — provide an HF token

Models load from the Hugging Face Hub via `AutoModel.from_pretrained(..., trust_remote_code=True)`
under the **`genslm-test`** org, which is **gated/private** (HTTP 401 without auth). Set an HF
token with access before loading any checkpoint:
```bash
export HF_TOKEN=hf_...            # account with genslm-test access
# or: hf auth login
```
Checkpoints (README): `genslm-test/genslm-esmc-300M-contrastive-2026-10-step-200000` (30
layers, light — the smoke-test choice), `...-600M-...`, and `...-6B-...` (~25 GB fp32 / ~13 GB
bf16). Compute nodes have internet, so weights download at runtime once the token is set.

## Smoke test on a compute node (SLURM, once HF_TOKEN is set)

```bash
#!/bin/bash
#SBATCH --job-name=genslm
#SBATCH --partition=debug
#SBATCH --account=g245
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=/ritom/scratch/cscs/avasan/skill-verify/logs/genslm-%j.out   # shared scratch
# Token is read from ~/.cache/huggingface/token (hf auth login). DO NOT set HF_HOME
# (that moves token discovery too) -- use HF_HUB_CACHE for the weights cache instead.
export HF_HUB_CACHE=/ritom/scratch/cscs/avasan/skill-venvs/.hf_cache/hub
/ritom/scratch/cscs/avasan/skill-venvs/genslm-esm-venv/bin/python - <<'PY'
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from genslm_esm.data import FastaDataset, GenslmEsmcDataCollator
name = "genslm-test/genslm-esmc-300M-contrastive-2026-10-step-200000"
model = AutoModel.from_pretrained(name, trust_remote_code=True).eval().to("cuda").to(torch.bfloat16)
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
# GenslmEsmcModel.forward takes codon_input_ids / aminoacid_input_ids (NOT input_ids);
# build inputs with FastaDataset (codon = DNA, len multiple of 3) + the collator.
seqs = ["ATGAAGGTACTACCACAAGAAACTGTAAGAATTGGA"]
ds   = FastaDataset(sequences=seqs, return_codon=True, return_aminoacid=True)
coll = GenslmEsmcDataCollator(return_codon=True, return_aminoacid=True, tokenizer=tok)
for batch in DataLoader(ds, batch_size=1, collate_fn=coll, num_workers=0):
    with torch.no_grad():
        out = model(**batch.to("cuda"), decode_aminoacid_head=True)
    print("genslm-esm OK | cuda bf16 | loss", float(out.loss),
          "| hidden", tuple(out.last_hidden_state.shape))
print("GENSLM_SMOKE_OK")
PY
```

## Verified on Alps (2026-11-04, GH200, debug, job 5258619)
Full forward pass with real weights: `genslm-esmc-300M-contrastive-2026-10-step-200000` on
`cuda:0` in bf16 via `FastaDataset` + `GenslmEsmcDataCollator`, **loss 0.711, last_hidden_state
(4, 19, 960)** (960 = 300M hidden dim). PASS. (Token via `hf auth login`; weights auto-download
to `HF_HUB_CACHE`.) `trust_remote_code=True` executes the repo's modeling code — expected.

### HF token gotcha (cost me a run)
`huggingface_hub` reads the token from `$HF_HOME/token` (default `~/.cache/huggingface/token`).
Setting `HF_HOME=/ritom/scratch/...` in a job **moves token discovery there** → the job runs
unauthenticated → 401 `RepositoryNotFoundError`. Redirect only the cache with **`HF_HUB_CACHE`**
and leave `HF_HOME` at its default so the stored token is found.

## Differences from the Aurora skill
- Native CUDA `.venv` (torch 2.15.0+cu132) instead of the Aurora `venvs/aurora` on the XPU
  frameworks module; no `venvs/ci` split, no `module load`, no `ZE_AFFINITY_MASK`.
- Same package API (`GenslmEsmcModel`, `GenslmEsmcDataCollator`) and same HF checkpoints.
