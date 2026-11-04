---
name: rag-build-index-cscs
description: CSCS Alps/Daint port of rag-build-index — embed a JSONL corpus with all-MiniLM-L6-v2 and persist a FAISS inner-product index + on-disk HuggingFace dataset, using the prebuilt venv on Alps. Embedding can use a GH200 GPU (device="cuda") or CPU. Use before semantically searching or RAG-querying a corpus.
---

# RAG build-index on CSCS Alps/Daint

CSCS port of `rag-build-index`. The procedure and code (embed → `IndexFlatIP` → persist
index + dataset, keeping row order aligned) are **identical to the Aurora skill** — read
that `SKILL.md`. Only the environment differs.

## Environment (this machine)
- venv (python 3.12; faiss-cpu 1.15.1, datasets, sentence-transformers 6.2.0, torch
  2.15.0+cu132): `/ritom/scratch/cscs/avasan/skill-venvs/rag` (shared with rag-query-cscs)

```bash
PY=/ritom/scratch/cscs/avasan/skill-venvs/rag/bin/python
```

## GPU is optional but available
MiniLM is tiny, so CPU is fine (as upstream notes). On Alps you can also embed on a GH200:
```python
import torch
from sentence_transformers import SentenceTransformer
dev = "cuda" if torch.cuda.is_available() else "cpu"      # GH200 when in an allocation
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
```
- On the **login node** the 4 GH200s are visible, so `device="cuda"` works there for small
  corpora. For large corpora, run in an sbatch `debug`/`low` allocation and write the store
  to `/ritom/scratch` (shared with compute nodes).
- `faiss-cpu` is used (the index is CPU-side; that is normal and fast for flat IP).
- First run downloads the MiniLM encoder from the HF Hub (public model, **no HF token
  needed**). Login + compute nodes both have internet.

## Verified on Alps (2026-11-04, login node, GH200)
Built a 3-doc index embedding on **`cuda:0`**, `index.ntotal == 3`, query returned the
correct top hit (cosine 0.652). PASS.

## Notes
- Keep vector order == record order (retrieval maps FAISS row → dataset row).
- Output `<out>/faiss.index` + `<out>/dataset/` on `/ritom/scratch`.
- Upstream: **fetch-literature-cscs**. Downstream: **rag-query-cscs**.
