---
name: fetch-literature-cscs
description: CSCS Alps/Daint port of the fetch-literature skill — gather a literature corpus from PubMed E-utilities and Europe PMC as JSONL, using the prebuilt venv on Alps. Pure HTTP, no GPU. Use on an Alps login node (which has outbound internet) to collect papers/abstracts about a target before building a RAG store.
---

# Fetch literature on CSCS Alps/Daint

CSCS port of `fetch-literature`. The procedure, API endpoints, JSONL record shape and the
example code are **identical to the Aurora skill** — read that `SKILL.md` for the full
PubMed esearch/efetch + Europe PMC recipe. Only the environment differs.

## Environment (this machine)
- venv (python 3.12, `requests`): `/ritom/scratch/cscs/avasan/skill-venvs/fetch-literature`

```bash
PY=/ritom/scratch/cscs/avasan/skill-venvs/fetch-literature/bin/python
```

- **Run it on a login node** (e.g. `daint-ln004`). Alps login nodes have direct outbound
  internet — no proxy, no `http_proxy`/`https_proxy` (those were ALCF-only). Compute nodes
  also have internet, but there is no reason to spend an allocation on plain HTTP.
- No GPU, no `module load`.
- Keep the output JSONL on `/ritom/scratch` so the rag-build-index step (and any compute
  job) can read it.

## Verified on Alps (2026-11-04, login node)
Live PubMed `esearch` for "ubiquitin" returned PMIDs (e.g. 42978514, 42978422, 42978067)
through `$PY`. PASS.

## Notes
- Be a polite NCBI citizen: rate-limit (~3/s without a key) and pass `tool=` and
  `email=avasan@anl.gov` on E-utilities calls (as the Aurora skill shows).
- Downstream: feed the JSONL to **rag-build-index-cscs**, then query with **rag-query-cscs**.
