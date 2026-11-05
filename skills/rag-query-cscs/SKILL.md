---
name: rag-query-cscs
description: CSCS Alps/Daint port of rag-query — retrieve from a FAISS store built by rag-build-index-cscs and answer with an OpenAI-compatible LLM, using the prebuilt venv on Alps. The LLM endpoint can be the local Argo tunnel already wired on this machine. Supports free-text answers and structured JSON extraction.
---

# RAG query on CSCS Alps/Daint

CSCS port of `rag-query`. The retrieval + prompt-assembly + chat-completion procedure and
code are **identical to the Aurora skill** — read that `SKILL.md`. Only the environment
and the LLM endpoint differ.

## Environment (this machine)
- venv (same shared store as rag-build-index-cscs): `/ritom/scratch/cscs/avasan/skill-venvs/rag`

```bash
PY=/ritom/scratch/cscs/avasan/skill-venvs/rag/bin/python
```
Use the **same** `all-MiniLM-L6-v2` encoder + `normalize_embeddings=True` as at index time.
Embedding can use the GH200 (`device="cuda"`) or CPU; queries are single-shot so CPU is fine.

## LLM endpoint: use the local Argo tunnel
The upstream skill reads `BINDIFY_LLM_BASE_URL` (an OpenAI-compatible `/v1`). On this
machine an **Argo proxy is already reachable via an SSH tunnel on `localhost:44497`**
(set up earlier in this port; `ssh -N argo-proxy`, forwarding to homes-02:51900). Point the
query at it:

```bash
export BINDIFY_LLM_BASE_URL="http://localhost:44497/v1"
export BINDIFY_LLM_MODEL="argo:gpt-5.6-sol"     # or argo:gpt-6-luna, etc. (Argo model names)
# Argo proxy is no-auth over the tunnel; if your client requires a key, any placeholder works.
```
If the tunnel is down, re-establish it (`ssh -fN argo-proxy`) or set `BINDIFY_LLM_BASE_URL`
to another OpenAI-compatible endpoint. **Do not invent an endpoint** — if none is
configured and the tunnel is down, stop and ask (as upstream instructs).

- No ALCF endpoints, no proxy env vars.
- Run on the login node (tunnel + internet live there). A compute node cannot see the
  login-node `localhost:44497` tunnel, so run rag-query on the login node, or stand up a
  separate forward on the compute node if you must query from inside an allocation.

## Verified on Alps (2026-11-04)
Retrieval half verified end-to-end (embed on GH200 + FAISS search returns correct passage,
see rag-build-index-cscs). The Argo endpoint itself was verified earlier in this session
(luna answered real prompts over `localhost:44497`). The combined retrieve→LLM call uses
the standard upstream code against `BINDIFY_LLM_BASE_URL` above.

## Notes
- Low retrieval scores ⇒ corpus likely lacks the answer; widen corpus or distrust the reply.
- Structured extraction: validate the parsed JSON array before using it downstream.
- Upstream: **rag-build-index-cscs** (store) and **fetch-literature-cscs** (corpus).
