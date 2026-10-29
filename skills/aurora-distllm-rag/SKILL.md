---
name: aurora-distllm-rag
description: >-
  Build and query a retrieval corpus on Aurora with distllm / HiPerRAG — chunk,
  embed and retrieve over papers or campaign records. Use for the
  literature-grounding stage of a campaign, where a claim needs a source rather
  than recollection.
---

# distllm / HiPerRAG on Aurora

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_rag-2026-venv.sh
distllm --help
```

`distllm` covers embedding, chunking and retrieval; `parsl` is in the venv for
workflow fan-out across nodes.

Use it when a campaign requires literature grounding: fetch a corpus, chunk and
embed it, then retrieve per question and cite what came back. Retrieval gives
you a passage you can quote and attribute — an assertion from model memory is
not evidence, and a campaign that asks for grounding is asking for the former.

Keep the corpus and the index inside the run's own output directory so they are
hashed with the rest of the results; an index rebuilt elsewhere cannot be tied
to the claims it supported.

Source: `envs/VENVS.txt`. Not individually probed — run `distllm --help` and
confirm the subcommands before composing a pipeline.
