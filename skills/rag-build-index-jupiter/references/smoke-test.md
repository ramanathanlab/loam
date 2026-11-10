# Manual Loam-style smoke test: rag-build-index

Run from the workspace using the SDK environment and an already healthy MCP pool:

```bash
software/globus-compute/.venv/bin/python loam/scripts/smoke_jupiter_skills.py \
  --only rag-build-index --output "$PWD/software/verification/<fresh-audit-directory>"
```

The command dispatches a real small workload through MCP in this software's isolated environment, saves process/command logs and a JSON result, and inspects actual outputs. It does not start or stop allocations or modify campaign scientific state. Use one audit workload at a time alongside an active campaign. The selected test case is implemented at `$HPC_LOAM_HOME/ports/jupiter/smoke/case.py`, a checkout file rather than a skill-relative reference.

An import, CLI help, or CUDA matrix multiplication alone is not a scientific-workflow pass. Missing software, inaccessible checkpoints, network failures, timeouts and unsupported APIs remain explicit blockers/failures. Do not fabricate a pass or replace a trained model with random weights. The suite uses clearly labeled small fixtures and reduced inference settings only for installation validation, not production scientific qualification. Retrieval fixtures are synthetic; literature queries fetch real source records.

Results are persisted in `<audit-directory>/summary.json` and `rag-build-index/`, outside campaign evidence ledgers. These manual checks are not integrated into campaign preflight.

## Latest recorded manual validation

Status: **passed**. Workload: Pretrained embeddings, FAISS/Arrow persistence and reload. 

Evidence directory: `/e/project1/deepwing/meigooni1/agent/software/verification/jupiter-skills-smoke-20261110/rag-build-index`. Preserve failed attempts; the root `summary.json` records this run.

This status describes only the tested path and cache/auth state at audit time. Revalidate after environment or checkpoint changes.
