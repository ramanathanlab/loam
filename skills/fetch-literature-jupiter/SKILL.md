---
name: fetch-literature-jupiter
description: Use fetch-literature on Julich Jupiter ARM64 GH200 nodes with its dedicated software/fetch-literature/.venv environment.
---

# fetch-literature on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/fetch-literature"
PY="$REPO/.venv/bin/python"
"$PY" -c "import requests"
```

Use the PubMed/Europe PMC API recipes in the original skill. Internet access is required; preserve identifiers, source URLs, and retrieval dates.

## Packaged references

- [Jupiter execution and readiness](references/workflow.md): `skill_view(name="fetch-literature-jupiter", file_path="references/workflow.md")`.
- [Concrete API recipe](references/api.md): `skill_view(name="fetch-literature-jupiter", file_path="references/api.md")`.

All reference paths are inside this skill. Do not request `../` sibling paths. Use MCP for scientific execution, preserve assigned workspace paths and report real outputs. These retrieval/RAG recipes do not require a GPU for their small CPU tests.

## Manual smoke workload

[Loam-style smoke test](references/smoke-test.md): `skill_view(name="fetch-literature-jupiter", file_path="references/smoke-test.md")`. Run it manually after installation or changes; do not count import/CLI checks as a workflow pass.

[Exact smoke recipe](references/smoke-workload.py): `skill_view(name="fetch-literature-jupiter", file_path="references/smoke-workload.py")`. This is a self-contained snapshot of this tool's manual recipe, with public-input provenance and output assertions. Run it through MCP with this tool's own Python, `--out <fresh-workdir> --software <software-root>`; the suite additionally checks the login-side Argo answer for rag-query.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
