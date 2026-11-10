# fetch-literature on Jupiter

Use `software/fetch-literature/.venv/bin/python` and `requests` on compute workers through MCP. Source `jupiter_env.sh` to inherit the authenticated login-node HTTP/HTTPS proxy; do not print proxy URLs or credentials. No GPU is needed. Retrieve real PubMed IDs, then actual abstracts with efetch; title-only records do not exercise the corpus workflow. Europe PMC can provide complementary metadata/full text when available. Preserve URLs, IDs, retrieval dates and input/output hashes. Rate-limit NCBI calls; supply an appropriate tool identifier and contact only if configured rather than inventing an email. Output JSONL records with `text`, `pmid`, `title`, and `source` as described in the bundled API recipe. Raise on HTTP errors and validate returned content; a successful requests import is not a retrieval pass.

Read [the packaged API recipe](api.md) for concrete inputs and Python calls. Use absolute assigned workspace paths, stream logs, and preserve failures. Current payload/client deadlines are 6000/6300 seconds, but tiny retrieval tests should use a tighter appropriate bound. These are manual installation tests, not new campaign preflight gates.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
