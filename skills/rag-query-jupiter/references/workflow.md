# rag-query on Jupiter

Use `software/rag-query/.venv/bin/python` through MCP for real query embedding and FAISS retrieval. Match the index's encoder, dimension and normalization; reload the saved Arrow records and verify hit-to-document mapping. Answer only from retrieved passages. Argo's `localhost:50856` tunnel is on the login node, so localhost in a compute-node task is not that service. The Loam-style manual smoke test performs retrieval through MCP, then calls the configured Argo chat endpoint on the login node with those returned passages. This split matches the agent/execution architecture; it does not claim compute-node loopback can access Argo. For an all-remote variant an explicitly configured reachable LLM endpoint is required. Do not expose the loopback tunnel or alter service bindings automatically. Retain retrieved records, scores, model identity and the grounded answer. Retrieval alone does not pass the complete query/answer smoke test.

Read [the packaged API recipe](api.md) for concrete inputs and Python calls. Use absolute assigned workspace paths, stream logs, and preserve failures. Current payload/client deadlines are 6000/6300 seconds, but tiny retrieval tests should use a tighter appropriate bound. These are manual installation tests, not new campaign preflight gates.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
