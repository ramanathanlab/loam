# GenSLM-ESM on Jupiter

The installed `software/genslm-esm/.venv` supplies the upstream `genslm_esm` package. It is not the Aurora `genslm_esm_sae` fork. In particular, the original SAE example and campaign checkpoints do not establish SAE support here.

```bash
"$PY" -c 'import genslm_esm'
```

This only checks the base package import. For a requested embedding/training task, inspect the installed package's API, exact model/tokenizer definitions, checkpoint provenance and access. Codon versus amino-acid tokenization must match the selected checkpoint; do not silently translate or substitute formats. Run the installed CUDA-compatible path and preserve the assigned GPU.

If sparse-autoencoder inference is required, the exact compatible SAE source and trained checkpoints must first be supplied and verified. Stop with that specific blocker rather than fabricating features or treating base-package import as success. Do not apply the Aurora `embed_and_sae.py` script unchanged.

Record input data/tokenization, model/checkpoint IDs, selected layers, tensor shapes, device and outputs; validate actual features before interpretation. Historical MPI/PMIx and Lustre timing observations are retained as context, not a license to change launcher settings or claim Jupiter performance. Use the campaign's authorized environment and timeout policy.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For campaign execution, stage with `write_files` and use independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task; `run_code` is reserved for staging fallback, harvesting and diagnosis. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.

## Current trained-checkpoint blocker

The real MCP test loaded the installed local `GenslmEsmcModel` class and attempted the upstream README's `genslm-test/genslm-esmc-300M-contrastive-2026-10-step-200000`. Hugging Face refused access (`RepositoryNotFoundError`/401); no trained embedding was produced. An inaccessible repository can be private or absent, so do not assume authentication alone resolves it. Obtain an accessible compatible trained checkpoint or an approved local copy, retain model/config/tokenizer provenance, then rerun the native embedding workload. Never replace this with random weights or an unrelated ESM model. The Aurora SAE fork/checkpoints remain separate and unverified.

The two upstream-documented older alternatives, `genslm-test/genslm-esmc-300M-contrastive` and `genslm-test/genslm-esmc-300M-joint`, were also checked through the current Hugging Face client and both returned `RepositoryNotFoundError`/401. A public metadata check confirmed the same responses. An accessible compatible checkpoint is required.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
