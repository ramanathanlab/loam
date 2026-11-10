# PyRosetta on Jupiter: availability gate

PyRosetta scoring is CPU work. The public installer package is not the licensed PyRosetta runtime, and an ARM64-compatible licensed distribution has not been established for this port. Do not claim this skill is runnable merely because the wrapper or an installer exists.

After the required distribution is supplied, check its version/import and initialization using its dedicated `software/pyrosetta/.venv` through the authorized HPC path. Until then, report the software-availability blocker; do not fabricate Rosetta scores or silently substitute another metric.

The bundled Aurora notes describe total/per-term and per-residue scoring, FastRelax and InterfaceAnalyzerMover API semantics. They are historical: constructor signatures and installed options must be verified against the actual licensed build. `pyrosetta.init(extra_options=...)` preserves the intended init option interface. Initialize once per process and preserve output structures and scoring configuration.

Rosetta energy units are REU; they are not kcal/mol MM/GBSA estimates or measured affinity. For an interface report, retain the explicit partner-chain grouping, score function, packing/relaxation settings and inspected structure. Do not run the historical PBS submission helper or copy Aurora absolute paths on Jupiter. No GPU needs to be requested solely for CPU scoring.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For campaign execution, stage with `write_files` and use independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task; `run_code` is reserved for staging fallback, harvesting and diagnosis. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
