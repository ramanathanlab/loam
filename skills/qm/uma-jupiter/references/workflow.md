# UMA/fairchem on Jupiter

Use the native CUDA environment in `software/fairchem/.venv`; source `jupiter_env.sh` for `FAIRCHEM_CACHE_DIR`. Confirm exact checkpoint access and permissions before inference. The Aurora Nitro/XPU fast path, IPEX assumptions, local checkpoint locations and timings are not Jupiter recipes.

```bash
"$PY" -c 'import fairchem.core'
```

An import does not verify model evaluation. Before building a calculator, read the installed fairchem API and model documentation at the installed version. Select the correct `task_name` for the chemistry and use the native CUDA calculator. Different task heads, charge/spin settings and unit conventions are not interchangeable; retain the experiment's requested model and numerical settings.

Start from chemically valid structures with recorded atomic identity, coordinates, cell/PBC where relevant, charge and spin as required by the task. Inspect actual finite energies/forces and their units before using them in optimization or MD. Validate the required force path and chemistry applicability; do not replace a requested method with a convenient preset merely to make execution succeed.

For simulations retain input/checkpoint identity, precision, device, settings, trajectories and failure evidence. Model potential energies are not protein-binding free energies or automatically comparable across different systems. Historical API examples are packaged as notes; verify their compatibility against the native installed source before adapting them.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For campaign execution, stage with `write_files` and use independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task; `run_code` is reserved for staging fallback, harvesting and diagnosis. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.

The manual installation test uses `pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda", inference_settings="batch")` and `FAIRChemCalculator(predictor, task_name="omol")` for a neutral singlet water fixture (`atoms.info` charge 0, spin 1). It requires actual checkpoint access and finite energy/forces. Batch mode avoids assuming the optional Nitro fast path is available; this is an explicit smoke-test setting, not evidence of production accuracy.

## Current blocker: checkpoint access

The actual MCP smoke attempted `facebook/UMA`'s
`checkpoints/uma-s-1p2p1.pt` and received `GatedRepoError: 401`. The Python
installation imports, but energy/force inference has **not** passed. This is
checkpoint authentication/authorization, not evidence that reinstalling PyTorch
will help. Do not claim UMA ready or substitute an unrelated model.

The user must obtain access to the intended model on Hugging Face, then log in
interactively on the login node using the same shared cache as the worker:

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
umask 077
"$LOAM_SOFTWARE_ROOT/fairchem/.venv/bin/hf" auth login
```

Keep tokens private; never put them in skill files, campaign prompts, command
arguments, logs or Git. Rerun the manual UMA MCP workload after access is granted.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
