# Reactra on Jupiter: deployment and availability

The exact source repository/revision used by the original Aurora skill is not currently established as accessible in this port. A generic wheel or `reactra --help` alone does not supply the required MECH schema/package, native OpenMM/UMA/PLUMED providers, model weights or execution profile. Treat these as prerequisites rather than claiming a working scientific runtime.

Reactra is an enzyme/reaction-sampling workflow, not a generic protein-binder affinity tool. Its intended control interface takes a run YAML and MECH mechanism, with deployment separated into an execution profile. Review scientific intent, exact installed CLI/schema and provider readiness before invoking a run. Do not recreate the whole scientific engine in an agent script.

The bundled original notes and their example/recovery guides are self-contained historical documentation. They describe the original execution contract and provenance semantics, but their Aurora deployment paths, shell wrapper, native Python, proxy, XPU backend and scheduler settings must not be executed unchanged on Jupiter.

If the exact source and dependency prerequisites are supplied later, use the dedicated `software/reactra/.venv`, a reviewed native CUDA/CPU deployment profile as appropriate, and the campaign's existing MCP execution tools. Preserve MECH/YAML hashes, inputs, checkpoint/provenance outputs and partial failures. Resume only compatible work; never publish or send results externally as a side effect of validation. Missing prerequisites are not scientific refutation.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For current extended tests, scientific tasks may use `run_code(timeout=6000)` or independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.
