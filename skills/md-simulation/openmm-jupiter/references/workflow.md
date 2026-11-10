# OpenMM molecular dynamics on Jupiter

Use `software/openmm/.venv` and the CUDA platform. The Aurora installation uses OpenCL; do not copy that backend selection, plugin paths or device masks.

```python
from openmm import Platform
platform = Platform.getPlatformByName("CUDA")
properties = {"Precision": "mixed"}
# Supply validated topology/system/integrator and coordinates to Simulation.
# Preserve the one-GPU task's CUDA_VISIBLE_DEVICES; do not select a physical GPU.
```

Run `"$PY" -m openmm.testInstallation` through MCP when diagnosing platform availability; this is a manual check, not evidence of stable production dynamics. Build the scientific system from actual coordinates using the appropriate OpenMM readers (for example AmberPrmtopFile/AmberInpcrdFile), and preserve box vectors for periodic systems. Resolve missing atoms, parameterization, protonation and restraints deliberately; do not silently alter the experimental system.

Minimize before dynamics. Use the experiment's timestep, constraints, thermostat, ensemble and duration. Write reporters, checkpoints and effective settings to the campaign workspace. Inspect actual coordinates and energies for non-finite values/clashes before reporting a successful run. Restart only from compatible saved state/system settings, retaining prior outputs.

An energy from a single force-field evaluation is not an MM/GBSA binding estimate. Compute the requested energy method with proper complex/receptor/binder preparation and report force field, solvent model, frames and units. No dynamics or affinity claim follows from an import or platform self-test.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For current extended tests, scientific tasks may use `run_code(timeout=6000)` or independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.

## Required CUDA plugin

The base `openmm==8.6.1` wheel imports successfully but does not install a CUDA platform. Install the matching `openmm[cuda12]==8.6.1` extra in this skill's own venv. From the checkout run `bash scripts/install_jupiter_openmm_cuda.sh`; its pinned requirements repair both isolated MD environments. This is an installation step, not a campaign preflight. Validate a real CUDA simulation through the manual MCP smoke workload before claiming GPU readiness. Do not silently fall back to CPU after a missing-CUDA error.
