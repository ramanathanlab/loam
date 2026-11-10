# molecular-simulations on Jupiter

Use `software/molecular-simulations/.venv`. `jupiter_env.sh` sets `AMBERHOME` to the separate ARM64 AmberTools installation in `software/ambertools` and adds its binaries to PATH. Builders and AmberTools are separate from a Python-package import. Confirm the actual required tools before claiming a complete simulation path.

## Build and run

The package exposes `molecular_simulations.build.ExplicitSolvent` / `ImplicitSolvent` and `molecular_simulations.simulate.Simulator` / `ImplicitSimulator` / `Minimizer`. Inspect installed signatures and the requested system assumptions before calling them; do not copy Aurora step counts as a production protocol.

```python
from molecular_simulations.build import ExplicitSolvent
from molecular_simulations.simulate import Simulator
# pdb and out_dir must be verified campaign-specific paths.
ExplicitSolvent(path=str(out_dir), pdb=str(pdb), padding=10.0).build()
# Set all scientific step counts explicitly from the agreed protocol.
sim = Simulator(path=out_dir, platform="CUDA", **simulation_settings)
sim.run()
```

`simulation_settings` is the experiment's explicit settings dictionary, not an installed preset supplied by this document. On Jupiter use `platform="CUDA"`, not the historical `OpenCL` overrides. Preserve the assigned CUDA visibility.

Builders produce topology/coordinate files such as `system.prmtop` and `system.inpcrd`. Read `leap.log` (or `build/leap.log`) for AmberTools failures rather than relying only on a Python exception. Validate chain/residue identity, caps, disulfides, ligands, solvent and protonation. Ligand builders require their additional dependencies; do not infer availability from protein-only imports.

Record minimization, equilibration, production and restart settings; inspect actual output files/checkpoints. Separate structural confidence, force-field energy and MM/GBSA binding-energy estimates. A static minimized-frame estimate is not an ensemble free energy or experimental affinity. Preserve failed preparations and partial stages. The original Aurora capped-peptide failures and performance numbers are historical observations, not Jupiter measurements.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For campaign execution, stage with `write_files` and use independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task; `run_code` is reserved for staging fallback, harvesting and diagnosis. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.

## Required CUDA plugin

The base `openmm==8.6.1` wheel imports successfully but does not install a CUDA platform. Install the matching `openmm[cuda12]==8.6.1` extra in this skill's own venv. From the checkout run `bash scripts/install_jupiter_openmm_cuda.sh`; its pinned requirements repair both isolated MD environments. This is an installation step, not a campaign preflight. Validate a real CUDA simulation through the manual MCP smoke workload before claiming GPU readiness. Do not silently fall back to CPU after a missing-CUDA error.

For a short minimization of an existing Amber build, the installed API is:

```python
from molecular_simulations.simulate import Minimizer
Minimizer(topology=str(out_dir / "system.prmtop"),
          coordinates=str(out_dir / "system.inpcrd"),
          out=str(out_dir / "min.pdb"), platform="CUDA").minimize()
```

Pass builder paths as strings; this installed builder does not handle a `Path` object consistently. `Minimizer` takes topology/coordinates/output arguments, not `path=...`.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
