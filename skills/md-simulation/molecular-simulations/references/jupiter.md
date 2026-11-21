# molecular-simulations on Jupiter (ARM64 GH200, CUDA)

Use the tool's own venv and `platform="CUDA"`, not the Aurora `OpenCL` overrides.

## Environment

The Site profile's `worker_init` runs on every worker before your code. It sets `AMBERHOME` to the
separate ARM64 AmberTools 26 install (default `$LOAM_SOFTWARE_ROOT/ambertools`) and puts its
binaries on `PATH`. The redacted example is `examples/site-profiles/jupiter.yaml`.

```bash
REPO="$LOAM_SOFTWARE_ROOT/molecular-simulations"
PY="$REPO/.venv/bin/python"
"$PY" -c "import molecular_simulations"    # an import alone does not verify the builders
```

Confirm the AmberTools binaries the build needs are present before claiming a complete simulation
path.

## Build and run

```python
from molecular_simulations.build import ExplicitSolvent
from molecular_simulations.simulate import Simulator, Minimizer
# pdb and out_dir must be verified campaign-specific paths.
ExplicitSolvent(path=str(out_dir), pdb=str(pdb), padding=10.0).build()
Simulator(path=out_dir, platform="CUDA", **simulation_settings).run()

# A short minimization of an existing Amber build:
Minimizer(topology=str(out_dir / "system.prmtop"),
          coordinates=str(out_dir / "system.inpcrd"),
          out=str(out_dir / "min.pdb"), platform="CUDA").minimize()
```

Pass builder paths as strings; this installed builder does not handle a `Path` object consistently.
Inspect the installed signatures before calling them, and do not copy Aurora step counts as a
production protocol. The Aurora capped-peptide failures and performance numbers are historical
observations, not Jupiter measurements.

## Required CUDA plugin

The base `openmm==8.6.1` wheel imports successfully but does not install a CUDA platform. The
matching `openmm[cuda12]==8.6.1` extra must be installed in this skill's own venv (the existing
installations were repaired with it). If CUDA is missing, report the failed prerequisite so the
operator can repair the environment; this is an installation step, not a campaign preflight.
Validate a real CUDA simulation through the smoke workload below before claiming GPU readiness, and
do not silently fall back to CPU after a missing-CUDA error.

## Execution contract on Jupiter

- Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir, with
  absolute campaign paths. Do not submit scheduler jobs, use SSH, replace the scientific software,
  or apply Aurora XPU shims.
  Do not use Aurora modules, `ZE_AFFINITY_MASK`, XPU launchers, ALCF proxies or PBS submissions.
- Jupiter has four GH200 GPUs per node. Keep the `CUDA_VISIBLE_DEVICES` assigned by the launcher;
  inside a one-GPU task use `cuda:0`.
- Stage the script and inputs with `write_files` (only under the campaign's `HPC_BASE_DIR`;
  read-only inspection roots do not grant write access to installed software), then run the science
  through `run_batch(gpus=1, timeout_each=6000)`, one run per row, with a subprocess deadline no
  larger than 6000 seconds (or the campaign YAML's timeouts; allocation walltime still limits
  completion). Activate this tool's own venv inside each row (and before any upstream shell wrapper)
  so the Globus worker's Python is not inherited. Poll `batch_status` until `unfinished == 0`, then
  inspect the actual output files before reporting success.
- `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis only.
- Never pad a batch with duplicate jobs. Keep each row within the remaining Slurm walltime; queue
  waits are separate, so do not cancel queued work just because it waits.
- Stream logs to workspace files and preserve failed attempts.

## Smoke workload

`scripts/jupiter/smoke-workload.py` fetches Trp-cage (PDB 1L2Y), builds it with `ExplicitSolvent`
(8 Å padding) through AmberTools, minimizes it with `Minimizer` on the CUDA platform, and asserts the
topology and finite minimized coordinates. Stage it into a fresh run directory and run it with this
tool's Python through `run_batch`:

```bash
"$PY" smoke-workload.py --out <fresh-absolute-directory> --software "$LOAM_SOFTWARE_ROOT"
```

Then inspect `result.json` and `system/min.pdb`. It verifies installation plumbing only; it does not
count as campaign evidence.

Latest recorded manual validation (on the `moeen` branch's Jupiter port, not a fresh run of this
checkout): **passed** — AmberTools explicit-solvent build and CUDA minimization. Evidence directory:
`$LOAM_SOFTWARE_ROOT/verification/jupiter-skills-repairs-20261110/molecular-simulations`. This
describes only the tested path at audit time (the evidence root's `summary.json` records the run); revalidate after
environment changes. Readable documentation does not establish that the software is installed or
that a run will succeed.

## Resource index

- `read_skill(name="molecular-simulations", file="SKILL.md")` — the System-neutral method
- `read_skill(name="molecular-simulations", file="references/jupiter.md")` — this file
- `read_skill(name="molecular-simulations", file="scripts/jupiter/smoke-workload.py")` — the smoke workload
