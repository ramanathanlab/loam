---
name: molecular-simulations
description: Build AMBER systems with tleap (explicit OPC water or implicit solvent) and run the full OpenMM MD pipeline (minimize, heat, restrained equilibration, NPT production with automatic restart) on a GPU. Use to prepare a protein PDB for MD, or to launch or restart Simulator runs.
systems: [aurora, jupiter]
---

# molecular-simulations

## Before you run it: read the reference file for this System

The environment, the AmberTools location, the GPU platform name, device selection and launch command
differ per System. Read `references/<HPC_SYSTEM>.md` (for example
`read_skill(name="molecular-simulations", file="references/aurora.md")`) and follow it; it is
authoritative over this file for anything System-specific. This skill has reference files for
**aurora** and **jupiter**. It is **unsupported** on any other System: report the step as blocked
rather than adapting another System's instructions.

Software is found only through Site profile variables: the package is an editable install under
`$LOAM_SOFTWARE_ROOT/molecular-simulations` (`src/molecular_simulations/{build,simulate,analysis}`)
with OpenMM in its own venv; AmberTools (tleap, cpptraj, pdb4amber, antechamber, sqm, parmchk2,
MMPBSA.py) is a separate install found through `$AMBERHOME`. Never edit the install directory; write
outputs to the campaign workspace. Upstream skills live in `<pkg>/skills/` (build-systems,
run-simulations, analyze-*, parsl-hpc); their APIs are accurate but they default to
`platform='CUDA'`, which is not every System's GPU platform.

## Core API

```python
from molecular_simulations.build import ExplicitSolvent      # or ImplicitSolvent(path, pdb, protein=True)
from molecular_simulations.simulate import Simulator          # or ImplicitSimulator, Minimizer

ExplicitSolvent(path=out_dir, pdb=pdb, padding=10.0).build()  # writes out_dir/system.prmtop/.inpcrd
Simulator(path=out_dir, platform=GPU_PLATFORM, **simulation_settings).run()   # platform: see the reference file
```

`simulation_settings` is the experiment's explicit settings, set from the agreed protocol — not a
preset, and not the reduced step counts of a smoke test.

## Arguments

`Simulator`: `heat_steps` (100k), `equil_steps` (1.25M @ 2 fs), `n_equil_cycles` (3), `prod_steps`
(250M), `prod_dt_in_ps` (0.004, HMR 1.5 amu), `temperature` (300), `eq_reporter_frequency` and
`prod_reporter_frequency`, `membrane`, `ff` (`'amber'` | `'charmm'`), `platform` (**always pass the
System's GPU platform**). `run()` skips equilibration if `eq.*` files exist and resumes production
from `prod.rst.chk`, so re-running it restarts.

`ExplicitSolvent`: `path`, `pdb`, `padding` (Å), `amberhome`, `debug=True` (keeps `tleap.in`).
Defaults give OPC water, ~150 mM NaCl. Without `$AMBERHOME` (or `amberhome=...`) the builders raise
`ValueError`.

`Minimizer(topology=..., coordinates=..., out=..., platform=...)` minimizes an existing Amber build;
it takes topology/coordinates/output arguments, not `path=...`.

## Outputs

```
<out_dir>/
  system.prmtop, system.inpcrd        # the built AMBER system
  build/                              # intermediates, leap.log
  eq.{dcd,log,state,chk}              # equilibration
  prod.{dcd,log,state,chk}            # production trajectory + restart checkpoint
```

## Must-know

- **Always pass the platform explicitly.** The `Simulator` default `'CUDA'` raises an exception on a
  System whose GPU platform has another name, as does the upstream `examples/run_omm_locally.py`.
- **When tleap fails, Python only shows `CalledProcessError ... exit status 31`.** Read
  `<out>/leap.log` (or `build/leap.log`) and grep for `FATAL`.
- Validate chain/residue identity, caps, disulfides, ligands, solvent and protonation of the built
  system. Importing the Python package does not verify the builders or AmberTools.
- Ligand builders (`ComplexBuilder`, `LigandBuilder`, `LigandSolutionBuilder`) need rdkit and
  openbabel; do not infer their availability from a protein-only import.
- The progress column in `prod.log` counts from the equilibration end step, so a short test shows
  >100%. Cosmetic.
- Tiny boxes under very short equilibration shrink noticeably under the barostat — fine for a smoke
  test, but use the default step counts (or the experiment's agreed protocol) for real runs.
- Record minimization, equilibration, production and restart settings; inspect the actual output
  files and checkpoints. Preserve failed preparations and partial stages.
- Keep structural confidence, force-field energy and MM/GBSA estimates separate. A static
  minimized-frame energy is not an ensemble free energy or an experimental affinity.
