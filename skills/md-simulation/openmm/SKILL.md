---
name: openmm
description: Run raw OpenMM molecular dynamics (custom systems, forces, integrators, minimization or benchmarks from AMBER/CHARMM/GROMACS/PDB inputs) on a GPU, or check which OpenMM platforms and devices are available. Use for hand-written OpenMM simulations; for the build-and-run pipeline use the molecular-simulations skill.
systems: [aurora, jupiter]
---

# OpenMM

## Before you run it: read the reference file for this System

The environment, GPU platform name, device selection and launch command differ per System. Read
`references/<HPC_SYSTEM>.md` (for example `read_skill(name="openmm", file="references/aurora.md")`)
and follow it; it is authoritative over this file for anything System-specific. This skill has
reference files for **aurora** and **jupiter**. It is **unsupported** on any other System: report the
step as blocked rather than adapting another System's instructions.

Software is found only through Site profile variables: the install is under `$LOAM_SOFTWARE_ROOT`.
Never edit or rebuild the install directory; write outputs to the campaign workspace.

## Core API

```python
from openmm.app import *; from openmm import *; from openmm.unit import *
platform = Platform.getPlatformByName(GPU_PLATFORM)   # the System's GPU platform: see the reference file
sim = Simulation(top, system, integrator, platform, {'Precision': 'mixed'})
print(sim.context.getPlatform().getName())             # confirm you are on the GPU platform
```

`Precision` is `single` | `mixed` | `double`; `mixed` is the usual choice. Never select a physical
device by index: restrict the process to one GPU the way the reference file says, and that GPU
appears as device 0.

## Inputs

The standard OpenMM app readers: `AmberPrmtopFile`/`AmberInpcrdFile`, `GromacsTopFile`,
`CharmmPsfFile`, or `PDBFile` with `ForceField('amber14-all.xml', 'amber14/tip3pfx.xml')`. Build the
scientific system from actual campaign coordinates and pass box vectors for periodic systems
(`periodicBoxVectors=inpcrd.boxVectors`). Resolve missing atoms, parameterization, protonation and
restraints deliberately; do not silently alter the experimental system. To parameterize a system
with AmberTools, use the molecular-simulations skill's builders.

## Outputs

Reporters write them: `DCDReporter`, `XTCReporter`, `StateDataReporter`, `CheckpointReporter`, plus
`simulation.saveState()` / `saveCheckpoint()`. Write reporters, checkpoints and the effective settings
to the campaign workspace.

## Must-know

- Minimize before dynamics (`simulation.minimizeEnergy()`). Use the experiment's timestep,
  constraints, thermostat, ensemble and duration; HBonds constraints allow ≤2 fs (≤4 fs with HMR).
- NaN or exploding energies: minimize first, check the timestep and constraints, and check the box
  vectors were passed.
- Inspect actual coordinates and energies for non-finite values and clashes before reporting a
  successful run. Restart only from a saved state whose system settings match, and keep prior outputs.
- The first `step()` call compiles the GPU kernels (a few seconds). Warm up before timing.
  `StateDataReporter` speed reads `0` on the first report; that is normal.
- `python -m openmm.testInstallation` (run from outside any OpenMM source tree) shows which platforms
  load. It is a diagnostic, not evidence of stable production dynamics.
- Don't `pip install` a different numpy into the tool's venv — OpenMM is tied to it and you get
  "compiled using NumPy 1.x" ABI errors.
- A single force-field energy is not an MM/GBSA binding estimate. Compute the requested energy
  method with proper complex/receptor/binder preparation, and report the force field, solvent model,
  frames and units. No dynamics or affinity claim follows from an import or a platform self-test.
