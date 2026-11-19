---
name: openmm
description: Run raw OpenMM 8.5 molecular dynamics on Aurora Intel GPU Max tiles, where the GPU backend is the "OpenCL" platform (no CUDA, no "SYCL" platform). Use for custom OpenMM simulations, forces, integrators or benchmarks from AMBER/CHARMM/GROMACS/PDB inputs on Aurora, or to check which platforms and devices are available.
---

# OpenMM on Aurora (Intel PVC via OpenCL)

Install: `/flare/FRAME-IDP/msinclair/software/openmm-sycl` (OpenMM 8.5.0, git 847515e, built with
icx/icpx and `-DOPENMM_BUILD_OPENCL_LIB=ON`). Do not modify or rebuild it. Despite the directory
name, the GPU platform is **`OpenCL`**; the available platforms are `Reference`, `CPU` and `OpenCL`.

## Environment

```bash
source /flare/FRAME-IDP/msinclair/software/openmm-sycl/.venv/bin/activate
# or call /flare/FRAME-IDP/msinclair/software/openmm-sycl/.venv/bin/python directly
export ZE_AFFINITY_MASK=0      # ONE tile (0-11 in FLAT mode); ALWAYS set this
```

No `module load` or `LD_LIBRARY_PATH` is needed for the Python API. Nothing downloads — it works
offline. `openmmforcefields` is **not** installed (use the `molecular-simulations` skill's builders
for ligand parameterization). Example inputs:
`/flare/FRAME-IDP/msinclair/software/openmm-sycl/examples/python-examples/` (`input.prmtop`/
`input.inpcrd` is a 2269-atom solvated alanine dipeptide; also `.gro/.top`, `.pdb`, CHARMM psf).

## Example

```bash
# Installation self-test (~15 s; all 3 platforms should agree within tolerance)
cd ~ && ZE_AFFINITY_MASK=0 \
  /flare/FRAME-IDP/msinclair/software/openmm-sycl/.venv/bin/python -m openmm.testInstallation

# Minimal MD runner: scripts/openmm_xpu_smoke.py <prmtop> <inpcrd> <steps> <out_dir>
export ZE_AFFINITY_MASK=0
E=/flare/FRAME-IDP/msinclair/software/openmm-sycl/examples/python-examples
/flare/FRAME-IDP/msinclair/software/openmm-sycl/.venv/bin/python scripts/openmm_xpu_smoke.py \
    $E/input.prmtop $E/input.inpcrd 5000 ./out
```

Core API pattern:

```python
from openmm.app import *; from openmm import *; from openmm.unit import *
platform = Platform.getPlatformByName('OpenCL')
sim = Simulation(top, system, integrator, platform, {'Precision': 'mixed'})
print(platform.getPropertyValue(sim.context, 'DeviceName'))   # confirm you are on the GPU
```

## Arguments

Valid OpenCL platform properties: `Precision` (`single` | `mixed` | `double`; `mixed` is the usual
choice), `DeviceIndex`, `OpenCLPlatformIndex`, `UseCpuPme`.

- **Inputs** use the standard OpenMM app readers: `AmberPrmtopFile`/`AmberInpcrdFile`,
  `GromacsTopFile`, `CharmmPsfFile`, or `PDBFile` with
  `ForceField('amber14-all.xml', 'amber14/tip3pfx.xml')`.
- **Outputs** come from reporters: `DCDReporter`, `XTCReporter`, `StateDataReporter`,
  `CheckpointReporter`, plus `simulation.saveState()` / `saveCheckpoint()`.

## Must-know

- **The platform name is `OpenCL`.** `getPlatformByName('CUDA')` or `'SYCL'` raises an exception.
  Omitting the platform lets OpenMM pick the fastest available, which is OpenCL on a GPU node.
- **Always set `ZE_AFFINITY_MASK` before starting Python.** Otherwise OpenCL sees all 12 tiles and
  device 0 may be one someone else is using. With the mask set, `DeviceIndex` is `0` — don't pass
  `DeviceIndex=N`.
- The first `step()` call JIT-compiles the OpenCL kernels (a few seconds). Warm up before timing.
- Run `testInstallation` from outside the source `build/python` directory.
- `StateDataReporter` speed reads `0` on the first report. Normal.
- If `OpenCL` is missing from `testInstallation` you are in the wrong python/venv; as an unverified
  fallback, set `OPENMM_PLUGIN_DIR=<install>/build/install/lib/plugins` and add
  `<install>/build/install/lib` to `LD_LIBRARY_PATH`.
- Don't `pip install` a different numpy into this venv — the build is tied to the venv's numpy and
  you get "compiled using NumPy 1.x" ABI errors.
- NaN or exploding energies: minimize first (`simulation.minimizeEnergy()`), use HBonds constraints
  with ≤2 fs (≤4 fs with HMR), and check the box vectors were passed
  (`periodicBoxVectors=inpcrd.boxVectors`).
- The `molecular-simulations` venv ships the same OpenMM 8.5 with the same three platforms.
