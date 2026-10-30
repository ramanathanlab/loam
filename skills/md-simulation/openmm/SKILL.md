---
name: openmm
description: Run OpenMM 8.5 molecular dynamics on Aurora's Intel Data Center GPU Max (PVC) tiles using the locally built OpenMM, whose GPU backend is the "OpenCL" platform (no CUDA, no platform named "SYCL"). Use when you need to run, benchmark, or debug raw OpenMM simulations (AMBER/CHARMM/GROMACS/PDB inputs, custom forces/integrators) on an Aurora compute node, or need to check which OpenMM platforms and devices are available there.
---

# OpenMM on Aurora (Intel PVC via OpenCL)

Source and build: `/flare/FRAME-IDP/msinclair/software/openmm-sycl` (OpenMM 8.5.0, git 847515e).
It was built with `make_build.sh` (icx/icpx, `-DOPENMM_BUILD_OPENCL_LIB=ON`) and `install_openmm.sh`.
The directory is called "sycl", but the GPU platform is **`OpenCL`**, running on the
Intel OpenCL Graphics runtime. The available platforms are `Reference`, `CPU` and `OpenCL`.
Do not modify or rebuild anything in that directory.

## Environment

```bash
source /flare/FRAME-IDP/msinclair/software/openmm-sycl/.venv/bin/activate
# or call /flare/FRAME-IDP/msinclair/software/openmm-sycl/.venv/bin/python directly
export ZE_AFFINITY_MASK=0      # pin to ONE GPU tile (0..11 in FLAT mode); ALWAYS set this
```

- You do not need `module load` or `LD_LIBRARY_PATH` for the Python API. The venv package
  finds its plugins without them (checked on this node with the default Aurora environment).
- With `ZE_AFFINITY_MASK=N` set, only that tile is visible, so OpenCL `DeviceIndex` is `0`.
  Don't pass `DeviceIndex=N`.
- There is no checkpoint/data download. It works offline. The example inputs are in
  `/flare/FRAME-IDP/msinclair/software/openmm-sycl/examples/python-examples/`
  (`input.prmtop`/`input.inpcrd`: 2269-atom solvated alanine dipeptide; also `.gro/.top`, `.pdb`, CHARMM psf).
- The molecular-simulations venv (`/flare/FRAME-IDP/msinclair/software/molecular-simulations/.venv`)
  ships the same OpenMM 8.5 with the same 3 platforms.
- `openmmforcefields` is NOT installed in this venv.

## Verified commands (2026-10-29, debug node)

Installation self-test (about 15 s; all 3 platforms agree within tolerance):
```bash
cd ~ && ZE_AFFINITY_MASK=0 /flare/FRAME-IDP/msinclair/software/openmm-sycl/.venv/bin/python -m openmm.testInstallation
```

Minimal MD (PME, HBonds, LangevinMiddle 2 fs, mixed precision on OpenCL). The script is
`scripts/openmm_xpu_smoke.py` in this skill folder:
```bash
export ZE_AFFINITY_MASK=0
E=/flare/FRAME-IDP/msinclair/software/openmm-sycl/examples/python-examples
/flare/FRAME-IDP/msinclair/software/openmm-sycl/.venv/bin/python scripts/openmm_xpu_smoke.py \
    $E/input.prmtop $E/input.inpcrd 5000 ./out
```
Result: `Platform: OpenCL`, `DeviceName = Intel(R) Data Center GPU Max 1550`. 5000 steps took 2.1 s,
which is about 416 to 460 ns/day for 2269 atoms. Total wall time was about 10 s including startup
and kernel JIT. It also passed on `ZE_AFFINITY_MASK=1`.

Core API pattern:
```python
from openmm.app import *; from openmm import *; from openmm.unit import *
platform = Platform.getPlatformByName('OpenCL')
sim = Simulation(top, system, integrator, platform, {'Precision': 'mixed'})
print(platform.getPropertyValue(sim.context, 'DeviceName'))   # confirm you are on the GPU
```
Valid OpenCL properties are `Precision` (`single`|`mixed`|`double`), `DeviceIndex`,
`OpenCLPlatformIndex` and `UseCpuPme`. `mixed` is the usual choice.

## Inputs and outputs
- Inputs are the standard OpenMM app readers: `AmberPrmtopFile`/`AmberInpcrdFile`, `GromacsTopFile`,
  `CharmmPsfFile`, and `PDBFile` with `ForceField('amber14-all.xml','amber14/tip3pfx.xml')`.
- Outputs come from reporters: `DCDReporter`, `XTCReporter`, `StateDataReporter`, `CheckpointReporter`,
  and `simulation.saveState()`/`saveCheckpoint()`.

## Aurora/XPU gotchas
- **Platform name is `OpenCL`.** `getPlatformByName('CUDA')` or `'SYCL'` raises an exception.
  If you omit the platform, OpenMM picks the fastest available one, which is OpenCL on a GPU node.
- **Always set `ZE_AFFINITY_MASK`** before starting Python. If you don't, OpenCL sees all 12 tiles
  and device 0 may be a tile someone else is using.
- The first `step()` call JIT-compiles the OpenCL kernels, which takes a few seconds. Do a warm-up
  before timing anything.
- Run `testInstallation` from outside the source `build/python` directory (see `install_openmm.sh`).
- `StateDataReporter` speed reads `0` on the first report. That is normal.

## Troubleshooting
- `No Platform with that name`: wrong platform name. Use `Reference`, `CPU` or `OpenCL`.
- `OpenCL` missing from `testInstallation`: you are in the wrong python or venv, or the plugin dir
  isn't found. Try `export OPENMM_PLUGIN_DIR=/flare/FRAME-IDP/msinclair/software/openmm-sycl/build/install/lib/plugins`
  and `LD_LIBRARY_PATH=.../build/install/lib:$LD_LIBRARY_PATH` (unverified fallback; it wasn't needed here).
- NumPy ABI errors ("compiled using NumPy 1.x"): the build is tied to the venv's numpy. Don't
  `pip install` a different numpy into this venv.
- NaN or exploding energies: minimize first (`simulation.minimizeEnergy()`), use HBonds constraints with
  ≤2 fs (or ≤4 fs with HMR), and check the box vectors were passed (`periodicBoxVectors=inpcrd.boxVectors`).
