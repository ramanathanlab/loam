---
name: openmm-cscs
description: Run OpenMM 8.6 molecular dynamics on CSCS Alps/Daint (NVIDIA GH200, aarch64) through the OpenCL platform. CSCS port of the Aurora `openmm` skill. Use to run/benchmark/debug raw OpenMM simulations (AMBER/CHARMM/GROMACS/PDB inputs, custom forces/integrators) on an Alps GH200 compute node, or to check which OpenMM platforms are available there.
---

# OpenMM on CSCS Alps/Daint (GH200 via OpenCL)

CSCS port of the Aurora `openmm` skill. Installed from the **PyPI `openmm` wheel**
(`uv pip install openmm`), version **8.6.1**, python 3.12.

## IMPORTANT: GPU platform is `OpenCL`, not `CUDA`

The aarch64 PyPI OpenMM wheel ships **`Reference`, `CPU`, `OpenCL`** — there is **no
`CUDA` platform** in it (a CUDA-platform build would need a conda-forge/source build with
the CUDA toolkit; not done here). On the GH200, OpenCL runs on the NVIDIA OpenCL runtime
and uses the GPU, so it is the GPU path. `getPlatformByName('CUDA')` raises. This mirrors
the Aurora situation (also `OpenCL`), just on NVIDIA instead of Intel hardware.

```bash
PY=/ritom/scratch/cscs/avasan/skill-venvs/openmm/bin/python
```

No `module load`, no `LD_LIBRARY_PATH`, no `ZE_AFFINITY_MASK`. On a whole-node Alps job all
4 GH200s are visible; OpenCL device 0 is used. To pin a GPU, set `CUDA_VISIBLE_DEVICES`
before python (NVIDIA's OpenCL honors it), or pass `DeviceIndex`.

## Running on a compute node (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=openmm
#SBATCH --partition=debug
#SBATCH --account=g245
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out        # on /ritom/scratch (shared), never /tmp (node-local)
PY=/ritom/scratch/cscs/avasan/skill-venvs/openmm/bin/python
$PY -m openmm.testInstallation
$PY your_md.py
```

Core API pattern (OpenCL):
```python
from openmm.app import *; from openmm import *; from openmm.unit import *
platform = Platform.getPlatformByName('OpenCL')
sim = Simulation(top, system, integrator, platform, {'Precision': 'mixed'})
print(platform.getPropertyValue(sim.context, 'DeviceName'))   # -> NVIDIA GH200 120GB
```
Valid OpenCL properties: `Precision` (`single`|`mixed`|`double`), `DeviceIndex`,
`OpenCLPlatformIndex`, `UseCpuPme`. `mixed` is the usual choice.

## Verified on Alps (2026-11-04, GH200, debug, job 5257565)
- `openmm.testInstallation`: all 3 platforms (Reference/CPU/OpenCL) computed forces and
  agreed within tolerance (max median diff ~7e-6).
- OpenCL MD smoke (500-particle LJ, LangevinMiddle, 2 fs): ran on
  **`DeviceName = NVIDIA GH200 120GB`**, 2000 steps, stable potential energy. PASS.

## Inputs / outputs / gotchas
- Standard app readers (`AmberPrmtopFile`/`AmberInpcrdFile`, `GromacsTopFile`,
  `CharmmPsfFile`, `PDBFile` + `ForceField(...)`) and reporters (`DCDReporter`,
  `StateDataReporter`, `CheckpointReporter`, `saveState`/`saveCheckpoint`) — same as upstream.
- `openmmforcefields` is **not** installed; add it to this venv if you need SMIRNOFF/GAFF
  templating (`uv pip install --python $PY openmmforcefields`).
- First `step()` JIT-compiles OpenCL kernels (a few seconds) — warm up before timing.
- NaN/exploding energies: minimize first, HBonds + ≤2 fs (or ≤4 fs with HMR), pass box vectors.

## Differences from the Aurora skill
- No Intel/XPU anything: drop `ZE_AFFINITY_MASK`, `OPENMM_PLUGIN_DIR`, the sycl paths.
- Platform name is still `OpenCL`, device is the GH200. Precision/property API identical.
- Version is 8.6.1 (PyPI) vs the Aurora 8.5 source build; APIs used here are unchanged.
