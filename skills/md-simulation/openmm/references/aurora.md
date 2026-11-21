# OpenMM on Aurora (Intel PVC via OpenCL)

Install: `$LOAM_SOFTWARE_ROOT/openmm-sycl` (OpenMM 8.5.0, git 847515e, built with icx/icpx and
`-DOPENMM_BUILD_OPENCL_LIB=ON`). Do not modify or rebuild it. Despite the directory name, the GPU
platform is **`OpenCL`**; the available platforms are `Reference`, `CPU` and `OpenCL`.

## Environment

```bash
source $LOAM_SOFTWARE_ROOT/openmm-sycl/.venv/bin/activate
# or call $LOAM_SOFTWARE_ROOT/openmm-sycl/.venv/bin/python directly
export ZE_AFFINITY_MASK=0      # ONE tile (0-11 in FLAT mode); ALWAYS set this
```

No `module load` or `LD_LIBRARY_PATH` is needed for the Python API. Nothing downloads — it works
offline. Example inputs: `$LOAM_SOFTWARE_ROOT/openmm-sycl/examples/python-examples/`
(`input.prmtop`/`input.inpcrd` is a 2269-atom solvated alanine dipeptide; also `.gro/.top`, `.pdb`,
CHARMM psf).

## Example

```bash
# Installation self-test (~15 s; all 3 platforms should agree within tolerance)
cd ~ && ZE_AFFINITY_MASK=0 \
  $LOAM_SOFTWARE_ROOT/openmm-sycl/.venv/bin/python -m openmm.testInstallation

# Minimal MD runner: scripts/aurora/openmm_xpu_smoke.py <prmtop> <inpcrd> <steps> <out_dir>
export ZE_AFFINITY_MASK=0
E=$LOAM_SOFTWARE_ROOT/openmm-sycl/examples/python-examples
$LOAM_SOFTWARE_ROOT/openmm-sycl/.venv/bin/python scripts/aurora/openmm_xpu_smoke.py \
    $E/input.prmtop $E/input.inpcrd 5000 ./out
```

```python
platform = Platform.getPlatformByName('OpenCL')
sim = Simulation(top, system, integrator, platform, {'Precision': 'mixed'})
print(platform.getPropertyValue(sim.context, 'DeviceName'))   # Intel(R) Data Center GPU Max 1550
```

Valid OpenCL platform properties: `Precision`, `DeviceIndex`, `OpenCLPlatformIndex`, `UseCpuPme`.

## Must-know

- **The platform name is `OpenCL`.** `getPlatformByName('CUDA')` or `'SYCL'` raises an exception.
  Omitting the platform lets OpenMM pick the fastest available, which is OpenCL on a GPU node.
- **Always set `ZE_AFFINITY_MASK` before starting Python.** Otherwise OpenCL sees all 12 tiles and
  device 0 may be one someone else is using. With the mask set, `DeviceIndex` is `0` — don't pass
  `DeviceIndex=N`.
- If `OpenCL` is missing from `testInstallation` you are in the wrong python/venv; as an unverified
  fallback, set `OPENMM_PLUGIN_DIR=<install>/build/install/lib/plugins` and add
  `<install>/build/install/lib` to `LD_LIBRARY_PATH`.
- The `molecular-simulations` venv ships the same OpenMM 8.5 with the same three platforms.
- Under Loam `run_batch`, reference the runner by its worker-visible copy
  (`$HPC_LOAM_HOME/skills/md-simulation/openmm/scripts/aurora/openmm_xpu_smoke.py`) — the Aurora
  worker cannot see a local skill directory.

## Resource index

- `read_skill(name="openmm", file="scripts/aurora/openmm_xpu_smoke.py")` — minimal MD runner (PME,
  HBonds, LangevinMiddle 2 fs, mixed precision on OpenCL); about 416–460 ns/day on the 2269-atom
  example
