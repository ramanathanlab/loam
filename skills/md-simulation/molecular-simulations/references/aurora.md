# molecular-simulations on Aurora (Intel XPU)

Install (editable): `$LOAM_SOFTWARE_ROOT/molecular-simulations`, using OpenMM 8.5 from its own venv.
On Aurora the GPU platform is **`OpenCL`**; `'CUDA'` does not exist here.

## Environment

```bash
source $LOAM_SOFTWARE_ROOT/molecular-simulations/.venv/bin/activate
# AmberTools (tleap, cpptraj, pdb4amber, antechamber, sqm, parmchk2, MMPBSA.py) is a separate env
# beside the software root, not inside it.
export AMBERHOME="${AMBERHOME:-$(dirname "$LOAM_SOFTWARE_ROOT")/envs/ambertools}"
export ZE_AFFINITY_MASK=0        # ONE tile; ALWAYS set it before python starts
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy   # only to fetch structures
```

AmberTools is not in the venv and not on `PATH`; without `AMBERHOME` the builders raise
`ValueError` (or pass `amberhome=...` to the builder). Installed in the venv: openmm 8.5
(`Reference`/`CPU`/`OpenCL`), parsl, MDAnalysis, mdtraj, pdbfixer. **rdkit and openbabel are not
installed**, so `ComplexBuilder`, `LigandBuilder` and `LigandSolutionBuilder` are unavailable.
If `$AMBERHOME/bin/tleap` does not exist, report the step as blocked rather than guessing a path.

## Example

```bash
# Minimal runner: scripts/aurora/molsim_xpu_smoke.py <input.pdb> <out_dir>
cd <a directory you own>         # everything is written here
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy
curl -s -o 1L2Y_raw.pdb https://files.rcsb.org/download/1L2Y.pdb                    # Trp-cage, 20 res
awk '/^MODEL/{m++} m<=1 && /^ATOM/ && $NF!="H"' 1L2Y_raw.pdb > 1L2Y.pdb; echo END >> 1L2Y.pdb
export ZE_AFFINITY_MASK=0 AMBERHOME="${AMBERHOME:-$(dirname "$LOAM_SOFTWARE_ROOT")/envs/ambertools}"
$LOAM_SOFTWARE_ROOT/molecular-simulations/.venv/bin/python \
    $HPC_LOAM_HOME/skills/md-simulation/molecular-simulations/scripts/aurora/molsim_xpu_smoke.py 1L2Y.pdb ./trpcage
```

The runner builds with `ExplicitSolvent` (8 Å padding, about 5700 atoms) and runs `Simulator` with
tiny step counts on `platform='OpenCL'`; it took about 44 s, with production at about 385 ns/day
(4 fs HMR).

```python
Simulator(path=out_dir, platform='OpenCL', **simulation_settings).run()   # platform is MANDATORY on Aurora
```

## Must-know

- **Always pass `platform='OpenCL'`.** The `Simulator` default `'CUDA'` raises
  `There is no registered Platform called "CUDA"` on Aurora. `Minimizer` defaults to `'OpenCL'`
  (not exercised here).
- **Device selection is by `ZE_AFFINITY_MASK` only.** In the OpenCL branch `Simulator` ignores
  `device_ids` and sets only `{'Precision': 'mixed'}`, so OpenMM takes the first visible device.
  One mask per process (e.g. Parsl workers each with their own).
- **Capped test peptides in the repo fail to build** (`tests/data/pdb/alanine_dipeptide.pdb`,
  `two_chain_saltbridge.pdb`): the builder's pdb4amber/cpptraj prep mangles ACE/NME caps
  (`FATAL: Atom .R<ACE 1>.A<N 7> does not have a type`). Use a real protein like 1L2Y to smoke-test.
- The first build on a cold node can take ~3 min (loading AmberTools off Lustre); later builds are
  under a second.
- Run hangs or is slow: check `ZE_AFFINITY_MASK` is set, and check `xpu-smi`/`sycl-ls` for tile
  contention.
- Constant-pH MD, EVB, coarse-grained (CALVADOS), MM/PBSA, free energy and the `analysis/` module
  are in the package but untested on Aurora.
- Under Loam `run_batch`, reference the runner by its worker-visible copy (as in the example
  above) — the Aurora worker cannot see a local skill directory.

## Resource index

- `read_skill(name="molecular-simulations", file="scripts/aurora/molsim_xpu_smoke.py")` — minimal
  build-and-simulate runner
