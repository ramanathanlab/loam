---
name: molecular-simulations
description: Build AMBER systems (tleap, explicit OPC water or implicit solvent) and run the full OpenMM MD pipeline (minimize, heat, restrained equilibration, NPT production with automatic restart) with the molecular-simulations package on Aurora Intel GPU tiles. Use when preparing a protein PDB for MD, launching or restarting Simulator/ImplicitSimulator runs on Aurora XPUs, or adapting the package's CUDA-oriented skills/examples to Aurora (platform='OpenCL', ZE_AFFINITY_MASK, AMBERHOME).
---

# molecular-simulations on Aurora

Package (editable install): `/flare/FRAME-IDP/msinclair/software/molecular-simulations`
(`src/molecular_simulations/{build,simulate,analysis}`). It uses OpenMM 8.5 from its own venv.
On Aurora the GPU platform is `OpenCL`.
The upstream skills live in `<pkg>/skills/` (build-systems, run-simulations, analyze-*, parsl-hpc).
Their APIs are accurate, but they default to `platform='CUDA'`, which **does not exist on Aurora**.

## Environment

```bash
source /flare/FRAME-IDP/msinclair/software/molecular-simulations/.venv/bin/activate
export AMBERHOME=/lus/flare/projects/FRAME-IDP/msinclair/envs/ambertools   # tleap, cpptraj, pdb4amber, antechamber, sqm, parmchk2, MMPBSA.py
export ZE_AFFINITY_MASK=0        # pin to ONE tile; ALWAYS set before python starts
```
- AmberTools is not in the venv and not on PATH. Without `AMBERHOME`, the builders raise
  `ValueError`. You can also pass `amberhome=...` to the builder.
- Installed in the venv: openmm 8.5 (`Reference`/`CPU`/`OpenCL`), parsl, MDAnalysis, mdtraj, pdbfixer.
- **Not installed:** rdkit and openbabel. That means the `[ligand]` extra is missing, so
  `ComplexBuilder`, `LigandBuilder` and `LigandSolutionBuilder` are unavailable or untested.
- You only need internet to fetch structures:
  `export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128`.

## Verified minimal workflow (2026-10-29, debug node, tile 1)

The script is `scripts/molsim_xpu_smoke.py` in this skill folder. It runs
ExplicitSolvent build, then Simulator with tiny step counts on `platform='OpenCL'`.
```bash
cd <a directory you own>     # everything below is written here; no shared dir to reuse
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
curl -s -o 1L2Y_raw.pdb https://files.rcsb.org/download/1L2Y.pdb          # Trp-cage, 20 res
awk '/^MODEL/{m++} m<=1 && /^ATOM/ && $NF!="H"' 1L2Y_raw.pdb > 1L2Y.pdb; echo END >> 1L2Y.pdb   # model 1, drop H
export ZE_AFFINITY_MASK=1 AMBERHOME=/lus/flare/projects/FRAME-IDP/msinclair/envs/ambertools
/flare/FRAME-IDP/msinclair/software/molecular-simulations/.venv/bin/python \
    <skill>/scripts/molsim_xpu_smoke.py 1L2Y.pdb ./trpcage
```
Result: PASS. The build took 0.4 s and produced 5715 atoms (OPC water, about 150 mM NaCl, 8 Å padding).
Simulate (heat 1000, equil 1000, 1 cycle, prod 2000 steps) took 39 s, and the whole run took 44 s.
Production speed was about 385 ns/day (4 fs HMR). Outputs in `trpcage/`: `system.prmtop/.inpcrd`,
`build/` (intermediates, leap.log), `eq.{dcd,log,state,chk}`, `prod.{dcd,log,state,chk}`.

Core API:
```python
from molecular_simulations.build import ExplicitSolvent      # or ImplicitSolvent(path, pdb, protein=True)
from molecular_simulations.simulate import Simulator         # or ImplicitSimulator, Minimizer
ExplicitSolvent(path=out_dir, pdb=pdb, padding=10.0).build()  # writes out_dir/system.prmtop/.inpcrd
Simulator(path=out_dir, equil_steps=..., prod_steps=..., platform='OpenCL').run()
```
Main `Simulator` args: `heat_steps` (default 100k), `equil_steps` (1.25M @2 fs), `n_equil_cycles` (3),
`prod_steps` (250M), `prod_dt_in_ps` (0.004, HMR 1.5 amu), `temperature` (300),
`eq_reporter_frequency` and `prod_reporter_frequency`, `membrane`, `ff` (`'amber'`|`'charmm'`).
`run()` skips equilibration if `eq.*` files exist and resumes production from `prod.rst.chk`.
To restart, run it again.
The `Minimizer` class defaults to `platform='OpenCL'` (not exercised here).

## Aurora/XPU gotchas
- **Always pass `platform='OpenCL'`.** The `Simulator` default `'CUDA'` raises an exception on Aurora.
  Same for the upstream `examples/run_omm_locally.py`, which uses the default.
- **Device selection is by `ZE_AFFINITY_MASK` only.** In the OpenCL branch, `Simulator` ignores
  `device_ids` and sets only `{'Precision': 'mixed'}`, so OpenMM uses the first visible device.
  Set `ZE_AFFINITY_MASK` to one tile per process, e.g. Parsl workers each with their own mask.
- The progress column in `prod.log` counts from the equilibration end step, so a short test shows
  125% to 200%. This is cosmetic.
- Capped test peptides in the repo (`tests/data/pdb/alanine_dipeptide.pdb`, `two_chain_saltbridge.pdb`)
  FAIL in `ExplicitSolvent.build()`. The tleap error is `FATAL: Atom .R<ACE 1>.A<N 7> does not have a type`,
  because the builder's pdb4amber/cpptraj prep mangles the ACE/NME caps. A capped dipeptide
  isn't a good smoke test. Use a real protein like 1L2Y.
- When tleap fails, Python only shows `CalledProcessError ... exit status 31`. Read `<out>/leap.log`
  (or `build/leap.log`) and grep for `FATAL`. `debug=True` keeps `tleap.in`.
- The first build on a cold node took about 3 min, because loading AmberTools from Lustre is slow.
  Later builds took less than 1 s.
- Tiny boxes with very short equilibration shrink a lot under the barostat (62.8 to 53.7 nm³ here).
  That is fine for a smoke test. Use the default step counts for real runs.

## Troubleshooting
- `ValueError` about AMBERHOME: export `AMBERHOME` as shown above.
- `Exception: There is no registered Platform called "CUDA"`: pass `platform='OpenCL'`.
- `ImportError` on ligand builders: rdkit and openbabel aren't installed (unverified path).
- Run hangs or is slow: check you set `ZE_AFFINITY_MASK`, and check `xpu-smi`/`sycl-ls` for tile contention.
