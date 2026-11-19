---
name: molecular-simulations
description: Build AMBER systems with tleap (explicit OPC water or implicit solvent) and run the full OpenMM MD pipeline (minimize, heat, restrained equilibration, NPT production with automatic restart) on Aurora Intel GPU tiles. Use to prepare a protein PDB for MD or to launch/restart Simulator runs on Aurora.
---

# molecular-simulations on Aurora (Intel XPU)

Install (editable): `/flare/FRAME-IDP/msinclair/software/molecular-simulations`
(`src/molecular_simulations/{build,simulate,analysis}`), using OpenMM 8.5 from its own venv.
Do not edit the install directory. Upstream skills live in `<pkg>/skills/` (build-systems,
run-simulations, analyze-*, parsl-hpc); their APIs are accurate but they default to
`platform='CUDA'`, **which does not exist on Aurora**.

## Environment

```bash
source /flare/FRAME-IDP/msinclair/software/molecular-simulations/.venv/bin/activate
export AMBERHOME=/lus/flare/projects/FRAME-IDP/msinclair/envs/ambertools   # tleap, cpptraj, pdb4amber, antechamber, sqm, parmchk2, MMPBSA.py
export ZE_AFFINITY_MASK=0        # ONE tile; ALWAYS set it before python starts
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy   # only to fetch structures
```

AmberTools is not in the venv and not on `PATH`; without `AMBERHOME` the builders raise
`ValueError` (or pass `amberhome=...` to the builder). Installed in the venv: openmm 8.5
(`Reference`/`CPU`/`OpenCL`), parsl, MDAnalysis, mdtraj, pdbfixer. **rdkit and openbabel are not
installed**, so `ComplexBuilder`, `LigandBuilder` and `LigandSolutionBuilder` are unavailable.

## Example

```bash
# Minimal runner: scripts/molsim_xpu_smoke.py <input.pdb> <out_dir>
cd <a directory you own>         # everything is written here
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy
curl -s -o 1L2Y_raw.pdb https://files.rcsb.org/download/1L2Y.pdb                    # Trp-cage, 20 res
awk '/^MODEL/{m++} m<=1 && /^ATOM/ && $NF!="H"' 1L2Y_raw.pdb > 1L2Y.pdb; echo END >> 1L2Y.pdb
export ZE_AFFINITY_MASK=1 AMBERHOME=/lus/flare/projects/FRAME-IDP/msinclair/envs/ambertools
/flare/FRAME-IDP/msinclair/software/molecular-simulations/.venv/bin/python \
    <this skill>/scripts/molsim_xpu_smoke.py 1L2Y.pdb ./trpcage
```

Core API:

```python
from molecular_simulations.build import ExplicitSolvent      # or ImplicitSolvent(path, pdb, protein=True)
from molecular_simulations.simulate import Simulator          # or ImplicitSimulator, Minimizer

ExplicitSolvent(path=out_dir, pdb=pdb, padding=10.0).build()  # writes out_dir/system.prmtop/.inpcrd
Simulator(path=out_dir, platform='OpenCL').run()              # platform is MANDATORY on Aurora
```

## Arguments

`Simulator`: `heat_steps` (100k), `equil_steps` (1.25M @ 2 fs), `n_equil_cycles` (3), `prod_steps`
(250M), `prod_dt_in_ps` (0.004, HMR 1.5 amu), `temperature` (300), `eq_reporter_frequency` and
`prod_reporter_frequency`, `membrane`, `ff` (`'amber'` | `'charmm'`), `platform`
(**pass `'OpenCL'`**). `run()` skips equilibration if `eq.*` files exist and resumes production
from `prod.rst.chk`, so re-running it restarts.

`ExplicitSolvent`: `path`, `pdb`, `padding` (Å), `amberhome`, `debug=True` (keeps `tleap.in`).
Defaults give OPC water, ~150 mM NaCl.

## Outputs

```
<out_dir>/
  system.prmtop, system.inpcrd        # the built AMBER system
  build/                              # intermediates, leap.log
  eq.{dcd,log,state,chk}              # equilibration
  prod.{dcd,log,state,chk}            # production trajectory + restart checkpoint
```

## Must-know

- **Always pass `platform='OpenCL'`.** The `Simulator` default `'CUDA'` raises an exception on
  Aurora, as does the upstream `examples/run_omm_locally.py`.
- **Device selection is by `ZE_AFFINITY_MASK` only.** In the OpenCL branch `Simulator` ignores
  `device_ids` and sets only `{'Precision': 'mixed'}`, so OpenMM takes the first visible device.
  One mask per process (e.g. Parsl workers each with their own).
- **When tleap fails, Python only shows `CalledProcessError ... exit status 31`.** Read
  `<out>/leap.log` (or `build/leap.log`) and grep for `FATAL`.
- **Capped test peptides in the repo fail to build** (`tests/data/pdb/alanine_dipeptide.pdb`,
  `two_chain_saltbridge.pdb`): the builder's pdb4amber/cpptraj prep mangles ACE/NME caps
  (`FATAL: Atom .R<ACE 1>.A<N 7> does not have a type`). Use a real protein like 1L2Y to smoke-test.
- The first build on a cold node can take ~3 min (loading AmberTools off Lustre); later builds are
  under a second.
- The progress column in `prod.log` counts from the equilibration end step, so a short test shows
  >100%. Cosmetic.
- Tiny boxes under very short equilibration shrink noticeably under the barostat — fine for a smoke
  test, but use the default step counts for real runs.
- Constant-pH MD, EVB, coarse-grained (CALVADOS), MM/PBSA, free energy and the `analysis/` module
  are in the package but untested on Aurora.
