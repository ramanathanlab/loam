---
name: uma
description: Run Meta FAIR's UMA universal machine-learning interatomic potential (fairchem-core) on a GPU for near-DFT energies, forces, geometry optimizations and MD of molecules, materials, catalysts, MOFs or molecular crystals through the ASE FAIRChemCalculator. Use for fast near-DFT energies/forces or relaxations.
systems: [aurora, jupiter]
---

# UMA (fairchem)

## Before you run it: read the reference file for this System

The install directory, device name, device selection, checkpoint cache and launch command differ per
System. Read `references/<HPC_SYSTEM>.md` (for example
`read_skill(name="uma", file="references/aurora.md")`) and follow it; it is authoritative over this
file for anything System-specific. This skill has reference files for **aurora** and **jupiter**. It
is **unsupported** on any other System: report the step as blocked rather than adapting another
System's instructions.

Software is found only through Site profile variables: the install is under `$LOAM_SOFTWARE_ROOT`
and the UMA checkpoints in the cache the reference file names. Never edit the install directory;
write outputs to the campaign workspace.

## Inputs

ASE `Atoms` built from chemically valid structures, with atomic identity, coordinates, and the cell
and PBC where relevant. `task_name` picks the UMA head:

| `task_name` | for | needs |
|---|---|---|
| `omol` | molecules (ωB97M-V) | `atoms.info["charge"]` and `atoms.info["spin"]` (spin multiplicity) |
| `omat` | inorganic bulk (PBE/PBE+U) | cell + PBC |
| `oc20` | catalyst surfaces + adsorbates | cell + PBC |
| `odac` | MOFs | cell + PBC |
| `omc` | molecular crystals | cell + PBC |

Models: `uma-s-1p2p1` (recommended), `uma-s-1p2`, `uma-s-1p1`, `uma-m-1p1`, and the
`uma-s-1p2p1-ft-d4rev` / `uma-s-1p2p1-ft-r2scan` fine-tunes. Use the model and task the experiment
asks for; different heads, charge/spin settings and unit conventions are not interchangeable.

## Method

```python
from ase.build import molecule
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2p1", inference_settings="batch", device=DEVICE)   # DEVICE: see the reference file
calc = FAIRChemCalculator(predictor, task_name="omol")

atoms = molecule("CH3CH2OH")
atoms.info.update({"charge": 0, "spin": 1})                     # set these explicitly
atoms.calc = calc
print(atoms.get_potential_energy(), atoms.get_forces())         # eV, eV/Å
LBFGS(atoms, trajectory="opt.traj").run(fmax=0.02, steps=200)
```

`inference_settings="batch"` is the general path on every System: it serves many different
molecules from one predictor and does not depend on an optional fast path. Any faster path is
System-specific and is described in the reference file.

## Outputs

Standard ASE results: `atoms.get_potential_energy()` in **eV**, `atoms.get_forces()` in **eV/Å**,
and an optimizer trajectory file (`opt.traj`) when one is requested. The relaxed geometry is in
`atoms` after `.run()`. Record the model, task, charge/spin, inference settings and device with the
results.

## Must-know

- Without `charge`/`spin` in `atoms.info`, `omol` silently uses (0, 1). Set them explicitly.
- Check that energies and forces are finite, and their units, before using them in an optimization
  or MD. Validate a new inference setting's forces against `batch` (or CPU) before trusting it.
- `UnknownCheckpointError`, an HF 401 or a gated-repository error means the checkpoint cache is
  wrong or not authorized: `facebook/UMA` is gated. Don't substitute an unrelated model.
- An import does not verify model evaluation, and a readable skill does not establish that the
  software is installed or a run will succeed.
- UMA potential energies are not protein-binding free energies, and are not comparable across
  different systems or task heads.
