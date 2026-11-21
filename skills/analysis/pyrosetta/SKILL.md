---
name: pyrosetta
description: Score, relax and analyze protein structures with PyRosetta (ref2015) on CPUs — total and per-term energies, per-residue energies, FastRelax refinement, and binder/target interface metrics (dG, dSASA, shape complementarity, packstat, unsat H-bonds), plus a parallel batch scorer for ranking 2-chain complexes. Use to score, relax or rank PDBs by interface quality.
systems: [aurora]
---

# PyRosetta

## Before you run it: read the reference file for this System

The install, the interpreter and the batch launcher differ per System. Read
`references/<HPC_SYSTEM>.md` (for example `read_skill(name="pyrosetta", file="references/aurora.md")`)
and follow it; it is authoritative over this file for anything System-specific, and it defines the
`$PY` used below. This skill has a reference file for **aurora** only. It is **unsupported** on any
other System: report the step as blocked rather than adapting Aurora's instructions.

On Jupiter in particular the installed package is `pyrosetta-installer`, not a licensed ARM64
PyRosetta runtime, so there is no scoring engine there. Report that software blocker rather than
fabricating Rosetta scores or silently substituting another metric.

Software is found only through Site profile variables: the install is under `$LOAM_SOFTWARE_ROOT`.
Never edit the install directory; write outputs to the campaign workspace.

PyRosetta scoring is **CPU work** — do not request a GPU for it.

## Core API

Every process must `init` first, and **must pass flags as `extra_options=`**:

```python
import pyrosetta
pyrosetta.init(extra_options="-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false")
pose  = pyrosetta.pose_from_pdb("in.pdb")
sfxn  = pyrosetta.create_score_function("ref2015")
total = sfxn(pose)                                  # REU

# Weighted per-term breakdown
from pyrosetta.rosetta.core.scoring import name_from_score_type
emap  = pose.energies().total_energies()
terms = {name_from_score_type(st): emap[st] * sfxn.get_weight(st)
         for st in sfxn.get_nonzero_weighted_scoretypes()}

# Per-residue energies (the decompose option is REQUIRED for the sum to equal the total)
emo = sfxn.energy_method_options()
emo.hbond_options().decompose_bb_hb_into_pair_energies(True)
sfxn.set_energy_method_options(emo); sfxn(pose)
for i in range(1, pose.total_residue() + 1):
    tot      = pose.energies().residue_total_energy(i)     # weighted
    per_term = pose.energies().residue_total_energies(i)   # EMapVector, UNWEIGHTED
    chain, resnum = pose.pdb_info().chain(i), pose.pdb_info().number(i)

# FastRelax
from pyrosetta.rosetta.protocols.relax import FastRelax
fr = FastRelax(sfxn, 1)                      # 1 repeat (default protocol is 5, ~5x slower)
fr.constrain_relax_to_start_coords(True)     # optional: keep the backbone near the input
fr.apply(pose); pose.dump_pdb("relaxed.pdb")
rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(ref_pose, pose)

# Interface analysis (multi-chain)
from pyrosetta.rosetta.core.pose import DockingPartners
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
partners = DockingPartners.docking_partners_from_string("A_B")    # or "AB_C"
# positional: (partners, tracer, sfxn, compute_packstat, pack_input, pack_separated)
iam = InterfaceAnalyzerMover(partners, True, sfxn, True, False, False)
iam.set_compute_interface_sc(True); iam.set_calc_dSASA(True)
iam.apply(pose)
d  = iam.get_all_data()         # vector1s: [1]=total, [2]=side1, [3]=side2
dG, dSASA = iam.get_interface_dG(), iam.get_interface_delta_sasa()
sc, packstat, nres = d.sc_value, d.packstat, d.interface_nres[1]
hb, unsat = d.interface_hbonds, d.delta_unsat_hbonds
```

Constructor signatures and available options vary between builds; check them against the installed
build rather than assuming.

## Example

The bundled runner is `scripts/<HPC_SYSTEM>/rosetta_tool.py`, one structure per call, JSON to
stdout. The reference file says how to address it on a worker; `$TOOL` below is that path.

```bash
$PY $TOOL score     in.pdb [--json out.json] [--sfxn ref2015]
$PY $TOOL perres    in.pdb --csv perres.csv
$PY $TOOL relax     in.pdb --out relaxed.pdb [--rounds 1] [--max-iter N] [--constrain]
$PY $TOOL interface in.pdb --partners A_B [--pack-input] [--pack-separated] [--json out.json]
```

The reference file names the batch scorer for ranking many 2-chain complexes, and its launcher.

## Arguments and outputs

`rosetta_tool.py` prints JSON. The `interface` subcommand defaults to **no repacking**, which is
deterministic; add `--pack-input --pack-separated` to match the repo protocol.

Batch scorer output is `{"results": [...], "errors": [...]}` sorted by `interface_dG` ascending
(most favorable first). Fields: `binder_id`, `pdb`, `leaderboard_score`, `total_score_ref2015`,
`interface_dG`, `dSASA_interface`, `dhSASA_interface`, `shape_complementarity`, `packstat`,
`delta_unsat_hbonds`, `interface_hbonds`, `nres_interface`,
`complex_normalized_dG` (= dG / nres_interface), `seconds`. The CSV has the same columns.

## Must-know

- **Rosetta Energy Units are not kcal/mol MM/GBSA estimates and not measured affinity.** For an
  interface report, record the partner-chain grouping, the score function, the packing/relaxation
  settings and the structure inspected.
- **`pyrosetta.init("opts")` and `init(extra_options="opts")` give different results.** The
  positional `options` default is `-ex1 -ex2aro`; passing flags positionally silently drops extra
  rotamers, and anything that packs (FastRelax, IAM with `pack_input`/`pack_separated`) then shifts
  by tens of REU. Always use `extra_options=`.
- **Call `pyrosetta.init` once per process**, not per structure — init dominates small jobs. The
  batch scorer does it in a `ProcessPoolExecutor` initializer.
- **Per-residue sums need `decompose_bb_hb_into_pair_energies(True)`.** Without it, backbone
  H-bonds are not assigned to residues and the per-residue sum does not match the pose total.
- `residue_total_energies(i)` terms are **unweighted** — multiply by `sfxn.get_weight(st)`.
- **There is no `InterfaceAnalyzerMover("A_B")` string constructor** in the installed build
  (TypeError). Wrap the string with `DockingPartners.docking_partners_from_string`.
- **Repacked interface dG is stochastic** (~1–3 REU spread, and it depends on RNG state from prior
  calls in the same worker). Don't over-interpret small rank differences, and compare only numbers
  from the same protocol and init flags. No-pack IAM is deterministic; `-constant_seed -jran N`
  makes a fresh process reproducible.
- **Unrelaxed predicted or crystal models score huge positive totals** from clashes. Relax, or at
  least repack, before trusting absolute numbers.
- The license banner prints even with `-mute all`; filter with `grep -v -e '│' -e '┌' -e '└'`.
- Memory is ~0.86 GB RSS per worker on a ~250-residue complex, so keep `--workers` ≤ the cores you
  own.
- **PyRosetta is free for academic/non-profit use; commercial use requires a license**
  (license@uw.edu).
