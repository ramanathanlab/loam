---
name: pyrosetta
description: Score, relax and analyze protein structures with PyRosetta (ref2015) on Aurora CPUs — total and per-term energies, per-residue energies, FastRelax refinement, and binder/target interface metrics (dG, dSASA, shape complementarity, packstat, unsat H-bonds), plus a parallel batch scorer for ranking 2-chain complexes. Use to score, relax or rank PDBs by interface quality.
---

# PyRosetta on Aurora (CPU only)

Install: `/flare/FRAME-IDP/msinclair/software/pyrosetta_energy` (do not edit; write outputs
elsewhere). CPU-only — never needs a GPU, so do not set `ZE_AFFINITY_MASK`.

## Environment

```bash
PY=/flare/FRAME-IDP/msinclair/software/pyrosetta_energy/.venv/bin/python   # Python 3.12, uv venv
# or: source /flare/FRAME-IDP/msinclair/software/pyrosetta_energy/.venv/bin/activate
```

`pyrosetta-2026.34+release.de92a3c0de`, plus numpy. **No pandas**, so `pyrosetta.distributed` and
`get_energies_table`-style helpers are unavailable — use the pure-pose APIs below. The venv's base
interpreter is an absolute `/opt/aurora/...2025.3.1` path, so it works without `module load`.

## Example

```bash
# Minimal runner: scripts/rosetta_tool.py — one structure, JSON to stdout
$PY scripts/rosetta_tool.py score     in.pdb [--json out.json] [--sfxn ref2015]
$PY scripts/rosetta_tool.py perres    in.pdb --csv perres.csv
$PY scripts/rosetta_tool.py relax     in.pdb --out relaxed.pdb [--rounds 1] [--max-iter N] [--constrain]
$PY scripts/rosetta_tool.py interface in.pdb --partners A_B [--pack-input] [--pack-separated] [--json out.json]
```

Core API — every process must `init` first, and **pass flags as `extra_options=`**:

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

Batch scorer (the repo pipeline), for ranking many 2-chain complexes:

```bash
$PY /flare/FRAME-IDP/msinclair/software/pyrosetta_energy/scripts/compute_rosetta_energy.py \
  --top100-dir DIR --out out.json [--out-csv out.csv] [--workers 32]
```

`DIR/leaderboard.json` is a list of `{binder_id, pdb, score, ...}`; each PDB is looked up as
`DIR/<basename(pdb)>` and must have exactly 2 chains, **A = target, B = binder** (`A_B` is
hardcoded). `scripts/make_leaderboard.py DIR` generates that file for any folder of 2-chain PDBs.
`scripts/submit_energy.pbs` is a PBS template (`qsub -v TOP_DIR=...,WORKERS=96`).

## Arguments and outputs

`rosetta_tool.py` prints JSON. The `interface` subcommand defaults to **no repacking**, which is
deterministic; add `--pack-input --pack-separated` to match the repo protocol.

Batch scorer output is `{"results": [...], "errors": [...]}` sorted by `interface_dG` ascending
(most favorable first). Fields: `binder_id`, `pdb`, `leaderboard_score`, `total_score_ref2015`,
`interface_dG`, `dSASA_interface`, `dhSASA_interface`, `shape_complementarity`, `packstat`,
`delta_unsat_hbonds`, `interface_hbonds`, `nres_interface`,
`complex_normalized_dG` (= dG / nres_interface), `seconds`. The CSV has the same columns.

## Must-know

- **`pyrosetta.init("opts")` and `init(extra_options="opts")` give different results.** The
  positional `options` default is `-ex1 -ex2aro`; passing flags positionally silently drops extra
  rotamers, and anything that packs (FastRelax, IAM with `pack_input`/`pack_separated`) then shifts
  by tens of REU. Always use `extra_options=`.
- **Call `pyrosetta.init` once per process**, not per structure — init dominates small jobs. The
  batch scorer does it in a `ProcessPoolExecutor` initializer.
- **Per-residue sums need `decompose_bb_hb_into_pair_energies(True)`.** Without it, backbone
  H-bonds are not assigned to residues and the per-residue sum does not match the pose total.
- `residue_total_energies(i)` terms are **unweighted** — multiply by `sfxn.get_weight(st)`.
- **There is no `InterfaceAnalyzerMover("A_B")` string constructor** in this build (TypeError).
  Wrap the string with `DockingPartners.docking_partners_from_string`.
- **Repacked interface dG is stochastic** (~1-3 REU spread, and it depends on RNG state from prior
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
- Durable test inputs: `$MPNN/inputs/PDB_monomers/pdbs/6MRR.pdb`,
  `.../PDB_complexes/pdbs/3HTN.pdb` (3 chains), and
  `/flare/FRAME-IDP/msinclair/software/ProtonPottsMPNN/inference/examples/pdl1_seed_binder.pdb` (A/B).
