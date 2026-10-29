---
name: pyrosetta
description: Rosetta energy analysis of protein structures with PyRosetta 2026.34 (ref2015) on ALCF Aurora CPUs - total score and per-term energies, per-residue energies, FastRelax refinement, and binder/target interface metrics (InterfaceAnalyzerMover dG, dSASA, shape complementarity, packstat, unsat H-bonds), plus the pyrosetta_energy repo's parallel batch scorer for ranking leaderboards of 2-chain complexes. Use when you need to score, relax, or compare Rosetta energies or binding-interface quality of PDBs (e.g. ranking CoFold/AF-style binder designs), or to run/extend the repo's top-N energy pipeline under PBS.
---

# PyRosetta (pyrosetta_energy) on Aurora

CPU-only. Never needs GPUs. All results below were verified on an Aurora debug compute node (2026-10-29)
unless marked UNVERIFIED.

## Environment

```bash
PY=/flare/FRAME-IDP/msinclair/software/pyrosetta_energy/.venv/bin/python   # Python 3.12.12, uv venv
# or: source /flare/FRAME-IDP/msinclair/software/pyrosetta_energy/.venv/bin/activate
```
- Package: `pyrosetta-2026.34+release.de92a3c0de` (PyRosetta4.Release.python312.ubuntu, 2026-08-19). Only other package: numpy 2.x (**no pandas** -> `pyrosetta.distributed` / `get_energies_table`-style pandas helpers not available; use the pure-pose APIs below).
- The venv's base interpreter is `/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python3.12` (absolute path in pyvenv.cfg). It works **without** `module load` (verified with a different frameworks module loaded). If that /opt image path ever disappears, the venv breaks.
- Do not modify the repo dir; write outputs elsewhere.

## Cost / timing (measured, 1 core each)
| step | wall |
|---|---|
| `import pyrosetta` + `init` (first, cold Lustre) | ~11-20 s |
| same, warm | ~3 s |
| score + per-residue, 68-res monomer (6MRR) | <1 s after init |
| FastRelax 1 round, 68 res | 3.8-5.6 s |
| FastRelax 1 round + coord constraints, ~200-res 2-chain complex (pdl1_seed_binder) | 25-33 s |
| InterfaceAnalyzerMover (pack_input+pack_separated) per complex, 2-chain ~150-250 res | 4-10 s (repo top100 run: 8-58 s, mean 17.8 s) |

Init dominates small jobs: in a pipeline, call `pyrosetta.init` **once per process** (repo does it in a ProcessPool initializer).

## Verified snippets

Every process must `init` first. Recommended flags (same as repo) - pass them as **`extra_options=`**:
```python
import pyrosetta
pyrosetta.init(extra_options="-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false")
# NOT pyrosetta.init("...") : the positional arg REPLACES the defaults "-ex1 -ex2aro" (extra rotamers)
pose = pyrosetta.pose_from_pdb("in.pdb")
sfxn = pyrosetta.create_score_function("ref2015")
total = sfxn(pose)                                    # REU
```
The license banner is printed even with `-mute all` (stdout); filter with `grep -v -e '│' -e '┌' -e '└'`.

### Weighted per-term breakdown
```python
from pyrosetta.rosetta.core.scoring import name_from_score_type
emap = pose.energies().total_energies()
terms = {name_from_score_type(st): emap[st]*sfxn.get_weight(st) for st in sfxn.get_nonzero_weighted_scoretypes()}
```

### Per-residue energies (sum must equal total)
```python
emo = sfxn.energy_method_options()
emo.hbond_options().decompose_bb_hb_into_pair_energies(True)   # REQUIRED, see gotcha
sfxn.set_energy_method_options(emo)
sfxn(pose)
for i in range(1, pose.total_residue()+1):
    tot = pose.energies().residue_total_energy(i)              # weighted
    per_term = pose.energies().residue_total_energies(i)       # EMapVector, UNWEIGHTED; index with ScoreType
    chain, resnum = pose.pdb_info().chain(i), pose.pdb_info().number(i)
```
Gotcha (verified): without the decompose option, backbone H-bonds (hbond_sr_bb/lr_bb) are not assigned
to residues; on 6MRR the per-residue sum was -83.6 vs pose total -129.5. With it: -129.50 == -129.50.

### FastRelax (short)
```python
from pyrosetta.rosetta.protocols.relax import FastRelax
fr = FastRelax(sfxn, 1)                     # 1 repeat (default protocol = 5; ~5x slower)
fr.constrain_relax_to_start_coords(True)    # optional: keep backbone near input
# fr.max_iter(200)                          # optional minimizer cap
fr.apply(pose); pose.dump_pdb("relaxed.pdb")
rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(ref_pose, pose)
```
6MRR: -129.5 -> -204.6 REU, CA-RMSD 0.71 A. pdl1_seed_binder (constrained): +254.6 -> -693.7, CA-RMSD 1.09 A.
Results vary run to run (Monte Carlo packing); a positional-init run of 6MRR gave -211.0.

### Interface analysis (multi-chain)
```python
from pyrosetta.rosetta.core.pose import DockingPartners
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
partners = DockingPartners.docking_partners_from_string("A_B")   # or "AB_C" etc.
# positional: (partners, tracer, sfxn, compute_packstat, pack_input, pack_separated)
iam = InterfaceAnalyzerMover(partners, True, sfxn, True, False, False)
iam.set_compute_interface_sc(True); iam.set_calc_dSASA(True)
iam.apply(pose)
d = iam.get_all_data()          # vector1s: [1]=total, [2]=side1, [3]=side2
dG, dSASA = iam.get_interface_dG(), iam.get_interface_delta_sasa()   # == d.dG[1], d.dSASA[1]
sc, packstat, nres = d.sc_value, d.packstat, d.interface_nres[1]
hb, unsat = d.interface_hbonds, d.delta_unsat_hbonds
```
**Gotcha:** this build has NO `InterfaceAnalyzerMover("A_B")` string constructor (TypeError);
wrap the string with `DockingPartners.docking_partners_from_string`. `AB_C` grouping verified on 3HTN.

## Helper scripts (tested) - `scripts/` next to this file
`scripts/rosetta_tool.py` (single structure; prints JSON):
```bash
S=<skill>/scripts                 # this skill's own scripts/, next to SKILL.md
$PY $S/rosetta_tool.py score     in.pdb [--json out.json] [--sfxn ref2015]
$PY $S/rosetta_tool.py perres    in.pdb --csv perres.csv           # per-residue weighted total + unweighted terms
$PY $S/rosetta_tool.py relax     in.pdb --out relaxed.pdb [--rounds 1] [--max-iter N] [--constrain]
$PY $S/rosetta_tool.py interface in.pdb --partners A_B [--pack-input] [--pack-separated] [--json out.json]
```
(interface default = no repacking -> deterministic; add both pack flags to mimic the repo script.)

`scripts/make_leaderboard.py DIR` - writes `DIR/leaderboard.json` for any folder of 2-chain PDBs so the repo
batch scorer can be used outside the CoFold pipeline (refuses to overwrite). `scripts/submit_energy.pbs` -
generalized PBS template of the repo job (`qsub -v TOP_DIR=...,WORKERS=96`); its body was run locally with
`bash` (WORKERS=2) - PASS; the actual `qsub` submission is UNVERIFIED.

## The repo pipeline: `/flare/FRAME-IDP/msinclair/software/pyrosetta_energy`
- `scripts/compute_rosetta_energy.py --top100-dir DIR --out out.json [--out-csv out.csv] [--workers 32]`
  - Input: `DIR/leaderboard.json` = list of `{binder_id, pdb, score, ...}`; PDB is looked up as `DIR/<basename(pdb)>`
    (missing files -> warning, skipped). Every PDB must be 2 chains: **A = target, B = binder** (hardcoded `A_B`).
  - Per structure: ref2015 total score, then `InterfaceAnalyzerMover(A_B, tracer, sfxn, packstat=True, pack_input=True, pack_separated=True)`.
  - Output JSON `{"results": [...], "errors": [...]}` sorted by `interface_dG` ascending (most favorable first);
    fields: binder_id, pdb, leaderboard_score, total_score_ref2015, interface_dG, dSASA_interface, dhSASA_interface,
    shape_complementarity, packstat, delta_unsat_hbonds, interface_hbonds, nres_interface,
    complex_normalized_dG (= dG / nres_interface), seconds. CSV has the same columns.
  - Parallelism: `ProcessPoolExecutor(workers)`, `pyrosetta.init` once per worker in the initializer. One banner per worker in the log.
- `scripts/submit_top100_energy.pbs`: 1 debug node (`select=1:ncpus=208`, 45 min), `TMPDIR=/tmp`, `--workers 96`,
  input `.../cofold_8730212_20260803T152231Z/top100`, output `top100/rosetta_energy/top100_rosetta_energy.{json,csv}`,
  log tee'd to `logs/$PBS_JOBID/job.log`. Job 8809862: 100/100 succeeded in 3 min wall (15:10:52 -> 15:13:52).
  The Lmod `ONEAPI_DEVICE_SELECTOR` warnings in `rosetta_top100.o8809862` are harmless.
- Verified rerun on 4 of those PDBs (`--workers 4`): 12.2 s wall, 4/4 OK; `total_score_ref2015` identical to the
  stored run to 3 decimals; `nres_interface` identical; `interface_dG` differed by 0.3-2.7 REU (see next gotcha).

## Gotchas
0. **`pyrosetta.init("opts")` vs `init(extra_options="opts")` changes results.** The positional `options`
   default is `-ex1 -ex2aro`; passing your flags positionally silently drops extra rotamers. Anything that packs
   (IAM with pack_input/pack_separated, FastRelax) then differs a lot: BND_0149 repo-protocol dG = +86 REU
   with positional init vs -64.9 with `extra_options` (the repo uses `extra_options`; stored value -63.5);
   BND_0103 +6..+9 vs -30.0. Scoring a fixed pose (total, per-residue, no-pack IAM) is unaffected.
1. **Repacked interface dG has some noise.** With `pack_input/pack_separated=True` (repo default) the packer
   is stochastic and depends on RNG state (prior calls in the same worker): BND_0103 gave -27.3, -30.0, -31.1;
   BND_0149 -63.5/-64.1/-64.9; pdl1_seed_binder +20.0/+21.4. Expect ~1-3 REU spread; don't over-interpret
   small rank differences. No-pack IAM is deterministic. Compare only numbers from the same protocol + init flags.
   `-constant_seed -jran N` makes a fresh process reproducible (verified), not worker-order independent.
2. Unrelaxed predicted/crystal models score huge positive totals (clashes): e.g. 3HTN total +3630, AB_C dG +3645;
   pdl1 seed no-pack dG +268. Relax (or at least repack) before trusting absolute numbers; after 1-round
   constrained FastRelax pdl1 dG became -66.5 (no-pack) / -57.2 to -58.6 (repo protocol).
3. Per-residue sums need `decompose_bb_hb_into_pair_energies(True)` (above).
4. `residue_total_energies(i)` terms are unweighted; multiply by `sfxn.get_weight(st)`.
5. String IAM constructor missing -> use DockingPartners.
6. Repo script assumes exactly chains A/B and `leaderboard.json`; other chain IDs require a modified copy (don't edit the repo).
7. Memory: one process scoring+repacked IAM on a ~250-res complex peaked at 0.86 GB RSS, so 96 workers ~ 80+ GB (fits an Aurora node). Keep `--workers` <= cores you own.

## License
PyRosetta is free for academic/non-profit use; **commercial use requires a license** (license@uw.edu; see
`.venv/lib/python3.12/site-packages/pyrosetta-2026.34+release.de92a3c0de.dist-info/licenses/`). The banner printed at init states this.

## Test inputs
A verified run produces `score_6MRR.json`, `perres_6MRR.csv`, `6MRR_relax1.pdb`,
`pdl1_relax1.pdb` and `iface_*.json`; write them somewhere you own. These inputs
are durable and reusable: `/flare/FRAME-IDP/msinclair/software/ProteinMPNN/inputs/PDB_monomers/pdbs/6MRR.pdb`,
`.../PDB_complexes/pdbs/3HTN.pdb` (3 chains), `/flare/FRAME-IDP/msinclair/software/ProtonPottsMPNN/inference/examples/pdl1_seed_binder.pdb` (A/B).
