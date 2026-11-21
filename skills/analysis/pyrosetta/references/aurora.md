# PyRosetta on Aurora (CPU only)

Install: `$LOAM_SOFTWARE_ROOT/pyrosetta_energy` (do not edit; write outputs elsewhere). CPU-only —
it never needs a GPU, so do not set `ZE_AFFINITY_MASK`.

## Environment

```bash
PY=$LOAM_SOFTWARE_ROOT/pyrosetta_energy/.venv/bin/python   # Python 3.12, uv venv
# or: source $LOAM_SOFTWARE_ROOT/pyrosetta_energy/.venv/bin/activate
```

`pyrosetta-2026.34+release.de92a3c0de`, plus numpy. **No pandas**, so `pyrosetta.distributed` and
`get_energies_table`-style helpers are unavailable — use the pure-pose APIs in SKILL.md. The venv's
base interpreter is an absolute `/opt/aurora/...2025.3.1` path, so it works without `module load`.

## Example

```bash
SKILL=$HPC_LOAM_HOME/skills/analysis/pyrosetta
TOOL=$SKILL/scripts/aurora/rosetta_tool.py        # SKILL.md's $TOOL
$PY $TOOL interface in.pdb --partners A_B --json out.json
```

## Batch scorer

For ranking many 2-chain complexes, the install's own pipeline:

```bash
$PY $LOAM_SOFTWARE_ROOT/pyrosetta_energy/scripts/compute_rosetta_energy.py \
  --top100-dir DIR --out out.json [--out-csv out.csv] [--workers 32]
```

`DIR/leaderboard.json` is a list of `{binder_id, pdb, score, ...}`; each PDB is looked up as
`DIR/<basename(pdb)>` and must have exactly 2 chains, **A = target, B = binder** (`A_B` is
hardcoded). `scripts/aurora/make_leaderboard.py DIR` generates that file for any folder of 2-chain
PDBs. `scripts/aurora/submit_energy.pbs` is a PBS template with no account or queue of its own:
pass your Site profile's `allocation.account` and `allocation.queue` on the command line
(`qsub -A <account> -q <queue> -v TOP_DIR=...,WORKERS=96 submit_energy.pbs`). Nothing exports them
into a worker's environment, so read them from the profile you are using.

## Must-know

- PBS: this is CPU work, so request whole nodes and plenty of `ncpus`.
- Durable test inputs: `$LOAM_SOFTWARE_ROOT/ProteinMPNN/inputs/PDB_monomers/pdbs/6MRR.pdb`,
  `.../PDB_complexes/pdbs/3HTN.pdb` (3 chains), and
  `$LOAM_SOFTWARE_ROOT/ProtonPottsMPNN/inference/examples/pdl1_seed_binder.pdb` (2 chains, A/B —
  the one that suits `interface --partners A_B`).
- Under Loam `run_batch`, reference the runners by their worker-visible copies (as `$SKILL` above) —
  the Aurora worker cannot see a local skill directory.

## Resource index

- `read_skill(name="pyrosetta", file="scripts/aurora/rosetta_tool.py")` — score / perres / relax /
  interface on one structure, JSON to stdout
- `read_skill(name="pyrosetta", file="scripts/aurora/make_leaderboard.py")` — build the
  `leaderboard.json` the batch scorer reads
- `read_skill(name="pyrosetta", file="scripts/aurora/submit_energy.pbs")` — PBS template for the
  batch scorer
