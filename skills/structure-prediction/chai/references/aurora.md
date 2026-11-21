# Chai-1 on Aurora (Intel XPU)

Install: `$LOAM_SOFTWARE_ROOT/chai-lab` (fork with the device hard-coded to `xpu:0`).
Do not edit the install directory; write outputs elsewhere.

## Environment

```bash
CHAI_PY=$LOAM_SOFTWARE_ROOT/chai-lab/.venv/bin/python   # no module load needed
LAUNCH=$HPC_LOAM_HOME/skills/structure-prediction/chai/scripts/aurora/chai_xpu.py   # REQUIRED wrapper, not the `chai` CLI
export ZE_AFFINITY_MASK=3                   # ONE tile; it appears as xpu:0
export CHAI_DOWNLOADS_DIR=$LOAM_SOFTWARE_ROOT/chai-lab/downloads
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy   # only for --use-msa-server
```

Weights are already in `$CHAI_DOWNLOADS_DIR` (models_v2, `conformers_v1.apkl`, 5.7 GB traced ESM2-3B).
Nothing downloads.

## Example

```bash
# Minimal runner: scripts/aurora/run_chai.sh <input.fasta> <out_dir> [tile] [extra args...]
bash scripts/aurora/run_chai.sh examples/ubq.fasta out 3

# Equivalent direct call
$CHAI_PY $LAUNCH fold examples/ubq.fasta out --num-trunk-recycles 3 --num-diffn-timesteps 200 \
  --num-diffn-samples 5 --seed 42 --device xpu:0
```

## Aurora-specific arguments

| flag | default | meaning |
|---|---|---|
| `--device` | xpu:0 | parsed but forced to `xpu:0`; select a tile with `ZE_AFFINITY_MASK` |
| `--use-msa-server` | off | needs the proxy env vars above |

## Must-know

- **Use `scripts/aurora/chai_xpu.py`, never the venv's `chai` CLI.** The stock CLI hits four
  breakages in this environment (missing IPEX, duplicate antipickle adapter, pandas-3 MSA `KeyError`,
  RDKit-2026 ETKDG renames). The launcher monkeypatches all four and takes identical arguments.
- The RDKit breakage is what makes a ligand vanish silently here: grep the log for `Failed to tokenize`.
- Under Loam `run_batch` the launcher must be referenced by its worker-visible copy
  (`$HPC_LOAM_HOME/skills/structure-prediction/chai/scripts/aurora/chai_xpu.py`, emitted verbatim so
  it resolves on the worker) — the Aurora worker cannot see a local skill directory.

## Resource index

- `read_skill(name="chai", file="scripts/aurora/chai_xpu.py")` — the required launcher
- `read_skill(name="chai", file="scripts/aurora/run_chai.sh")` — minimal one-tile runner
- `read_skill(name="chai", file="examples/ubq.fasta")`, `examples/ubq_lig.fasta` — example inputs (any System)
