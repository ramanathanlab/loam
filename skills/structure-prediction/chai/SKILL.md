---
name: chai
description: Fold or co-fold proteins, protein-ligand (SMILES), DNA/RNA and glycan complexes with Chai-1 on Aurora Intel XPU, writing mmCIF plus pTM/ipTM scores. Supports MSAs, contact/pocket restraints and glycans. Use when asked to predict a structure or complex with Chai-1 on Aurora.
---

# Chai-1 on Aurora (Intel XPU)

Install: `/flare/FRAME-IDP/msinclair/software/chai-lab` (fork with the device hard-coded to `xpu:0`).
Do not edit the install directory; write outputs elsewhere.

## Environment

```bash
CHAI_PY=/flare/FRAME-IDP/msinclair/software/chai-lab/.venv/bin/python   # no module load needed
LAUNCH=<this skill>/scripts/chai_xpu.py                                 # REQUIRED wrapper, not the `chai` CLI
export ZE_AFFINITY_MASK=3                   # ONE tile; it appears as xpu:0
export CHAI_DOWNLOADS_DIR=/flare/FRAME-IDP/msinclair/software/chai-lab/downloads
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy   # only for --use-msa-server
```

Weights are already in `$CHAI_DOWNLOADS_DIR` (models_v2, `conformers_v1.apkl`, 5.7 GB traced ESM2-3B).
Nothing downloads.

## Example

```bash
# Minimal runner: scripts/run_chai.sh <input.fasta> <out_dir> [tile] [extra args...]
bash scripts/run_chai.sh ubq.fasta out 3

# Equivalent direct call
$CHAI_PY $LAUNCH fold ubq.fasta out --num-trunk-recycles 3 --num-diffn-timesteps 200 \
  --num-diffn-samples 5 --seed 42 --device xpu:0
```

Input is a FASTA with entity headers; chains are lettered A, B, C… in input order and each `name=`
must be unique. Entity types: `protein`, `ligand` (SMILES), `dna`, `rna`, `glycan`. Modified residues
use `AAA(SEP)AAA`. Examples: `scripts/ubq.fasta`, `scripts/ubq_lig.fasta`.

```
>protein|name=ubq
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>ligand|name=asp
CC(=O)Oc1ccccc1C(=O)O
```

## Arguments (`fold FASTA OUT_DIR`)

| flag | default | meaning |
|---|---|---|
| `--num-trunk-recycles` | 3 | trunk recycling passes |
| `--num-diffn-timesteps` | 200 | diffusion steps |
| `--num-diffn-samples` | 5 | structures generated |
| `--num-trunk-samples` | 1 | trunk samples |
| `--seed` | none | reproducibility |
| `--use-esm-embeddings` / `--no-…` | on | ESM2-3B single-sequence embeddings |
| `--use-msa-server` | off | ColabFold MSAs (needs the proxy) |
| `--msa-directory DIR` | — | user `.aligned.pqt` MSAs |
| `--constraint-path CSV` | — | contact/pocket restraints (format: `<install>/examples/restraints/`) |
| `--use-templates-server`, `--template-hits-path` | off | templates |
| `--device` | xpu:0 | parsed but forced to `xpu:0`; select a tile with `ZE_AFFINITY_MASK` |
| `--low-memory` / `--no-low-memory` | on | memory/speed tradeoff |
| `--full-data` | off | also store pae/plddt arrays in the scores npz |

## Outputs

```
out/
  pred.model_idx_{i}.cif     # one per diffusion sample; per-atom pLDDT (0-100) in the B-factor column
  scores.model_idx_{i}.npz   # aggregate_score, ptm, iptm, per_chain_ptm, per_chain_pair_iptm,
                             # has_inter_chain_clashes, chain_chain_clashes
  msas/*.aligned.pqt, msa_depth.pdf      # only with --use-msa-server
```

`aggregate_score = 0.2*ptm + 0.8*iptm` minus a clash penalty. Read the npz with `np.load(f)` then
`d[k].tolist()` (the boolean clash arrays break `np.round` over the whole dict).

## Must-know

- **Use `scripts/chai_xpu.py`, never the venv's `chai` CLI.** The stock CLI hits four breakages in
  this environment (missing IPEX, duplicate antipickle adapter, pandas-3 MSA `KeyError`, RDKit-2026
  ETKDG renames). The launcher monkeypatches all four and takes identical arguments.
- **A ligand can vanish silently.** The RDKit breakage logs only `ERROR … Failed to tokenize input`
  and the run still "succeeds" without the ligand. Always grep the log for `Failed to tokenize`.
- For monomers `iptm = 0`, so `aggregate_score` sits near 0.19 even for a perfect fold. Judge
  monomers by `ptm` and pLDDT.
- `OUT_DIR` must be empty or nonexistent.
- Under Loam `run_batch` the launcher must be referenced by its worker-visible copy
  (`$HPC_LOAM_HOME/skills/structure-prediction/chai/scripts/chai_xpu.py`, emitted verbatim so it
  resolves on the worker) — the Aurora worker cannot see a local skill directory.
