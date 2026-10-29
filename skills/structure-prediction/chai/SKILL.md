---
name: chai
description: Predict all-atom structures of proteins, complexes, protein-ligand (SMILES), DNA/RNA and glycans with Chai-1 on ALCF Aurora Intel XPUs, optionally with ColabFold MSAs and contact/pocket restraints. Use when the user wants to fold or co-fold molecules with Chai-1 on Aurora with the local install at /flare/FRAME-IDP/msinclair/software/chai-lab. You MUST use the bundled scripts/chai_xpu.py launcher, because the plain `chai` CLI is broken in this environment.
---

# Chai-1 on Aurora (Intel XPU)

Install: `/flare/FRAME-IDP/msinclair/software/chai-lab` is a fork with the device hard-coded to `xpu:0`.
Do not edit files in the install directory.

## IMPORTANT: use the launcher, not `chai`
In the current environment (aurora_frameworks-2026.1.0 with pandas 3.1 and RDKit 2026.09), the stock
`.venv/bin/chai` hits up to four breakages. The launcher `scripts/chai_xpu.py`, located next to this file, monkeypatches all of them at runtime and takes the same CLI arguments as `chai`:

| Symptom with plain `chai` | Cause | Fix in launcher |
|---|---|---|
| `ModuleNotFoundError: intel_extension_for_pytorch` at import | chai_lab imports IPEX (unused); IPEX is not in the 2026 frameworks | injects an empty stub module |
| `AssertionError` in `antipickle/base.py register_adapter` while loading `conformers_v1.apkl` | antipickle 0.2.2 already ships a "torch" adapter, and chai registers a second one | skips duplicate adapters |
| `KeyError: 'source_database'` with `--use-msa-server` / `--msa-directory` | pandas 3 `groupby.apply` drops the group column | applies the MSA quota itself, then calls the original parser |
| **Ligand silently missing** from the output. The log only shows `ERROR ... Failed to tokenize input ... Cannot set unknown attribute 'useChirality'` (then `'maxAttempts'`) | RDKit 2026.09 renamed these ETKDG params | maps them to `enforceChirality` / `maxIterations` |

The last case does not crash. The run "succeeds" without the ligand, so always grep logs for `Failed to tokenize`.
Alternative for imports only: put an empty `intel_extension_for_pytorch/__init__.py` on `PYTHONPATH`. That fixes only the first row.

## Environment
```bash
CHAI_PY=/flare/FRAME-IDP/msinclair/software/chai-lab/.venv/bin/python   # system frameworks python + venv; no module load needed
LAUNCH=<skill>/scripts/chai_xpu.py    # travels with this skill
export ZE_AFFINITY_MASK=3     # ONE tile; it appears as xpu:0 (the code always uses xpu:0)
export CHAI_DOWNLOADS_DIR=/flare/FRAME-IDP/msinclair/software/chai-lab/downloads
# for --use-msa-server only:
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
```

## Weights
All weights are already in `/flare/FRAME-IDP/msinclair/software/chai-lab/downloads` (this is the default location even without the env var):
`models_v2/{feature_embedding,token_embedder,trunk,diffusion_module,confidence_head,bond_loss_input_proj}.pt`,
`conformers_v1.apkl`, `esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt` (5.7 GB ESM2-3B, used by default for embeddings).
No downloads are needed.

## Input (FASTA with entity headers)
```
>protein|name=ubq
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>ligand|name=asp
CC(=O)Oc1ccccc1C(=O)O
```
Entity types are `protein`, `ligand` (SMILES), `dna`, `rna`, and `glycan`. Each `name=` must be unique. Modified residues use the form `AAA(SEP)AAA`.
Chains are lettered A, B, C... in input order. Example inputs are in `scripts/ubq.fasta` and `scripts/ubq_lig.fasta`, and more are in
`<install>/input_tests/*.fasta`.
Restraints: a CSV given with `--constraint-path` (see `<install>/examples/restraints/README.md` for the contact/pocket format). This was not tested here.
Custom MSAs: `.aligned.pqt` files given with `--msa-directory` (see `<install>/examples/msas/README.md`). The code path is the same as the MSA server, but a user-supplied directory was not tested directly.

## Verified commands (Aurora, tile 3, Oct 2026)
1. Protein only, single sequence + ESM embeddings, 76 aa. Wall time 50 s (about 35 s of that is loading the models and ESM from Lustre). pTM 0.935, pLDDT about 97:
```bash
$CHAI_PY $LAUNCH fold ubq.fasta out --num-trunk-recycles 1 --num-diffn-timesteps 50 \
  --num-diffn-samples 1 --seed 42 --device xpu:0
```
2. Protein + SMILES ligand + ColabFold MSA server. Wall time 51 s. The ligand is present (13 HETATM records), ptm 0.91:
```bash
$CHAI_PY $LAUNCH fold ubq_lig.fasta out_lig --use-msa-server --num-trunk-recycles 1 \
  --num-diffn-timesteps 50 --num-diffn-samples 2 --seed 42 --device xpu:0
```
For production, use the defaults (`--num-trunk-recycles 3 --num-diffn-timesteps 200 --num-diffn-samples 5`) with the MSA server. Those settings were not timed here.

## Main options (`fold FASTA OUT_DIR`)
`--use-esm-embeddings/--no-use-esm-embeddings` (default on), `--use-msa-server`, `--msa-server-url`, `--msa-directory`,
`--constraint-path`, `--use-templates-server`, `--template-hits-path`, `--num-trunk-recycles` (3), `--num-diffn-timesteps` (200),
`--num-diffn-samples` (5), `--num-trunk-samples` (1), `--seed`, `--device` (the code forces xpu:0 anyway), `--low-memory/--no-low-memory`,
`--full-data` (also stores pae/plddt arrays in the scores npz; not tested). OUT_DIR must be empty or nonexistent.

## Output layout
```
out/
  pred.model_idx_{i}.cif     # one per diffusion sample; per-atom pLDDT (0-100) is in the B-factor column
  scores.model_idx_{i}.npz   # aggregate_score, ptm, iptm, per_chain_ptm, per_chain_pair_iptm,
                             # has_inter_chain_clashes, chain_chain_clashes
  msas/*.aligned.pqt, msa_depth.pdf   # only with --use-msa-server
```
`aggregate_score = 0.2*ptm + 0.8*iptm` (minus a clash penalty). For monomers iptm=0, so the score is about 0.19 even for a perfect fold. Judge monomers by ptm and pLDDT instead.
Loading the npz: `np.load(f)`, then `d[k].tolist()`. The boolean clash arrays break `np.round` on the whole dict.

## Aurora/XPU gotchas
- Pin to one tile with `ZE_AFFINITY_MASK`. The fork hard-codes `xpu:0`, so `--device` cannot select a tile.
- Expected noise that you can ignore: the `aten::topk` kernel-override warnings at import.
- The first part of a run is I/O-bound (5.7 GB ESM plus about 1.2 GB of models from Lustre, with low CPU use). It is not hung.
- The MSA server needs the ALCF proxy env vars.

## Troubleshooting
- Any of the four errors in the table: you ran plain `chai`. Use the launcher.
- Ligand missing / iptm=0 with a ligand present: grep the log for `Failed to tokenize`. The SMILES might be invalid, or the RDKit patch is not active.
- `name=... used more than once`: entity names must be unique.
