---
name: aurora-chai-fold
description: >-
  Co-fold a protein complex (target + designed binder, any number of chains)
  with Chai-1 on an Aurora XPU tile and read the interface metrics out of the
  scores .npz. Use for every fold in a binder campaign, including the
  candidate-vs-control pairs that make a comparison mean something.
---

# Chai-1 complex folding on Aurora

## Activate (no activate script for this venv)

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
module use /soft/modulefiles; module load frameworks/2026.1.0
source $R/envs/chai-2026-venv/bin/activate
```

## Invoke

`chai fold` takes two REQUIRED positional arguments, in this order:

```bash
chai fold <fasta_file> <output_dir> --device xpu:0 --seed 42 \
  --num-diffn-samples 1 --num-trunk-samples 1 --num-trunk-recycles 3 \
  --num-diffn-timesteps 80 --no-use-msa-server --low-memory
```

Flags that matter, from `chai fold --help` (verified rc=0 on Aurora):

- `--use-esm-embeddings` / `--no-use-esm-embeddings` — **default is ON**.
- `--use-msa-server` / `--no-use-msa-server` — **default is OFF**. Leave it off:
  the compute nodes should not depend on an external MSA service mid-campaign.
  `--msa-directory <path>` supplies precomputed MSAs instead.
- `--constraint-path <path>` — residue/contact constraints. This is how you keep
  a binder on the epitope you intend instead of letting it relocate.
- `--template-hits-path <path>`, `--use-templates-server`.

## FASTA format

One record per chain, header `>protein|<name>`. Both chains of a complex go in
ONE file:

```
>protein|EGFR_human_ectodomain
LEEKKVCQGTSNKLTQ...
>protein|binder_v1
H(AIB)EGTFTSDVSSYLEGQ...
```

Non-standard residues go in parentheses by CCD code, e.g. `(AIB)`.

## Outputs and metrics

In `<output_dir>`:

- `pred.model_idx_0.cif` — the predicted complex
- `scores.model_idx_0.npz` — `iptm`, `ptm`, `per_chain_pair_iptm`,
  `aggregate_score`

Read the `.npz` and report the numbers; do not infer confidence from the
structure alone. `per_chain_pair_iptm` is the interface value you want for a
two-chain complex, not the global `ptm`. For ipSAE you need the full PAE —
`ipSAE.py` lives in the molecular-simulations tree.

## Cost and scaling

A ~620-residue ectodomain plus a short binder is minutes per tile. One fold uses
one tile, so a batch row needs `gpus=1` and 12 rows fill a node. For more than a
handful of folds use `run_batch`, never a loop of `run_code`.

## Pitfalls

- Writing results into the same directory you pass as a `run_code` `workdir`
  will corrupt an output manifest: the server drops a temporary script there and
  deletes it. Run in `scratch/`, write results to `outputs/` by absolute path.
- `--num-diffn-samples 1` gives ONE model (`model_idx_0`). Raise it for seed
  replicates; a single sample is not a confidence estimate.
- Expect torch XPU kernel-override warnings on stderr. They are noise, not
  failure; check the return code and the `.npz`.
