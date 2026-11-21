---
name: chai
description: Fold or co-fold proteins, protein-ligand (SMILES), DNA/RNA and glycan complexes with Chai-1, writing mmCIF plus pTM/ipTM scores. Supports MSAs, contact/pocket restraints and glycans. Use when asked to predict a structure or complex with Chai-1.
systems: [aurora, jupiter]
---

# Chai-1

## Before you run it: read the reference file for this System

The environment, device selection and launch command differ per System. Read
`references/<HPC_SYSTEM>.md` (for example `read_skill(name="chai", file="references/aurora.md")`)
and follow it; it is authoritative over this file for anything System-specific. This skill has
reference files for **aurora** and **jupiter**. It is **unsupported** on any other System: report the
step as blocked rather than adapting another System's instructions.

Software is found only through Site profile variables: the install is under `$LOAM_SOFTWARE_ROOT`,
and model assets under `$CHAI_DOWNLOADS_DIR`. Never edit the install directory; write outputs to the
campaign workspace.

## Input

A FASTA with entity headers. Chains are lettered A, B, C… in input order and each `name=` must be
unique. Entity types: `protein`, `ligand` (SMILES), `dna`, `rna`, `glycan`. Modified residues use
`AAA(SEP)AAA`. Examples: `examples/ubq.fasta`, `examples/ubq_lig.fasta`.

```
>protein|name=ubq
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>ligand|name=asp
CC(=O)Oc1ccccc1C(=O)O
```

For a campaign complex, write one record per entity from the verified target sequence and the
actual designed binder sequence. Never fabricate a binder for a folding test.

## Arguments (`fold FASTA OUT_DIR`)

| flag | default | meaning |
|---|---|---|
| `--num-trunk-recycles` | 3 | trunk recycling passes |
| `--num-diffn-timesteps` | 200 | diffusion steps |
| `--num-diffn-samples` | 5 | structures generated |
| `--num-trunk-samples` | 1 | trunk samples |
| `--seed` | none | reproducibility |
| `--use-esm-embeddings` / `--no-…` | on | ESM2-3B single-sequence embeddings |
| `--use-msa-server` / `--no-use-msa-server` | off | ColabFold MSAs (needs network access) |
| `--msa-directory DIR` | — | user `.aligned.pqt` MSAs |
| `--constraint-path CSV` | — | contact/pocket restraints (format: `<install>/examples/restraints/`) |
| `--use-templates-server`, `--template-hits-path` | off | templates |
| `--low-memory` / `--no-low-memory` | on | memory/speed tradeoff |
| `--full-data` | off | also store pae/plddt arrays in the scores npz |

Production settings are the defaults above (3 recycles, 200 steps, 5 samples). Reduced settings
(1 recycle, 50 steps, 1 sample) test plumbing only; never draw scientific conclusions from them.

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

- `OUT_DIR` must be empty or nonexistent.
- A process return code alone is not success. Check that every intended entity is present with its
  expected sequence and chain mapping, and that coordinates are finite.
- **A ligand can vanish silently**: a tokenization failure logs `ERROR … Failed to tokenize input`
  and the run still "succeeds" without the ligand. Always grep the log for `Failed to tokenize`.
- For monomers `iptm = 0`, so `aggregate_score` sits near 0.19 even for a perfect fold. Judge
  monomers by `ptm` and pLDDT.
- Confidence and interface metrics are not binding free energy or measured affinity. A valid
  prediction with low interface confidence is evidence against a reliable designed interface, not a
  runtime failure and not proof of binding.
- Record the MSA choice: single-sequence runs (no MSA server) are a limitation to report.
