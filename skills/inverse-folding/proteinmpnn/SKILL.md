---
name: proteinmpnn
description: Design amino-acid sequences for a given protein backbone (PDB) with ProteinMPNN. Supports multi-chain design, fixed positions, tied positions for homooligomers/symmetry, soluble and CA-only models, and scoring sequences against a backbone. Use when asked to design or redesign sequences for a structure.
systems: [aurora, jupiter]
---

# ProteinMPNN

## Before you run it: read the reference file for this System

The environment, the device selection and the launch command differ per System. Read
`references/<HPC_SYSTEM>.md` (for example
`read_skill(name="proteinmpnn", file="references/aurora.md")`) and follow it; it is authoritative
over this file for anything System-specific, and it defines the `$MPNN` (install directory) and
`$PY` (interpreter) used below. This skill has reference files for **aurora** and **jupiter**. It is
**unsupported** on any other System: report the step as blocked rather than adapting another
System's instructions.

Software is found only through Site profile variables: the install is under `$LOAM_SOFTWARE_ROOT`.
Never edit or rebuild the install directory; write outputs to the campaign workspace.

ProteinMPNN designs a sequence for a **fixed** backbone. It does not generate backbones (that is
the diffusion-models family) and it does not predict or prove binding.

## Model selection

`--model_name` takes `v_48_002`, `v_48_010`, `v_48_020` (default) or `v_48_030`; the number is the
training backbone noise (0.02–0.30 Å) — more noise gives more robust, less native-like designs.
Weight sets live in the install: `$MPNN/vanilla_model_weights/` (default),
`soluble_model_weights/` (`--use_soluble_model`), `ca_model_weights/` (`--ca_only`; 002/010/020
only), or `--path_to_model_weights DIR`.

## Inputs

A PDB (or a JSONL parsed from many PDBs) plus the chains to design. For a complex, **name the
designed chain(s) explicitly and keep the target native** — an empty chain selection redesigns
every chain.

```bash
# Single PDB, no JSONL
"$PY" $MPNN/protein_mpnn_run.py --pdb_path "$INPUT_PDB" --pdb_path_chains "$DESIGN_CHAINS" \
  --out_folder "$OUT" --num_seq_per_target 4 --sampling_temp "0.1" --seed 37 --batch_size 1

# Multi-chain: design A and B, hold the other chains as fixed context
"$PY" $MPNN/helper_scripts/parse_multiple_chains.py \
  --input_path=$PDB_DIR/ --output_path=$OUT/parsed.jsonl          # trailing / matters
"$PY" $MPNN/helper_scripts/assign_fixed_chains.py \
  --input_path=$OUT/parsed.jsonl --output_path=$OUT/assigned.jsonl --chain_list "A B"   # DESIGN these
"$PY" $MPNN/protein_mpnn_run.py --jsonl_path $OUT/parsed.jsonl --chain_id_jsonl $OUT/assigned.jsonl \
  --out_folder $OUT --num_seq_per_target 4 --sampling_temp "0.1 0.2" --seed 37 --batch_size 2

# Fixed positions (1-based WITHIN each chain's parsed sequence, not PDB numbering)
"$PY" $MPNN/helper_scripts/make_fixed_positions_dict.py \
  --input_path=$OUT/parsed.jsonl --output_path=$OUT/fixed.jsonl \
  --chain_list "A C" --position_list "1 2 3 4 5 23 25, 10 11 12 13 14 40"
"$PY" $MPNN/protein_mpnn_run.py --jsonl_path $OUT/parsed.jsonl --chain_id_jsonl $OUT/assigned.jsonl \
  --fixed_positions_jsonl $OUT/fixed.jsonl --out_folder $OUT \
  --num_seq_per_target 2 --sampling_temp "0.1" --seed 37 --batch_size 1

# Tied positions (symmetric homooligomer design)
"$PY" $MPNN/helper_scripts/make_tied_positions_dict.py \
  --input_path=$OUT/parsed.jsonl --output_path=$OUT/tied.jsonl --homooligomer 1
"$PY" $MPNN/protein_mpnn_run.py --jsonl_path $OUT/parsed.jsonl --tied_positions_jsonl $OUT/tied.jsonl \
  --out_folder $OUT --num_seq_per_target 2 --sampling_temp "0.2" --seed 37 --batch_size 1

# Score a sequence against a backbone instead of designing
"$PY" $MPNN/protein_mpnn_run.py --pdb_path "$INPUT_PDB" --pdb_path_chains "A B" \
  --score_only 1 --out_folder $OUT --num_seq_per_target 4 --seed 37 --batch_size 1
# add --path_to_fasta for your own sequences
```

Set every path and chain list from real campaign artifacts. ProteinMPNN reads **PDB**; a folding or
diffusion step that wrote mmCIF needs a conversion first (`gemmi.read_structure(cif).write_pdb(pdb)`,
with gemmi installed in this tool's own venv — the reference file says how to check). PDB format has
stricter identifier and size limits than mmCIF, so check the converted structure keeps the intended
chains, residue identities and finite coordinates, and record the chain/residue mapping; a
successful import is not a verified conversion.

## Arguments

| flag | meaning |
|---|---|
| `--pdb_path` / `--jsonl_path` | single PDB, or a parsed JSONL of many structures |
| `--pdb_path_chains` / `--chain_id_jsonl` | chains to design (empty = all) |
| `--fixed_positions_jsonl`, `--tied_positions_jsonl` | hold positions native; tie them across chains |
| `--specify_non_fixed` | invert `--fixed_positions_jsonl`: list the positions to DESIGN |
| `--num_seq_per_target N` | sequences per structure per temperature; must be divisible by `--batch_size` |
| `--batch_size B` | copies decoded in parallel; raise for throughput |
| `--sampling_temp "0.1 0.2"` | space-separated string; each temperature yields N sequences. Typical 0.1–0.3 |
| `--seed S` | 0 = random; set for reproducibility |
| `--model_name`, `--use_soluble_model`, `--ca_only`, `--path_to_model_weights` | model selection |
| `--omit_AAs "CX"` | globally forbid residues (default `X`) |
| `--bias_AA_jsonl`, `--bias_by_res_jsonl`, `--omit_AA_jsonl`, `--pssm_jsonl` | composition/position biases (helpers `make_bias_AA.py`, `make_bias_per_res_dict.py`, `make_pssm_input_dict.py`) |
| `--backbone_noise σ` | add Gaussian noise to the input backbone |
| `--score_only 1`, `--path_to_fasta` | score sequences instead of designing |
| `--save_score 1`, `--save_probs 1` | write per-sequence scores / per-position probabilities as `.npz` |
| `--conditional_probs_only 1`, `--unconditional_probs_only 1` | output probability arrays |
| `--max_length` | skip longer inputs |
| `--device` | **parsed but ignored** — the reference file says how the device is chosen |

## Outputs

`OUT/seqs/<name>.fa`. The first record is the native sequence:

```
>3HTN, score=1.1466, global_score=1.1931, fixed_chains=['C'], designed_chains=['A', 'B'], model_name=v_48_020, seed=37
<native designed-chain sequence(s), chains separated by '/'>
>T=0.1, sample=1, score=0.7154, global_score=0.9104, seq_recovery=0.6099
<designed sequence>
```

- `score` — mean negative log-probability over **designed** residues; lower is better.
- `global_score` — the same average over **all** residues, including fixed chains and positions.
- `seq_recovery` — fraction of designed positions identical to the native.
- Only designed chains appear, separated by `/` in alphabetical chain order; `X` marks residues
  missing from the PDB.
- `--score_only` writes `OUT/score_only/<name>_pdb.npz` with keys `score`, `global_score`, `S`,
  `seq_str` and prints mean/std.

## Must-know

- **Scores are sequence negative log-likelihoods, not binding free energies** and not affinities.
  Rank designs by them only against other designs from the same model and settings.
- **The first FASTA record is the native/reference sequence.** Select a generated sample, never that
  first record, and check the target chain stayed native and the designed sequence belongs to the
  intended chain.
- Fixed/tied positions are 1-based indices **within each chain's parsed sequence**, not PDB residue
  numbers. Check against `parsed.jsonl` → `seq_chain_X`. Comma-separated groups follow the
  `--chain_list` order, and custom tie lists must match in length.
- `num_seq_per_target < batch_size` gives zero batches and no output.
- For many structures, put them all in one JSONL (one process) rather than looping over processes —
  each process costs seconds of imports and model load.
- `chain_id_jsonl is NOT loaded` in `--pdb_path` mode is normal.
- Write to your own `--out_folder` (it is created, with `seqs/`); never write into the install.
- Inspect the weights used, the device actually selected, the seed, the sampled temperatures and the
  saved FASTA before reporting success.
