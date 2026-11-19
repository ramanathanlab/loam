---
name: proteinmpnn
description: Design amino-acid sequences for a given protein backbone (PDB) with ProteinMPNN on Aurora Intel XPU. Supports multi-chain design, fixed positions, tied positions for homooligomers/symmetry, soluble and CA-only models, and scoring sequences against a backbone. Use when asked to design or redesign sequences for a structure on Aurora.
---

# ProteinMPNN on Aurora (Intel XPU)

Install: `/flare/FRAME-IDP/msinclair/software/ProteinMPNN` (do not edit). Entry point
`protein_mpnn_run.py`; JSONL helpers in `helper_scripts/`; example inputs in `inputs/`
(`PDB_complexes/pdbs/{3HTN,4YOW}.pdb`, `PDB_monomers/pdbs/{5L33,6MRR}.pdb`,
`PDB_homooligomers/pdbs/{6EHB,4GYT}.pdb`).

## Environment

```bash
MPNN=/flare/FRAME-IDP/msinclair/software/ProteinMPNN
SKILL=<this skill>                          # directory holding SKILL.md and scripts/
export ZE_AFFINITY_MASK=8                   # ONE tile; it then appears as xpu:0
source $MPNN/.venv/bin/activate             # uv venv on frameworks/2026.1.0 (torch 2.13, system site-packages)
export PYTHONPATH=$SKILL/scripts/ipex_shim${PYTHONPATH:+:$PYTHONPATH}   # REQUIRED, see Must-know
```

`--model_name` takes `v_48_002`, `v_48_010`, `v_48_020` (default) or `v_48_030`; the number is the
training backbone noise (0.02-0.30 Å) — more noise gives more robust, less native-like designs.
Weight sets: `$MPNN/vanilla_model_weights/` (default), `soluble_model_weights/`
(`--use_soluble_model`), `ca_model_weights/` (`--ca_only`; 002/010/020 only), or
`--path_to_model_weights DIR`.

## Example

```bash
# Minimal runner: scripts/run_proteinmpnn.sh <pdb> <out_dir> [tile] [extra args...]
bash scripts/run_proteinmpnn.sh $MPNN/inputs/PDB_monomers/pdbs/5L33.pdb out 8

# Single PDB, no JSONL
python $MPNN/protein_mpnn_run.py --pdb_path $MPNN/inputs/PDB_monomers/pdbs/5L33.pdb \
  --pdb_path_chains "A" --out_folder out \
  --num_seq_per_target 4 --sampling_temp "0.1" --seed 37 --batch_size 1

# Multi-chain: design A and B, hold the other chains as fixed context
python $MPNN/helper_scripts/parse_multiple_chains.py \
  --input_path=$MPNN/inputs/PDB_complexes/pdbs/ --output_path=out/parsed.jsonl   # trailing / matters
python $MPNN/helper_scripts/assign_fixed_chains.py \
  --input_path=out/parsed.jsonl --output_path=out/assigned.jsonl --chain_list "A B"   # chains to DESIGN
python $MPNN/protein_mpnn_run.py --jsonl_path out/parsed.jsonl --chain_id_jsonl out/assigned.jsonl \
  --out_folder out --num_seq_per_target 4 --sampling_temp "0.1 0.2" --seed 37 --batch_size 2

# Fixed positions (1-based WITHIN each chain's parsed sequence, not PDB numbering)
python $MPNN/helper_scripts/make_fixed_positions_dict.py \
  --input_path=out/parsed.jsonl --output_path=out/fixed.jsonl \
  --chain_list "A C" --position_list "1 2 3 4 5 23 25, 10 11 12 13 14 40"
python $MPNN/protein_mpnn_run.py --jsonl_path out/parsed.jsonl --chain_id_jsonl out/assigned.jsonl \
  --fixed_positions_jsonl out/fixed.jsonl --out_folder out \
  --num_seq_per_target 2 --sampling_temp "0.1" --seed 37 --batch_size 1

# Tied positions (symmetric homooligomer design)
python $MPNN/helper_scripts/make_tied_positions_dict.py \
  --input_path=out/parsed.jsonl --output_path=out/tied.jsonl --homooligomer 1
python $MPNN/protein_mpnn_run.py --jsonl_path out/parsed.jsonl --tied_positions_jsonl out/tied.jsonl \
  --out_folder out --num_seq_per_target 2 --sampling_temp "0.2" --seed 37 --batch_size 1

# Score a sequence against a backbone instead of designing
python $MPNN/protein_mpnn_run.py --pdb_path $MPNN/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --pdb_path_chains "A B" --score_only 1 --out_folder out \
  --num_seq_per_target 4 --seed 37 --batch_size 1        # add --path_to_fasta for your own seqs
```

## Arguments

| flag | meaning |
|---|---|
| `--pdb_path` / `--jsonl_path` | single PDB, or a parsed JSONL of many structures |
| `--pdb_path_chains` / `--chain_id_jsonl` | chains to design (empty = all) |
| `--fixed_positions_jsonl`, `--tied_positions_jsonl` | hold positions native; tie them across chains |
| `--specify_non_fixed` | invert `--fixed_positions_jsonl`: list the positions to DESIGN |
| `--num_seq_per_target N` | sequences per structure per temperature; must be divisible by `--batch_size` |
| `--batch_size B` | copies decoded in parallel; raise for throughput (fine on 64 GB tiles) |
| `--sampling_temp "0.1 0.2"` | space-separated string; each temperature yields N sequences. Typical 0.1-0.3 |
| `--seed S` | 0 = random; set for reproducibility |
| `--model_name`, `--use_soluble_model`, `--ca_only`, `--path_to_model_weights` | model selection |
| `--omit_AAs "CX"` | globally forbid residues (default `X`) |
| `--bias_AA_jsonl`, `--bias_by_res_jsonl`, `--omit_AA_jsonl`, `--pssm_jsonl` | composition/position biases (helpers `make_bias_AA.py`, `make_bias_per_res_dict.py`, `make_pssm_input_dict.py`) |
| `--backbone_noise σ` | add Gaussian noise to the input backbone |
| `--score_only 1`, `--path_to_fasta` | score sequences instead of designing |
| `--save_score 1`, `--save_probs 1` | write per-sequence scores / per-position probabilities as `.npz` |
| `--conditional_probs_only 1`, `--unconditional_probs_only 1` | output probability arrays |
| `--max_length` | skip longer inputs |
| `--device` | **parsed but ignored** — use `ZE_AFFINITY_MASK` |

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

- **The IPEX shim is required.** `protein_mpnn_run.py` and `protein_mpnn_utils.py` do
  `import intel_extension_for_pytorch as ipex` but never use it, and IPEX is absent from the 2026
  frameworks stack. `scripts/ipex_shim/` is an empty stand-in package; without it on `PYTHONPATH`
  the run fails with `ModuleNotFoundError`. Native `torch.xpu` does the real work.
- **`--device` is ignored.** The device is hard-coded to `xpu:0` if available, else CPU — so it
  **silently falls back to CPU**. Check the log line `model on device: xpu:0`. Pick the tile with
  `ZE_AFFINITY_MASK`.
- Fixed/tied positions are 1-based indices **within each chain's parsed sequence**, not PDB residue
  numbers. Check against `parsed.jsonl` → `seq_chain_X`. Comma-separated groups follow the
  `--chain_list` order, and custom tie lists must match in length.
- `num_seq_per_target < batch_size` gives zero batches and no output.
- For many structures, put them all in one JSONL (one process) rather than looping over processes —
  each process costs ~3 s of imports and model load.
- `chain_id_jsonl is NOT loaded` in `--pdb_path` mode is normal.
- Write to your own `--out_folder` (it is created, with `seqs/`); never write into `$MPNN/outputs`.
- Harmless noise: the `aten::topk` kernel-override warning and a `torch.cross` deprecation warning
  (CA model).
