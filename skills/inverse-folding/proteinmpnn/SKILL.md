---
name: proteinmpnn
description: Run ProteinMPNN inverse folding on ALCF Aurora (Intel Max 1550 XPU) to design amino-acid sequences for a given protein backbone (PDB). Use when asked to design/redesign sequences for a structure, design selected chains of a complex, keep some residues fixed, tie positions across chains (homooligomers/symmetry), use the soluble or CA-only ProteinMPNN models, or score native/given sequences against a backbone (score_only, per-residue log-probs).
---

# ProteinMPNN on Aurora

Code: `/flare/FRAME-IDP/msinclair/software/ProteinMPNN` (do not edit). Entry point: `protein_mpnn_run.py`; JSONL helpers in `helper_scripts/`; example inputs in `inputs/` (`PDB_complexes/pdbs/{3HTN,4YOW}.pdb`, `PDB_monomers/pdbs/{5L33,6MRR}.pdb`, `PDB_homooligomers/pdbs/{6EHB,4GYT}.pdb`).

## Environment (verified)

```bash
MPNN=/flare/FRAME-IDP/msinclair/software/ProteinMPNN
SKILL=<skill>                             # this skill's own directory
export ZE_AFFINITY_MASK=8                 # pin to one tile; it then appears as xpu:0
source $MPNN/.venv/bin/activate           # uv venv on top of aurora_frameworks-2026.1.0 (torch 2.13, system site-packages)
export PYTHONPATH=$SKILL/scripts/ipex_shim${PYTHONPATH:+:$PYTHONPATH}   # REQUIRED, see below
```

**IPEX shim is required.** `protein_mpnn_run.py` and `protein_mpnn_utils.py` do `import intel_extension_for_pytorch as ipex` but never use it. IPEX is not installed in the 2026 frameworks stack, so without the shim the run fails with `ModuleNotFoundError: No module named 'intel_extension_for_pytorch'`. `scripts/ipex_shim/` holds an empty stand-in package. Native `torch.xpu` does the actual work.

## Device: what actually happens
- The device is hard-coded: `torch.device("xpu:0" if torch.xpu.is_available() else "cpu")`. **The `--device` flag is parsed but ignored.** Choose the tile with `ZE_AFFINITY_MASK`.
- Verified: the log prints `model on device: xpu:0` (that is physical tile 8 under the mask). If XPU is unavailable it silently falls back to CPU, so check that line.
- Ignore the `Overriding a previously registered kernel ... aten::topk` UserWarning and the `torch.cross` deprecation warning (CA model). Both are harmless.

## Model weights
`--model_name` takes one of `v_48_002`, `v_48_010`, `v_48_020` (default), `v_48_030`. The number is the training backbone noise (0.02 to 0.30 Å); more noise gives more robust, less native-like designs.
- vanilla: `$MPNN/vanilla_model_weights/` (default)
- soluble: `$MPNN/soluble_model_weights/` (pass `--use_soluble_model`)
- CA-only: `$MPNN/ca_model_weights/` (pass `--ca_only`; has only 002/010/020)
- Custom folder: `--path_to_model_weights DIR`

## Verified workflows (all PASS on tile 8, 2026-10-29)
Full suite: `bash $SKILL/scripts/run_tests.sh [OUTDIR]` (about 57 s total, including about 3 s import/model load per call). It sets up the environment itself.

### 1. Multi-chain design (design A,B; other chains fixed as context)
```bash
D=out/multichain; mkdir -p $D
python $MPNN/helper_scripts/parse_multiple_chains.py --input_path=$MPNN/inputs/PDB_complexes/pdbs/ --output_path=$D/parsed.jsonl
python $MPNN/helper_scripts/assign_fixed_chains.py --input_path=$D/parsed.jsonl --output_path=$D/assigned.jsonl --chain_list "A B"
python $MPNN/protein_mpnn_run.py --jsonl_path $D/parsed.jsonl --chain_id_jsonl $D/assigned.jsonl \
  --out_folder $D --num_seq_per_target 4 --sampling_temp "0.1 0.2" --seed 37 --batch_size 2
```
`parse_multiple_chains.py` parses every PDB in a folder (the trailing `/` matters). `--chain_list` = chains to DESIGN.

### 2. Fixed positions
```bash
python $MPNN/helper_scripts/make_fixed_positions_dict.py --input_path=$D/parsed.jsonl --output_path=$D/fixed.jsonl \
  --chain_list "A C" --position_list "1 2 3 4 5 6 7 8 23 25, 10 11 12 13 14 15 16 17 18 19 20 40"
python $MPNN/protein_mpnn_run.py --jsonl_path $D/parsed.jsonl --chain_id_jsonl $D/assigned.jsonl \
  --fixed_positions_jsonl $D/fixed.jsonl --out_folder $D --num_seq_per_target 2 --sampling_temp "0.1" --seed 37 --batch_size 1
```
Positions are **1-based indices within each chain's parsed sequence, not PDB residue numbers**. Comma-separated groups follow the `--chain_list` order. The inverse also works: `--specify_non_fixed` lists the positions to design (see `examples/submit_example_4_non_fixed.sh`; not tested here). Verified that 4YOW chain A positions 1-8 stayed native (`MRIVAADT`).

### 3. Homooligomer, tied positions (symmetric design)
```bash
python $MPNN/helper_scripts/parse_multiple_chains.py --input_path=$MPNN/inputs/PDB_homooligomers/pdbs/ --output_path=$D/parsed.jsonl
python $MPNN/helper_scripts/make_tied_positions_dict.py --input_path=$D/parsed.jsonl --output_path=$D/tied.jsonl --homooligomer 1
python $MPNN/protein_mpnn_run.py --jsonl_path $D/parsed.jsonl --tied_positions_jsonl $D/tied.jsonl \
  --out_folder $D --num_seq_per_target 2 --sampling_temp "0.2" --seed 37 --batch_size 1
```
Custom ties (from upstream example 5, not run here): `make_tied_positions_dict.py --chain_list "A C" --position_list "1 2 3, 1 2 3"`. The lists must match in length.

### 4. Single PDB, no JSONL (soluble model)
```bash
python $MPNN/protein_mpnn_run.py --pdb_path $MPNN/inputs/PDB_monomers/pdbs/5L33.pdb --pdb_path_chains "A" \
  --use_soluble_model --out_folder $D --num_seq_per_target 4 --sampling_temp "0.1" --seed 37 --batch_size 1
```

### 5. CA-only model
```bash
python $MPNN/protein_mpnn_run.py --ca_only --pdb_path $MPNN/inputs/PDB_monomers/pdbs/5L33.pdb \
  --out_folder $D --num_seq_per_target 2 --sampling_temp "0.1" --seed 37 --batch_size 1
```

### 6. Score only (native sequence against backbone)
```bash
python $MPNN/protein_mpnn_run.py --pdb_path $MPNN/inputs/PDB_complexes/pdbs/3HTN.pdb --pdb_path_chains "A B" \
  --score_only 1 --out_folder $D --num_seq_per_target 4 --seed 37 --batch_size 1
```
Writes `$D/score_only/3HTN_pdb.npz` with keys `score` (shape num_seq), `global_score`, `S`, `seq_str`, and prints mean/std to stdout. To score your own sequences, add `--path_to_fasta seqs.fa`, with chains sorted alphabetically and separated by `/` (`examples/submit_example_3_score_only_from_fasta.sh`; not tested here).

## Key flags
| flag | meaning |
|---|---|
| `--num_seq_per_target N` | sequences per structure per temperature; must be divisible by `--batch_size` (`NUM_BATCHES = N // batch_size`) |
| `--batch_size B` | copies decoded in parallel; raise it for throughput (fine on 64 GB tiles) |
| `--sampling_temp "0.1 0.2"` | space-separated string; each temperature gives N sequences. Typical values 0.1-0.3 |
| `--seed S` | 0 = random; set it for reproducibility |
| `--omit_AAs "CX"` | globally forbid residues (default `X`) |
| `--bias_AA_jsonl`, `--bias_by_res_jsonl`, `--omit_AA_jsonl`, `--pssm_jsonl` | composition/position biases (helpers `make_bias_AA.py`, `make_bias_per_res_dict.py`, `make_pssm_input_dict.py`); not tested here |
| `--backbone_noise σ` | add Gaussian noise to the input backbone |
| `--save_score 1`, `--save_probs 1` | write per-sequence scores / per-position probabilities as `.npz` |
| `--conditional_probs_only 1`, `--unconditional_probs_only 1` | output probability arrays instead of sequences (untested) |
| `--max_length` | skip longer inputs |

## Output format
`OUT/seqs/<name>.fa`. The first record is the native sequence:
```
>3HTN, score=1.1466, global_score=1.1931, fixed_chains=['C'], designed_chains=['A', 'B'], model_name=v_48_020, git_hash=..., seed=37
<native designed-chain sequence(s), chains separated by '/'>
>T=0.1, sample=1, score=0.7154, global_score=0.9104, seq_recovery=0.6099
<designed sequence>
```
- `score`: mean negative log-probability over **designed** residues. Lower is better (more confident).
- `global_score`: the same average over **all** residues (including fixed chains and positions).
- `seq_recovery`: fraction of designed positions identical to the native.
- Only designed chains appear in the sequence lines, separated by `/` in alphabetical chain order. `X` marks missing residues (gaps in the PDB).

## Performance (tile 8, measured)
2 seqs × 960 aa (6EHB homotetramer, tied) in 4.3 s; 4 × 106 aa in 1.5 s; each process costs about 3 s for imports and model load. For many structures, put them all in one JSONL (one process) rather than looping over processes.

## Troubleshooting
- `No module named 'intel_extension_for_pytorch'`: the IPEX shim is not on `PYTHONPATH` (see Environment).
- `model on device: cpu`: XPU is not visible. Check `ZE_AFFINITY_MASK` (must be a valid tile on this node) and `python -c "import torch;print(torch.xpu.is_available())"`.
- `--device xpu:3` has no effect. Use `ZE_AFFINITY_MASK` instead.
- Fixed positions on the wrong residues: the indices are positions within the chain sequence (1-based), not PDB numbering. Check against `parsed.jsonl` `seq_chain_X`.
- `chain_id_jsonl is NOT loaded` with the `--pdb_path` mode is normal. Chains come from `--pdb_path_chains`, and when it is empty all chains are designed.
- Zero outputs: `num_seq_per_target < batch_size` gives 0 batches.
- Outputs go to `--out_folder` (it is created, along with `seqs/`). Never write into `$MPNN/outputs`.
