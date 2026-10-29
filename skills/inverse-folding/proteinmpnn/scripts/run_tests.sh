#!/bin/bash
# Verified ProteinMPNN test suite on Aurora (single XPU tile).
set -euo pipefail
MPNN=/flare/FRAME-IDP/msinclair/software/ProteinMPNN
OUT=${1:-/lus/flare/projects/FRAME-IDP/msinclair/testing_grounds/env_tests/test_runs/proteinmpnn}
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-8}
source $MPNN/.venv/bin/activate
SKILL_SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH=$SKILL_SCRIPTS/ipex_shim${PYTHONPATH:+:$PYTHONPATH}
mkdir -p $OUT; cd $OUT
python -c "import torch;print('torch',torch.__version__,'xpu',torch.xpu.is_available(),torch.xpu.device_count())"

# 1. Multi-chain design: all PDBs in PDB_complexes, design chains A B
D=$OUT/multichain; mkdir -p $D
python $MPNN/helper_scripts/parse_multiple_chains.py --input_path=$MPNN/inputs/PDB_complexes/pdbs/ --output_path=$D/parsed.jsonl
python $MPNN/helper_scripts/assign_fixed_chains.py --input_path=$D/parsed.jsonl --output_path=$D/assigned.jsonl --chain_list "A B"
/usr/bin/time -f "WALL multichain %e s" python $MPNN/protein_mpnn_run.py --jsonl_path $D/parsed.jsonl --chain_id_jsonl $D/assigned.jsonl \
  --out_folder $D --num_seq_per_target 4 --sampling_temp "0.1 0.2" --seed 37 --batch_size 2

# 2. Fixed positions (chains A C designed; listed positions kept native; 1-based index within chain)
D=$OUT/fixed; mkdir -p $D
python $MPNN/helper_scripts/parse_multiple_chains.py --input_path=$MPNN/inputs/PDB_complexes/pdbs/ --output_path=$D/parsed.jsonl
python $MPNN/helper_scripts/assign_fixed_chains.py --input_path=$D/parsed.jsonl --output_path=$D/assigned.jsonl --chain_list "A C"
python $MPNN/helper_scripts/make_fixed_positions_dict.py --input_path=$D/parsed.jsonl --output_path=$D/fixed.jsonl --chain_list "A C" \
  --position_list "1 2 3 4 5 6 7 8 23 25, 10 11 12 13 14 15 16 17 18 19 20 40"
/usr/bin/time -f "WALL fixed %e s" python $MPNN/protein_mpnn_run.py --jsonl_path $D/parsed.jsonl --chain_id_jsonl $D/assigned.jsonl \
  --fixed_positions_jsonl $D/fixed.jsonl --out_folder $D --num_seq_per_target 2 --sampling_temp "0.1" --seed 37 --batch_size 1

# 3. Homooligomer, tied positions
D=$OUT/homooligomer; mkdir -p $D
python $MPNN/helper_scripts/parse_multiple_chains.py --input_path=$MPNN/inputs/PDB_homooligomers/pdbs/ --output_path=$D/parsed.jsonl
python $MPNN/helper_scripts/make_tied_positions_dict.py --input_path=$D/parsed.jsonl --output_path=$D/tied.jsonl --homooligomer 1
/usr/bin/time -f "WALL homooligomer %e s" python $MPNN/protein_mpnn_run.py --jsonl_path $D/parsed.jsonl --tied_positions_jsonl $D/tied.jsonl \
  --out_folder $D --num_seq_per_target 2 --sampling_temp "0.2" --seed 37 --batch_size 1

# 4. Single PDB, soluble model
D=$OUT/soluble; mkdir -p $D
/usr/bin/time -f "WALL soluble %e s" python $MPNN/protein_mpnn_run.py --pdb_path $MPNN/inputs/PDB_monomers/pdbs/5L33.pdb --pdb_path_chains "A" \
  --use_soluble_model --out_folder $D --num_seq_per_target 4 --sampling_temp "0.1" --seed 37 --batch_size 1

# 5. CA-only model
D=$OUT/ca_only; mkdir -p $D
/usr/bin/time -f "WALL ca_only %e s" python $MPNN/protein_mpnn_run.py --ca_only --pdb_path $MPNN/inputs/PDB_monomers/pdbs/5L33.pdb \
  --out_folder $D --num_seq_per_target 2 --sampling_temp "0.1" --seed 37 --batch_size 1

# 6. Score only (native sequence scoring)
D=$OUT/score_only; mkdir -p $D
/usr/bin/time -f "WALL score_only %e s" python $MPNN/protein_mpnn_run.py --pdb_path $MPNN/inputs/PDB_complexes/pdbs/3HTN.pdb --pdb_path_chains "A B" \
  --score_only 1 --out_folder $D --num_seq_per_target 4 --seed 37 --batch_size 1
echo ALL_TESTS_DONE
