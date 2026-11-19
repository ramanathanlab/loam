#!/bin/bash
# Minimal ProteinMPNN sequence design on Aurora (one XPU tile).
# Usage: run_proteinmpnn.sh <input.pdb> <out_dir> [tile=8] [extra protein_mpnn_run.py args...]
set -eo pipefail
MPNN=/flare/FRAME-IDP/msinclair/software/ProteinMPNN
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PDB=$1; OUT=$2; TILE=${3:-8}; shift 3 2>/dev/null || shift $#

export ZE_AFFINITY_MASK=$TILE
export PYTHONPATH="$HERE/ipex_shim${PYTHONPATH:+:$PYTHONPATH}"   # empty IPEX stub; required

mkdir -p "$OUT"
"$MPNN/.venv/bin/python" "$MPNN/protein_mpnn_run.py" \
  --pdb_path "$PDB" --out_folder "$OUT" \
  --num_seq_per_target 4 --sampling_temp "0.1" --seed 37 --batch_size 1 "$@"

# Confirm the GPU was actually used: the log must say "model on device: xpu:0", not cpu.
