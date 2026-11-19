#!/bin/bash
# Minimal ESMFold2 single-sequence fold on Aurora (one XPU tile).
# Usage: run_esmfold2_example.sh [out_dir=.] [tile=4]
set -eo pipefail
REPO=/flare/FRAME-IDP/msinclair/software/esm
OUT=${1:-$PWD}; TILE=${2:-4}

export ZE_AFFINITY_MASK=$TILE
export PYTHONPATH="$REPO/transformers/src:$REPO:${PYTHONPATH:-}"   # fork must come first
export HF_HOME=$REPO/.hf_cache
export HF_HUB_OFFLINE=1

mkdir -p "$OUT"; cd "$OUT"
$REPO/.venv/bin/python $REPO/scripts/aurora/fold_sequence.py \
  --seq MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE \
  --loops 20 --steps 100 --out gb1_pred.cif
