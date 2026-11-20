#!/bin/bash
# Usage: run_boltz_xpu.sh <input.yaml|.fasta|dir> <out_dir> [tile=2] [extra boltz args...]
# Tested on Aurora (aurora_frameworks-2026.1.0, PVC FLAT mode).
set -e
IN=$1; OUT=$2; TILE=${3:-2}; shift 3 || shift $#
export ZE_AFFINITY_MASK=$TILE
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
/flare/FRAME-IDP/msinclair/software/boltz/.venv/bin/boltz predict "$IN" \
  --out_dir "$OUT" --cache ~/.boltz --accelerator xpu --devices 1 \
  --output_format pdb --num_workers 0 "$@"
