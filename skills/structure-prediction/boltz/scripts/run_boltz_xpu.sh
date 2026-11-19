#!/bin/bash
# Minimal Boltz-2 prediction on Aurora (one XPU tile).
# Usage: run_boltz_xpu.sh <input.yaml|.fasta|dir> <out_dir> [tile=2] [extra boltz args...]
set -eo pipefail
IN=$1; OUT=$2; TILE=${3:-2}; shift 3 2>/dev/null || shift $#

export ZE_AFFINITY_MASK=$TILE
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128

/flare/FRAME-IDP/msinclair/software/boltz/.venv/bin/boltz predict "$IN" \
  --out_dir "$OUT" --cache ~/.boltz --accelerator xpu --devices 1 \
  --output_format pdb --num_workers 0 --override "$@"
