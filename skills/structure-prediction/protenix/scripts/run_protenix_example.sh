#!/bin/bash
# Verified minimal Protenix run on Aurora XPU (plain `protenix` CLI entry point).
REPO=/flare/FRAME-IDP/msinclair/software/Protenix
export LD_LIBRARY_PATH=/opt/aurora/26.26.0/oneapi/2025.3/lib:${LD_LIBRARY_PATH:-}
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-4}
export PROTENIX_ROOT_DIR=/flare/FRAME-IDP/msinclair/software/protenix_data
export PYTHONUNBUFFERED=1
MODEL=${MODEL:-protenix_mini_esm_v0.5.0}
OUT=${OUT:-out_cli_mini}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${1:-$PWD}         # a directory you own; defaults to the current one
mkdir -p "$WORK"; cd "$WORK"
# The input travels with the skill, so this does not depend on a prepared dir.
[[ -f gb1.json ]] || cp "$HERE/example_gb1.json" gb1.json
time $REPO/.venv/bin/protenix pred -i gb1.json -o $OUT -s 101 \
  -n $MODEL -c 4 -p 20 -e 1 --use_msa false \
  --trimul_kernel torch --triatt_kernel torch
