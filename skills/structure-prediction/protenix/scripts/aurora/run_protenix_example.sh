#!/bin/bash
# Minimal Protenix prediction on Aurora (one XPU tile).
# Usage: run_protenix_example.sh [work_dir=.] [tile=4]
set -eo pipefail
: "${LOAM_SOFTWARE_ROOT:?set by the Site profile (software_root)}"
REPO="$LOAM_SOFTWARE_ROOT/Protenix"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${1:-$PWD}; TILE=${2:-4}

export LD_LIBRARY_PATH=/opt/aurora/26.26.0/oneapi/2025.3/lib:${LD_LIBRARY_PATH:-}   # venv/module skew
export ZE_AFFINITY_MASK=$TILE
export PROTENIX_ROOT_DIR="$LOAM_SOFTWARE_ROOT/protenix_data"
export PYTHONUNBUFFERED=1

mkdir -p "$WORK"; cd "$WORK"        # ESM embeddings are cached into the CWD
[[ -f gb1.json ]] || cp "$HERE/../../examples/example_gb1.json" gb1.json

"$REPO"/.venv/bin/protenix pred -i gb1.json -o out -s 101 \
  -n protenix_mini_esm_v0.5.0 -c 10 -p 200 -e 5 --use_msa false \
  --trimul_kernel torch --triatt_kernel torch        # torch kernels are mandatory on XPU
