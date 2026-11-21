#!/bin/bash
# Minimal Chai-1 fold on Aurora (one XPU tile).
# Usage: run_chai.sh <input.fasta> <out_dir> [tile=3] [extra chai args...]
set -eo pipefail
: "${LOAM_SOFTWARE_ROOT:?set by the Site profile (software_root)}"
IN=$1; OUT=$2; TILE=${3:-3}; shift 3 2>/dev/null || shift $#
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ZE_AFFINITY_MASK=$TILE
export CHAI_DOWNLOADS_DIR="$LOAM_SOFTWARE_ROOT/chai-lab/downloads"
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128

# The launcher is mandatory: the stock `chai` CLI is broken in this environment.
"$LOAM_SOFTWARE_ROOT"/chai-lab/.venv/bin/python "$HERE/chai_xpu.py" \
  fold "$IN" "$OUT" --device xpu:0 "$@"
