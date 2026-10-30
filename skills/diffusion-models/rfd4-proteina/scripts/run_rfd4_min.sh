#!/usr/bin/env bash
# Minimal RFD4-Proteina generation on Aurora (one XPU tile).
set -eo pipefail
REPO=/flare/FRAME-IDP/msinclair/software/RFD4-Proteina
# BASE is a writable directory you own; it holds shim/, data_path/ and hf/. There
# is no shared copy to inherit, so it must be set. The shim is installed from this
# skill's own scripts/ below, so no manual copy step is needed.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${BASE:?set BASE to a writable directory you own (it will hold shim/, data_path/, hf/)}
mkdir -p "$BASE/shim"
[[ -f $BASE/shim/sitecustomize.py ]] || cp "$HERE/sitecustomize.py" "$BASE/shim/"
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-7}
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export CLUSTER=local
export DATA_PATH=$BASE/data_path
export HF_HOME=$BASE/hf
export PYTHONPATH=$BASE/shim${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
time "$REPO/.venv/bin/rfd4-generate" \
  inputs=data/benchmarks/monomer \
  subset_to_examples=monomer-short \
  num_replicates=1 batch_size=1 \
  out_dir="$BASE/out_smoke" \
  ~metrics compile_lengths=null verbose=true
