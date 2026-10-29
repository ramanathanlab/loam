#!/usr/bin/env bash
# Minimal RFD4-Proteina generation on Aurora (one XPU tile).
set -eo pipefail
REPO=/flare/FRAME-IDP/msinclair/software/RFD4-Proteina
# BASE must be a writable dir holding shim/sitecustomize.py (copy it from this
# skill's scripts/ next to this file). Override on the command line if needed.
BASE=${BASE:-/lus/flare/projects/FRAME-IDP/msinclair/testing_grounds/env_tests/test_runs/rfd4-proteina}
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
