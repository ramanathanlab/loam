#!/bin/bash
# Verified minimal Protenix run on Aurora XPU (plain `protenix` CLI entry point).
REPO=/flare/FRAME-IDP/msinclair/software/Protenix
export LD_LIBRARY_PATH=/opt/aurora/26.26.0/oneapi/2025.3/lib:${LD_LIBRARY_PATH:-}
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-4}
export PROTENIX_ROOT_DIR=/flare/FRAME-IDP/msinclair/software/protenix_data
export PYTHONUNBUFFERED=1
MODEL=${MODEL:-protenix_mini_esm_v0.5.0}
OUT=${OUT:-out_cli_mini}
cd /lus/flare/projects/FRAME-IDP/msinclair/testing_grounds/env_tests/test_runs/protenix
time $REPO/.venv/bin/protenix pred -i gb1.json -o $OUT -s 101 \
  -n $MODEL -c 4 -p 20 -e 1 --use_msa false \
  --trimul_kernel torch --triatt_kernel torch
