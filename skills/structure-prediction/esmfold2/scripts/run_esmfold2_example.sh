#!/bin/bash
REPO=/flare/FRAME-IDP/msinclair/software/esm
export ZE_AFFINITY_MASK=4
export PYTHONPATH="$REPO/transformers/src:$REPO:${PYTHONPATH:-}"
export HF_HOME=$REPO/.hf_cache
export HF_HUB_OFFLINE=1
cd /lus/flare/projects/FRAME-IDP/msinclair/testing_grounds/env_tests/test_runs/esmfold2
# Villin headpiece-ish / small 56-aa protein (GB1)
time $REPO/.venv/bin/python $REPO/scripts/aurora/fold_sequence.py \
  --seq MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE \
  --loops 4 --steps 20 --out gb1_pred.cif
