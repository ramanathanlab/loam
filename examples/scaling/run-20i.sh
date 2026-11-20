#!/usr/bin/env bash
# Best qualified 20-node recipe: 51.06% whole-parent sampled GPU busy.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${1:?Usage: run-20i.sh NEW_EXPERIMENT_ROOT [extra stage arguments, optionally --submit]}
shift
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
exec "${LOAM_PYTHON:-$repo/.venv/bin/python}" "$repo/scripts/stage_single_allocation.py" "$root" \
  --nodes 20 --inference 8 --gpus-per-node 4 --cpus-per-node 288 \
  --label "${LOAM_RUN_LABEL:-recipe-20i}" --walltime 08:00:00 \
  --concurrency 48 --top-k 144 --initial-hypotheses 288 --new-per-round 72 \
  --tournament-workers 16 --generation-batch-size 4 --science-workers-per-gpu 2 "$@"
