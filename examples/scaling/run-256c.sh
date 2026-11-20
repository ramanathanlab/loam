#!/usr/bin/env bash
# Best observed 256-node recipe: 40.51% GPU busy; claim-provenance qualification failed.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${1:?Usage: run-256c.sh NEW_EXPERIMENT_ROOT [extra stage arguments, optionally --submit]}
shift
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
exec "${LOAM_PYTHON:-$repo/.venv/bin/python}" "$repo/scripts/stage_single_allocation.py" "$root" \
  --nodes 256 --inference 102 --gpus-per-node 4 --cpus-per-node 288 \
  --label "${LOAM_RUN_LABEL:-recipe-256c}" --walltime 09:00:00 \
  --concurrency 616 --top-k 1848 --initial-hypotheses 3696 --new-per-round 924 \
  --tournament-workers 204 --generation-batch-size 4 --science-workers-per-gpu 2 \
  --tournament-max-seconds 10800 --controller-gc-threshold0 10000 "$@"
