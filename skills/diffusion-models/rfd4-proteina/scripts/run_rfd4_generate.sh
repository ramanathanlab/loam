#!/usr/bin/env bash
# Run RFD4-Proteina `rfd4-generate` on one Aurora XPU tile, UNDER Loam's
# run_batch/run_code worker. Handles BOTH Aurora blockers so the code generator
# does not have to assemble the environment by hand:
#
#   1. PALS/PMIx rank-env scrub -- the worker inherits a live PALS/PMIx launch
#      context (PMIX_RANK, PMIX_NAMESPACE, PALS_*), so rfd4-generate's torch/
#      frameworks stack runs MPI_Init_thread as "one rank of a many-rank job" and
#      aborts with `PMIx_Fence returned -33` (seen as rc=15/143). Unsetting these
#      is REQUIRED here and is the single most common reason rfd4-generate dies.
#   2. the sitecustomize.py shim -- lightning 2.3.0dev seed_everything()/XPU-
#      without-IPEX compatibility (same shim run_rfd4_min.sh installs).
#
# Design length, hotspots and conditioning come from the INPUT annotated CIF
# (build it with make_rfd4_input.py); they are NOT CLI flags here.
#
# Usage (on the worker, in RFD4's venv is NOT needed -- this calls it directly):
#   run_rfd4_generate.sh --inputs <input_annotated.cif> --out-dir <dir> \
#       --base <writable_dir> --num-designs 8 [--tile 7] [--seed 0]
set -eo pipefail

REPO=/flare/FRAME-IDP/msinclair/software/RFD4-Proteina
# Pre-downloaded RFD4 weights for this campaign (avoids a slow/fragile HF fetch).
DEFAULT_HF=/lus/flare/projects/FRAME-IDP/agentic_campaign_09_26/binder_design/models/rfd4_proteina
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

INPUTS=""; OUT_DIR=""; NUM=8; BATCH=1; BASE="${BASE:-}"; TILE="${ZE_AFFINITY_MASK:-7}"
SEED=""; HF="${HF_HOME:-$DEFAULT_HF}"; EXTRA=()
while [[ $# -gt 0 ]]; do case "$1" in
  --inputs)       INPUTS="$2"; shift 2;;
  --out-dir)      OUT_DIR="$2"; shift 2;;
  --num-designs)  NUM="$2"; shift 2;;      # rfd4 num_replicates (backbones per input)
  --batch-size)   BATCH="$2"; shift 2;;
  --base)         BASE="$2"; shift 2;;     # writable dir for shim/, data_path/
  --tile)         TILE="$2"; shift 2;;     # ZE_AFFINITY_MASK
  --seed)         SEED="$2"; shift 2;;
  --hf-home)      HF="$2"; shift 2;;
  --)             shift; EXTRA+=("$@"); break;;
  *)              EXTRA+=("$1"); shift;;
esac; done

: "${INPUTS:?--inputs <annotated.cif | benchmark dir> required (build the CIF with make_rfd4_input.py)}"
: "${OUT_DIR:?--out-dir <dir> required}"
: "${BASE:?--base <writable dir> required (holds shim/ and data_path/); or set BASE}"

# ---- blocker #1: scrub the inherited PALS/PMIx rank env (MPI_Init_thread else aborts) ----
unset PMIX_RANK PMIX_NAMESPACE PMIX_SERVER_URI PMIX_SECURITY_MODE PMIX_GDS_MODULE \
      PMI_RANK PMI_SIZE PMI_FD RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE \
      PALS_RANKID PALS_NODEID PALS_LOCAL_RANKID PALS_LOCAL_SIZE PARSL_WORKER_RANK

# ---- blocker #2: the sitecustomize.py shim on PYTHONPATH ----
mkdir -p "$BASE/shim"
[[ -f $BASE/shim/sitecustomize.py ]] || cp "$HERE/sitecustomize.py" "$BASE/shim/"
export PYTHONPATH="$BASE/shim${PYTHONPATH:+:$PYTHONPATH}"

# ---- the rest of the verified run_rfd4_min.sh environment ----
export ZE_AFFINITY_MASK="$TILE"
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export CLUSTER=local
export DATA_PATH="$BASE/data_path"
export HF_HOME="$HF"
mkdir -p "$OUT_DIR" "$DATA_PATH"

HYDRA=( inputs="$INPUTS" out_dir="$OUT_DIR" num_replicates="$NUM" batch_size="$BATCH"
        '~metrics' compile_lengths=null verbose=true )
[[ -n "$SEED" ]] && HYDRA+=( seed="$SEED" )

cd "$REPO"
echo "[run_rfd4_generate] tile=$TILE inputs=$INPUTS out_dir=$OUT_DIR num=$NUM seed=${SEED:-default} hf=$HF" >&2
exec "$REPO/.venv/bin/rfd4-generate" "${HYDRA[@]}" "${EXTRA[@]}"
