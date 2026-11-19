#!/bin/bash
# RFD4-Proteina resident-worker launch on Aurora, verified under Loam's
# run_batch / Ensemble-Launcher path. Uses David Kastner's pinned read-only
# deployment (frameworks/2026.1.0, Lightning 2.6.6, in-code SingleDeviceStrategy)
# -- NOT the old sitecustomize/Lightning-2.3 wrapper, which DEADLOCKS under the
# MPI/PALS launcher. Source of truth: RFD4_Aurora_Handoff/ (dkastner, FRAME-IDP).
#
# One independent sampler per assigned XPU tile; the tile comes from the
# EL-assigned ZE_AFFINITY_MASK (fall back to 0). Do NOT distribute one model
# across MPI ranks: the adapter stays single-device by construction.
#
# Args: $1=plan.json (absolute)  $2=fresh_run_dir (absolute, must NOT exist)
#       $3=bounded_worker_seconds (e.g. 700 for one cold design)

# --- make `module` available even in a non-login EL subprocess ---
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/00-modulepath.sh /opt/aurora/*/modules/lmod/lmod/init/bash; do
    [ -f "$f" ] && source "$f"
  done
fi

export RFD4_PLAN="$1"
export RFD4_RUN="$2"
export RFD4_SECONDS="$3"
export RFD4_TILE="${ZE_AFFINITY_MASK:-0}"   # respect the EL-pinned tile
echo "[rfd4-resident] host=$(hostname -s) tile(ZE_AFFINITY_MASK)=$RFD4_TILE plan=$RFD4_PLAN run=$RFD4_RUN PBS_JOBID=${PBS_JOBID:-UNSET}"

set -eo pipefail
: "${RFD4_PLAN:?absolute path to samples JSON}"
: "${RFD4_RUN:?fresh absolute output directory}"
: "${RFD4_SECONDS:?bounded worker seconds}"
case "$RFD4_TILE" in 0|1|2|3|4|5|6|7|8|9|10|11) ;; *) echo "bad tile $RFD4_TILE"; exit 2 ;; esac
[[ "$RFD4_SECONDS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$RFD4_PLAN" == /* && "$RFD4_RUN" == /* ]] || exit 2
[[ -f "$RFD4_PLAN" && ! -e "$RFD4_RUN" ]] || { echo "plan missing or run dir exists"; exit 2; }
: "${PBS_JOBID:?run inside a genuine Aurora allocation}" "${PBS_NODEFILE:?}"
[[ "$PBS_JOBID" == *aurora* && -r "$PBS_NODEFILE" ]] || exit 2
RFD4_HOST=$(hostname -s)
[[ "$RFD4_HOST" == x* ]] || exit 2
awk -v node="$RFD4_HOST" -F. '$1 == node {found=1} END {exit !found}' "$PBS_NODEFILE" || exit 2
module load frameworks/2026.1.0
set -u

# ---- pinned read-only deployment (do NOT modify; hashes verified in handoff) ----
RFD4_SHARE=/lus/flare/projects/FRAME-IDP/dkastner/Projects/IDeA
RFD4_PYTHON="$RFD4_SHARE/envs/rfd4-xpu-2026.1.0/bin/python"
RFD4_ADAPTER="$RFD4_SHARE/experiments/pmd_design_20261026T070010Z/source/v34"
RFD4_ENGINE="$RFD4_SHARE/experiments/aurora_performance_20261023T194443Z/source/rfd4_phase1"
export PYTHONPATH="$RFD4_ADAPTER:$RFD4_ADAPTER/src:$RFD4_ENGINE/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export IDEA_COMPUTE_SITE=aurora IDEA_XPU_DEVICE=0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK="$RFD4_TILE"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENBLAS_CORETYPE=Haswell PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 DISABLE_CUEQUIVARIANCE=1
export HF_HOME="$RFD4_SHARE/cache/huggingface" HF_HUB_OFFLINE=1
export DATA_PATH="$RFD4_SHARE/cache/data" PYTORCH_DEBUG_XPU_FALLBACK=1
export MPIR_CVAR_ENABLE_GPU=0

mkdir "$RFD4_RUN"
mkdir "$RFD4_RUN/tmp" "$RFD4_RUN/cache"
export TMPDIR="$RFD4_RUN/tmp" XDG_CACHE_HOME="$RFD4_RUN/cache"
cd "$RFD4_ADAPTER"
sha256sum -c > "$RFD4_RUN/adapter_check.log" <<'HASHES'
2a0a9392c805a941cf85695c5191e1e943697ad830aefcbe2b9077d4f7d742f0  src/idea/engines/rfd4_persistent.py
ef8e69f47bc5d0232cc9aa1f0b224d4fac6fbb8d0a5fbbf0503bf4f90d7bf190  src/idea/engines/rfd4_xpu.py
HASHES

set +e   # capture rc explicitly so diagnostics always print; real code preserved via exit
# Import/tile probe (300s: cold imports off Lustre can exceed the handoff's 90s).
timeout --signal=TERM --kill-after=15s 300s "$RFD4_PYTHON" -u - \
  "$RFD4_ADAPTER" "$RFD4_ENGINE" > "$RFD4_RUN/imports.log" 2>&1 <<'PY'
import sys
from pathlib import Path
import idea.engines.rfd4_persistent as adapter
import rfproteina.generate as engine
import torch
assert Path(adapter.__file__).resolve() == (Path(sys.argv[1]) / 'src/idea/engines/rfd4_persistent.py').resolve()
assert Path(engine.__file__).resolve() == (Path(sys.argv[2]) / 'src/rfproteina/generate.py').resolve()
assert torch.xpu.is_available() and torch.xpu.device_count() == 1
print('Pinned imports OK; exactly one visible XPU tile', flush=True)
PY
IRC=$?
echo "[rfd4-resident] import-probe rc=$IRC"
if [[ $IRC -ne 0 ]]; then echo "[rfd4-resident] import probe FAILED"; tail -5 "$RFD4_RUN/imports.log"; exit $IRC; fi

# Resident worker: load the model once, run every sample in the plan serially.
timeout --signal=TERM --kill-after=20s "${RFD4_SECONDS}s" \
  "$RFD4_PYTHON" -u -X faulthandler -m idea.engines.rfd4_persistent \
  --plan "$RFD4_PLAN" --output "$RFD4_RUN/predictions" \
  --device 0 --precision bf16-mixed --matmul highest --deterministic \
  --recycle-strategy medium/midpeak \
  > "$RFD4_RUN/worker.log" 2>&1
WRC=$?
echo "[rfd4-resident] worker rc=$WRC"
echo "[rfd4-resident] summary.json:"; cat "$RFD4_RUN/predictions/summary.json" 2>&1 || echo "NONE"
exit $WRC
