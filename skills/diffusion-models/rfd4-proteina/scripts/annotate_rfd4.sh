#!/usr/bin/env bash
# Build an RFD4-Proteina annotated-CIF binder input from a folded target, in the
# SAME pinned engine env as run_rfd4_resident.sh (dkastner's deployment). This is
# the annotation half of rfd4_proteina_generate: it drives RFD4's own
# write_condition_spec_cif via make_rfd4_input.py (CPU, no MPI, no GPU), so the
# tool can go target -> annotated CIF -> generation without a hand-made input.
#
# Usage (called by rfd4_run_row.py):
#   annotate_rfd4.sh TARGET TARGET_CHAIN TARGET_RANGE HOTSPOTS BINDER_LEN OUT [HOTSPOT_ATOMS]
# TARGET_RANGE is "<start>-<end>" (required: the engine contig rejects a bare
# chain id). HOTSPOTS is a comma list of residue indices (may be empty).
set -eo pipefail

# make `module` available even in a non-login EL subprocess
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh \
           /etc/profile.d/00-modulepath.sh /opt/aurora/*/modules/lmod/lmod/init/bash; do
    [ -f "$f" ] && source "$f"
  done
fi

TARGET="$1"; TARGET_CHAIN="${2:-A}"; TARGET_RANGE="$3"; HOTSPOTS="$4"
BINDER_LEN="${5:-55}"; OUT="$6"; HOTSPOT_ATOMS="${7:-CA}"
: "${TARGET:?target structure path}"
: "${TARGET_RANGE:?target residue range start-end}"
: "${OUT:?output annotated CIF path}"
[[ "$TARGET" == /* && "$OUT" == /* ]] || { echo "paths must be absolute"; exit 2; }
[[ -f "$TARGET" ]] || { echo "target not found: $TARGET"; exit 2; }

module load frameworks/2026.1.0

# Pinned deployment paths (match run_rfd4_resident.sh; dkastner FRAME-IDP).
RFD4_SHARE="${RFD4_SHARE:-/lus/flare/projects/FRAME-IDP/dkastner/Projects/IDeA}"
RFD4_PYTHON="${RFD4_PYTHON:-$RFD4_SHARE/envs/rfd4-xpu-2026.1.0/bin/python}"
RFD4_ADAPTER="$RFD4_SHARE/experiments/pmd_design_20261026T070010Z/source/v34"
RFD4_ENGINE="$RFD4_SHARE/experiments/aurora_performance_20261023T194443Z/source/rfd4_phase1"
export PYTHONPATH="$RFD4_ADAPTER:$RFD4_ADAPTER/src:$RFD4_ENGINE/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$RFD4_SHARE/cache/huggingface" HF_HUB_OFFLINE=1
export DATA_PATH="$RFD4_SHARE/cache/data"
export DISABLE_CUEQUIVARIANCE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

MAKE="$(dirname "$0")/make_rfd4_input.py"
args=(--target "$TARGET" --target-chain "$TARGET_CHAIN" --target-range "$TARGET_RANGE"
      --binder-len "$BINDER_LEN" --out "$OUT" --hotspot-atoms "$HOTSPOT_ATOMS")
[ -n "$HOTSPOTS" ] && args+=(--hotspots "$HOTSPOTS")
echo "[annotate_rfd4] target=$TARGET chain=$TARGET_CHAIN range=$TARGET_RANGE hotspots=${HOTSPOTS:-none} binder_len=$BINDER_LEN -> $OUT"
exec "$RFD4_PYTHON" "$MAKE" "${args[@]}"
