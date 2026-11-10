#!/usr/bin/env bash
export HPC_LOAM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LOAM_SOFTWARE_ROOT="${LOAM_SOFTWARE_ROOT:-$(dirname "$HPC_LOAM_HOME")/software}"
export HF_HOME="${HF_HOME:-$LOAM_SOFTWARE_ROOT/.hf-cache}"
export CHAI_DOWNLOADS_DIR="${CHAI_DOWNLOADS_DIR:-$LOAM_SOFTWARE_ROOT/chai-lab/downloads}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$LOAM_SOFTWARE_ROOT/boltz/cache}"
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$LOAM_SOFTWARE_ROOT/fairchem/cache}"
export DATA_PATH="${DATA_PATH:-$LOAM_SOFTWARE_ROOT/RFD4-Proteina/data_path}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export FOUNDRY_CHECKPOINT_DIRS="${FOUNDRY_CHECKPOINT_DIRS:-$LOAM_SOFTWARE_ROOT/foundry/checkpoints}"
export AMBERHOME="${AMBERHOME:-$LOAM_SOFTWARE_ROOT/ambertools}"
export PATH="$AMBERHOME/bin:$PATH"
export LOAM_HOOK_PYTHON="$LOAM_SOFTWARE_ROOT/jnana-engine/.venv/bin/python"
export LAYERNORM_TYPE=torch
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$LOAM_SOFTWARE_ROOT/protenix/data}"

# Proxy-aware tools use login-node internet; internal services bypass the proxy.
if [[ -f "$LOAM_SOFTWARE_ROOT/internet-proxy/env.sh" ]]; then
    source "$LOAM_SOFTWARE_ROOT/internet-proxy/env.sh"
fi

# Open Babel's ARM64 wheel needs X11 libraries absent on compute nodes.
# Dedicated pinned runtime, installed by install_jupiter_rfd4_runtime.sh.
export RFD4_RUNTIME_LIB_DIR="$LOAM_SOFTWARE_ROOT/RFD4-Proteina/runtime/lib"
if [[ -d "$RFD4_RUNTIME_LIB_DIR" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$RFD4_RUNTIME_LIB_DIR:"*) ;;
        *) export LD_LIBRARY_PATH="$RFD4_RUNTIME_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi
