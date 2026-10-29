#!/usr/bin/env bash
# Minimal verified RFD3 unconditional design on Aurora (one tile).
set -eo pipefail
module use /opt/aurora/26.26.0/modulefiles
module swap frameworks/2026.1.0 frameworks/2025.3.1 2>/dev/null || module load frameworks/2025.3.1
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-6}
export FOUNDRY_CHECKPOINT_DIRS=/flare/FRAME-IDP/msinclair/software/checkpoints
FOUNDRY=/flare/FRAME-IDP/msinclair/software/foundry
OUT=${1:-$PWD/out_uncond50}      # a directory you own; defaults under the current one
cd "$FOUNDRY"
time "$FOUNDRY/.venv/bin/rfd3" design \
  inputs=null +specification.length=50 \
  out_dir="$OUT" \
  diffusion_batch_size=1 n_batches=1 \
  inference_sampler.num_timesteps=50 \
  skip_existing=False prevalidate_inputs=True
