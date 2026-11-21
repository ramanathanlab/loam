#!/usr/bin/env bash
# The cluster agent's launcher on Systems whose HPC_MPI_FLAVOR is srun-overlap: every srun
# step EL starts, its node-local copy steps included, shares the endpoint's allocation.
set -euo pipefail
exec srun --overlap --cpu-bind=none "$@"
