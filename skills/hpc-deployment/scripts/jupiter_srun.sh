#!/usr/bin/env bash
# Nested EL child/copy steps share the Globus pilot allocation.
set -euo pipefail
exec srun --overlap --cpu-bind=none "$@"
