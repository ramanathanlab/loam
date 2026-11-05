#!/usr/bin/env bash
# Source this file; configuration only. This never starts an endpoint or a job.
export HPC_LOAM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LOAM_SOFTWARE_ROOT="${LOAM_SOFTWARE_ROOT:-$(dirname "$HPC_LOAM_HOME")/software}"
export GLOBUS_COMPUTE_VENV="$LOAM_SOFTWARE_ROOT/globus-compute/.venv"
export GLOBUS_COMPUTE_USER_DIR="$LOAM_SOFTWARE_ROOT/globus-compute/endpoints"
export GLOBUS_COMPUTE_ENDPOINT_NAME=loam-julich-jupiter
export TMPDIR=/tmp
source "$GLOBUS_COMPUTE_VENV/bin/activate"
# The UUID is created by Globus registration during the first user-started run.
if [[ -z "${HPC_GLOBUS_ENDPOINT_ID:-}" && -f "$GLOBUS_COMPUTE_USER_DIR/$GLOBUS_COMPUTE_ENDPOINT_NAME/endpoint.json" ]]; then
    export HPC_GLOBUS_ENDPOINT_ID="$("$GLOBUS_COMPUTE_VENV/bin/python" -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["GLOBUS_COMPUTE_USER_DIR"])/os.environ["GLOBUS_COMPUTE_ENDPOINT_NAME"]/"endpoint.json").read_text())["endpoint_id"])')"
fi
