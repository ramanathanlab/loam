#!/usr/bin/env bash
# Source this file in the MCP server shell. The UUID is discovered after the first user-started registration.
export HPC_LOAM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LOAM_SOFTWARE_ROOT="${LOAM_SOFTWARE_ROOT:-$(dirname "$HPC_LOAM_HOME")/software}"
export GLOBUS_COMPUTE_VENV="$LOAM_SOFTWARE_ROOT/globus-compute/.venv"
export GLOBUS_COMPUTE_USER_DIR="${GLOBUS_COMPUTE_USER_DIR:-$LOAM_SOFTWARE_ROOT/globus-compute/endpoints}"
export GLOBUS_COMPUTE_ENDPOINT_NAME=loam-julich-jupiter
export HPC_VENV="$GLOBUS_COMPUTE_VENV"
if [[ -z "${HPC_GLOBUS_ENDPOINT_ID:-}" && -f "$GLOBUS_COMPUTE_USER_DIR/$GLOBUS_COMPUTE_ENDPOINT_NAME/endpoint.json" ]]; then
    export HPC_GLOBUS_ENDPOINT_ID="$("$HPC_VENV/bin/python" -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["GLOBUS_COMPUTE_USER_DIR"])/os.environ["GLOBUS_COMPUTE_ENDPOINT_NAME"]/"endpoint.json").read_text())["endpoint_id"])')"
fi
export HPC_SYSTEM=jupiter
export HPC_ACCOUNT="${HPC_ACCOUNT:-deepwing}"
export HPC_QUEUE="${HPC_QUEUE:-booster}"
export HPC_WALLTIME="${HPC_WALLTIME:-00:15:00}"
export HPC_NNODES="${HPC_NNODES:-1}"
export HPC_POOL_SIZE="${HPC_POOL_SIZE:-4}"
export HPC_BASE_DIR="${HPC_BASE_DIR:-$(dirname "$HPC_LOAM_HOME")/campaigns/jupiter-smoke}"
export HPC_INSPECT_ROOTS="$HPC_BASE_DIR:$LOAM_SOFTWARE_ROOT"
export HPC_CAMPAIGN_ID="${HPC_CAMPAIGN_ID:-jupiter-smoke}"
export ACADEMY_EXCHANGE_TYPE="${ACADEMY_EXCHANGE_TYPE:-redis}"
if [[ "$ACADEMY_EXCHANGE_TYPE" == redis && -f "$LOAM_SOFTWARE_ROOT/redis/env.sh" ]]; then
    source "$LOAM_SOFTWARE_ROOT/redis/env.sh"
fi
export ACADEMY_EXCHANGE_URL="${ACADEMY_EXCHANGE_URL:-https://exchange.academy-agents.org/v1}"
export HPC_MCP_TRANSPORT=http
export HPC_ALLOWED_QUEUES=booster,largebooster
export HPC_MAX_NODES="${HPC_MAX_NODES:-1}"
export HPC_MAX_WALLTIME="${HPC_MAX_WALLTIME:-00:30:00}"
export HPC_CONTROL_DIR="$HPC_BASE_DIR/control"
export HPC_ENV_FILE="$HPC_LOAM_HOME/skills/hpc-deployment/set_env_jupiter.sh"
export LOAM_PREFLIGHT_PYTHON="$HPC_VENV/bin/python"
export ALCF_PROXY=0
export LOAM_HOOK_PYTHON="$LOAM_SOFTWARE_ROOT/jnana-engine/.venv/bin/python"
# JSON construction preserves quoting and spaces in paths. Worker env is explicit.
export HPC_ENDPOINT_CONFIG_JSON="$("$HPC_VENV/bin/python" - <<'PYCONFIG'
import json, os, shlex
e=os.environ
worker='export TMPDIR=/tmp; ' + '; '.join('export '+k+'='+shlex.quote(e[k]) for k in ['HPC_LOAM_HOME','LOAM_SOFTWARE_ROOT','GLOBUS_COMPUTE_USER_DIR'])
if e.get('ACADEMY_HOME'):
    worker+='; export ACADEMY_HOME='+shlex.quote(e['ACADEMY_HOME'])
proxy_env=e['LOAM_SOFTWARE_ROOT']+'/internet-proxy/env.sh'
worker+='; if [ -f '+shlex.quote(proxy_env)+' ]; then source '+shlex.quote(proxy_env)+'; fi'
worker+='; source '+shlex.quote(e['HPC_VENV']+'/bin/activate')
worker+='; export PYTHONPATH='+shlex.quote(e['HPC_LOAM_HOME']+'/skills/hpc-deployment/scripts')+':${PYTHONPATH:-}; export OMP_NUM_THREADS=4'
print(json.dumps(dict(account=e['HPC_ACCOUNT'], queue=e['HPC_QUEUE'], partition=e['HPC_QUEUE'], walltime=e['HPC_WALLTIME'], nodes_per_block=int(e['HPC_NNODES']), max_workers_per_node=int(e['HPC_POOL_SIZE'])+1, init_blocks=0, min_blocks=0, max_blocks=1, worker_init=worker)))
PYCONFIG
)"
# Keep endpoint identity user-supplied; never silently use another site's UUID.
if [[ -z "${HPC_GLOBUS_ENDPOINT_ID:-}" ]]; then
    echo 'Endpoint UUID is not registered yet. Start loam-julich-jupiter yourself, then source this file again before starting MCP.' >&2
fi
