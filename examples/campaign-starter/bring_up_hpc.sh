#!/usr/bin/env bash
#
# bring_up_hpc.sh -- self-service bring-up of the loam HPC plane on CSCS Alps.
#
# Does, in order, what you otherwise ask Claude to do by hand:
#   1. ensure you are logged in to Globus Compute (and warn if the Academy
#      exchange login cache is missing -- that one is interactive);
#   2. CREATE the templatable Globus Compute endpoint from the repo config if it
#      does not exist yet, else START/RESTART it (auto-clearing a stale
#      daemon.pid left by a crash);
#   3. write the endpoint's UUID back into the HPC env file so the MCP server
#      targets the right endpoint (important after a fresh create -- a new
#      registration mints a NEW UUID);
#   4. start the local FastMCP server (run_code / run_batch / ...), foreground.
#
# Run it yourself:
#   ./bring_up_hpc.sh            # endpoint up + MCP server (default: 'up')
#   ./bring_up_hpc.sh endpoint   # endpoint only, no server
#   ./bring_up_hpc.sh server     # MCP server only (endpoint assumed up)
#   ./bring_up_hpc.sh status     # show endpoint + whether :8000 answers
#   ./bring_up_hpc.sh stop       # stop the endpoint (server is Ctrl-C)
#
# Per-launch allocation overrides pass straight through the env file, e.g.:
#   HPC_QUEUE=normal HPC_WALLTIME=04:00:00 HPC_NNODES=2 ./bring_up_hpc.sh
#
# NOTE: SLURM allows ONE running job per user. Stop a previous server
# (Ctrl-C) and let its block drain before bringing up a new one on a different
# shape. The MCP server runs in the FOREGROUND; Ctrl-C triggers a clean EL
# shutdown. The Globus endpoint keeps running after the server exits -- use
# 'stop' to take it down, or leave it up for the next run.

set -euo pipefail

# =========================================================================
# CONFIG -- edit these for your checkout / endpoint. Absolute on purpose, so
# this script works wherever it is copied.
# =========================================================================
LOAM_REPO="${LOAM_REPO:-/ritom/scratch/cscs/avasan/loam}"
LOAM_VENV="${LOAM_VENV:-/ritom/scratch/cscs/avasan/loam_venv}"
ENDPOINT_NAME="${ENDPOINT_NAME:-loam-cscs-alps}"
ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-loam-cscs-alps-flex}"
CONTACT_EMAIL="${CONTACT_EMAIL:-avasan@anl.gov}"
# The operator env file the MCP server sources (HPC_* knobs). Its
# HPC_GLOBUS_ENDPOINT_ID line is rewritten to match the live endpoint.
HPC_ENV_FILE="${HPC_ENV_FILE:-$LOAM_REPO/skills/hpc-deployment/set_env_cscs.sh}"
# =========================================================================

# ---- derived -----------------------------------------------------------
GCE="$LOAM_VENV/bin/globus-compute-endpoint"
PYTHON="$LOAM_VENV/bin/python"
CONFIG_SRC="$LOAM_REPO/globus_compute_endpoint_configs/cscs_alps"
HPC_DEPLOY_DIR="$LOAM_REPO/skills/hpc-deployment"
MCP_SERVER="$HPC_DEPLOY_DIR/scripts/mcp_server.py"
GC_HOME="${GLOBUS_COMPUTE_DIR:-$HOME/.globus_compute}"
GC_DIR="$GC_HOME/$ENDPOINT_NAME"
ACADEMY_DB="$HOME/.local/share/academy/storage.db"

say()  { printf '\033[1;36m[hpc]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[hpc WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[hpc ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

check_prereqs() {
  [[ -x "$GCE" ]]    || die "globus-compute-endpoint not found at $GCE (check LOAM_VENV)"
  [[ -x "$PYTHON" ]] || die "python not found at $PYTHON (check LOAM_VENV)"
  [[ -d "$CONFIG_SRC" ]] || die "endpoint config source dir missing: $CONFIG_SRC"
  [[ -r "$HPC_ENV_FILE" ]] || die "HPC env file missing/unreadable: $HPC_ENV_FILE"
}

require_login() {
  if ! "$GCE" whoami >/dev/null 2>&1; then
    die "not logged in to Globus Compute. Run interactively first:
         $GCE login"
  fi
  if [[ ! -f "$ACADEMY_DB" ]]; then
    warn "Academy exchange login cache not found: $ACADEMY_DB"
    warn "The MCP server needs a one-time interactive Globus login for Academy."
    warn "If the server errors on the exchange, do that login once, then re-run."
  fi
}

endpoint_exists() { [[ -f "$GC_DIR/config.yaml" ]]; }

clear_stale_pid() {
  # A crashed endpoint leaves a dead PID in daemon.pid, which makes
  # list/stop/restart crash in psutil. Remove it only when the PID is truly gone.
  local pidfile="$GC_DIR/daemon.pid" pid
  [[ -f "$pidfile" ]] || return 0
  pid="$(tr -dc '0-9' < "$pidfile" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    warn "stale daemon.pid (dead PID '${pid:-empty}') -> removing it"
    rm -f "$pidfile"
  fi
}

create_endpoint() {
  say "creating templatable endpoint '$ENDPOINT_NAME' from $CONFIG_SRC"
  "$GCE" configure \
    --multi-user true \
    --display-name "$ENDPOINT_DISPLAY_NAME" \
    --contact-email "$CONTACT_EMAIL" \
    --core-config      "$CONFIG_SRC/config.yaml" \
    --template-config  "$CONFIG_SRC/user_config_template.yaml.j2" \
    --schema-config    "$CONFIG_SRC/user_config_schema.json" \
    --user-env-config  "$CONFIG_SRC/user_environment.yaml" \
    --id-mapping-config "$CONFIG_SRC/identity_mapping_config.json" \
    "$ENDPOINT_NAME"
}

endpoint_up() {
  if endpoint_exists; then
    say "endpoint '$ENDPOINT_NAME' exists -> restart"
    clear_stale_pid
    "$GCE" restart "$ENDPOINT_NAME" || {
      warn "restart failed; trying a clean start"
      clear_stale_pid
      "$GCE" start "$ENDPOINT_NAME"
    }
  else
    create_endpoint
    say "starting freshly-created endpoint"
    "$GCE" start "$ENDPOINT_NAME"
  fi
  wait_for_endpoint_json
}

wait_for_endpoint_json() {
  local i
  for i in $(seq 1 30); do
    [[ -s "$GC_DIR/endpoint.json" ]] && return 0
    sleep 1
  done
  warn "endpoint.json not written after 30s; UUID sync may be skipped"
  return 1
}

sync_uuid() {
  [[ -s "$GC_DIR/endpoint.json" ]] || { warn "no endpoint.json; skipping UUID sync"; return 0; }
  local new_id cur_id
  new_id="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["endpoint_id"])' "$GC_DIR/endpoint.json")" \
    || { warn "could not read endpoint_id from endpoint.json"; return 0; }
  cur_id="$(sed -n -E 's/^export HPC_GLOBUS_ENDPOINT_ID="([^"]*)".*/\1/p' "$HPC_ENV_FILE" | head -1)"
  if [[ "$new_id" == "$cur_id" ]]; then
    say "endpoint UUID matches HPC env file ($new_id)"
    return 0
  fi
  warn "endpoint UUID changed: env='$cur_id' -> live='$new_id'; updating $HPC_ENV_FILE"
  cp -p "$HPC_ENV_FILE" "$HPC_ENV_FILE.bak.$(date +%Y%m%d%H%M%S)"
  sed -i -E 's|^(export HPC_GLOBUS_ENDPOINT_ID=").*(".*)$|\1'"$new_id"'\2|' "$HPC_ENV_FILE"
  say "wrote HPC_GLOBUS_ENDPOINT_ID=$new_id (backup saved alongside the env file)"
}

start_server() {
  [[ -f "$MCP_SERVER" ]] || die "MCP server not found: $MCP_SERVER"
  say "starting MCP server (foreground; Ctrl-C for a clean EL shutdown)"
  cd "$HPC_DEPLOY_DIR"
  # shellcheck disable=SC1090
  source "$HPC_ENV_FILE"
  export PYTHONPATH="$HPC_DEPLOY_DIR/scripts:${PYTHONPATH:-}"
  exec "$PYTHON" "$MCP_SERVER"
}

status() {
  say "endpoints known to globus-compute-endpoint:"
  "$GCE" list 2>&1 || true
  if [[ -s "$GC_DIR/endpoint.json" ]]; then
    say "endpoint.json id: $("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["endpoint_id"])' "$GC_DIR/endpoint.json" 2>/dev/null || echo '?')"
  fi
  if curl -fsS -m 3 http://127.0.0.1:8000/mcp >/dev/null 2>&1; then
    say "MCP server: ANSWERING on http://127.0.0.1:8000/mcp"
  else
    say "MCP server: not answering on :8000 (not started, or still coming up)"
  fi
}

stop_endpoint() {
  clear_stale_pid
  say "stopping endpoint '$ENDPOINT_NAME'"
  "$GCE" stop "$ENDPOINT_NAME" || warn "stop reported an error (may already be down)"
  warn "SLURM block may persist (parsl owns it). If a job lingers: scancel <jobid> after this."
}

main() {
  local cmd="${1:-up}"; [[ $# -gt 0 ]] && shift || true
  check_prereqs
  case "$cmd" in
    up)       require_login; endpoint_up; sync_uuid; start_server ;;
    endpoint) require_login; endpoint_up; sync_uuid ;;
    server)   require_login; start_server ;;
    status)   status ;;
    stop)     stop_endpoint ;;
    *) die "unknown command '$cmd'. Use: up | endpoint | server | status | stop" ;;
  esac
}

main "$@"
