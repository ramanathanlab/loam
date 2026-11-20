#!/usr/bin/env bash
# Jupiter campaign launcher. ./launch.sh --check validates without submitting work.
set -eo pipefail
LOAM_ROOT=${LOAM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
CONFIG=${CONFIG:-$LOAM_ROOT/loam.yaml}
PYTHON=${LOAM_PYTHON:-$LOAM_ROOT/.venv/bin/python}
[[ -x "$PYTHON" ]] || { echo "Install first: python3.12 -m venv $LOAM_ROOT/.venv; $LOAM_ROOT/.venv/bin/python -m pip install -e '$LOAM_ROOT[dev,hpc]'" >&2; exit 1; }
source "$LOAM_ROOT/scripts/jupiter_env.sh"
# Model clients run on the internet-connected login node. Its shared forward
# proxy has a 10s idle cutoff, shorter than Tara's non-streaming responses.
# Compute workers still source jupiter_env.sh and retain their egress proxy.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}inference-api.alcf.anl.gov"
export no_proxy="${no_proxy:+$no_proxy,}inference-api.alcf.anl.gov"
export HPC_MIN_BATCH_ROWS=${HPC_MIN_BATCH_ROWS:-1}
export HPC_MIN_BATCH_DISTINCT=${HPC_MIN_BATCH_DISTINCT:-1}
# The typed HPC config propagates queue/startup allowances to the server.
export ALCF_TOKENS_BIN=${ALCF_TOKENS_BIN:-$LOAM_ROOT/.auth-venv/bin/alcf-tokens}
# Reuse the existing internal exchange without embedding credentials in YAML.
if [[ -f "$LOAM_SOFTWARE_ROOT/redis/env.sh" ]]; then
    source "$LOAM_SOFTWARE_ROOT/redis/env.sh"
    export ACADEMY_EXCHANGE_TYPE=${ACADEMY_EXCHANGE_TYPE:-redis}
fi
# Optional refresh for retained ALCF model routes; existing exported tokens work too.
if [[ "${1:-}" != --check && -z "${ALCF_INFERENCE_TOKEN:-}" ]] &&
   "$PYTHON" - "$CONFIG" <<'PYCONFIG'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
specs = [*c['models'].values(), c['tournament']['generation'], c['tournament']['ranking']]
sys.exit(0 if any(s.get('api_key_env') == 'ALCF_INFERENCE_TOKEN' for s in specs) else 1)
PYCONFIG
then
    if [[ -n "${AUTH_HELPER:-}" ]]; then
        ALCF_INFERENCE_TOKEN="$("${AUTH_PY:-$PYTHON}" "$AUTH_HELPER" get_access_token)"
    else
        TOKENS=${ALCF_TOKENS_BIN:-$LOAM_ROOT/.auth-venv/bin/alcf-tokens}
        [[ -x "$TOKENS" ]] || { echo "Install alcf-tokens in .auth-venv; see JUPITER.md" >&2; exit 1; }
        ALCF_INFERENCE_TOKEN="$("$TOKENS" get-token inference)" || {
            echo "Authenticate first: $TOKENS login inference" >&2; exit 1;
        }
    fi
    [[ -n "$ALCF_INFERENCE_TOKEN" ]] || { echo "ALCF inference token is empty" >&2; exit 1; }
    export ALCF_INFERENCE_TOKEN
fi
export PYTHONPATH="$LOAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$LOAM_ROOT/scripts/preflight.py" "$CONFIG" "$@"
[[ "${1:-}" != --check ]] || exit 0
cd "$(dirname "$CONFIG")"
mkdir -p logs
LOG="logs/run-$(date +%Y%m%dT%H%M%S).log"
ln -sf "$(basename "$LOG")" logs/latest.log
"$PYTHON" -u -m loam "$CONFIG" 2>&1 | tee "$LOG"
