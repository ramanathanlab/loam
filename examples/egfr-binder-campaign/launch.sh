#!/usr/bin/env bash
# Launch campaign1: EGFR binder design, execution.mode remote_mcp on Aurora.
#
# This wrapper only sets the variables this launch needs. Everything about WHAT
# the campaign does lives in files: the `campaign1` profile in ~/.hermes/loam.yaml,
# campaigns/binder-design.md, and rules/.
#
#   LOAM_REPO             run-campaign was COPIED here, so it cannot infer the
#                         checkout and must be told.
#   LOAM_PREFLIGHT_PYTHON interpreter for the readiness gate. It must import
#                         fastmcp: the gate's check_hpc opens a FastMCP Client
#                         against :8000, and hpc_control launches the local
#                         mcp_server with sys.executable -- the SAME interpreter
#                         -- so this one also needs academy and
#                         globus_compute_sdk. A stdlib-only python gets through
#                         the inference check and then fails on the HPC probe.
#   HPC_ENV_FILE          REQUIRED. run-campaign defaults this to
#                         skills/hpc-deployment/set_env.sh, which is a DIFFERENT
#                         endpoint (3724d6df...). Without this the preflight
#                         would start an allocation on somebody else's endpoint.
#   ALCF_PROXY=0          the ranking judge runs on the Argo tunnel, not the
#                         :51911 ALCF proxy, so there is nothing to start.
#
# NOT set, deliberately: LOAM_SKIP_PREFLIGHT. On a remote campaign the gate is
# what starts the HPC allocation and proves one real run_code round-trip;
# run-campaign's own docs call skipping it "launcher diagnostics only, never a
# scientific campaign."

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export LOAM_REPO=/Users/msinclair/github/loam
# The campaign's own uv venv, which carries fastmcp, academy-py,
# globus-compute-sdk, parsl, dill, yaml and jinja2. This replaces the
# ~/.loam-venv-312 that run-campaign defaults to and that does not exist here.
export LOAM_PREFLIGHT_PYTHON="${LOAM_PREFLIGHT_PYTHON:-$PWD/.venv/bin/python}"
export HPC_ENV_FILE="$LOAM_REPO/skills/hpc-deployment/set_env_msinclair.sh"
export ALCF_PROXY=0

fail() { echo "launch.sh: $1" >&2; shift; for l in "$@"; do echo "           $l" >&2; done; exit 1; }

# 1. The Argo tunnel: generation AND ranking both run on it. Fail loudly rather
#    than let Jnana report "generated_hypotheses: 0, status: completed".
curl -fsS -m 5 http://localhost:61612/v1/models >/dev/null 2>&1 ||
  fail "no Argo tunnel answering on http://localhost:61612" "start the ssh tunnel, then re-run."

# 2. The preflight interpreter must be able to run the local MCP server, not
#    just the stdlib parts of the gate.
[[ -x "$LOAM_PREFLIGHT_PYTHON" ]] ||
  fail "preflight Python is not executable: $LOAM_PREFLIGHT_PYTHON" \
       "build it, or export LOAM_PREFLIGHT_PYTHON at a env that has" \
       "fastmcp, academy-py and globus-compute-sdk installed."
"$LOAM_PREFLIGHT_PYTHON" -c 'import fastmcp, academy, globus_compute_sdk' 2>/dev/null ||
  fail "$LOAM_PREFLIGHT_PYTHON cannot import fastmcp/academy/globus_compute_sdk" \
       "hpc_control launches mcp_server.py with this same interpreter, so the" \
       "HPC server cannot start without them."

# 3. The env file must be the one for THIS endpoint, fully filled.
[[ -r "$HPC_ENV_FILE" ]] || fail "HPC env file missing: $HPC_ENV_FILE"
! grep -q "FILL_IN" "$HPC_ENV_FILE" ||
  fail "$HPC_ENV_FILE still has FILL_IN markers" "fill them before burning reservation time."

exec ./run-campaign "$@"
