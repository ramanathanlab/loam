#!/bin/bash
# Wrapper for the msinclair Reactra checkout's uv venv (control/software layer).
# The uv venv lacks mech-format; it is provided by an overlay installed OUTSIDE the
# software tree (see SKILL.md "One-time setup"). Usage: reactra_cli.sh <subcommand> [args...]
set -e
OVERLAY=${REACTRA_MECH_OVERLAY:-/lus/flare/projects/FRAME-IDP/msinclair/testing_grounds/env_tests/test_runs/reactra/mech_overlay}
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec /flare/FRAME-IDP/msinclair/software/reactra/.venv/bin/reactra "$@"
