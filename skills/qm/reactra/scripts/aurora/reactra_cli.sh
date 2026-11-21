#!/bin/bash
# Wrapper for the Reactra checkout's uv venv under $LOAM_SOFTWARE_ROOT (control/software layer).
# The uv venv lacks mech-format; it is provided by an overlay you build OUTSIDE the
# software tree (see references/aurora.md "One-time setup"). Usage: reactra_cli.sh <subcommand> [args...]
set -e
if [[ -z "${LOAM_SOFTWARE_ROOT:-}" ]]; then
  echo "reactra_cli.sh: LOAM_SOFTWARE_ROOT is unset; it comes from the Site profile." >&2
  exit 2
fi
if [[ -z "${REACTRA_MECH_OVERLAY:-}" ]]; then
  echo "reactra_cli.sh: REACTRA_MECH_OVERLAY is unset." >&2
  echo "  Build the mech-format overlay in a directory you own and export the path," >&2
  echo "  as described under 'One-time setup' in this skill. There is no shared copy:" >&2
  echo "  without it every MECH-dependent command fails with 'MECH inspection" >&2
  echo "  requires mech-format'." >&2
  exit 2
fi
if [[ ! -d "$REACTRA_MECH_OVERLAY" ]]; then
  echo "reactra_cli.sh: REACTRA_MECH_OVERLAY does not exist: $REACTRA_MECH_OVERLAY" >&2
  echo "  Build it (see 'One-time setup'), including the ../schema symlink." >&2
  exit 2
fi
export PYTHONPATH="$REACTRA_MECH_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "$LOAM_SOFTWARE_ROOT/reactra/.venv/bin/reactra" "$@"
