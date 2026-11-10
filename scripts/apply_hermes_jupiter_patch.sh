#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hermes_source="${1:-$(dirname "$repo")/software/hermes-agent}"
patch="$repo/ports/jupiter/hermes-parent-agent.patch"
if git -C "$hermes_source" apply --reverse --check "$patch" 2>/dev/null; then
    echo "Hermes gateway parent-agent patch is already applied."
elif git -C "$hermes_source" apply --check "$patch"; then
    git -C "$hermes_source" apply "$patch"
    echo "Applied Hermes gateway parent-agent patch; use a fresh campaign gateway."
else
    echo "Hermes source is incompatible with the patch; no files changed." >&2
    exit 1
fi
