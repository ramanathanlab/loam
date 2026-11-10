#!/usr/bin/env bash
# Install the native libraries required by RFD4's Open Babel wheel on compute nodes.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
software="${LOAM_SOFTWARE_ROOT:-$(dirname "$repo")/software}"
if [[ "$(uname -m)" != aarch64 || "$(uname -s)" != Linux ]]; then
    echo 'This lock targets Linux ARM64 Jupiter nodes.' >&2
    exit 1
fi
mamba="${MICROMAMBA_EXE:-$software/micromamba/bin/micromamba}"
if [[ ! -x "$mamba" ]]; then
    echo "Install micromamba at $mamba or set MICROMAMBA_EXE to its executable." >&2
    exit 1
fi
prefix="$software/RFD4-Proteina/runtime"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$software/micromamba/root}"
"$mamba" create --yes --prefix "$prefix" --file "$repo/ports/jupiter/rfd4-runtime-linux-aarch64.explicit.txt"
test -e "$prefix/lib/libXrender.so.1"
test -e "$prefix/lib/libX11.so.6"
test -e "$prefix/lib/libXext.so.6"
python3 "$repo/scripts/configure_jupiter_rfd4.py" --software-root "$software"
printf 'RFD4 native runtime installed at %s\nSource %s/scripts/jupiter_env.sh before RFD4 commands.\n' "$prefix" "$repo"
