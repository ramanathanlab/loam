#!/usr/bin/env bash
# Install CUDA support in both isolated MD environments; no campaign preflight.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
software="${LOAM_SOFTWARE_ROOT:-$(dirname "$repo")/software}"
uv_bin="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$uv_bin" ]]; then
    uv_bin="$software/RFD4-Proteina/.venv/bin/uv"
fi
test -x "$uv_bin"
for name in openmm molecular-simulations; do
    python="$software/$name/.venv/bin/python"
    test -x "$python"
    "$uv_bin" pip install --python "$python" -r "$repo/ports/jupiter/openmm-cuda-requirements.txt"
done
echo 'Run the openmm and molecular-simulations MCP smoke cases on a compute node.'
