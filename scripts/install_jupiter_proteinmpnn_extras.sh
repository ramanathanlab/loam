#!/usr/bin/env bash
# Add structure-conversion support without changing ProteinMPNN's model stack.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
software="${LOAM_SOFTWARE_ROOT:-$(dirname "$repo")/software}"
python="$software/proteinmpnn/.venv/bin/python"
test -x "$python"
uv_bin="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$uv_bin" ]]; then
    uv_bin="$software/RFD4-Proteina/.venv/bin/uv"
fi
if [[ ! -x "$uv_bin" ]]; then
    echo 'Install uv or set UV_BIN to its executable.' >&2
    exit 1
fi
"$uv_bin" pip install --python "$python" -r "$repo/ports/jupiter/proteinmpnn-conversion-requirements.txt"
"$python" -c 'import gemmi; print("ProteinMPNN conversion dependency ready: gemmi", gemmi.__version__)'
