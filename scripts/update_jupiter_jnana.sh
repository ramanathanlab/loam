#!/usr/bin/env bash
# Update an existing Jupiter Jnana install; dependencies are unchanged at this pin.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
software="${LOAM_SOFTWARE_ROOT:-$(dirname "$repo")/software}"
source_dir="$software/jnana-engine"
revision=4c4cdfb42bf042a462eaf091dfbdf24a2ce34598
test -x "$source_dir/.venv/bin/python"
git -C "$source_dir" diff --quiet
git -C "$source_dir" diff --cached --quiet
GIT_TERMINAL_PROMPT=0 git -C "$source_dir" fetch --no-tags origin "$revision"
git -C "$source_dir" merge-base --is-ancestor HEAD "$revision"
git -C "$source_dir" merge --ff-only "$revision"
"$source_dir/.venv/bin/python" - <<'PY'
import inspect
from jnana.protognosis.core.coscientist import CoScientist
from jnana.protognosis.periodic_tournament import run_incremental_tournament
inspect.signature(CoScientist.generate_hypotheses).bind(None, count=2, batch_size=2)
inspect.signature(run_incremental_tournament).bind(
    state_file="unused.json", llm_config=None, scheme="swiss", rounds=2, workers=2)
print("Jnana supports batched generation, Swiss tournaments and parallel matches")
PY
