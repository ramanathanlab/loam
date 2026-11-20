---
name: reactra
description: Unavailable on Jupiter: Reactra and mech-format sources and native providers are not installed.
---

# Reactra on Jupiter

No Reactra checkout or runtime is installed in `$LOAM_SOFTWARE_ROOT`. Do not select it for executable campaign stages. Exact source and native dependencies must be supplied first.

## Executable reference

Read `references/smoke-workload.py` with `read_skill(name="reactra", file="references/smoke-workload.py")` for a concrete Jupiter workflow with output assertions. Stage its contents into the assigned workspace before execution, using this tool's Python and `--out <absolute-output-directory> --software $LOAM_SOFTWARE_ROOT`. Reduced smoke settings validate plumbing; choose campaign settings for scientific evidence.
