# Manual Jupiter workflow validation

Read this skill's `references/smoke-workload.py` using `read_skill(name, file="references/smoke-workload.py")`. Stage that file into a fresh hypothesis run directory and invoke it with the tool's dedicated Python through the campaign's `run_batch` interface. Source the absolute `scripts/jupiter_env.sh` first and pass `--out <fresh-absolute-directory> --software "$LOAM_SOFTWARE_ROOT"`. Preserve assigned CUDA visibility, poll batch completion, and inspect `result.json` plus scientific outputs.

These recipes use reduced settings and real model/runtime paths. They verify installation plumbing and artifact generation; they do not establish binding efficacy or constitute current campaign evidence. Recorded results below come from the neighboring Jupiter port, not a fresh run of this checkout.

## Latest recorded manual validation

Status: **passed**. Workload: Single-sequence CUDA folding with native PyTorch kernels.

Evidence directory: `$LOAM_SOFTWARE_ROOT/verification/jupiter-skills-boltz-final-20261110/boltz`. Preserve failed attempts; the root `summary.json` records this run.

This status describes only the tested path and cache/auth state at audit time. Revalidate after environment or checkpoint changes.
