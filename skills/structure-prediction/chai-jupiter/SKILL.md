---
name: chai-jupiter
description: Use chai on Julich Jupiter ARM64 GH200 nodes with its dedicated software/chai-lab/.venv environment.
---

# chai on Jupiter

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/chai-lab"
PY="$REPO/.venv/bin/python"
"$REPO/.venv/bin/chai-lab" fold input.fasta output --num-trunk-recycles 1 --num-diffn-timesteps 50 --num-diffn-samples 1 --seed 42
```

Use entity FASTA headers such as >protein|name=target. First inference downloads model assets to CHAI_DOWNLOADS_DIR. Use an empty output directory. These reduced settings test plumbing; use production defaults for scientific conclusions.

Run campaign scientific workloads through `run_batch(gpus=1)` after staging with `write_files`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

## Packaged references through skill_view

These files are inside this skill and work without access to a sibling skill or installed documentation tree:

- [Jupiter workflow](references/workflow.md): `skill_view(name="chai-jupiter", file_path="references/workflow.md")`.
- [Original API/provenance notes](references/aurora-notes.md): `skill_view(name="chai-jupiter", file_path="references/aurora-notes.md")`. This is historical Aurora material; use the Jupiter workflow for execution.
- [Supporting-file index](references/resource-index.md): `skill_view(name="chai-jupiter", file_path="references/resource-index.md")`. It lists exact paths for every packaged helper and input example.

`file_path` is relative to this skill directory. Do not use `../`, invent an unlisted path, or assume files in the installed checkout are skill references. Original helper sources and examples are packaged under `references/aurora-resources/` for reading, not as Jupiter launchers. Platform-specific instructions in this SKILL.md and `references/workflow.md` take precedence over historical references.

Explicitly activate this tool's venv before invoking upstream shell wrappers. Preserve assigned GPU visibility, run one scientific task per remote call, and stream logs immediately. Current extended test limits are 6000-second payloads, 6300-second MCP requests and 14400-second outer Hermes tool deadlines; check remaining Slurm walltime before starting long work.

Check `$HPC_LOAM_HOME/JUPITER.md` and `$HPC_LOAM_HOME/ports/jupiter/software-smoke-status.json` for manual workflow readiness. The older `ports/jupiter/status.json` is an import/package inventory. Those are checkout files, not paths within this skill. Import/CLI success and readable references alone do not establish successful scientific execution.

## Manual smoke workload

[Loam-style smoke test](references/smoke-test.md): `skill_view(name="chai-jupiter", file_path="references/smoke-test.md")`. Run it manually after installation or changes; do not count import/CLI checks as a workflow pass.

[Exact smoke recipe](references/smoke-workload.py): `skill_view(name="chai-jupiter", file_path="references/smoke-workload.py")`. This is a self-contained snapshot of this tool's manual recipe, with public-input provenance and output assertions. Run it through MCP with this tool's own Python, `--out <fresh-workdir> --software <software-root>`; the suite additionally checks the login-side Argo answer for rag-query.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
