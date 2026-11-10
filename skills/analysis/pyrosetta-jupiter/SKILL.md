---
name: pyrosetta-jupiter
description: Unavailable until a licensed ARM64 PyRosetta runtime is supplied; the Jupiter venv currently contains only its installer.
---

# pyrosetta on Jupiter

**Unavailable until a licensed ARM64 PyRosetta runtime is supplied.** The installed `pyrosetta-installer` is not a scoring engine; do not select this skill for an executable campaign stage yet.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` before using this recipe. Software is installed beside the Loam checkout in `software/`, with a `.venv` inside each software directory.

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/pyrosetta"
PY="$REPO/.venv/bin/python"
"$PY" -c "import pyrosetta; pyrosetta.init(); print(pyrosetta.version())"
```

CPU scoring. A pyrosetta-installer installation is not PyRosetta itself: obtain a licensed ARM64 distribution and verify import before scoring. Do not submit the Aurora PBS example.

Run GPU workloads through `mcp__hpc__run_code` or `run_batch(gpus=1)`. Jupiter has four GH200 GPUs per node. Keep the CUDA_VISIBLE_DEVICES assigned by Ensemble Launcher; inside a one-GPU task use cuda:0. Write outputs to the campaign workspace. Do not use Aurora modules, ZE_AFFINITY_MASK, XPU launchers, ALCF proxies, or PBS submissions.

## Packaged references through skill_view

These files are inside this skill and work without access to a sibling skill or installed documentation tree:

- [Jupiter workflow](references/workflow.md): `skill_view(name="pyrosetta-jupiter", file_path="references/workflow.md")`.
- [Original API/provenance notes](references/aurora-notes.md): `skill_view(name="pyrosetta-jupiter", file_path="references/aurora-notes.md")`. This is historical Aurora material; use the Jupiter workflow for execution.
- [Supporting-file index](references/resource-index.md): `skill_view(name="pyrosetta-jupiter", file_path="references/resource-index.md")`. It lists exact paths for every packaged helper and input example.

`file_path` is relative to this skill directory. Do not use `../`, invent an unlisted path, or assume files in the installed checkout are skill references. Original helper sources and examples are packaged under `references/aurora-resources/` for reading, not as Jupiter launchers. Platform-specific instructions in this SKILL.md and `references/workflow.md` take precedence over historical references.

Explicitly activate this tool's venv before invoking upstream shell wrappers. Preserve assigned GPU visibility, run one scientific task per remote call, and stream logs immediately. Current extended test limits are 6000-second payloads, 6300-second MCP requests and 14400-second outer Hermes tool deadlines; check remaining Slurm walltime before starting long work.

Check `$HPC_LOAM_HOME/JUPITER.md` and `$HPC_LOAM_HOME/ports/jupiter/software-smoke-status.json` for manual workflow readiness. The older `ports/jupiter/status.json` is an import/package inventory. Those are checkout files, not paths within this skill. Import/CLI success and readable references alone do not establish successful scientific execution.

## Manual smoke workload

[Loam-style smoke test](references/smoke-test.md): `skill_view(name="pyrosetta-jupiter", file_path="references/smoke-test.md")`. Run it manually after installation or changes; do not count import/CLI checks as a workflow pass.

[Exact smoke recipe](references/smoke-workload.py): `skill_view(name="pyrosetta-jupiter", file_path="references/smoke-workload.py")`. This is a self-contained snapshot of this tool's manual recipe, with public-input provenance and output assertions. Run it through MCP with this tool's own Python, `--out <fresh-workdir> --software <software-root>`; the suite additionally checks the login-side Argo answer for rag-query.
