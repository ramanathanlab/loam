---
name: rfd4-proteina-jupiter
description: Generate protein designs and target-conditioned binders with RFD4-Proteina on Jupiter ARM64 GH200 workers, using its dedicated venv, native runtime libraries, and correct Hydra configuration groups.
---

# RFD4-Proteina on Jupiter

Use the dedicated `software/RFD4-Proteina/.venv` through HPC MCP. Run all scientific commands with `mcp__hpc__run_code` or `run_batch(gpus=1)`; preserve Ensemble Launcher's `CUDA_VISIBLE_DEVICES` and use `cuda:0` inside a one-GPU task. Write artifacts under the assigned campaign workspace. Do not use Aurora modules, XPU settings, ALCF proxies, SSH, or scheduler submissions from a campaign agent.

## Read these packaged references

These are real files inside this skill, accessible with `skill_view`:

- [Protein-binder workflow](references/tasks/protein-binder.md): `skill_view(name="rfd4-proteina-jupiter", file_path="references/tasks/protein-binder.md")`.
- [Annotated-CIF construction](references/annotated-cif.md): `skill_view(name="rfd4-proteina-jupiter", file_path="references/annotated-cif.md")`.

Their paths are relative to this skill, not to the installed software checkout. Upstream files named within a reference must be read at the indicated checkout path through MCP; they are not implied `skill_view` paths.

## Environment

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/RFD4-Proteina"
source "$REPO/.venv/bin/activate"
export CLUSTER=local WANDB_MODE=disabled OMP_NUM_THREADS=4
cd "$REPO"
```

Activate this venv explicitly before upstream shell wrappers: they otherwise preserve an unrelated active environment such as Globus's Python. Prefer absolute `.venv/bin/python` and `.venv/bin/rfd4-generate` paths. Read the checkout's `AGENTS.md` for model-specific constraints. RFdiffusion3 is not an available campaign tool on this port.

## Correct Hydra invocation for a protein binder

After constructing and validating a real target-conditioned CIF, set `INPUT` and `OUT` to absolute campaign workspace paths; `OUT` must be fresh:

```bash
"$REPO/.venv/bin/rfd4-generate" \
  inputs="$INPUT" \
  out_dir="$OUT" \
  recycle_strategy=medium/front_loaded \
  num_replicates=1 \
  batch_size=1 \
  compile_lengths=null \
  num_workers=0 \
  save_chain_of_interest=true \
  seed=42
```

**Use `recycle_strategy=medium/front_loaded` exactly.** This selects the existing Hydra configuration group `configs/recycle_strategy/medium/front_loaded.yaml`, which defines the actual sampling schedule. Do not use `ppi.recycle_strategy=medium` or `+ppi.recycle_strategy=medium`: the first targets an absent field; the second adds an unused field instead of selecting the schedule. Do not blindly add `+` to a rejected override. Inspect `configs/inference.yaml` and the named group file when diagnosing configuration errors.

The example requests one design at an explicit seed, batch size one, and eager execution. Keep the campaign's agreed settings and record effective overrides, checkpoint revision, input checksum, and output files. Do not silently substitute sampling presets or invent binder sequences/metrics.

## Task lifetime and logs

Use `subprocess.run(..., timeout=6000)` and `run_code(..., timeout=6000)`, or one independent design per `run_batch(gpus=1, timeout_each=6000)` row. Do not run multiple sequential 6000-second commands under one 6000-second remote call. Stream stdout/stderr directly to files while the command runs; do not rely on `capture_output=True` followed by a write only after completion. Use the assigned workdir. Queue waiting is separate from payload execution; do not cancel queued work merely because it waits.

Current test settings use MCP request deadlines of 6300 seconds and outer Hermes tool deadlines of 14400 seconds. These do not extend Slurm walltime: check sufficient remaining allocation time before starting long inference. Read named docs directly rather than recursively scanning `.venv`, caches, or the software tree.

## Native runtime installation

On Jupiter, run `$HPC_LOAM_HOME/scripts/install_jupiter_rfd4_runtime.sh` after installing RFD4. This downloads the pinned ARM64 X11 dependencies required by Open Babel into `software/RFD4-Proteina/runtime/`, independent of AmberTools. `jupiter_env.sh` adds its library directory to `LD_LIBRARY_PATH`. Login-node system libraries are not representative of compute nodes.

See `JUPITER.md` and `ports/jupiter/rfd4-runtime-validation.json` for installation and the manual compute-node import/CLI/generation validation. These are not extra campaign preflight gates. An unconditional model smoke test does not establish successful EGFR binder generation or binding affinity.

On Jupiter use `num_workers=0` for single-design inference: PyTorch loads data in the main process rather than spawning four loader subprocesses. This matches the successful manual installation test and avoids oversubscription under one-core task affinity. The runtime installer configures this default via `scripts/configure_jupiter_rfd4.py`; keep the explicit CLI override for auditable runs. It changes data-loading parallelism, not the intended model recycle schedule.

## Manual smoke workload

[Loam-style smoke test](references/smoke-test.md): `skill_view(name="rfd4-proteina-jupiter", file_path="references/smoke-test.md")`. Run it manually after installation or changes; do not count import/CLI checks as a workflow pass.

[Exact smoke recipe](references/smoke-workload.py): `skill_view(name="rfd4-proteina-jupiter", file_path="references/smoke-workload.py")`. This is a self-contained snapshot of this tool's manual recipe, with public-input provenance and output assertions. Run it through MCP with this tool's own Python, `--out <fresh-workdir> --software <software-root>`; the suite additionally checks the login-side Argo answer for rag-query.
