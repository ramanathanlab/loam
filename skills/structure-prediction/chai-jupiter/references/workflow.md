# Chai co-folding on Jupiter

Use the native `software/chai-lab/.venv/bin/chai-lab` CLI, not the historical Aurora `chai_xpu.py` launcher. Cache assets under `CHAI_DOWNLOADS_DIR` from `jupiter_env.sh`; asset downloads or cold filesystem reads can dominate startup.

## Input and command

Provide entity FASTA with unique headers such as `>protein|name=target` and `>protein|name=binder`. Use authoritative target sequence and an actually generated/designed binder sequence. Never fabricate a binder for a folding test. Retain the target crop and chain mapping used for generation.

```bash
"$REPO/.venv/bin/chai-lab" fold "$INPUT_FASTA" "$OUT" \
  --num-trunk-recycles 1 --num-diffn-timesteps 50 \
  --num-diffn-samples 1 --seed 42
```

These reduced settings are for a campaign that explicitly requests a smoke test; preserve the agreed settings for production work. `OUT` must be empty or fresh. Select MSA/template use explicitly according to the experiment; network access uses inherited proxy settings. Preserve the assigned GPU rather than hard-coding a physical device.

## Results

Inspect `pred.model_idx_*.cif` and `scores.model_idx_*.npz`. Check both intended entities are present with their expected sequences and chain mapping; a process return code alone is insufficient. Extract available pTM, ipTM, per-chain confidence and clash metrics from actual files, and binder pLDDT from its actual atoms. Handle boolean arrays separately from numeric rounding. Confirm ligand/entity presence if applicable; tokenization errors can invalidate an apparently completed prediction.

Confidence and interface metrics are not binding free energy or measured affinity. Historical XPU timing, patches, defaults and cache-presence claims in the bundled Aurora notes are not Jupiter validation.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For current extended tests, scientific tasks may use `run_code(timeout=6000)` or independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.
