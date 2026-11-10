# Protenix inference on Jupiter

Use `software/protenix/.venv/bin/protenix` and the native CUDA installation. Source `jupiter_env.sh` for `PROTENIX_ROOT_DIR` and `LAYERNORM_TYPE=torch`; the latter avoids eager compilation of an optional fused layernorm. Inspect the installed `pred --help` and the exact model configuration before choosing kernels or model names.

```bash
"$REPO/.venv/bin/protenix" pred --help
```

Inference accepts an input JSON list. Each task has a unique `name` and a `sequences` list; a proteinChain contains its sequence and count. Provide authoritative campaign sequences and record entity/chain identity. The bundled example JSON is a historical input fixture, not a newly designed candidate.

Run in the campaign workspace: embeddings and other caches may be written relative to CWD. Set model, seed, sample count, recycle count, diffusion steps and MSA/template usage explicitly using the installed CLI. Confirm the selected checkpoint and chemical-component assets are present before relying on offline execution. Native torch triangle kernels are an option when fused kernels are unavailable; validate the installed option names rather than copying obsolete flags.

Inspect actual predicted mmCIF and confidence JSON under the model's output tree, including expected entities and pLDDT/pTM/ipTM where available. Import/CLI checks do not validate inference. The original Aurora fork's kernel requirements, oneAPI workaround, checkpoints, historical fixes and timings do not establish Jupiter behavior. The pinned ARM64 Triton override is recorded in `ports/jupiter/protenix-overrides.txt` outside this skill.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For current extended tests, scientific tasks may use `run_code(timeout=6000)` or independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.

## Validated small CUDA recipe

The manual Jupiter smoke test used an authoritative 20-residue 1L2Y sequence,
`proteinChain` JSON, and the following native command:

```bash
"$REPO/.venv/bin/protenix" pred -i "$INPUT_JSON" -o "$OUT" \
  -n protenix_mini_esm_v0.5.0 -s 42 -c 1 -p 20 -e 1 \
  --use_msa false --trimul_kernel torch --triatt_kernel torch
```

This mini ESM model still needs the large `esm2_t36_3B_UR50D.pt` checkpoint,
its own folding checkpoint, and common CCD/cluster assets. The first cold run
including downloads took about 14 minutes; do not diagnose this as a GPU or
DataLoader stall solely from a quiet log. Inspect native download progress.
After interruption, verify or replace incomplete cache files before retrying;
the installed downloader checks file existence rather than full content for
common assets. Retain successful caches. The smoke produced a predicted CIF
with exactly 20 residues, 154 atoms and finite coordinates. This validates this
reduced single-sequence path, not arbitrary model presets or production accuracy.
