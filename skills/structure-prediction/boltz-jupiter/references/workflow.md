# Boltz inference on Jupiter

Use the native `software/boltz/.venv/bin/boltz` CLI with CUDA (`--accelerator gpu --devices 1 --num_workers 0 --no_kernels`), not the XPU launcher in the historical bundle. Set `BOLTZ_CACHE` using `jupiter_env.sh` and verify required checkpoints/assets; old Aurora cache locations do not imply local availability.

## Input and execution

Prefer the documented version-1 YAML schema with a `sequences` list and unique entity IDs. Protein entries contain sequence and an explicit MSA choice. `msa: empty` requests single-sequence input; supplied MSA paths or `--use_msa_server` must match the intended experiment. Ligands use a supported SMILES or CCD representation. The bundled sample YAML/FASTA files are examples, not campaign-specific scientific results.

```bash
"$REPO/.venv/bin/boltz" predict "$INPUT_YAML" \
  --out_dir "$OUT" --cache "$BOLTZ_CACHE" \
  --accelerator gpu --devices 1 --num_workers 0 --no_kernels
```

Select recycling/sampling parameters explicitly from the approved campaign budget and installed CLI. Use a fresh output directory; do not overwrite prior predictions merely to force a rerun. Network MSA or model-asset retrieval uses inherited login-proxy settings; keep TLS verification on.

## Results

Inspect the actual predicted PDB/mmCIF and confidence/NPZ files under `boltz_results_*/predictions/`. Verify all requested chains and ligands are present. Compare confidence scales and field names in the actual installed version rather than assuming Chai's scale. Preserve input/checkpoint IDs, seeds, MSA choice and failures.

Boltz's optional protein–ligand affinity head is not a protein-binder MM/GBSA calculation or measured binding affinity. Do not report it as an EGFR protein-binder free energy. Raw examples and historical outputs are available in the resource index, with Aurora-only launch scripts labeled accordingly.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For current extended tests, scientific tasks may use `run_code(timeout=6000)` or independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.

Jupiter's installed Boltz lacks the optional `cuequivariance_torch` triangle kernels. Its default prediction path fails at inference with `ModuleNotFoundError`, despite successful input preparation and CUDA detection. Use the supported `--no_kernels` flag to run the native PyTorch implementation on CUDA, and `--num_workers 0` for the single-case loader. This preserves the trained model; it does not replace inference with a mock. The manual smoke suite uses these flags.
