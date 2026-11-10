# ESMFold2 on Jupiter

Use `software/esmfold2/.venv`, with the native `esm` package providing `esm.models.esmfold2.EsmFold2Model`. This differs from the Aurora transformers fork. Do not run the bundled Aurora example script as a Jupiter recipe or substitute ESMFold v1.

```bash
"$PY" -c 'from esm.models.esmfold2 import EsmFold2Model'
```

That command is an import diagnostic only. Actual inference requires CUDA and access to the exact `biohub/ESMFold2` and `biohub/ESMC-6B` assets. Confirm checkpoint permissions/cache completeness and inspect the installed package's model-loading/inference API before constructing a task; the historical Aurora Python/CLI signatures are not guaranteed to match this installation.

Supply real campaign protein sequences, retain exact chain/domain definitions and report the source. Preserve the scheduler's GPU assignment. The ARM64 environment uses native PyTorch attention where the upstream x86-only flash-attention wheel is unavailable. This alone does not prove GPU inference or equivalent numerical performance.

For a completed run, inspect the actual output coordinates, chain/residue identities and available confidence fields; record model/checkpoint versions, sequence, sample settings, seed and failures. Predictions are not generated binder sequences, measured affinity or experimental validation. The bundled original document is available for historical semantics, with its Aurora launch assumptions explicitly non-authoritative on Jupiter.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For campaign execution, stage with `write_files` and use independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task; `run_code` is reserved for staging fallback, harvesting and diagnosis. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.

## Validated native inference API

```python
from esm.models.esmfold2 import (EsmFold2Model, ESMFold2InputBuilder,
                                ProteinInput, StructurePredictionInput)
model = EsmFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
result = ESMFold2InputBuilder().fold(
    model, StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)]),
    num_loops=1, num_sampling_steps=20, num_diffusion_samples=1, seed=42)
output_cif.write_text(result.complex.to_mmcif())
```

Here `sequence` is an authoritative protein sequence and `output_cif` a fresh `Path` in the assigned workspace. This reduced native CUDA path passed for the 20-residue 1L2Y fixture, producing 153 atoms with finite coordinates. First-use file fetching and native-PyTorch fallback warnings are expected; an FP8-on-CPU loading warning alone does not mean the subsequent `.cuda()` inference is running on CPU. The exact smoke recipe includes structural assertions.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
