---
name: rfd4-proteina
description: Generate protein designs and target-conditioned binders with RFD4-Proteina on Jupiter ARM64 GH200 workers, using its dedicated venv, native runtime libraries, and correct Hydra configuration groups.
---

# RFD4-Proteina on Jupiter

Use the dedicated `software/RFD4-Proteina/.venv` through the campaign execution tools. Run all campaign scientific commands through `run_batch(gpus=1)` after staging with `write_files`; preserve Ensemble Launcher's `CUDA_VISIBLE_DEVICES` and use `cuda:0` inside a one-GPU task. Write artifacts under the assigned campaign workspace. Do not use Aurora modules, XPU settings, ALCF proxies, SSH, or scheduler submissions from a campaign agent.

## Read these packaged references

These are real files inside this skill, accessible with `read_skill`:

- [Protein-binder workflow](references/tasks/protein-binder.md): `read_skill(name="rfd4-proteina", file="references/tasks/protein-binder.md")`.
- [Annotated-CIF construction](references/annotated-cif.md): `read_skill(name="rfd4-proteina", file="references/annotated-cif.md")`.

Their paths are relative to this skill, not to the installed software checkout. Upstream files named within a reference must be read at the indicated checkout path through MCP; they are not implied `read_skill` paths.

Absolute install: `$LOAM_SOFTWARE_ROOT/RFD4-Proteina`. For both `register_tool` and `run_batch`, `venv_path` is `$LOAM_SOFTWARE_ROOT/RFD4-Proteina/.venv`. There is no `software/` directory inside the loam checkout.

Native annotated CIFs renumber the target and use condition annotation column names, not literal C_HOT/C_SEQ/C_CRDGRP strings. Read the annotated-CIF reference and the helper's `.report.json` to validate source-to-input numbering, hotspot preservation, and the requested contig.

## Environment

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/RFD4-Proteina"
source "$REPO/.venv/bin/activate"
export CLUSTER=local WANDB_MODE=disabled OMP_NUM_THREADS=4
cd "$REPO"
```

Activate this venv explicitly before upstream shell wrappers: they otherwise preserve an unrelated active environment such as Globus's Python. Prefer absolute `.venv/bin/python` and `.venv/bin/rfd4-generate` paths. RFdiffusion3 is not an available campaign tool on this port.

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


## Executable reference

Read `references/smoke-workload.py` with `read_skill(name="rfd4-proteina", file="references/smoke-workload.py")` for a concrete Jupiter workflow with output assertions. Stage its contents into the assigned workspace before execution, using this tool's Python and `--out <absolute-output-directory> --software $LOAM_SOFTWARE_ROOT`. Reduced smoke settings validate plumbing; choose campaign settings for scientific evidence.

## Campaign execution

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` in a bash subprocess before activating the tool environment. Preserve the assigned `CUDA_VISIBLE_DEVICES`; use `cuda:0` within a one-GPU row. Stage scripts and inputs with `write_files`, execute real sweeps with `run_batch(gpus=1)`, poll completion, then inspect actual output files. Batch floors are configured by the deployment server; consult the campaign setup. Never pad with duplicate jobs. Keep each row within the remaining Slurm walltime.

The portable input helper is available at `$HPC_LOAM_HOME/skills/diffusion-models/rfd4-proteina/scripts/make_rfd4_input.py`. Hotspots accept comma-separated author residue IDs or inclusive ranges such as `360-386`. Validation is enabled by default; `--validate` makes this explicit.

For reusable generation stages, read `scripts/run_binder_generation.py` with `read_skill` and stage its contents in your assigned run directory. It accepts a row JSON filename with `input_cif` (absolute verified input), `seed`, and optional `binder_length` (default 60), `target_chain` (A), `binder_chain` (B). It writes only into the assigned batch-row directory and requires real, finite binder-target structures before writing `generation_manifest.json`. Register with `params={"input_cif":"str","seed":"int"}`, `skip_smoke=true`, the RFD4 venv, and `expects=["generation_manifest.json", "generation/inference_samples/*/*generated.cif"]`. Pass each hypothesis's own verified input explicitly. A previously registered tool with hard-coded input/output paths for a different hypothesis is unsuitable for reuse.

For GPU inference stages, use `register_tool(skip_smoke=true, expects=[...])` and validate their real outputs through `run_batch(gpus=1)`. Registration smoke tests run on an unpinned submitter and have a 120-second limit; they are not a suitable place to run model inference. Preserve the batch output contract rather than dropping it to avoid a failed stage.

## Bundled production pipeline

`scripts/run_binder_pipeline.py` bundles one complete candidate into one assigned GPU row: the verified generation helper, target-fixed ProteinMPNN (chain B designed, A fixed), Chai with 3 recycles/200 diffusion steps/5 samples, and checked interface geometry. This avoids serial model-driven transitions between short stages. It does not reduce scientific sampling. Use it for the verified EGFR 205-residue crop and 60-residue binder; its hotspot geometry is specific to that mapped target.

Each JSON row contains `id`, `condition` (`dual`, `one_region`, or `ridge`), integer `seed`, absolute `input_cif`, `input_sha256`, `target_fasta`, `generation_script`, and `generation_script_sha256`. The generation script is a frozen copy of `scripts/run_binder_generation.py`; the checksums must match the actual files. Prepare separate validated annotated inputs for each condition, preserving target geometry and sequence. Use globally distinct seeds and record them; do not reuse or pad candidates to fill slots.

For the operator-prepared scale campaign, `scripts/make_campaign_manifest.py` computes all row paths, checksums and twelve distinct seeds from the actual hypothesis UUID/run ID. Its frozen copy is in each hypothesis's data/. Invoke it once with `run_code` from YOUR RUN DIRECTORY using `import runpy; from pathlib import Path; runpy.run_path(str(Path.cwd().parents[1]/'data/make_campaign_manifest.py'), run_name='__main__')`. It writes `candidate-inputs.jsonl` and provenance in the assigned run and returns the exact input path. This is CPU metadata preparation; no GPU result is claimed. Submit that path to the registered bundled pipeline. This avoids repeatedly asking the model to transcribe long manifests or compute hashes itself.

After submission, call `batch_status` once to obtain the actual batch `dir`. For a bounded wait use `scripts/wait_for_batch.py`, also frozen in data/: `import runpy,json; from pathlib import Path; helper=runpy.run_path(str(Path.cwd().parents[1]/'data/wait_for_batch.py')); print(json.dumps(helper['wait_batch'](ACTUAL_BATCH_DIR, max_seconds=600, interval=15)))`. Replace ACTUAL_BATCH_DIR with the literal directory from `batch_status`. This read-only CPU helper checks real done/status files and returns immediately when every row is terminal, including failures. Then call MCP `batch_status` again to attest success. Never run an unconditional sleep loop for600seconds: small analysis batches can finish in less than a second, and even a large batch may finish early. The scientific workload and sampling budget remain unchanged.

Register the bundled script with all row fields except `id` declared in `params` (`seed` is int, others str), `skip_smoke=true`, the RFD4 venv, and `expects=["summary.json", "generation_stage/generation_manifest.json", "mpnn/seqs/*.fa", "fold/pred.model_idx_*.cif", "fold/scores.model_idx_*.npz"]`. Run with `gpus=1`, the same venv, and `timeout_each=5000`. The script asserts five finite, sequence-matched complex models and native hotspot presence. It writes full provenance and stage timings in `summary.json`, plus `progress.json` for diagnosis. Resolve actual row output paths from `batch_status.dir`; successful outputs are promoted out of the attempt directory.

Contact counts, ipTM and target-aligned model-ensemble RMSDs remain computational observations. Report matched condition budgets and failed candidates; do not interpret these metrics as affinity or silently replace a failed pipeline with an easier tool.

`scripts/summarize_binder_pipeline.py` provides a failure-aware CPU-only summary. Its row fields are `batch_dir` (the actual owned batch output directory) and `expected_per_condition` (normally 4). Register with `expects=["analysis.json", "full_analysis.json"]`, the RFD4 venv and `skip_smoke=true`, then run with `gpus=0`. It checks terminal row status and model hashes, preserves failed candidates and reports an incomplete comparison if condition budgets were not met.

## Intact EGFR context before energetic ranking

The source 3NJP structure explicitly records disulfides A482–491 and A486–499 crossing the A286–490 crop. Do not treat this crop as a chemically intact EGFR construct for MM/GBSA. Screen on the crop, then re-fold selected actual designed sequences with intact canonical EGFR25–646 (622 residues) and verify target disulfides before energetic evaluation.

`scripts/run_intact_egfr_followup.py` performs this later funnel stage. It accepts `id`, `seed`, `screen_summary`, `screen_summary_sha256`, `target_annotation`, `target_annotation_sha256`, `stages_directory`, `cofold_script_sha256`, and `mmgbsa_script_sha256`. Source summary paths must refer to actual completed candidate rows. The target annotation is an accession-verified UniProt P00533 JSON file retrieved once. Frozen stage scripts and their source provenance are in `references/intact-egfr-energy/`; copy them into the hypothesis's data directory before use. The reference requires a cysteine-free 60-residue binder; other binders need an independently verified binder disulfide topology.

Register with the RFD4 venv, `skip_smoke=true`, and `expects=["followup-summary.json", "validation-report.json", "intact-geometry.json", "mmgbsa/report.json", "mmgbsa/FINAL_RESULTS_MMPBSA.dat"]`; run `gpus=1`, `timeout_each=5000`. Chai retains 3/200/5 sampling. OpenMM uses CUDA with ff14SB/GBn2, native target disulfides, 0.15M salt and up to 500 minimization iterations. MM/GBSA uses a single minimized frame and omits entropy and glycans. Report it as a static energy estimate, never as measured affinity or a validated binding free energy. Keep failures and ineligible chemistry explicit.

The follow-up measures native hotspot contacts and target-crop-aligned binder RMSD for all five intact Chai models, and separately for the minimized energy frame. A favorable energy at another receptor site is not evidence for the intended epitope. Intact target native IDs equal the verified source-author hotspot IDs; the screen instead subtracts 285. The full target sequence slice 286–490 must equal the screen target exactly. Amber may discard chain labels when writing the minimized PDB, so the analysis verifies the complete sequence and splits residues 1–622 (target) and 623–682 (binder). The Amber topology must contain zero covalent bonds across that boundary. `followup-summary.json` distinguishes the input pose from post-minimization geometry; no new ipTM is assigned to a minimized structure.


### Intact-follow-up metadata helper
The campaign may provide `data/make_intact_followup_manifest.py`. Read it with `read_skill` file="scripts/make_intact_followup_manifest.py". From the assigned run directory, call its `make_manifest([absolute_per_candidate_summary_paths])` through CPU `run_code`. Supply real `summary.json` files from this hypothesis's recorded screen batch, never cohort `analysis.json` or `full_analysis.json`. It verifies candidate/model hashes, source hypothesis, binder chemistry and real annotation/stage paths, then writes the correctly checksummed JSONL for the registered intact tool. Selection and confidence gates remain the validator's responsibility; preparing metadata is not scientific evidence.
