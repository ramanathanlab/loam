# RFD4-Proteina on Jupiter (ARM64 GH200, CUDA)

Use the tool's own venv through the campaign execution tools. The Aurora resident-worker wrapper,
its pinned adapter, XPU settings and tile pinning are not Jupiter recipes; do not copy them.

## Environment

```bash
REPO="$LOAM_SOFTWARE_ROOT/RFD4-Proteina"
PY="$REPO/.venv/bin/python"                # or $REPO/.venv/bin/rfd4-generate
```

For both `register_tool` and `run_batch`, `venv_path` is `$LOAM_SOFTWARE_ROOT/RFD4-Proteina/.venv`.
Activate that venv explicitly before any upstream shell wrapper: they otherwise keep whatever
environment is active, such as the Globus worker's Python. Prefer the absolute `.venv/bin/...`
paths. Set `CLUSTER=local WANDB_MODE=disabled OMP_NUM_THREADS=4` and run from `$REPO`.

The Site profile's `worker_init` runs on every worker before your code and exports the caches and
runtime library paths this tool needs — `DATA_PATH` (default
`$LOAM_SOFTWARE_ROOT/RFD4-Proteina/data_path`), `HF_HOME`, and RFD4's bundled
`runtime/lib` on `LD_LIBRARY_PATH`, which Open Babel's ARM64 wheel needs. The redacted example is
`examples/site-profiles/jupiter.yaml`. Nothing in this skill sources an environment script of its
own.

## Correct Hydra invocation for a protein binder

After constructing and validating a real target-conditioned CIF
([annotated-cif.md](annotated-cif.md)), set `INPUT` and `OUT` to absolute campaign workspace paths;
`OUT` must be fresh:

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

**Use `recycle_strategy=medium/front_loaded` exactly.** This selects the existing Hydra
configuration group `configs/recycle_strategy/medium/front_loaded.yaml`, which defines the actual
sampling schedule (60 sampling steps, 180 total trunk passes). Do not use
`ppi.recycle_strategy=medium` or `+ppi.recycle_strategy=medium`: the first targets an absent field;
the second adds an unused field instead of selecting the schedule. Do not blindly add `+` to a
rejected override. Inspect `configs/inference.yaml` and the named group file when diagnosing
configuration errors.

The example requests one design at an explicit seed, batch size one and eager execution. Keep the
campaign's agreed settings and record the effective overrides, checkpoint revision, input checksum
and output files. **Do not silently substitute sampling presets, switch a preset to force success,
or invent binder sequences or metrics.**

`num_workers=0` is right for single-design inference here: PyTorch loads data in the main process
instead of spawning four loader subprocesses, which matches the successful installation test and
avoids oversubscription under one-core task affinity. It changes data loading, not the recycle
schedule. `save_chain_of_interest=true` saves the generated chain for downstream use.

Default PPI conditioning is target backbone-only internal coordinates (`C_CRDGRP`), full target
sequence (`C_SEQ`) and selected hotspot atoms (`C_HOT`): target internal geometry preserved, global
pose and sidechains free, only the binder chain generated. Keep the campaign's explicit conditions.

The upstream example `docs/how-to/input_prep/examples/condition_spec_ppi.yaml` in the checkout
illustrates these annotations; it is a PD-L1 example, never target-specific campaign evidence.
Upstream `ppi_refolding` metrics need the configured AF3 oracle and target templating — Chai does
not supply that oracle. Hotspot metrics assess input hotspot obedience. Consult the installed
`docs/how-to/evaluate-with-metrics.md` and the effective metric configuration before selecting
metric tags.

## Outputs

```
<OUT>/inference_samples/<name>/<name>_generated.cif    # designed binder chain + the fixed target chain(s)
```

Inspect the actual CIF before reporting success: identify the target and the generated chain, the
generated residue count, finite coordinates, and the conditioning/provenance annotations. **A
command starting, or an input CIF being written, is not a generated binder.** Save input and output
checksums, source and checkpoint revisions, the seed, the effective overrides and the failure logs.

ProteinMPNN sequence design, co-folding, structural metrics and energy evaluation are separate
stages; a generated structure is not validated binding.

## Deriving hotspots from an existing complex

Identify target and binding partner from the source mmCIF entity descriptions and chain mappings
before measuring contacts. A biological ligand may itself be a protein: EGF is a polymeric ligand
of EGFR, and a non-polymer-only filter selects glycans, ions or crystallization compounds instead.
Measure the target crop against the entire verified partner chain, exclude the target itself and
unrelated receptor copies, and keep the partner identity plus residue-level contacts in the report.
Gemmi `het_flag` is a character (`A`, `H` or `W`), so testing its truth value does not distinguish
protein from non-protein. Do not label a single target amino acid as the ligand or count target
self-distances as an interface. If the requested contact mechanism is incompatible with the actual
structure, report the contradiction to the validator before generation.

## Reusable generation stage

`scripts/jupiter/run_binder_generation.py` is one real generation row. Stage its contents in the
assigned run directory. Its row JSON takes `input_cif` (absolute verified input), `seed`, and
optional `binder_length` (default 60), `target_chain` (A) and `binder_chain` (B). It writes only
into the assigned batch-row directory and requires real, finite binder-target structures before
writing `generation_manifest.json`. Register with `params={"input_cif":"str","seed":"int"}`,
`skip_smoke=true`, the RFD4 venv, and
`expects=["generation_manifest.json", "generation/inference_samples/*/*generated.cif"]`. Pass each
hypothesis's own verified input explicitly: a previously registered tool with hard-coded
input/output paths for a different hypothesis is unsuitable for reuse.

For GPU inference stages generally, use `register_tool(skip_smoke=true, expects=[...])` and
validate real outputs through `run_batch(gpus=1)`. Registration smoke tests run on an unpinned
submitter with a 120-second limit; they are not a place to run model inference. Preserve the batch
output contract rather than dropping it to avoid a failed stage.

## Bundled production pipeline

`scripts/jupiter/run_binder_pipeline.py` bundles one complete candidate into one assigned GPU row:
the verified generation helper, target-fixed ProteinMPNN (chain B designed, A fixed), Chai with 3
recycles/200 diffusion steps/5 samples, and checked interface geometry. It avoids serial
model-driven transitions between short stages and does not reduce scientific sampling. Each stage
runs in its own tool venv, resolved as `$LOAM_SOFTWARE_ROOT/<tool>`
(`RFD4-Proteina`, `proteinmpnn`, `chai-lab`), so the row needs `LOAM_SOFTWARE_ROOT` and nothing
else. It is written for the verified EGFR 205-residue crop and 60-residue binder; its hotspot
geometry is specific to that mapped target.

Each JSON row contains `id`, `condition` (`dual`, `one_region` or `ridge`), integer `seed`,
absolute `input_cif`, `input_sha256`, `target_fasta`, `generation_script` and
`generation_script_sha256`. The generation script is a frozen copy of
`scripts/jupiter/run_binder_generation.py`; the checksums must match the actual files. Prepare
separate validated annotated inputs per condition, preserving target geometry and sequence. Use
globally distinct seeds and record them; do not reuse or pad candidates to fill slots.

Register it with all row fields except `id` declared in `params` (`seed` is int, others str),
`skip_smoke=true`, the RFD4 venv, and
`expects=["summary.json", "generation_stage/generation_manifest.json", "mpnn/seqs/*.fa",
"fold/pred.model_idx_*.cif", "fold/scores.model_idx_*.npz"]`. Run with `gpus=1`, the same venv and
`timeout_each=5000`. The script asserts five finite, sequence-matched complex models and native
hotspot presence, and writes full provenance and stage timings in `summary.json`, plus
`progress.json` for diagnosis. Resolve actual row output paths from `batch_status.dir`; successful
outputs are promoted out of the attempt directory.

Contact counts, ipTM and target-aligned model-ensemble RMSDs remain computational observations.
Report matched condition budgets and failed candidates; never read these metrics as affinity, and
never replace a failed pipeline with an easier tool.

`scripts/jupiter/summarize_binder_pipeline.py` is the failure-aware, CPU-only cohort summary. Its
row fields are `batch_dir` (the actual owned batch output directory) and `expected_per_condition`
(normally 4). Register with `expects=["analysis.json", "full_analysis.json"]`, the RFD4 venv and
`skip_smoke=true`, then run with `gpus=0`. It checks terminal row status and model hashes,
preserves failed candidates, and reports an incomplete comparison when condition budgets were not
met.

## Intact EGFR context before energetic ranking

The source 3NJP structure explicitly records disulfides A482–491 and A486–499 crossing the A286–490
crop, so that crop is not a chemically intact EGFR construct for MM/GBSA. Screen on the crop, then
re-fold selected actual designed sequences with intact canonical EGFR 25–646 (622 residues) and
verify target disulfides before energetic evaluation.

`scripts/jupiter/run_intact_egfr_followup.py` performs this later funnel stage. It accepts `id`,
`seed`, `screen_summary`, `screen_summary_sha256`, `target_annotation`,
`target_annotation_sha256`, `stages_directory`, `cofold_script_sha256` and `mmgbsa_script_sha256`.
Source summary paths must refer to actual completed candidate rows. The target annotation is an
accession-verified UniProt P00533 JSON file retrieved once. The frozen stage scripts and their
provenance are in `scripts/jupiter/intact-egfr-energy/`; copy them into the hypothesis's data directory
before use and hash the copies. The reference requires a cysteine-free 60-residue binder; other
binders need an independently verified binder disulfide topology.

Register with the RFD4 venv, `skip_smoke=true`, and
`expects=["followup-summary.json", "validation-report.json", "intact-geometry.json",
"mmgbsa/report.json", "mmgbsa/FINAL_RESULTS_MMPBSA.dat"]`; run `gpus=1`, `timeout_each=5000`. Chai
retains 3/200/5 sampling. OpenMM uses CUDA with ff14SB/GBn2, native target disulfides, 0.15 M salt
and up to 500 minimization iterations. MM/GBSA uses a single minimized frame and omits entropy and
glycans: report it as a static energy estimate, never as measured affinity or a validated binding
free energy. Keep failures and ineligible chemistry explicit.

The follow-up measures native hotspot contacts and target-crop-aligned binder RMSD for all five
intact Chai models, and separately for the minimized energy frame. A favorable energy at another
receptor site is not evidence for the intended epitope. Intact target native IDs equal the verified
source-author hotspot IDs; the screen instead subtracts 285. The full target sequence slice 286–490
must equal the screen target exactly. Amber may discard chain labels when writing the minimized
PDB, so the analysis verifies the complete sequence and splits residues 1–622 (target) and 623–682
(binder), and the Amber topology must contain zero covalent bonds across that boundary.
`followup-summary.json` distinguishes the input pose from post-minimization geometry; no new ipTM is
assigned to a minimized structure.

## Execution contract on Jupiter

- Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir, with
  absolute campaign paths. Do not submit scheduler jobs, use SSH, replace the scientific software,
  or apply Aurora XPU shims. Do not use Aurora modules, `ZE_AFFINITY_MASK`, XPU launchers, ALCF
  proxies or PBS submissions.
- Jupiter has four GH200 GPUs per node. Keep the `CUDA_VISIBLE_DEVICES` assigned by the launcher;
  inside a one-GPU task use `cuda:0`, and do not oversubscribe a GPU.
- Stage scripts and inputs with `write_files` (only under the campaign's `HPC_BASE_DIR`; read-only
  inspection roots do not grant write access to installed software), then run the science through
  `run_batch(gpus=1, timeout_each=6000)`, one design per row, streaming output to a log file
  immediately. Poll `batch_status` until `unfinished == 0`, then inspect the actual output files
  before reporting success.
- `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis only.
- Never pad a batch with duplicate jobs. Keep each row within the remaining Slurm walltime; queue
  waits are separate, so do not cancel queued work just because it waits.
- Stream logs to workspace files and preserve failed attempts. Retain input/checkpoint identity,
  seed, effective overrides, source revisions, and failure logs.

## Installation failures

A missing `libXrender.so.1` means the native runtime is absent or the Site profile's `worker_init`
did not put RFD4's `runtime/lib` on `LD_LIBRARY_PATH`. An upstream shell preflight that cannot
import AtomWorks while the absolute RFD4 Python can is an environment-selection problem: activate
the RFD4 venv explicitly and set `CLUSTER=local` before invoking the wrapper. Do not reinstall a
functioning model environment on the strength of that wrapper error.

## Smoke workload

`scripts/jupiter/smoke-workload.py` generates one backbone from the checkout's own
`data/benchmarks/monomer/monomer-short.cif.zst` with zero loader workers, and asserts a finite
structure. Stage it into a fresh run directory and run it with this tool's Python through
`run_batch`:

```bash
"$PY" smoke-workload.py --out <fresh-absolute-directory> --software "$LOAM_SOFTWARE_ROOT"
```

Then inspect `result.json` and the generated CIF. Reduced settings validate plumbing only; they do
not establish binding efficacy and do not count as campaign evidence.

Latest recorded manual validation (on the `moeen` branch's Jupiter port, not a fresh run of this
checkout): **passed** — trained-model monomer backbone generation with zero loader workers.
Evidence directory: `$LOAM_SOFTWARE_ROOT/verification/jupiter-skills-smoke-20261110/rfd4-proteina`
(the evidence root's `summary.json` records the run). This describes only the tested path and cache
state at audit time; revalidate after environment or checkpoint changes.

## Resource index

- `read_skill(name="rfd4-proteina", file="SKILL.md")` — the System-neutral method
- `read_skill(name="rfd4-proteina", file="references/jupiter.md")` — this file
- `read_skill(name="rfd4-proteina", file="references/annotated-cif.md")` — building a valid input
- `read_skill(name="rfd4-proteina", file="scripts/jupiter/make_rfd4_input.py")` — the input helper
- `read_skill(name="rfd4-proteina", file="scripts/jupiter/run_binder_generation.py")` — one generation row
- `read_skill(name="rfd4-proteina", file="scripts/jupiter/run_binder_pipeline.py")` — the bundled candidate pipeline
- `read_skill(name="rfd4-proteina", file="scripts/jupiter/summarize_binder_pipeline.py")` — the cohort summary
- `read_skill(name="rfd4-proteina", file="scripts/jupiter/run_intact_egfr_followup.py")` — the intact-EGFR follow-up
- `read_skill(name="rfd4-proteina", file="scripts/jupiter/intact-egfr-energy/provenance.json")` — the frozen stage scripts
- `read_skill(name="rfd4-proteina", file="scripts/jupiter/smoke-workload.py")` — the smoke workload
