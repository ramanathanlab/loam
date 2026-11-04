---
name: rfd4-proteina
description: Run RFD4-Proteina (Baker Lab x NVIDIA flow-matching all-atom protein design) inference on the ALCF Aurora Intel XPU install at /flare/FRAME-IDP/msinclair/software/RFD4-Proteina. Covers the venv and module situation (it uses Aurora's default frameworks/2026.1.0 - no module swap), ZE_AFFINITY_MASK tile pinning, the Hugging Face checkpoint download over the ALCF proxy, the lightning seed_everything incompatibility and its shim, `rfd4-generate` Hydra options, benchmark/annotated-CIF inputs (unconditional monomer, motif, enzyme, small-molecule and protein binders) and the output layout. Use whenever asked to generate or refine protein designs with RFD4 / RFproteina on Aurora. For RFdiffusion3 use the rfdiffusion3 skill instead.
---

# RFD4-Proteina on Aurora

RFD4-Proteina is a flow-matching all-atom design model. One entry point,
`rfd4-generate`, covers unconditional monomers/complexes, folding and inverse
folding, motif scaffolding, enzyme (active-site) design, small-molecule binders,
protein binders and DNA binders. `rfd4-evaluate` scores designs. All arguments
are Hydra `key=value` — **never** `--key value`.

Install: `/flare/FRAME-IDP/msinclair/software/RFD4-Proteina` (read-only for your
purposes: do not write runs, `.env.local`, or `data_path` into it).

## 1. Environment

Unlike the Foundry/RFD3 install, this venv was built against Aurora's **current
default** module, `frameworks/2026.1.0`
(`home = /opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin`,
`include-system-site-packages = true`), so **no `module swap` is needed** — the
default login/PBS environment works. Verified: torch `2.13.0a0+gitcf30153`,
`torch.xpu.is_available() == True`.

```bash
REPO=/flare/FRAME-IDP/msinclair/software/RFD4-Proteina
BASE=<your writable run dir>

export ZE_AFFINITY_MASK=7                      # pin ONE tile
export CLUSTER=local                            # else: "No .env file found for cluster ..."
export DATA_PATH=$BASE/data_path                # native ckpts + optional assets
# HF bundle is PRE-STAGED on flare -- point HF_HOME at it and run OFFLINE: no
# download, no proxy. (Weights are NOT bundled in the install; this cache holds
# the nvidia/RFD4-Proteina bundle at revision a06e2509..., the inference default.)
export HF_HOME=/lus/flare/projects/FRAME-IDP/agentic_campaign_09_26/binder_design/models/rfd4_proteina
export HF_HUB_OFFLINE=1
# Only to RE-FETCH instead of using the staged cache: unset HF_HUB_OFFLINE and
# export http_proxy/https_proxy=http://proxy.alcf.anl.gov:3128 (then HF_HOME may
# be any writable dir).
export PYTHONPATH=$BASE/shim${PYTHONPATH:+:$PYTHONPATH}   # see §3
# CRITICAL under an MPI/PMIx launcher (Ensemble Launcher / Parsl / Globus-Compute
# worker -- i.e. Loam's run_batch / run_code path): scrub the rank env, or
# Lightning auto-detects a multi-rank job and DEADLOCKS on a collective with no
# peers. Symptom: builds the model, logs "Guidance is turned off", then hangs
# forever with the XPU idle (no structure, any walltime/thread count). The EL
# task inherits a live PALS/PMIx context (PMIX_RANK, PMIX_NAMESPACE,
# PMIX_SERVER_URI, PALS_*); these make it think it is one rank of a many-rank job.
# Confirmed: with this scrub, one binder design runs in ~150 s on a tile.
# NOT needed for a plain single-rank `qsub -l select=1` job (no PMIx launch ctx).
unset PMIX_RANK PMIX_NAMESPACE PMI_RANK PMI_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE \
      PALS_RANKID PALS_NODEID PARSL_WORKER_RANK
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
# Keep BLAS/OMP single-threaded: the work is on the XPU, so CPU threads are only
# glue. At the default (~104) 12 rows/node oversubscribe the 208 CPUs and the run
# crawls; =1 lets 12 designs run concurrently at full speed (~145 s each).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$REPO"
$REPO/.venv/bin/rfd4-generate <hydra overrides>
```

`source .venv/bin/activate` also works; the absolute `.venv/bin/rfd4-*` path is
equivalent and safer inside scripts. Commands must run from the repo root (it
holds `.project-root`, the Hydra config tree, and `data/benchmarks`).

The repo has **no `.env.local`** (only `.env.local.example`). With
`CLUSTER=local` and `DATA_PATH`/`HF_HOME` exported, the exported values win and
no env file is needed. `atomworks` prints harmless warnings that
`CCD_MIRROR_PATH` / `PDB_MIRROR_PATH` are unset — ignore them for plain
generation.

## 2. Checkpoints

`configs/inference.yaml` defaults to the **Hugging Face bundle**
`hf_ckpt_path=nvidia/RFD4-Proteina` pinned at revision
`a06e2509c7033b010fc43614e9fd840303588502`, which overrides the native
checkpoint paths. This bundle (~1.7 GB) is **pre-staged** at
`/lus/flare/projects/FRAME-IDP/agentic_campaign_09_26/binder_design/models/rfd4_proteina`
(the `HF_HOME` set above); with `HF_HUB_OFFLINE=1` it loads from there with **no
download and no proxy** — the robust path for many concurrent runs. Only if you
must re-fetch does it download into `$HF_HOME` on first run (~1.7 GB, ~16 s, needs
the proxy); then point `HF_HOME` at a shared flare directory so you pay once.

Native checkpoints are the alternative (`hf_ckpt_path=null`):
`${DATA_PATH}/flow_checkpoints/rfd4-proteina-flow-release-2026-10-19.ckpt` plus
the matching autoencoder
`${DATA_PATH}/ae_checkpoints/rfd4-proteina-ae-release-2026-10-19.ckpt`; fetch
them with `./scripts/download_assets.sh "$DATA_PATH"`. Neither is present on
this install — **the HF path is the verified one here**. Always keep the flow
checkpoint and its AE paired.

RFD4 does **not** use the Foundry `checkpoints/` directory or
`FOUNDRY_CHECKPOINT_DIRS`; those are RFD3/RF3 only.

## 3. Aurora blockers — the `sitecustomize.py` shim (required)

Out of the box **nothing runs**: the venv's `lightning` is an Intel-patched
**2.3.0dev**, far older than what `src/rfproteina` expects, and the frameworks
module provides torch 2.13 with *native* XPU support and **no
`intel_extension_for_pytorch`**. Three separate failures result, in this order:

1. `TypeError: seed_everything() got an unexpected keyword argument 'verbose'`
   — `generate.py:642` calls `L.seed_everything(cfg.seed, verbose=...)`;
   lightning 2.3's signature is `(seed, workers)`. Happens for any `verbose`
   value.
2. `MisconfigurationException: No supported gpu backend found!` —
   `generate.py:708` hardcodes `L.Trainer(accelerator="gpu")`. This lightning
   *does* have an `XPUAccelerator`, but both
   `lightning/{pytorch,fabric}/accelerators/xpu.py` gate it behind a
   `RequirementCache("intel_extension_for_pytorch")`, which is unsatisfiable
   here, so `is_available()` is `False`. The next error after patching only the
   pytorch copy is `You requested gpu: [0] But your machine only has: []` from
   `lightning/fabric/utilities/device_parser.py` — the fabric copy and
   `_get_all_visible_xpu_devices` must be patched too.
3. `IndexError: pop from empty list` in `rich/console.py:clear_live`, from
   lightning 2.3's `RichProgressBar._init_progress()` against the newer `rich`
   in frameworks/2026.1.0 (reached through the repo's `GenerationProgressBar`
   at `trainer.validate()`).

Since the software tree must not be modified, all three are patched from
`PYTHONPATH` via a `sitecustomize.py`. Use the tested copy at
the worker-visible copy in the /flare loam clone (run_batch runs on the Aurora
worker, which cannot see the local skill dir):

```bash
mkdir -p $BASE/shim && cp ${HPC_LOAM_HOME:?set HPC_LOAM_HOME to the loam clone on /flare}/skills/diffusion-models/rfd4-proteina/scripts/sitecustomize.py $BASE/shim/
export PYTHONPATH=$BASE/shim${PYTHONPATH:+:$PYTHONPATH}
```

It drops the unsupported `verbose` kwarg, forces `_IPEX_AVAILABLE`/
`is_available()`/`num_xpu_devices()`/`_get_all_visible_xpu_devices()` in both
accelerator modules to use `torch.xpu.*`, and makes `Console.clear_live()`
swallow the `IndexError`. Each patch is a no-op if the underlying version
already behaves.

Root cause: this install's `pyproject.toml` has `torch>=2.11` and
`lightning>=2.6.0` commented out of `dependencies` (so torch comes from the
frameworks module), which left an Intel-patched lightning 2.3.0dev in the venv
against code that expects >= 2.4.

**The proper fix is upgrading `lightning` in the RFD4 venv** to the version
`pyproject.toml` expects (>= 2.4, which has `verbose`, native-XPU accelerator
support and a rich-compatible progress bar). That was deliberately not attempted
here, to avoid perturbing a working XPU environment near the end of a job.
Treat the shim as a workaround, not the intended configuration, and re-test it
after any venv change.

## 4. Verified minimal run

Script: `scripts/run_rfd4_min.sh` beside this skill. It performs the exports in
§1 (including the shim) and runs:

```bash
cd /flare/FRAME-IDP/msinclair/software/RFD4-Proteina
.venv/bin/rfd4-generate \
  inputs=data/benchmarks/monomer \
  subset_to_examples=monomer-short \
  num_replicates=1 batch_size=1 \
  out_dir=$BASE/out_smoke \
  ~metrics compile_lengths=null verbose=true
```

`monomer-short` is a 100-residue unconditional monomer — the smallest shipped
benchmark example. `~metrics` deletes the metrics config group (no oracle
needed); `compile_lengths=null` disables `torch.compile` bucket padding, which
matters on XPU (see §7).

**Verified 2026-10-29 on an Aurora debug node, tile 7: PASS, 1m22s wall**
(`user` 88 min — the XPU tile is well fed; checkpoint already cached in
`HF_HOME`, add ~20 s for the first download). Log markers:
`GPU available: True (xpu), used: True`, then
`✓ Generation complete! Results saved to ...`.

Output:

```
out_smoke/inference_samples/monomer-short/monomer-short_generated.cif
out_smoke/inference_metrics/per_example_metrics.csv
```

The CIF is a real all-atom design: 801 `ATOM` records, one chain, a 100-residue
designed sequence
(`SSKEELISDISELVSEKLSSFSEDDQVDLTLVISSLFVAIHEEDIDEIISCLKEIDEIDHSLCLEILELILPYLSEDQRLKLLSAFKHNSEMSDLIKEYL`),
so XPU numerics produce sane structures, not just a non-crashing run.
`per_example_metrics.csv` is written even with `~metrics`: it records the
resolved generation config per design (seed, `nsteps=60`, `tsr_k`, checkpoint
`nvidia/RFD4-Proteina` / `model.safetensors`, input path), which is worth
keeping for provenance.

## 5. Inputs

`inputs=` takes either a directory of benchmark examples or a single
`.cif` / `.cif.zst`. Pick named keys inside a directory with
`subset_to_examples=<key>` (a key is the file stem, e.g. `monomer-short`).

| Task | Example input under `data/benchmarks/` |
|---|---|
| Unconditional monomer | `monomer` (`monomer-short` 100 aa, `-medium` 150, `-long` 250) |
| Fold-class-conditioned monomer | `adherence/monomer_helical`, `adherence/monomer_ig` |
| Motif scaffolding | `motif_scaffolding/motif_indexed`, `motif_unindexed`, `motif_indexed_internal_coords` |
| Enzyme / active site | `mcsa/mcsa_37`, `mcsa_37_fixed_ligand`, `mcsa_37_no_ligand` |
| Small-molecule binder | `small_molecule_binder/sm_binder`, `SAM_hbonds.cif.zst` |
| Protein binder (PPI) | `ppi/`, plus `docs/how-to/input_prep/examples/condition_spec_ppi.yaml` |
| DNA binder | `dna_binder/dna_binder_denovo_val` |
| RNA / structure prediction | `rna/`, `structure_prediction/` |

For anything custom — a specific length, hotspots, fixed ligands, unindexed
motifs, H-bond or RASA conditioning — the input format is an **annotated CIF**
with per-atom conditioning columns, built with the repo's
`skills/rfd4-proteina-annotated-cif`. There is no contig mini-language as in
RFD3; conditioning lives in the CIF. Design length is set by the input, so
"design a 70-residue monomer" means building an annotated CIF, not a CLI flag.

Conditioning conventions worth preserving (from the repo's AGENTS.md): for PPI,
default to target backbone-only internal coordinates (`C_CRDGRP`), full target
sequence (`C_SEQ`) and the requested hotspots; fixing sidechains also fixes
residue identity, so use backbone-only conditioning when the sequence is to be
redesigned.

Unconditional/motif/binder commands beyond the verified monomer case are
**documented but unverified on this machine** — see the per-task references in
`skills/rfd4-proteina-generate/references/tasks/`.

## 6. Key options

| Argument | Default | Meaning |
|---|---|---|
| `inputs` | — | Benchmark dir or annotated CIF. |
| `out_dir` | `./runs/<run_name>` | Use an absolute path outside the repo. |
| `subset_to_examples` | null | Run only these example keys. |
| `num_replicates` | 1 | Designs (backbones) per example from ONE process on ONE tile. 1 is too small for real work; a design funnel needs a large starting library, so the **default is >1000 backbones per hypothesis per round**. Treat it as a lever to tune (§6.1), not a fixed rule. Reach the count by fanning out across tiles rather than one long serial job: keep per-job `num_replicates` modest (~8) and get the total from the number of parallel jobs (an Aurora node has 12 usable tiles, so ~12 jobs × 8 = ~96 backbones per wave per node). |
| `batch_size` | 1 | Raise to ~8 only after the longest input passes at 1. |
| `seed` | null | Reproducibility. |
| `skip_existing` | true | Resume; use a fresh `out_dir` when changing settings. |
| `prevalidate_inputs` | false | Fail fast on bad inputs. |
| `precision` | `bf16-mixed` | Left as-is on XPU. |
| `compile_lengths` | 15-bucket list | `null` disables compile padding (used here). |
| `recycle_strategy` | `thinking_medium` | Task presets in `configs/recycle_strategy/`; see `docs/how-to/run-the-model.md#recycling-schedules`. |
| `generation.args.nsteps` | 60 | Model evaluations per design. |
| `generation.model.bb_ca.simulation_step_params.tsr_k` | 16 | Backbone temperature; lower = more diverse. |
| `generation.model.local_latents.simulation_step_params.tsr_k` | 4 | Latent temperature. |
| `generation.args.partial_trajectory_fraction_steps` | null | Partial diffusion as a fraction of `nsteps`. |
| `generation.model.*.gt.mode=ode` | sde | Deterministic sampling. |
| `metrics_tags` | `[]` | Inline metrics; every `*_refolding` tag needs the AF3/OF3 oracle container (**not installed here**). |
| `compress_outputs` | false | Compress CIFs. |
| `save_chain_of_interest` | false | Required for later foldseek clustering. |
| `inference_backend` | `native` | `bioir` is a CUDA-oriented path — do not use on XPU. |
| `verbose` | false | `true` prints the composed config and diagnostics. |
| `hardware.ngpus_per_node_` / `hardware.nnodes_` | 1 / 1 | Trainer devices/nodes. Leave at 1 — multi-tile is untested. |

## 6.1 Design counts and the parameters you control (batch size, length)

How many backbones to generate — and at what length — are **yours to choose and
to vary between rounds**; this skill fixes neither. Pick deliberately and adjust
as the campaign tells you more.

**Batch size.** `num_replicates=1` is far too small for real work: a design
funnel loses most candidates between backbone and the confident, filtered
survivors (typically RFD4 → sequence design → structure/metric validation →
filters), so the starting library has to be large. The **default is >1000
backbones per hypothesis per round** — scale up further when the pool is large
and idle or a direction is clearly paying off, down when the pool is contended or
you are just probing.

RFD4 here is **one process on one tile** (`hardware.ngpus_per_node_=1`, multi-tile
untested — §7.10), and one long serial job is exactly what the 1 h debug-scaling
reclaim kills mid-run. So reach your chosen count by **fanning out across tiles**
(the `aurora-multinode-fold` pattern), not by one huge call:

- Submit many independent `rfd4-generate` jobs via `run_batch(gpus=1 …)`, each
  on its own tile, each with a modest `num_replicates` (~8), a **distinct `seed`**
  and a **distinct `out_dir`**. An Aurora node has 12 usable tiles, so ~12 jobs ×
  `r=8` = ~96 backbones per wave on one node; run successive waves (and/or more
  nodes) to reach the >1000 target instead of one long serial job.
- `skip_existing=true` + a per-job `out_dir` makes the set resume cleanly after a
  reclaim: only unfinished jobs rerun.
- **Match the fan-out to the tiles actually available** — call the HPC server's
  `pool_status` (it reports `total_tiles`, `queue`, `walltime`) and size the number
  of parallel jobs to it, so you don't over-queue a wave the pool can't clear
  before the next reclaim or leave idle tiles empty.

**Length.** Design length is set by the input CIF, not a CLI flag (§5), so it is
easy to leave it pinned at whatever benchmark you started from — don't. Design
length is a real variable: **vary it across rounds** (build annotated CIFs
at a few lengths, e.g. ~20 / 40 / 60 / 80 / 100 / 120 aa) rather than generating every
hypothesis at one length, and let the downstream scores show what the target prefers.

The same per-tile fan-out applies to the downstream validation/scoring screen, so
budget tiles across generation *and* validation when both are in flight.

## 7. Aurora / XPU gotchas

1. **Do not swap frameworks modules for this repo.** It wants the default
   `frameworks/2026.1.0`. (The sibling Foundry/RFD3 install wants
   `frameworks/2025.3.1` — the two cannot share one shell. Run them in separate
   shells or separate scripts.)
2. **lightning `verbose` crash** — §3; this is the one hard blocker.
3. **`bash skills/rfd4-proteina-setup/scripts/preflight.sh` does not work
   here**: it goes through `scripts/rfproteina_exec.sh`, which did not pick up
   the `.venv`, and failed with `ModuleNotFoundError: No module named
   'atomworks'` even though `atomworks` is installed in `.venv`. Verify the
   environment with `.venv/bin/python -c "import torch, atomworks"` instead.
   Also note its GPU check is written for CUDA/NVIDIA drivers.
4. **Pin a tile** with `ZE_AFFINITY_MASK=<n>` before launching; otherwise the
   run takes tile 0 and collides with other users.
5. **Internet** needs `http_proxy`/`https_proxy=http://proxy.alcf.anl.gov:3128`
   on compute nodes for the HF checkpoint fetch. Pre-warm `HF_HOME` if you plan
   offline runs.
6. **`compile_lengths=null`** was used for the verified run. The default padding
   list exists for `torch.compile`, and torch.compile/Triton-XPU is the most
   fragile part of the stack here; leave it `null` unless you are deliberately
   testing compilation.
7. **`inference_backend=bioir`** and its CUDA-graph settings are NVIDIA-specific
   — keep `native`.
8. Expect **~2 minutes of imports** before anything happens, plus noisy
   deprecation warnings and an `aten::topk` XPU kernel-override warning from the
   frameworks module. Not errors.
9. **Refolding metrics are unavailable** on this install: no AF3/OpenFold3
   container or weights, and `scripts/download_assets.sh` + Apptainer were not
   attempted. Any `metrics_tags=[*_refolding]` run will fail.
10. **Multi-tile / multi-node and the SLURM launchers are untested.** The repo's
    `scripts/launch_*.sh` assume SLURM; Aurora uses PBS, so a PBS wrapper would
    have to be written.

## 8. Outputs

Under `out_dir`:

```
out_dir/inference_samples/<example-key>/<example-key>_generated.cif
out_dir/inference_metrics/per_example_metrics.csv
```

(As observed: one subdirectory per example key, and the metrics CSV is written
even with `~metrics` — it then holds the resolved config rather than metric
values. The repo's `references/outputs.md` describes CIFs landing directly in
`out_dir`; the actual layout is the nested one above.)

(`rfd4-evaluate` writes `evaluation_samples/` and `evaluation_metrics/`.) Each
design is an annotated CIF carrying per-atom provenance columns `SA_SRC`,
`SA_CRDSRC`, `SA_IDXGRPSRC`, `SA_UII` that tie output atoms back to the input
spec. Check the structures and metrics, not just the exit code.

## 9. Troubleshooting

- `seed_everything() got an unexpected keyword argument 'verbose'` — §3 shim.
- `No .env file found for cluster '<name>'` — `export CLUSTER=local`.
- `--inputs`/`--out_dir` not recognized — use `=`, Hydra style.
- Checkpoint not found — check `HF_HOME` + proxy, or `DATA_PATH` and
  `ckpt_name`/`ckpt_path` for the native path; keep flow + AE paired.
- `ModuleNotFoundError: atomworks` from the preflight script — §7.3.
- OOM — lower `batch_size`; confirm the longest input at `batch_size=1`.
- Segfault in rdkit/posebusters — the repo `Makefile` notes `LD_PRELOAD` must
  point at the bundled `libstdc++.so.6` (**unverified here**).
- Nothing regenerated — `skip_existing=true`; use a fresh `out_dir`.

## 10. Further reading (in the install)

- `AGENTS.md` / `CLAUDE.md` — repo-wide agent rules and conditioning defaults.
- `skills/rfd4-proteina-generate/SKILL.md` + `references/tasks/*.md` — per-task
  commands and overrides (the authoritative detail).
- `skills/rfd4-proteina-annotated-cif/` — build conditioned inputs.
- `skills/rfd4-proteina-search/`, `.../-evaluate/`, `.../-campaign/` — beam
  search, metrics, multi-stage campaigns.
- `docs/quickstart.rst`, `docs/how-to/run-the-model.md`,
  `docs/how-to/evaluate-with-metrics.md`, `configs/inference.yaml`.
