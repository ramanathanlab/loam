---
name: rfd4-proteina
description: Run RFD4-Proteina (Baker Lab x NVIDIA flow-matching all-atom protein design) inference on ALCF Aurora Intel XPUs, under Loam, via the VERIFIED resident-worker route — David Kastner's pinned read-only deployment (frameworks/2026.1.0, Lightning 2.6.6, in-code SingleDeviceStrategy). Covers the pinned env + shared paths, one-resident-worker-per-tile execution, the Loam run_batch fan-out (rfd4_run_row.py + run_rfd4_resident.sh), annotated-CIF binder inputs, CLI/precision options, and outputs. Use whenever asked to generate protein/binder backbones with RFD4 / RFproteina on Aurora. For RFdiffusion3 use the rfdiffusion3 skill instead.
---

# RFD4-Proteina on Aurora (resident-worker route)

RFD4-Proteina is a flow-matching all-atom design model (unconditional monomers/
complexes, folding/inverse-folding, motif scaffolding, enzyme design, small-
molecule binders, protein binders, DNA binders).

**On Aurora, under Loam, use ONLY the resident-worker route documented here.**
It runs the pinned, read-only deployment maintained by David Kastner (FRAME-IDP)
and delivered as `RFD4_Aurora_Handoff/` at the loam repo root. The adapter is
single-device **by construction** (explicit `SingleDeviceStrategy` +
`LightningEnvironment`, `num_workers=0`), so it does not deadlock when launched
from an MPI/PALS worker — which is exactly how Loam's `run_batch` / Ensemble
Launcher calls it.

> **Why the old route was removed.** The previous skill drove the msinclair
> install (`/flare/FRAME-IDP/msinclair/software/RFD4-Proteina`) through a
> `sitecustomize.py` Lightning-2.3 shim and `run_rfd4_generate.sh`, which scrubbed
> the inherited PMIX_*/PALS_* rank env. Under Loam's run_batch/EL launcher that
> approach **deadlocks**: the model builds, logs "Guidance is turned off /
> Running inference / Validation", then hangs ~18 min with the XPU idle until it
> is killed — producing **0 backbones**. Unsetting env vars is structurally
> insufficient; the fix has to force single-device *in code*, which the resident
> adapter does. That hang was the campaign's generation-stage failure. Do **not**
> reintroduce the sitecustomize shim or mix the two stacks.

## Verified result (2026-11-05)

Through Loam's MCP `run_batch` / EL path, on a 2-node debug block:

- 1 worker: rc=0, 1 backbone + receipt, ~200 s cold.
- **24-tile saturation: 24/24 rc=0, 24 backbones, 0 failures, ~175 s wall**
  (per-row 74–156 s); both nodes covered tiles 0–11 with no overlap.
- HER2 PPI benchmark input; binder length sampled 60–99 aa (mean ~82) from the
  contig's length range, target chain fixed at 237 aa.

Qualified through the handoff's acceptance step 2 (1 task → 2 nodes). NOT yet
qualified: several designs per worker served serially (warm reuse), bounded
failure/retry, and non-HER2 / ligand-conditioned inputs. BF16 replay determinism
and large-scale stability are not promised — treat a second unexplained failure
of a config as a stop, per the handoff.

## 1. Pinned deployment — shared paths and access

Read-only; do **not** modify, reinstall, or `pip install` into it. Requires
`FRAME-IDP` group membership (check from the running account: `id -nG | tr ' ' '\n'
| grep FRAME-IDP`, then `test -r` the files below). Stop and report permission
errors rather than changing permissions.

```bash
RFD4_SHARE=/lus/flare/projects/FRAME-IDP/dkastner/Projects/IDeA
RFD4_PYTHON="$RFD4_SHARE/envs/rfd4-xpu-2026.1.0/bin/python"     # frameworks/2026.1.0, Lightning 2.6.6
RFD4_ADAPTER="$RFD4_SHARE/experiments/pmd_design_20261026T070010Z/source/v34"
RFD4_ENGINE="$RFD4_SHARE/experiments/aurora_performance_20261023T194443Z/source/rfd4_phase1"
```

Verified identities (checked by `run_rfd4_resident.sh` before each run):
- adapter `src/idea/engines/rfd4_persistent.py` sha256 `2a0a9392…d742f0`
- adapter `src/idea/engines/rfd4_xpu.py`        sha256 `ef8e69f4…bf190`
- checkpoint `nvidia/RFD4-Proteina` rev `a06e2509…`, bundle sha256 `5b549dd6…68fb0`
- upstream RFD4 source rev `274c366a0981d680bf3ba79924d99da5e91ae107`
- HF cache `$RFD4_SHARE/cache/huggingface` (used with `HF_HUB_OFFLINE=1`)

## 2. Execution model

- **One independent resident worker per assigned XPU tile**, normally 12/node.
  These are independent samplers — **never distribute one model across MPI ranks.**
- The worker is a **fresh subprocess** with the full env block; `ZE_AFFINITY_MASK`
  is set to the EL-assigned tile **before** Torch import (changing it after import
  is too late). With one tile visible the adapter selects logical `xpu:0`.
- Keep BLAS/OMP single-threaded (`OMP/MKL/OPENBLAS_NUM_THREADS=1`); the adapter
  itself calls `torch.set_num_threads(4)`. Allocate ~4 CPUs per worker.
- The worker loads the model **once** and runs every sample in its plan serially,
  amortizing startup. Do not run concurrent calls against one resident process.

## 3. Running it under Loam (run_batch fan-out)

Two scripts in `scripts/` do everything (deploy the skill to `$HPC_LOAM_HOME` on
flare so Aurora can see them):

- `run_rfd4_resident.sh <plan.json> <fresh_run_dir> <seconds>` — the pinned-stack
  launch: module load, env block, tile from `ZE_AFFINITY_MASK`, hash + import/tile
  probe (300 s), then the resident worker. Self-contained.
- `rfd4_run_row.py` — the `run_batch` row runner: reads one JSONL row, writes a
  1-sample plan, invokes the wrapper on this row's tile, records
  `result.json {host,tile,seconds,rc,n_receipts,n_cifs}`.

**Fan out across tiles with `run_batch(gpus=1)`** — one row per tile, up to the
pool's `total_tiles` (nodes×12) concurrently. Each JSONL row:

```json
{"id":"sat_00","input":"/abs/annotated.cif","seed":101,"seconds":900}
```

Submit (fastmcp client against the hpc-deployment MCP server; `run_batch` resolves
`script_path`/`inputs_path` on **Aurora's** filesystem):

```python
await c.call_tool("run_batch", {
    "script_path": f"{HPC_LOAM_HOME}/skills/diffusion-models/rfd4-proteina/scripts/rfd4_run_row.py",
    "inputs_path": "<aurora>/rfd4_rows.jsonl",   # one row per tile
    "gpus": 1, "timeout_each": 1200,
    "labels": {"tool": "rfd4-proteina", "stage": "rfd4_generate"}})
```

The row runner resolves the wrapper via `row["wrapper"]` → `$RFD4_WRAPPER` →
`$HPC_LOAM_HOME/.../run_rfd4_resident.sh`. Set `HPC_LOAM_HOME` (or `RFD4_WRAPPER`)
in the pool/worker env. Size each fan-out to the live free tiles (`pool_status`
capacity minus in-flight work from `list_batches`); each row must finish within
the block walltime (debug/debug-scaling reclaim ~hourly).

**Design count = rows × samples-per-plan.** Distinct `seed`s are distinct designs;
reusing a seed is a replay, not new diversity. A design funnel wants a large
library — reach it by fanning out across tiles (and, once qualified, several
samples per plan for warm reuse), not one long serial job.

## 4. Inputs — annotated CIF

RFD4 inputs are **annotated CIFs** (`SA_*` columns + `pdbx_PDB_model_num` + a
condition spec: contig `"<binder_len>, /0, <target>"` with `C_CRDGRP`/`C_SEQ`/
`C_HOT`). A generic PDB does not encode fixed atoms, sequence constraints or
ligand conditioning — it will fail.

- **Build a binder input:** `scripts/make_rfd4_input.py --target <pdb> --target-chain
  <ch> --hotspots <resids> --binder-len <N|range> --target-range <a-b> --out <cif>`
  wraps the engine's `write_condition_spec_cif`. `--binder-len` sets/narrows the
  designed length (the HER2 benchmark uses a *range*, hence the 60–99 aa spread).
  NOTE: validate a freshly built CIF against this engine rev (`274c366`) on a
  1-row smoke before fanning out.
- **Ready-made benchmark inputs** (matched to this engine, guaranteed valid) live
  under `$RFD4_ENGINE/data/benchmarks/` — e.g. protein binders in `ppi/ppi_bb_only/`
  (`28_HER2_AAV.cif.zst`, …), plus `monomer/`, `motif_scaffolding/`,
  `small_molecule_binder/`, `dna_binder/`, `rna/`. Use these for smoke tests.

## 5. CLI / precision (what the adapter runs)

`python -m idea.engines.rfd4_persistent --plan <plan.json> --output <dir>
--device 0 --precision {32|bf16-mixed} --matmul {highest|high|medium}
--deterministic --recycle-strategy medium/midpeak [--mode full|partial|inverse_folding]
[--compile-length N …]`. Plan: `{"samples":[{"input":"<cif>","seed":N}, …]}`.

Internally it composes `inference_experiment=quickstart` with `num_replicates=1
batch_size=1 num_workers=0 nn.tri_mult_backend=torch nn.attention_type=torch
+idea_keep_model_on_xpu=true` — native eager Torch kernels, model resident on XPU.
Measured warm throughput (handoff, 388-aa PPI): ~64 s/design BF16, ~112 s FP32.
Keep compilation/dynamic shapes off initially.

## 6. Outputs

Per sample: `<run_dir>/predictions/sample_NNNN/output/inference_samples/<name>/<name>_generated.cif`
plus `receipt.json`, and a batch `summary.json` with the full protocol
(adapter/engine/checkpoint hashes, precision, device, seed, memory). The generated
CIF has the designed binder chain + the fixed target chain(s). **Output existence
alone is not completion** — require rc=0 **and** `summary.json` + the expected
`receipt.json`(s) with matching input/seed identities.

## 7. Troubleshooting (from the handoff stall table)

| Observation | Check / response |
| --- | --- |
| Hangs after model build, idle tile | You are on the OLD stack. Confirm `$RFD4_PYTHON` + adapter paths; the resident adapter forces single-device. Do not reintroduce the sitecustomize shim. |
| Import probe times out (rc=124) | Cold imports off Lustre; the probe budget is 300 s. Raise it, or accept the worker's own cold import under its `seconds` budget. Not a hang. |
| `seed_everything(verbose=…)` / XPU errors | You loaded the old Lightning-2.3 env, not this deployment. Use `$RFD4_PYTHON`. |
| Many workers slow together | Check CPU affinity, thread counts (OMP=1), unique tiles, and Lustre/checkpoint I/O contention on simultaneous cold imports. |
| CIF exists but run didn't return | Not a completion receipt — verify rc=0 + summary.json + receipt.json. Keep partial receipts; never count missing output as done. |
| Timeout / SIGABRT | Preserve logs/seed/input; terminate only the owned process group. At most one retry after a concrete diagnosis; a second unexplained failure disables that config and reports to the operator. |

Do not enable broad CPU fallback or run on a login node to bypass the allocation
check (the adapter verifies it is on a node in `$PBS_NODEFILE`). SVD falls back to
CPU by design on this Torch build (logged); that is not permission for general
fallback.

## 8. Provenance / further reading

Source of truth: `RFD4_Aurora_Handoff/` at the loam repo root (dkastner) —
`SKILL.md`, `deployment_notes.md`, and `rfd4-aurora-verified/`. The handoff's
performance report (`reports/aurora_performance_20261023.md`, private Dezyme repo)
records the measured timings. This route has been qualified inside Loam only to
acceptance step 2 (see "Verified result" above); extend qualification (warm
multi-sample, failure/retry, new inputs) before relying on it at larger scale.
