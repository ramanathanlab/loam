---
name: rfd4-proteina
description: Generate protein backbones with RFD4-Proteina (flow-matching all-atom design) on Aurora Intel XPU — monomers, protein/DNA/small-molecule binders, motif scaffolding and enzyme design — from an annotated CIF, one resident worker per tile. Use for backbone/binder generation on Aurora, including Loam run_batch fan-out across tiles.
---

# RFD4-Proteina on Aurora (resident-worker route)

Runs against a **pinned, read-only deployment** maintained by David Kastner (FRAME-IDP), delivered
as `RFD4_Aurora_Handoff/` at the loam repo root. Requires `FRAME-IDP` group membership
(`id -nG | tr ' ' '\n' | grep FRAME-IDP`). Do not modify, reinstall or `pip install` into it;
report permission errors rather than changing permissions.

The adapter is single-device **by construction** (explicit `SingleDeviceStrategy` +
`LightningEnvironment`, `num_workers=0`), so it does not deadlock when launched from an MPI/PALS
worker — which is how Loam's `run_batch` calls it.

## Environment

```bash
RFD4_SHARE=/lus/flare/projects/FRAME-IDP/dkastner/Projects/IDeA
RFD4_VENV="$RFD4_SHARE/envs/rfd4-xpu-2026.1.0"                # the venv ROOT
RFD4_PYTHON="$RFD4_VENV/bin/python"                           # frameworks/2026.1.0, Lightning 2.6.6
RFD4_ADAPTER="$RFD4_SHARE/experiments/pmd_design_20261026T070010Z/source/v34"
RFD4_ENGINE="$RFD4_SHARE/experiments/aurora_performance_20261023T194443Z/source/rfd4_phase1"
```

**`run_batch(venv_path=...)` takes `$RFD4_VENV`, the ROOT — never `$RFD4_PYTHON`.**
`hpc_agent._python()` appends `bin/python` to whatever it is given, so passing the
interpreter yields `.../bin/python/bin/python` ("Not a directory") and every row
fails to launch. Observed 2026-11-11: a 48-row `rfd4_generation` batch
(`b-52527742c634`) was submitted with the interpreter path. `$RFD4_PYTHON` is for
direct shell invocation only.

`scripts/run_rfd4_resident.sh` sets the full env block itself (PYTHONPATH, `ZE_AFFINITY_MASK`,
single-threaded BLAS/OMP, `HF_HOME` + `HF_HUB_OFFLINE=1`, run-local `TMPDIR`), verifies the pinned
adapter/engine hashes, and runs an import+tile probe before the worker. Deployed worker-visible
copies, byte-identical to `scripts/`:

```
/flare/FRAME-IDP/msinclair/loam-hpc/rfd4/rfd4_run_row.py        # run_batch row runner
/flare/FRAME-IDP/msinclair/loam-hpc/rfd4/run_rfd4_resident.sh   # the wrapper
```

## Example

```bash
# One design on one tile.  run_rfd4_resident.sh <plan.json> <fresh_run_dir> <seconds>
echo '{"samples":[{"input":"/abs/annotated.cif","seed":101}]}' > plan.json
ZE_AFFINITY_MASK=0 bash scripts/run_rfd4_resident.sh "$PWD/plan.json" "$PWD/run01" 900
```

Fan out across tiles with `run_batch(gpus=1)` — one JSONL row per tile, up to the pool's
`total_tiles` (nodes×12) concurrently:

```json
{"id":"sat_00","input":"/abs/annotated.cif","seed":101,"seconds":900,
 "wrapper":"/flare/FRAME-IDP/msinclair/loam-hpc/rfd4/run_rfd4_resident.sh"}
```

```python
await c.call_tool("run_batch", {
    "script_path": "/flare/FRAME-IDP/msinclair/loam-hpc/rfd4/rfd4_run_row.py",
    "inputs_path": "<aurora>/rfd4_rows.jsonl",     # one row per tile
    "gpus": 1, "timeout_each": 1200,
    "tool": "rfd4_generate",            # registered with expects, see below
    "labels": {"tool": "rfd4-proteina", "stage": "rfd4_generate"}})
```

**Register the row runner once, with its output contract**, then submit batches
against it. Without a contract a row that exits 0 having written nothing counts as
a success, `batch_status` withholds `ok`, and no validator may claim `executed` —
a batch that really did produce 48 CIFs scores as nothing (observed 2026-11-11,
`b-764d0bb44360`: 48/48 rows `rc=0`, one 1.6 MB CIF each, `ok` withheld).

```python
await c.call_tool("register_tool", {
    "name": "rfd4_generate",
    "script_path": "/flare/FRAME-IDP/msinclair/loam-hpc/rfd4/rfd4_run_row.py",
    "params": {"id": "str", "input": "str", "seed": "int", "seconds": "int",
               "wrapper": "str"},
    "description": "RFD4-Proteina resident-worker backbone generation, one row per tile",
    "example": {"id": "sat_00", "input": "/abs/annotated.cif", "seed": 101,
                "seconds": 900,
                "wrapper": "/flare/FRAME-IDP/msinclair/loam-hpc/rfd4/run_rfd4_resident.sh"},
    "venv_path": "$RFD4_VENV",          # the ROOT
    "skip_smoke": True,                  # needs an XPU the client node may lack
    "expects": ["rfd4_run/predictions/**/*_generated.cif"]})
```

Globs are relative to the row's **attempt** directory (the row script's cwd) and are
checked *before* a successful attempt is promoted to the row top level, so match
`rfd4_run/...`, not the promoted path you see on disk afterwards. `list_tools`
shows what is already registered — register once, reuse across rounds.

## Inputs — annotated CIF

RFD4 inputs are **annotated CIFs** (`SA_*` columns + `pdbx_PDB_model_num` + a condition spec:
contig `"<binder_len>, /0, <target>"` with `C_CRDGRP`/`C_SEQ`/`C_HOT`). A generic PDB does not
encode fixed atoms, sequence constraints or ligand conditioning and will fail.

- **Build a binder input:** `scripts/make_rfd4_input.py --target <pdb> --target-chain <ch>
  --hotspots <resids> --binder-len <N|range> --target-range <a-b> --out <cif>`, which wraps the
  engine's `write_condition_spec_cif`. Validate a freshly built CIF on a 1-row smoke before
  fanning out.
- **Ready-made benchmark inputs** (matched to this engine, guaranteed valid):
  `$RFD4_ENGINE/data/benchmarks/` — `ppi/ppi_bb_only/` (e.g. `28_HER2_AAV.cif.zst`), `monomer/`,
  `motif_scaffolding/`, `small_molecule_binder/`, `dna_binder/`, `rna/`.

## Arguments (what the wrapper passes to the adapter)

`python -m idea.engines.rfd4_persistent --plan <plan.json> --output <dir> --device 0`

| flag | default in wrapper | meaning |
|---|---|---|
| `--plan` | required | `{"samples":[{"input":"<cif>","seed":N}, …]}` |
| `--output` | required | run dir (must not exist) |
| `--device` | 0 | logical XPU index; the physical tile comes from `ZE_AFFINITY_MASK` |
| `--precision` | `bf16-mixed` | or `32` (~1.75× slower) |
| `--matmul` | `highest` | `highest`/`high`/`medium` |
| `--deterministic` | on | fixed-seed execution |
| `--recycle-strategy` | `medium/midpeak` | recycling schedule |
| `--mode` | full | `full`/`partial`/`inverse_folding` |
| `--compile-length N` | off | keep compilation/dynamic shapes off initially |

Internally it composes `inference_experiment=quickstart` with `num_replicates=1 batch_size=1
num_workers=0 nn.tri_mult_backend=torch nn.attention_type=torch +idea_keep_model_on_xpu=true` —
native eager Torch kernels, model resident on XPU.

## Outputs

```
<run_dir>/predictions/sample_NNNN/output/inference_samples/<name>/<name>_generated.cif
<run_dir>/predictions/sample_NNNN/.../receipt.json
<run_dir>/predictions/summary.json      # adapter/engine/checkpoint hashes, precision, device, seed, memory
<run_dir>/{adapter_check.log,imports.log,worker.log}
```

The generated CIF holds the designed binder chain plus the fixed target chain(s). The row runner
also records `result.json {host,tile,seconds,rc,n_receipts,n_cifs}`.

## Must-know

- **Output existence alone is not completion.** Require `rc=0` **and** `summary.json` **and** the
  expected `receipt.json`(s) with matching input/seed identities.
- **One independent resident worker per assigned tile** (normally 12/node). These are independent
  samplers — never distribute one model across MPI ranks. Allocate ~4 CPUs per worker and keep
  `OMP/MKL/OPENBLAS_NUM_THREADS=1`.
- `ZE_AFFINITY_MASK` must be set **before** Torch import (the wrapper does this); changing it after
  import is too late.
- Each worker loads the model once and runs its plan serially. Do not issue concurrent calls against
  one resident process.
- **Set `"wrapper"` explicitly in every `run_batch` row.** The row runner's fallback chain ends at
  `$HPC_LOAM_HOME/.../run_rfd4_resident.sh`, and `HPC_LOAM_HOME` is not set in the worker env.
- Design count = rows × samples-per-plan, and distinct seeds are distinct designs (reusing a seed is
  a replay). Reach a large library by fanning out across tiles, not one long serial job.
- Size a fan-out to live free tiles (`pool_status` capacity minus in-flight work from
  `list_batches`); each row must finish inside the block walltime.
- Hangs after model build with an idle tile mean you are on an older stack — confirm `$RFD4_PYTHON`
  and the adapter paths, and do not reintroduce a `sitecustomize.py` Lightning shim.
- An import probe timeout (rc=124) is cold Lustre imports, not a hang; the probe budget is 300 s.
- Do not enable broad CPU fallback or run on a login node to bypass the allocation check (the
  adapter verifies the host is in `$PBS_NODEFILE`). SVD falls back to CPU by design on this Torch
  build and is logged; that is not permission for general fallback.
- On timeout/SIGABRT, preserve logs/seed/input and terminate only the owned process group. At most
  one retry after a concrete diagnosis; a second unexplained failure of a config should stop and
  report.
- Source of truth: `RFD4_Aurora_Handoff/` at the loam repo root (`SKILL.md`,
  `deployment_notes.md`, `rfd4-aurora-verified/`).
