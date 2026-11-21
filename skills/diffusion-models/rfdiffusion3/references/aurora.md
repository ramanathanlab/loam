# RFdiffusion3 on Aurora (Intel XPU)

> **UNRESOLVED under `run_batch`: rows hang, do not use this for width.**
> Every row stalls inside `initialize()` right after
> `foundry.utils.ddp: Intel XPU detected - using XPU accelerator`, then burns its
> whole `timeout_each`. Measured 2026-11-11 at 48-row width: 46 rows hung with
> leases held, 2 exited `rc=15` at 142s, no output. `trainer.strategy=xpu_single`
> (below) removes the MPI *abort* but NOT the stall, so it makes the failure slow
> and silent instead of fast and loud.
>
> **For backbone generation at width use the `rfd4-proteina` skill instead.** Its
> resident-worker route is verified under `run_batch` and produced 48/48 real
> backbones at ~290s/row on 2026-11-11. Use RFD3 only interactively, on one tile,
> where a stall is visible and cheap.

Checkout: `$LOAM_SOFTWARE_ROOT/foundry` (do not edit it; write outputs elsewhere).

## Environment

The venv is built against the node-default `frameworks/2026.1.0` module and takes torch/lightning
from it through `include-system-site-packages`, so **do not swap frameworks modules**. Precision
stays `bf16-mixed`. Pin one GPU tile with `ZE_AFFINITY_MASK=<int>` — a plain integer (`4.0` hides
the device).

Checkpoints are found through `$FOUNDRY_CHECKPOINT_DIRS`; set it in the Site profile's `worker_init`
(see `examples/site-profiles/aurora.yaml`), else the runner falls back to
`$LOAM_SOFTWARE_ROOT/checkpoints` and refuses if neither is set. That directory holds
`rfd3_latest.ckpt` (alias `rfd3`, the default), `rfd3na_1190.ckpt` (nucleic acids),
`rf3_foundry_01_24_latest_remapped.ckpt` (RF3 prediction) and `proteinmpnn_v_48_020.pt`.

## How to run

Use `scripts/aurora/run_rfd3_api.py`. It drives the engine in-process and is the only supported
entrypoint here: the `rfd3` CLI cannot reach `trainer.strategy`, so it inherits DDP from the
checkpoint's training config and `fabric.launch()` initialises MPI (the script's docstring has the
full argument).

```python
import sys; sys.path.insert(0, "<this skill>/scripts/aurora")
from run_rfd3_api import run_design

rec = run_design(spec="/abs/spec.json", out_dir="/abs/out", seed=1234, timesteps=200)
# rec["ok"] is True only if a structure was written; rec["produced"] lists the CIFs.
```

or as a command: `python run_rfd3_api.py --spec SPEC.json --out-dir DIR [--seed N]`.
`--check-only` constructs the engine and verifies the strategy override without loading a
checkpoint or touching a GPU.

One process per XPU tile. In a `run_batch` row use `gpus=1` and let the row inherit its pinned
`ZE_AFFINITY_MASK` — subject to the hang above.

## What is known about the hang

Nothing in the inference path overrides `trainer.strategy`, so it inherits DDP from the checkpoint's
training config and `fabric.launch()` initialises MPI. Inside a `run_batch` row (already under EL's
`MpiExecLauncher`) that MPI job cannot route over the Slingshot NIC, and the row died with
`MPIDI_OFI_handle_cq_error ... nic=cxi0: No route to host`, rc=143. `run_rfd3_api.build_engine` sets
`trainer.strategy=xpu_single` in the only window where it is reachable — after construction, before
`initialize()` — and that DOES remove the `cxi0` abort. It does NOT make the row complete: it then
stalls in `initialize()` indefinitely. Dropping `PMI_*`/`PMIX_*`/`PALS_*` from the row environment
also removes the abort and also does not fix the stall.

Still unidentified: what `initialize()` blocks on once distributed init is sidestepped — checkpoint
load off Lustre and a distributed barrier are both consistent with the evidence, and nobody has
instrumented it. Resolve that before trusting RFD3 at width again.

**`set -u` + Lmod don't mix.** A wrapper that runs `module load` under `set -euo pipefail` dies in
Lmod's init; use `set -eo pipefail` until after the module is loaded.

`xpu_single` is registered by `foundry/utils/xpu/single_xpu_strategy.py`; override it with
`$RFD3_STRATEGY` only to test a hypothesis about the stall.

## Resource index

- `read_skill(name="rfdiffusion3", file="SKILL.md")` — the System-neutral method, spec format and outputs
- `read_skill(name="rfdiffusion3", file="references/aurora.md")` — this file
- `read_skill(name="rfdiffusion3", file="scripts/aurora/run_rfd3_api.py")` — the in-process runner
