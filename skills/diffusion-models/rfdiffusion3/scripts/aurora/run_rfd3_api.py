#!/usr/bin/env python
"""Run RFD3 in-process via the Python API. THE way to run RFD3 inside a run_batch row.

Import `run_design` from a row script, or call this file as a CLI.

WHY NOT THE `rfd3` BINARY
------------------------
`run_inference.py` does `RFD3InferenceEngine(**RFD3InferenceConfig(**cfg))` and runs
it immediately. No `RFD3InferenceConfig` field reaches `trainer.strategy`, and any
extra top-level key is splatted into the dataclass and raises
`TypeError: ... unexpected keyword argument`. So `trainer.strategy` keeps whatever
the CHECKPOINT's training config carried -- DDP -- and
`BaseInferenceEngine._construct_trainer` calls `trainer.fabric.launch()`, which
initialises MPI. Inside a `run_batch` row (already launched under EL's
`MpiExecLauncher`) that MPI job cannot route over the Slingshot NIC:

    MPIDI_OFI_handle_cq_error(595): OFI poll failed (default nic=cxi0: No route to host)

and the row dies with rc=143 after ~15-60s.

`BaseInferenceEngine` is lazily initialised: construction only fills
`self.overrides`; the checkpoint config is merged later inside `initialize()`
(`cfg = merge(cfg, self.overrides)`). That window is the only place
`trainer.strategy` can be set, and it is reachable from the API but not the CLI.
Running in-process also avoids spawning a second MPI-inheriting subprocess per row.

`trainer_overrides=` cannot be passed as a kwarg: rfd3's `__init__` hardcodes it in
its own `super()` call, so supplying it raises "multiple values for keyword
argument". `_assign_override` after construction is the way in.

STATUS: the override mechanism is verified (the key lands in
`engine.overrides["trainer"]["strategy"]`, which is what `initialize()` merges).
Whether `xpu_single` fully avoids `MPI_Init` on a node was NOT yet confirmed at the
time of writing -- if a row still dies on `cxi0`, try `FI_PROVIDER=shm` /
`I_MPI_FABRICS=shm` in the row environment as well, and record what you find here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Single-device XPU strategy, registered by
# foundry/utils/xpu/single_xpu_strategy.py (strategy_name = "xpu_single").
STRATEGY = os.environ.get("RFD3_STRATEGY", "xpu_single")
DEFAULT_CKPT = "rfd3"


def checkpoint_dirs() -> str:
    """Where the checkpoint registry looks, from the Site profile -- never a baked-in path."""
    configured = os.environ.get("FOUNDRY_CHECKPOINT_DIRS")
    if configured:
        return configured
    software = os.environ.get("LOAM_SOFTWARE_ROOT")
    if not software:
        raise SystemExit("Set FOUNDRY_CHECKPOINT_DIRS, or LOAM_SOFTWARE_ROOT so that "
                         "$LOAM_SOFTWARE_ROOT/checkpoints can be used.")
    return str(Path(software) / "checkpoints")


def _stage(path: Path | None, stage: str, **kw: Any) -> None:
    """Append a durable stage marker. A row killed by the PBS wall still leaves a trail."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps({"stage": stage, "t": time.time(), **kw}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        pass


def build_engine(
    *,
    ckpt: str = DEFAULT_CKPT,
    seed: int = 0,
    timesteps: int = 200,
    diffusion_batch_size: int = 1,
    low_memory_mode: bool = True,
    dump_trajectories: bool = False,
    strategy: str = STRATEGY,
):
    """Construct the engine as the CLI would, then force a single-device strategy."""
    from rfd3.engine import RFD3InferenceEngine

    os.environ["FOUNDRY_CHECKPOINT_DIRS"] = checkpoint_dirs()
    engine = RFD3InferenceEngine(
        skip_existing=False,
        json_keys_subset=None,
        prevalidate_inputs=True,
        diffusion_batch_size=diffusion_batch_size,
        inference_sampler={"num_timesteps": timesteps},
        specification=None,
        global_prefix=None,
        cleanup_guideposts=True,
        cleanup_virtual_atoms=True,
        read_sequence_from_sequence_head=False,
        output_full_json=False,
        dump_prediction_metadata_json=True,
        dump_trajectories=dump_trajectories,
        align_trajectory_structures=False,
        low_memory_mode=low_memory_mode,
        ckpt_path=ckpt,
        num_nodes=1,
        devices_per_node=1,
        seed=seed,
    )
    # Reachable only before initialize(); see module docstring.
    engine._assign_override("trainer.strategy", strategy)
    return engine


def run_design(
    *,
    spec: str | Path,
    out_dir: str | Path,
    seed: int = 0,
    timesteps: int = 200,
    diffusion_batch_size: int = 1,
    n_batches: int = 1,
    ckpt: str = DEFAULT_CKPT,
    strategy: str = STRATEGY,
    stage_log: str | Path | None = None,
) -> dict[str, Any]:
    """Generate backbones for one design spec. Returns a row-record-shaped dict.

    `ok` is true only when a structure was actually written: a clean exit that
    produced nothing is NOT a success, so a caller can trust this field and
    declare the same globs as run_batch `expects`.
    """
    out = Path(out_dir)
    marker = Path(stage_log) if stage_log else out / "rfd3_stages.jsonl"
    record: dict[str, Any] = {
        "spec": str(spec),
        "out_dir": str(out),
        "seed": seed,
        "strategy_requested": strategy,
        "host": os.uname().nodename,
        "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK"),
    }
    _stage(marker, "started", host=record["host"], ze=record["ze_affinity_mask"])
    try:
        out.mkdir(parents=True, exist_ok=True)
        engine = build_engine(
            ckpt=ckpt, seed=seed, timesteps=timesteps,
            diffusion_batch_size=diffusion_batch_size, strategy=strategy,
        )
        record["strategy_applied"] = (
            engine.overrides.get("trainer", {}).get("strategy") == strategy
        )
        _stage(marker, "engine_constructed", strategy_applied=record["strategy_applied"])

        engine.initialize()          # _construct_trainer -> fabric.launch(): MPI risk is HERE
        _stage(marker, "engine_initialized")

        engine.run(inputs=str(spec), out_dir=str(out), n_batches=n_batches)
        _stage(marker, "run_returned")

        produced = sorted(str(p) for p in out.glob("*.cif*"))
        metrics = sorted(str(p) for p in out.glob("*.json")
                         if Path(p).name not in {"spec.json", "metadata.json"})
        record.update(produced=produced, metrics=metrics, ok=bool(produced))
        _stage(marker, "done", n_produced=len(produced))
    except BaseException as exc:      # noqa: BLE001 - a row reports, never raises
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"[:600]
        record["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        _stage(marker, "error", error=record["error"])
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="RFD3 design spec JSON")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--diffusion-batch-size", type=int, default=1)
    ap.add_argument("--n-batches", type=int, default=1)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--strategy", default=STRATEGY)
    ap.add_argument("--check-only", action="store_true",
                    help="construct + verify the strategy override, then stop "
                         "(no checkpoint load, no GPU needed)")
    args = ap.parse_args()

    if args.check_only:
        try:
            engine = build_engine(ckpt=args.ckpt, seed=args.seed, strategy=args.strategy)
            landed = engine.overrides.get("trainer", {}).get("strategy") == args.strategy
            print(json.dumps({"check_only": True, "strategy_applied": landed,
                              "trainer_overrides": engine.overrides.get("trainer", {}),
                              "ok": landed}, indent=1, default=str))
            return 0 if landed else 1
        except BaseException as exc:  # noqa: BLE001
            print(json.dumps({"check_only": True, "ok": False,
                              "error": f"{type(exc).__name__}: {exc}"}, indent=1))
            return 1

    record = run_design(
        spec=args.spec, out_dir=args.out_dir, seed=args.seed,
        timesteps=args.timesteps, diffusion_batch_size=args.diffusion_batch_size,
        n_batches=args.n_batches, ckpt=args.ckpt, strategy=args.strategy,
    )
    print(json.dumps(record, indent=1, default=str))
    return 0 if record.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
