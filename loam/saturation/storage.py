"""Where staged jobs live, and the one ownership ledger every job directory shares.

An experiment's control root holds `experiment.json`, the ledger of the job
IDs it owns. Outputs may live on another filesystem, but only if that output
root links back to the same ledger, so ownership is never forked or replaced.
"""
from __future__ import annotations

from pathlib import Path

LEDGER = 'experiment.json'


def output_root(control_root: Path, requested: Path | None) -> Path:
    """The output root for new jobs, linked to the control root's ledger if it is elsewhere."""
    control_root = control_root.resolve()
    destination = Path(requested).resolve() if requested else control_root
    if destination == control_root:
        return destination
    ledger = control_root / LEDGER
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    linked = destination / LEDGER
    if linked.exists() or linked.is_symlink():
        if not linked.is_symlink() or linked.resolve() != ledger:
            raise RuntimeError(f'{destination} must share the exact existing ownership ledger {ledger}')
    else:
        linked.symlink_to(ledger)
    return destination


def experiment_control_root(job: Path) -> Path:
    """The control root that owns `job`, following an output root's ledger link back to it."""
    ledger = (Path(job).resolve().parent / LEDGER).resolve(strict=True)
    if ledger.name != LEDGER:
        raise RuntimeError(f'Ownership ledger of {job} resolves to {ledger}, not an {LEDGER}')
    return ledger.parent
