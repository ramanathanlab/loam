"""Observe every assigned physical GPU before starting services inside an allocation.

Ported from the moeen branch's single-allocation startup, where the device count was
Jupiter's four; here the System (with the campaign's node shape) says how many GPUs a
node has and how to read a sample. A sample is one host's line in
`<campaign>/telemetry/<job_id>-<host>.jsonl`. A System whose GPU telemetry loam cannot
read cannot be started this way.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from .systems import System

# Only the end of a telemetry file is read: an earlier attempt may have left a long one.
TAIL_BYTES = 65536
# How old (or, allowing for clock skew, how far ahead) a sample may be and still count as live.
FRESH_SECONDS = 30
SKEW_SECONDS = 5


def _tail_lines(path: Path) -> list[str]:
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - TAIL_BYTES))
        return stream.read().decode(errors='replace').splitlines()


def _healthy(row: dict, allocation: dict, *, host: str, science: set[str], system: System, now: float) -> bool:
    """A fresh, successful sample from this allocation and host role, covering every GPU on the node."""
    timestamp = float(row['timestamp'])
    if (row['job_id'] != allocation['job_id']
            or row.get('role') != ('science' if host in science else 'inference')
            or not math.isfinite(timestamp)
            or timestamp < float(allocation['started_at'])
            or not -SKEW_SECONDS <= now - timestamp <= FRESH_SECONDS):
        return False
    return system.every_gpu_reported(row)


def telemetry_ready_hosts(campaign: Path, allocation: dict, *, system: System,
                          now: float | None = None) -> set[str]:
    """The allocation's hosts whose latest sample shows each of the System's GPUs healthy.

    A previous allocation's samples, partial appends, wrong roles, failed queries, and
    device lists of the wrong size cannot satisfy startup readiness.
    """
    system.check_gpu_telemetry()
    now = time.time() if now is None else now
    expected = set(allocation['all_nodes'])
    science = set(allocation['science_nodes'])
    ready = set()
    for path in (Path(campaign) / 'telemetry').glob(allocation['job_id'] + '-*.jsonl'):
        try:
            lines = _tail_lines(path)
        except OSError:
            continue
        for line in reversed(lines):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('host'), str):
                    continue
                host = row['host'].split('.')[0]
                if host not in expected or host in ready:
                    continue
                if _healthy(row, allocation, host=host, science=science, system=system, now=now):
                    ready.add(host)
                    break
            except (KeyError, TypeError, ValueError):
                continue
    return ready


async def wait_for_allocation_telemetry(process: Any, campaign: Path, allocation: dict, *, system: System,
                                        timeout: float = 900) -> set[str]:
    """Wait until every host shows all its GPUs; never submit, cancel, or restart the allocation.

    `process` is the telemetry collector (anything with `poll()`); if it exits first, startup fails.
    """
    system.check_gpu_telemetry()
    deadline = time.monotonic() + timeout
    expected = set(allocation['all_nodes'])
    if not expected:
        raise ValueError('Cannot establish telemetry readiness for an empty allocation')
    while True:
        if process.poll() is not None:
            raise RuntimeError('Allocation telemetry process exited before service startup')
        ready = telemetry_ready_hosts(campaign, allocation, system=system)
        if ready == expected:
            return ready
        if time.monotonic() >= deadline:
            raise TimeoutError(f'No fresh {system.gpus_per_node}-GPU telemetry on assigned nodes: '
                               + ', '.join(sorted(expected - ready)))
        await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))
