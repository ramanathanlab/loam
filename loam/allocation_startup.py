"""Observe every assigned physical GPU before starting allocation services."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import time
from pathlib import Path
from typing import Any


def telemetry_ready_hosts(campaign: Path, allocation: dict, *, now: float | None = None) -> set[str]:
    """Read fresh, successful four-GPU samples belonging to this allocation.

    A previous allocation's samples, partial appends, wrong roles, failed NVML
    queries, and incomplete device lists cannot satisfy startup readiness.
    Reading is bounded even when a prior attempt left a long telemetry file.
    """
    now = time.time() if now is None else now
    expected = set(allocation['all_nodes'])
    science = set(allocation['science_nodes'])
    ready = set()
    for path in (Path(campaign) / 'telemetry').glob(allocation['job_id'] + '-*.jsonl'):
        try:
            with path.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                lines = stream.read().decode(errors='replace').splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('host'), str):
                    continue
                host = row['host'].split('.')[0]
                timestamp = float(row['timestamp'])
                if (host not in expected or host in ready or row['job_id'] != allocation['job_id']
                        or row.get('role') != ('science' if host in science else 'inference')
                        or row.get('nvidia_returncode') != 0 or not math.isfinite(timestamp)
                        or timestamp < float(allocation['started_at']) or not -5 <= now - timestamp <= 30):
                    continue
                devices = list(csv.reader(io.StringIO(row.get('gpu_csv', ''))))
                if (len(devices) != 4 or any(len(gpu) < 8 for gpu in devices)
                        or {int(gpu[0]) for gpu in devices} != {0, 1, 2, 3}
                        or not all(math.isfinite(float(gpu[2])) and 0 <= float(gpu[2]) <= 100 for gpu in devices)):
                    continue
                ready.add(host)
                break
            except (KeyError, TypeError, ValueError, csv.Error):
                continue
    return ready


async def wait_for_allocation_telemetry(process: Any, campaign: Path, allocation: dict,
                                       *, timeout: float = 900) -> set[str]:
    """Wait inside an already granted job; never submit, cancel, or restart it."""
    deadline = time.monotonic() + timeout
    expected = set(allocation['all_nodes'])
    if not expected:
        raise ValueError('Cannot establish telemetry readiness for an empty allocation')
    while True:
        if process.poll() is not None:
            raise RuntimeError('Allocation telemetry process exited before service startup')
        ready = telemetry_ready_hosts(campaign, allocation)
        if ready == expected:
            return ready
        if time.monotonic() >= deadline:
            raise TimeoutError('No fresh four-GPU telemetry on assigned nodes: ' + ', '.join(sorted(expected - ready)))
        await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))
