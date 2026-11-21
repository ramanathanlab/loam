"""Per-host telemetry inside a single-allocation job: CPU, memory and every GPU, every few seconds.

    python -m loam.telemetry CAMPAIGN --allocation JOB/allocation.json

runs once on each host of the allocation (the coordinator starts it as one step across all of
them) and appends one JSON line per sample to CAMPAIGN/telemetry/<job_id>-<host>.jsonl. The GPU
query and its format are the System's; `loam.allocation_startup` waits for these samples before
starting services, and `loam.saturation` reads them afterwards. Ported from the moeen branch's
sample_jupiter_telemetry.py.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import time
from pathlib import Path

from .allocation import read_allocation, validate_role
from .config import load
from .systems import System


def cpu_counters() -> list[int]:
    return [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]


def memory_used_percent() -> float:
    memory = {line.split(':', 1)[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    return 100 * (1 - memory['MemAvailable'] / memory['MemTotal'])


def sample(system: System, *, host: str, job_id: str, role: str, previous: list[int] | None = None,
           now: float | None = None) -> tuple[dict, list[int] | None]:
    """One telemetry row for this host, and the CPU counters to diff the next sample against."""
    row = {'timestamp': time.time() if now is None else now, 'host': host, 'job_id': job_id, 'role': role}
    counters = previous
    try:
        counters = cpu_counters()
        row['host_memory_used_percent'] = memory_used_percent()
        row['logical_cpus'] = os.cpu_count()
        if previous:
            total = sum(counters) - sum(previous)
            if total > 0:
                idle, iowait = counters[3] - previous[3], counters[4] - previous[4]
                row['cpu_busy_percent'] = 100 * (1 - (idle + iowait) / total)
                row['cpu_iowait_percent'] = 100 * iowait / total
    except (OSError, ValueError, KeyError, IndexError) as error:
        row['cpu_memory_error'] = type(error).__name__
    row.update(system.gpu_sample())
    return row, counters


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='python -m loam.telemetry', description=__doc__.split('\n\n')[0])
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--allocation', type=Path, required=True, help="the job's allocation.json")
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args(argv)
    system = load(args.campaign / 'campaign.yaml').execution.hpc.system_definition()
    system.check_gpu_telemetry()
    host = socket.gethostname().split('.')[0]
    allocation = read_allocation(args.allocation)
    role = 'inference' if host in allocation['inference_nodes'] else 'science'
    job_id = os.environ.get(system.job_id_variable)
    validate_role(args.allocation, role, job_id=job_id, hostname=host)
    folder = args.campaign / 'telemetry'
    folder.mkdir(exist_ok=True)
    lock = (folder / f'{job_id}-{host}.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another sampler already records job {job_id} on {host}') from None
    destination = folder / f'{job_id}-{host}.jsonl'
    previous = None
    while True:
        row, previous = sample(system, host=host, job_id=job_id, role=role, previous=previous)
        with destination.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
