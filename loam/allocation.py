"""One allocation that is already running, split into science and inference roles.

Ported from the moeen branch's single-allocation mode. The manifest is written by
whoever splits the allocation (the single-allocation coordinator); every process that
serves a role checks it runs in that job, on a host assigned to that role. Job ids are
the scheduler's own, read from the System's `job_id_variable`: numeric on some
schedulers, dotted on others (`4242.aurora-pbs-0001...`).
"""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path

ROLES = ('science', 'inference')
# A job id also names telemetry files, so it may not contain a path separator.
JOB_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')


def _unique(hosts: list[str]) -> bool:
    return len(hosts) == len(set(hosts))


def read_allocation(path: Path) -> dict:
    """The allocation manifest at `path`, if its roles partition its hosts."""
    data = json.loads(Path(path).resolve(strict=True).read_text())
    if data.get('mode') != 'single-allocation' or not JOB_ID.fullmatch(str(data.get('job_id', ''))):
        raise RuntimeError('Invalid single-allocation manifest')
    science, inference, nodes = data['science_nodes'], data['inference_nodes'], data['all_nodes']
    if not science or not inference or not _unique(nodes):
        raise RuntimeError('Empty roles or repeated allocation hosts')
    if (not _unique(science) or not _unique(inference) or set(science) & set(inference)
            or set(science) | set(inference) != set(nodes)):
        raise RuntimeError('Science and inference roles must partition the allocated hosts')
    return data


def validate_role(path: Path, role: str, *, job_id: str | None, hostname: str | None = None) -> dict:
    """The manifest, if this process runs in its job (`job_id`, None outside any) on a host assigned to `role`.

    The caller reads `job_id` from the environment, as `os.environ.get(system.job_id_variable)`.
    """
    data = read_allocation(path)
    host = (hostname or socket.gethostname()).split('.')[0]
    if job_id != data['job_id'] or role not in ROLES or host not in data[role + '_nodes']:
        raise RuntimeError('Process does not belong to its assigned job/role/host')
    return data


def pooled_overlap(science: dict, inference_jobs: list[dict], replicas: int) -> list[dict]:
    """When the science job and every inference replica ran at once: their intersection, never a sum.

    Jobs are `sacct` records with epoch start and end (`loam.systems.parse_sacct_parent_jobs`).
    Empty unless all `replicas` replicas are present and overlap the science job.
    """
    jobs = [science, *inference_jobs]
    if len(inference_jobs) != replicas or not all(job.get('start_epoch') and job.get('end_epoch') for job in jobs):
        return []
    start = max(job['start_epoch'] for job in jobs)
    end = min(job['end_epoch'] for job in jobs)
    if end <= start:
        return []
    return [{'start': start, 'end': end, 'seconds': end - start,
             'inference_jobs': [job['JobIDRaw'] for job in inference_jobs]}]
