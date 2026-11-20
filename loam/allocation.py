"""Explicit role ownership within one already granted Slurm allocation."""
from __future__ import annotations
import json, os, socket
from pathlib import Path


def validate_campaign_paths(cfg, campaign: Path) -> dict[str, str]:
    """Reject reference-state reuse before a single-allocation campaign starts."""
    base = Path(campaign).resolve()
    fields = {
        'state_dir': (cfg.state_dir, base / 'state'),
        'goal': (cfg.goal, base / 'goal.md'),
        'workspace_inputs': (cfg.workspace_inputs, base / 'inputs'),
        'execution.hpc.base_dir': (cfg.execution.hpc.base_dir, base),
        'execution.workdir': (cfg.execution.workdir, base / 'work'),
    }
    result = {}
    for name, (value, expected) in fields.items():
        actual = Path(value).resolve() if value is not None else None
        if actual != expected:
            raise RuntimeError(f'Single-allocation {name} must be {expected}; got {actual}')
        result[name] = str(actual)
    return result


def read_allocation(path: Path) -> dict:
    path=Path(path).resolve(strict=True);data=json.loads(path.read_text())
    if data.get('mode')!='single-allocation' or not str(data.get('job_id','')).isdigit():
        raise RuntimeError('Invalid single-allocation manifest')
    science=data['science_nodes'];inference=data['inference_nodes'];nodes=data['all_nodes']
    if not science or not inference or len(nodes)!=len(set(nodes)):
        raise RuntimeError('Empty roles or repeated allocation hosts')
    if len(science)!=len(set(science)) or len(inference)!=len(set(inference)) or set(science)&set(inference) or set(science)|set(inference)!=set(nodes):
        raise RuntimeError('Science and inference roles must partition the allocated hosts')
    return data


def validate_role(path: Path, role: str, *, hostname: str | None=None, job_id: str | None=None) -> dict:
    data=read_allocation(path);job=job_id or os.environ.get('SLURM_JOB_ID');host=(hostname or socket.gethostname()).split('.')[0]
    if job!=data['job_id'] or role not in ('science','inference') or host not in data[role+'_nodes']:
        raise RuntimeError('Process does not belong to its assigned Slurm job/role/host')
    return data
