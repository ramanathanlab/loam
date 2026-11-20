"""Requeued boot attempts retain their full allocated GPU-hour cost."""
import importlib
from pathlib import Path

import pytest


def test_counts_requeued_allocation_and_excludes_pending_and_steps(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=importlib.import_module('saturation_accounting')
    def row(job,state,start,end,elapsed,tres='billing=5760,gres/gpu:gh200=80,gres/gpu=80,node=20'):
        return '|'.join([job,'owned',state,'20',tres,'2026-11-16T21:00:00',start,end,str(elapsed),'node-list','0:0','None',''])
    boot=row('123','REQUEUED','2026-11-16T22:53:06','2026-11-16T22:53:42',36)
    completed=row('123','COMPLETED','2026-11-16T23:00:00','2026-11-17T00:00:00',3600)
    pending=row('123','PENDING','Unknown','Unknown',0,'')
    step=row('123.batch','COMPLETED','2026-11-16T23:00:00','2026-11-17T00:00:00',3600)
    foreign=row('456','COMPLETED','2026-11-16T23:00:00','2026-11-17T00:00:00',3600)
    result=module.parse_attempts('\n'.join([boot,boot,pending,pending,completed,step,foreign]),'123')
    assert result['allocation_attempts']==2
    assert result['total_allocated_seconds']==3636
    assert result['total_allocated_GPU_seconds']/3600==80.8
    assert len(result['unallocated_records'])==2
    assert all(r['JobIDRaw']=='123' for r in result['allocations'])
