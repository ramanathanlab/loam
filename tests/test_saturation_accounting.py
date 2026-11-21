"""Requeued boot attempts retain their full allocated GPU-hour cost."""
import pytest

from loam.systems import AURORA, parse_sacct_attempts


def row(job, state, start, end, elapsed, tres='billing=5760,gres/gpu:gh200=80,gres/gpu=80,node=20'):
    return '|'.join([job, 'owned', state, '20', tres, '2026-11-16T21:00:00', start, end, str(elapsed),
                     'node-list', '0:0', 'None', ''])


def test_counts_requeued_allocation_and_excludes_pending_and_steps():
    boot = row('123', 'REQUEUED', '2026-11-16T22:53:06', '2026-11-16T22:53:42', 36)
    completed = row('123', 'COMPLETED', '2026-11-16T23:00:00', '2026-11-17T00:00:00', 3600)
    pending = row('123', 'PENDING', 'Unknown', 'Unknown', 0, '')
    step = row('123.batch', 'COMPLETED', '2026-11-16T23:00:00', '2026-11-17T00:00:00', 3600)
    foreign = row('456', 'COMPLETED', '2026-11-16T23:00:00', '2026-11-17T00:00:00', 3600)
    result = parse_sacct_attempts('\n'.join([boot, boot, pending, pending, completed, step, foreign]), '123',
                            gpus_per_node=4)
    assert result['allocation_attempts'] == 2
    assert result['total_allocated_seconds'] == 3636
    assert result['total_allocated_GPU_seconds'] / 3600 == 80.8
    assert len(result['unallocated_records']) == 2
    assert all(r['JobIDRaw'] == '123' for r in result['allocations'])


def test_gpu_count_is_checked_against_the_systems_node_shape():
    # 20 nodes with 80 GPUs is four per node; a twelve-GPU node shape must reject it.
    completed = row('123', 'COMPLETED', '2026-11-16T23:00:00', '2026-11-17T00:00:00', 3600)
    with pytest.raises(ValueError, match='12 GPUs per node'):
        parse_sacct_attempts(completed, '123', gpus_per_node=12)
    tres = 'billing=1,gres/gpu=240,node=20'
    twelve = row('123', 'COMPLETED', '2026-11-16T23:00:00', '2026-11-17T00:00:00', 3600, tres)
    assert parse_sacct_attempts(twelve, '123', gpus_per_node=12)['total_allocated_GPU_seconds'] == 240 * 3600


def test_a_system_without_allocation_accounting_refuses_to_guess():
    with pytest.raises(NotImplementedError, match='allocation accounting on aurora'):
        AURORA.allocation_attempts('123', gpus_per_node=12)
    with pytest.raises(NotImplementedError, match='GPU telemetry on aurora'):
        AURORA.gpu_busy({'gpu_csv': ''}, gpus_per_node=12)
