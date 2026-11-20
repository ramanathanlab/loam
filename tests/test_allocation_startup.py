"""Startup must observe all physical devices, rather than trust a live srun."""
import asyncio
import json
from types import SimpleNamespace

import pytest

import loam.allocation_startup as startup


ALLOCATION = {'job_id': '123', 'all_nodes': ['science-a', 'inference-b'],
              'science_nodes': ['science-a'], 'inference_nodes': ['inference-b'], 'started_at': 100.0}


def sample(host='science-a', **changes):
    row = {'job_id': '123', 'host': host + '.jupiter.internal', 'timestamp': 105.0,
           'role': 'science' if host == 'science-a' else 'inference', 'nvidia_returncode': 0,
           'gpu_csv': '\n'.join(f'{i}, UUID-{i}, 0, 0, 10, 97871, 70, 35' for i in range(4))}
    row.update(changes)
    return row


def append(campaign, row, suffix=''):
    folder = campaign/'telemetry';folder.mkdir(exist_ok=True)
    path = folder/('123-' + str(row['host']) + '.jsonl')
    with path.open('a') as stream:stream.write(json.dumps(row) + '\n' + suffix)


def test_fresh_physical_samples_are_required_per_assigned_role(tmp_path):
    append(tmp_path, sample())
    assert startup.telemetry_ready_hosts(tmp_path, ALLOCATION, now=110) == {'science-a'}
    append(tmp_path, sample('inference-b'))
    assert startup.telemetry_ready_hosts(tmp_path, ALLOCATION, now=110) == set(ALLOCATION['all_nodes'])


@pytest.mark.parametrize('changes', [
    {'job_id': 'other'}, {'role': 'inference'}, {'host': None}, {'timestamp': 99}, {'timestamp': 70},
    {'timestamp': 120}, {'timestamp': float('nan')}, {'nvidia_returncode': 1},
    {'gpu_csv': '\n'.join(f'{i}, UUID, 0, 0, 0, 1, 2, 3' for i in range(3))},
    {'gpu_csv': '\n'.join('0, UUID, 0, 0, 0, 1, 2, 3' for _ in range(4))},
    {'gpu_csv': '\n'.join(f'{i}, UUID, NaN, 0, 0, 1, 2, 3' for i in range(4))},
    {'gpu_csv': '\n'.join(f'{i}, UUID, 101, 0, 0, 1, 2, 3' for i in range(4))},
    {'gpu_csv': '0, incomplete'},
])
def test_invalid_stale_or_foreign_data_cannot_release_startup(tmp_path, changes):
    row = sample();row.update(changes);append(tmp_path, row)
    assert startup.telemetry_ready_hosts(tmp_path, ALLOCATION, now=110) == set()


def test_old_samples_and_partial_appends_do_not_replace_live_evidence(tmp_path):
    append(tmp_path, sample('unassigned'))
    append(tmp_path, sample(timestamp=50))
    append(tmp_path, sample(), suffix='{"incomplete":')
    assert startup.telemetry_ready_hosts(tmp_path, ALLOCATION, now=110) == {'science-a'}
    assert startup.telemetry_ready_hosts(tmp_path, ALLOCATION, now=150) == set()


def test_wait_requires_every_node_and_propagates_collector_failure(tmp_path, monkeypatch):
    append(tmp_path, sample(timestamp=100))
    monkeypatch.setattr(startup.time, 'time', lambda: 110)
    calls = []
    async def provide_second_node(delay):
        calls.append(delay);append(tmp_path, sample('inference-b', timestamp=109))
    monkeypatch.setattr(startup.asyncio, 'sleep', provide_second_node)
    result = asyncio.run(startup.wait_for_allocation_telemetry(SimpleNamespace(poll=lambda: None), tmp_path, ALLOCATION))
    assert result == set(ALLOCATION['all_nodes']) and calls
    with pytest.raises(RuntimeError, match='process exited'):
        asyncio.run(startup.wait_for_allocation_telemetry(SimpleNamespace(poll=lambda: 1), tmp_path, ALLOCATION))


def test_timeout_cannot_silently_accept_a_partial_allocation(tmp_path):
    with pytest.raises(TimeoutError, match='inference-b'):
        asyncio.run(startup.wait_for_allocation_telemetry(SimpleNamespace(poll=lambda: None), tmp_path, ALLOCATION, timeout=0))
    with pytest.raises(ValueError, match='empty allocation'):
        asyncio.run(startup.wait_for_allocation_telemetry(SimpleNamespace(poll=lambda: None), tmp_path,
                    {**ALLOCATION, 'all_nodes': []}, timeout=0))
