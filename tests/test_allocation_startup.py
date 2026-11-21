"""Startup must observe every assigned physical GPU, rather than trust a live launcher.

Ported from the moeen branch, where the GPU count was fixed at Jupiter's four; here it is
the System's GPUs per node.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

import loam.allocation_startup as startup
from loam.systems import SYSTEMS

ALLOCATION = {'job_id': '123', 'all_nodes': ['science-a', 'inference-b'],
              'science_nodes': ['science-a'], 'inference_nodes': ['inference-b'], 'started_at': 100.0}
GPUS = 4


def devices(count):
    return '\n'.join(f'{i}, UUID-{i}, 0, 0, 10, 97871, 70, 35' for i in range(count))


def sample(host='science-a', gpus=GPUS, **changes):
    row = {'job_id': '123', 'host': host + '.cluster.internal', 'timestamp': 105.0,
           'role': 'science' if host == 'science-a' else 'inference', 'nvidia_returncode': 0,
           'gpu_csv': devices(gpus)}
    row.update(changes)
    return row


def append(campaign, row, suffix=''):
    folder = campaign / 'telemetry'
    folder.mkdir(exist_ok=True)
    with (folder / ('123-' + str(row['host']) + '.jsonl')).open('a') as stream:
        stream.write(json.dumps(row) + '\n' + suffix)


def ready(campaign, now=110, gpus_per_node=GPUS):
    return startup.telemetry_ready_hosts(campaign, ALLOCATION, gpus_per_node=gpus_per_node, now=now)


@pytest.mark.parametrize(('system', 'gpus'), [('aurora', 12), ('polaris', 4)])
def test_a_node_reporting_the_systems_gpu_count_is_ready(tmp_path, system, gpus):
    gpus_per_node = SYSTEMS[system].gpus_per_node
    append(tmp_path, sample(gpus=gpus))

    assert ready(tmp_path, gpus_per_node=gpus_per_node) == {'science-a'}


@pytest.mark.parametrize(('system', 'gpus'), [('aurora', 4), ('aurora', 11), ('polaris', 12), ('polaris', 3)])
def test_a_node_reporting_the_wrong_gpu_count_is_not_ready(tmp_path, system, gpus):
    append(tmp_path, sample(gpus=gpus))

    assert ready(tmp_path, gpus_per_node=SYSTEMS[system].gpus_per_node) == set()


def test_fresh_physical_samples_are_required_per_assigned_role(tmp_path):
    append(tmp_path, sample())
    assert ready(tmp_path) == {'science-a'}
    append(tmp_path, sample('inference-b'))
    assert ready(tmp_path) == set(ALLOCATION['all_nodes'])


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
    row = sample()
    row.update(changes)
    append(tmp_path, row)
    assert ready(tmp_path) == set()


def test_old_samples_and_partial_appends_do_not_replace_live_evidence(tmp_path):
    append(tmp_path, sample('unassigned'))
    append(tmp_path, sample(timestamp=50))
    append(tmp_path, sample(), suffix='{"incomplete":')
    assert ready(tmp_path) == {'science-a'}
    assert ready(tmp_path, now=150) == set()


def test_wait_requires_every_node_and_propagates_collector_failure(tmp_path, monkeypatch):
    append(tmp_path, sample(timestamp=100))
    monkeypatch.setattr(startup.time, 'time', lambda: 110)
    calls = []

    async def provide_second_node(delay):
        calls.append(delay)
        append(tmp_path, sample('inference-b', timestamp=109))

    monkeypatch.setattr(startup.asyncio, 'sleep', provide_second_node)
    running = SimpleNamespace(poll=lambda: None)
    result = asyncio.run(startup.wait_for_allocation_telemetry(running, tmp_path, ALLOCATION, gpus_per_node=GPUS))
    assert result == set(ALLOCATION['all_nodes']) and calls
    with pytest.raises(RuntimeError, match='process exited'):
        asyncio.run(startup.wait_for_allocation_telemetry(SimpleNamespace(poll=lambda: 1), tmp_path, ALLOCATION,
                                                          gpus_per_node=GPUS))


def test_timeout_cannot_silently_accept_a_partial_allocation(tmp_path):
    running = SimpleNamespace(poll=lambda: None)
    with pytest.raises(TimeoutError, match='12-GPU telemetry.*inference-b'):
        asyncio.run(startup.wait_for_allocation_telemetry(running, tmp_path, ALLOCATION, gpus_per_node=12,
                                                          timeout=0))
    with pytest.raises(ValueError, match='empty allocation'):
        asyncio.run(startup.wait_for_allocation_telemetry(running, tmp_path, {**ALLOCATION, 'all_nodes': []},
                                                          gpus_per_node=GPUS, timeout=0))
