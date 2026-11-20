"""Checkpoint coalescing must preserve every field and all final updates."""
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import loam.jnana_checkpoint as checkpoint


PAYLOAD = {'hypotheses': [{'hypothesis_id': 'h1', 'summary': 'β binder', 'elo_rating': 1200.5}],
           'experiments': {'e': {'values': [1, 2, None]}}, 'analyses': {'a': {'score': 0.3}},
           'papers': {'p': 'paper'}, 'metadata': {'research_goal': 'goal', 'updates': 0},
           'agent_states': {'agent': 'running'}, 'datasets': {'d': ['a', 'b']},
           'training_plans': {'t': {'epochs': 2}}, 'evaluations': {'v': False},
           'tournament_state': {'matches': [{'match_id': '1', 'winner': 'h1'}], 'statistics': {'n': 1}}}


class Hypothesis:
    def __init__(self, data):self.data = data
    def to_dict(self):return dict(self.data)


class Memory:
    def __init__(self, path):
        self.storage_path = str(path) if path else None
        self._lock = threading.RLock()
        data = copy.deepcopy(PAYLOAD)
        self.hypotheses = {x['hypothesis_id']: Hypothesis(x) for x in data.pop('hypotheses')}
        self.__dict__.update(data)
    def save(self):raise AssertionError('Original per-field writer should be replaced')
    def _save_if_needed(self):self.save()


def test_every_field_survives_and_explicit_phase_flush_bypasses_coalescing(tmp_path, monkeypatch):
    clock = [100.0];monkeypatch.setattr(checkpoint.time, 'monotonic', lambda: clock[0])
    path = tmp_path/'jnana.json';memory = Memory(path)
    assert checkpoint.install_checkpoint_policy(memory)
    memory._save_if_needed()
    assert json.loads(path.read_text()) == PAYLOAD
    memory.metadata['updates'] = 1;clock[0] = 101
    memory._save_if_needed()
    assert json.loads(path.read_text())['metadata']['updates'] == 0
    memory.save()
    assert json.loads(path.read_text())['metadata']['updates'] == 1
    memory.metadata['updates'] = 2;clock[0] = 107;memory._save_if_needed()
    assert json.loads(path.read_text())['metadata']['updates'] == 2
    records = [json.loads(s) for s in (tmp_path/'jnana-checkpoint-timings.jsonl').read_text().splitlines()]
    assert [r['reason'] for r in records] == ['autosave', 'explicit', 'autosave']
    assert records[-1]['json_bytes'] == path.stat().st_size


def test_failed_replace_keeps_old_valid_checkpoint_and_removes_only_its_temp(tmp_path, monkeypatch):
    path = tmp_path/'jnana.json';memory = Memory(path);checkpoint.install_checkpoint_policy(memory);memory.save()
    old = path.read_bytes();memory.metadata['updates'] = 3
    def fail(source, destination):raise OSError('simulated storage failure')
    monkeypatch.setattr(checkpoint.os, 'replace', fail)
    with pytest.raises(OSError, match='storage failure'):memory.save()
    assert path.read_bytes() == old and json.loads(path.read_text()) == PAYLOAD
    assert not list(tmp_path.glob('.jnana.json.*.tmp'))


def test_old_namespace_is_valid_until_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path/'jnana.json';memory = Memory(path);checkpoint.install_checkpoint_policy(memory);memory.save()
    memory.metadata['updates'] = 4;real_replace = checkpoint.os.replace;observed = []
    def inspect_replace(source, destination):
        observed.append(json.loads(Path(destination).read_text())['metadata']['updates'])
        assert json.loads(Path(source).read_text())['metadata']['updates'] == 4
        real_replace(source, destination)
    monkeypatch.setattr(checkpoint.os, 'replace', inspect_replace);memory.save()
    assert observed == [0] and json.loads(path.read_text())['metadata']['updates'] == 4


def test_parallel_updates_are_all_present_at_final_flush(tmp_path):
    path = tmp_path/'jnana.json';memory = Memory(path);checkpoint.install_checkpoint_policy(memory, interval_seconds=1000)
    def update(worker):
        for index in range(25):
            with memory._lock:
                memory.metadata['updates'] += 1
                memory.tournament_state['matches'].append({'match_id': f'{worker}-{index}'})
                memory._save_if_needed()
    with ThreadPoolExecutor(max_workers=8) as executor:list(executor.map(update, range(8)))
    memory.save();saved = json.loads(path.read_text())
    assert saved['metadata']['updates'] == 200
    assert len(saved['tournament_state']['matches']) == 201
    assert len({r['match_id'] for r in saved['tournament_state']['matches']}) == 201
    assert memory._loam_checkpoint_sequence == 2


def test_scientist_wrapper_preserves_constructor_and_existing_stop_flush(tmp_path):
    class Scientist:
        def __init__(self, path, marker):self.memory = Memory(path);self.marker = marker
        def stop(self):self.memory.save()
    wrapped = checkpoint.checkpointed_coscientist(Scientist, interval_seconds=1000)
    scientist = wrapped(tmp_path/'jnana.json', marker='preserved')
    scientist.memory._save_if_needed();scientist.memory.metadata['updates'] = 8;scientist.memory._save_if_needed()
    scientist.stop()
    assert isinstance(scientist, Scientist) and scientist.marker == 'preserved'
    assert json.loads((tmp_path/'jnana.json').read_text())['metadata']['updates'] == 8
    assert checkpoint.install_checkpoint_policy(scientist.memory)
    assert scientist.memory._loam_checkpoint_sequence == 2


@pytest.mark.parametrize('interval', [0, -1, float('inf'), float('nan')])
def test_invalid_intervals_are_rejected(tmp_path, interval):
    with pytest.raises(ValueError):checkpoint.install_checkpoint_policy(Memory(tmp_path/'x'), interval_seconds=interval)


def test_memory_without_storage_is_unchanged():
    memory = Memory(None);original = memory.save
    assert checkpoint.install_checkpoint_policy(memory) is False
    assert memory.save == original
