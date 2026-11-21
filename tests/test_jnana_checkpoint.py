"""Checkpoint coalescing must preserve every field and all final updates."""
import copy
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import loam.jnana_checkpoint as checkpoint
from loam.config import Config
from loam.tournament import Tournament

PAYLOAD = {'hypotheses': [{'hypothesis_id': 'h1', 'summary': 'β binder', 'elo_rating': 1200.5}],
           'experiments': {'e': {'values': [1, 2, None]}}, 'analyses': {'a': {'score': 0.3}},
           'papers': {'p': 'paper'}, 'metadata': {'research_goal': 'goal', 'updates': 0},
           'agent_states': {'agent': 'running'}, 'datasets': {'d': ['a', 'b']},
           'training_plans': {'t': {'epochs': 2}}, 'evaluations': {'v': False},
           'tournament_state': {'matches': [{'match_id': '1', 'winner': 'h1'}], 'statistics': {'n': 1}}}


class Hypothesis:
    def __init__(self, data):
        self.data = data
        self.hypothesis_id = data['hypothesis_id']

    def to_dict(self):
        return dict(self.data)


class Memory:
    def __init__(self, path):
        self.storage_path = str(path) if path else None
        self._lock = threading.RLock()
        data = copy.deepcopy(PAYLOAD)
        self.hypotheses = {x['hypothesis_id']: Hypothesis(x) for x in data.pop('hypotheses')}
        self.__dict__.update(data)

    def save(self):
        raise AssertionError('Original per-field writer should be replaced')

    def _save_if_needed(self):
        self.save()


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(checkpoint.time, 'monotonic', lambda: now[0])
    return now


def test_every_field_survives_and_explicit_phase_flush_bypasses_coalescing(tmp_path, clock):
    path = tmp_path / 'jnana.json'
    memory = Memory(path)
    assert checkpoint.install_checkpoint_policy(memory)
    memory._save_if_needed()
    assert json.loads(path.read_text()) == PAYLOAD
    memory.metadata['updates'] = 1
    clock[0] = 101
    memory._save_if_needed()
    assert json.loads(path.read_text())['metadata']['updates'] == 0
    memory.save()
    assert json.loads(path.read_text())['metadata']['updates'] == 1
    memory.metadata['updates'] = 2
    clock[0] = 107
    memory._save_if_needed()
    assert json.loads(path.read_text())['metadata']['updates'] == 2
    records = [json.loads(s) for s in (tmp_path / 'jnana-checkpoint-timings.jsonl').read_text().splitlines()]
    assert [r['reason'] for r in records] == ['autosave', 'explicit', 'autosave']
    assert records[-1]['json_bytes'] == path.stat().st_size


def test_failed_replace_keeps_old_valid_checkpoint_and_removes_only_its_temp(tmp_path, monkeypatch):
    path = tmp_path / 'jnana.json'
    memory = Memory(path)
    checkpoint.install_checkpoint_policy(memory)
    memory.save()
    old = path.read_bytes()
    memory.metadata['updates'] = 3

    def fail(source, destination):
        raise OSError('simulated storage failure')
    monkeypatch.setattr(checkpoint.os, 'replace', fail)
    with pytest.raises(OSError, match='storage failure'):
        memory.save()
    assert path.read_bytes() == old and json.loads(path.read_text()) == PAYLOAD
    assert not list(tmp_path.glob('.jnana.json.*.tmp'))


def test_old_namespace_is_valid_until_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / 'jnana.json'
    memory = Memory(path)
    checkpoint.install_checkpoint_policy(memory)
    memory.save()
    memory.metadata['updates'] = 4
    real_replace = checkpoint.os.replace
    observed = []

    def inspect_replace(source, destination):
        observed.append(json.loads(Path(destination).read_text())['metadata']['updates'])
        assert json.loads(Path(source).read_text())['metadata']['updates'] == 4
        real_replace(source, destination)
    monkeypatch.setattr(checkpoint.os, 'replace', inspect_replace)
    memory.save()
    assert observed == [0] and json.loads(path.read_text())['metadata']['updates'] == 4


def test_parallel_updates_are_all_present_at_final_flush(tmp_path):
    path = tmp_path / 'jnana.json'
    memory = Memory(path)
    checkpoint.install_checkpoint_policy(memory, interval_seconds=1000)

    def update(worker):
        for index in range(25):
            with memory._lock:
                memory.metadata['updates'] += 1
                memory.tournament_state['matches'].append({'match_id': f'{worker}-{index}'})
                memory._save_if_needed()
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(update, range(8)))
    memory.save()
    saved = json.loads(path.read_text())
    assert saved['metadata']['updates'] == 200
    assert len(saved['tournament_state']['matches']) == 201
    assert len({r['match_id'] for r in saved['tournament_state']['matches']}) == 201
    assert memory._loam_checkpoint_sequence == 2


def test_scientist_wrapper_preserves_constructor_and_existing_stop_flush(tmp_path):
    class Scientist:
        def __init__(self, path, marker):
            self.memory = Memory(path)
            self.marker = marker

        def stop(self):
            self.memory.save()
    wrapped = checkpoint.checkpointed_coscientist(Scientist, interval_seconds=1000)
    scientist = wrapped(tmp_path / 'jnana.json', marker='preserved')
    scientist.memory._save_if_needed()
    scientist.memory.metadata['updates'] = 8
    scientist.memory._save_if_needed()
    scientist.stop()
    assert isinstance(scientist, Scientist) and scientist.marker == 'preserved'
    assert json.loads((tmp_path / 'jnana.json').read_text())['metadata']['updates'] == 8
    assert checkpoint.install_checkpoint_policy(scientist.memory)
    assert scientist.memory._loam_checkpoint_sequence == 2


@pytest.mark.parametrize('interval', [0, -1, float('inf'), float('nan')])
def test_invalid_intervals_are_rejected(tmp_path, interval):
    with pytest.raises(ValueError):
        checkpoint.install_checkpoint_policy(Memory(tmp_path / 'x'), interval_seconds=interval)


def test_memory_without_storage_is_unchanged():
    memory = Memory(None)
    original = memory.save
    assert checkpoint.install_checkpoint_policy(memory) is False
    assert memory.save == original


def test_interrupted_campaign_resumes_generation_from_its_last_checkpoint(tmp_path, monkeypatch, clock):
    """Generation is interrupted after checkpointing two of four hypotheses; the next run tops up only two."""
    requested = []

    class JnanaMemory(Memory):  # loads its storage_path on construction, as Jnana's ContextMemory does
        def __init__(self, path):
            super().__init__(path)
            self.hypotheses = {}
            if Path(path).exists():
                data = json.loads(Path(path).read_text())
                self.hypotheses = {x['hypothesis_id']: Hypothesis(x) for x in data.pop('hypotheses')}
                self.__dict__.update(data)

    class CoScientist:
        def __init__(self, *, storage_path, **kwargs):
            self.memory = JnanaMemory(storage_path)

        def get_all_hypotheses(self):
            return list(self.memory.hypotheses.values())

        def generate_hypotheses(self, *, count, batch_size):
            requested.append(count)
            for _ in range(min(count, 2)):
                hypothesis_id = f'h{len(self.memory.hypotheses)}'
                self.memory.hypotheses[hypothesis_id] = Hypothesis(
                    {'hypothesis_id': hypothesis_id, 'summary': 's', 'content': 'c'})
                clock[0] += 10
                self.memory._save_if_needed()
            if count > 2:
                raise KeyboardInterrupt('allocation ended mid-generation')

        def wait_for_completion(self, timeout):
            return True
        set_research_goal = start = stop = lambda self, *a, **k: None

    periodic = SimpleNamespace(CoScientist=CoScientist, run_incremental_tournament=lambda **k: {'top_hypotheses': []})
    fakes = {'jnana.protognosis.core.coscientist': SimpleNamespace(CoScientist=CoScientist),
             'jnana.protognosis.core.multi_llm_config': SimpleNamespace(LLMConfig=lambda **k: k),
             'jnana.protognosis.periodic_tournament': periodic}
    for name in ['jnana', 'jnana.protognosis', 'jnana.protognosis.core', *fakes]:
        monkeypatch.setitem(sys.modules, name, fakes.get(name, SimpleNamespace()))
    test = {'model': 'test'}
    cfg = Config(goal='goal', state_dir=tmp_path / 'state',
                 models=dict.fromkeys(['validator', 'code_generator', 'reviewer', 'reasoner'], test),
                 tournament={'jnana_root': tmp_path, 'generation': test, 'ranking': test, 'initial_hypotheses': 4})

    with pytest.raises(KeyboardInterrupt):
        Tournament.for_campaign(cfg).run('goal', initial=True)
    assert set(Tournament.for_campaign(cfg).pool()) == {'h0', 'h1'}
    Tournament.for_campaign(cfg).run('goal', initial=True)
    assert requested == [4, 2]
