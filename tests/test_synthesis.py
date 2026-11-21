"""Round synthesis: every record reaches the reasoner once, no request exceeds the context budget,
and the reduction lineage stays on disk. Ported from the moeen branch's test_saturation.py."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from test_round import make_cfg

from loam.schemas import Brief
from loam.synthesis import CONTEXT_BYTES, partition, prompt, size, synthesize


def instructions(cfg) -> str:
    return cfg.goal_text() + cfg.rules_text() + (Path(__file__).parents[1] / 'loam/prompts/reasoner.md').read_text()


def test_oversized_unicode_record_is_losslessly_fragmented():
    original = {'kind': 'thread', 'claim': '界😀\\"' * 6000, 'source': 'batch/owned/artifact.json'}
    groups = partition([original], 16000, 0)
    assert len(groups) > 1
    assert all(size(prompt(g, 0)) <= 16000 for g in groups)
    fragments = [item for group in groups for item in group]
    assert json.loads(''.join(item['json_fragment'] for item in fragments)) == original


def test_large_synthesis_covers_all_records_and_bounds_every_request(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.max_history_bytes = 96000
    cfg.concurrency = 24
    cfg.synthesis_concurrency = 12
    seen = []
    active = peak = 0

    class Agents:
        async def run(self, role, text):
            nonlocal active, peak
            assert role == 'reasoner'
            seen.append(text)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.002)
            active -= 1
            return Brief(report='Derived scoped evidence; sources retained.', tournament_brief='Test discriminating neighbors.')

    digest = [{'hypothesis_id': str(i), 'accepted_claims': [{'claim': 'measured ' * 250,
               'source': f'batches/owned-{i}/result.json'}]} for i in range(616)]
    table = [{'hypothesis_id': str(i), 'priority': 1} for i in range(1232)]
    result = asyncio.run(synthesize(Agents(), cfg, digest, table, 'Prior uncertain evidence', 1))

    assert result['tournament_brief'] == 'Test discriminating neighbors.'
    receipt = Path(result['synthesis_receipt'])
    manifest = json.loads(receipt.read_text())
    assert manifest['status'] == 'complete' and manifest['levels'] >= 2
    assert peak == manifest['request_concurrency'] == 12
    assert all(size(text) <= manifest['prompt_budget_bytes'] for text in seen)
    raw = json.loads((receipt.parent / 'input.json').read_text())
    assert raw['threads'] == digest and raw['steering'] == table
    leaves = []
    for path in sorted(receipt.parent.glob('level-00-*-request.json')):
        leaves.extend(json.loads(json.loads(path.read_text())['prompt'].split('\n\n', 1)[1]))
    assert [x['value'] for x in leaves if x['kind'] == 'thread'] == digest
    assert [x['value'] for x in leaves if x['kind'] == 'steering'] == table


def test_synthesis_concurrency_never_exceeds_validator_concurrency(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.concurrency = 2
    cfg.synthesis_concurrency = 8

    class Agents:
        async def run(self, role, text):
            return Brief(report='r', tournament_brief='b')

    digest = [{'hypothesis_id': str(i), 'evidence': 'measured ' * 2000} for i in range(20)]
    result = asyncio.run(synthesize(Agents(), cfg, digest, [], '', 1))
    assert json.loads(Path(result['synthesis_receipt']).read_text())['request_concurrency'] == 2


def test_verbose_valid_reports_are_archived_and_compact_briefs_are_reduced(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.max_history_bytes = 96000
    long_report = 'Detailed evidence with source lineage. ' * 400

    class Agents:
        async def run(self, role, text):
            return Brief(report=long_report, tournament_brief='Retain h1 uncertainty; compare independent controls.')

    digest = [{'hypothesis_id': str(n), 'accepted_claims': [{'claim': 'measured ' * 180, 'source': f'owned/{n}.json'}]}
              for n in range(60)]
    result = asyncio.run(synthesize(Agents(), cfg, digest, [], '', 1))
    directory = Path(result['synthesis_receipt']).parent
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['levels'] >= 2
    assert result['report'] == long_report and len(long_report.encode()) > 8000
    for path in directory.glob('*-response.json'):
        assert json.loads(path.read_text())['report'] == long_report
    parents = list(directory.glob('level-01-*-request.json'))
    assert parents
    for path in parents:
        request = json.loads(path.read_text())['prompt']
        assert size(request) <= manifest['prompt_budget_bytes']
        for item in json.loads(request.split('\n\n', 1)[1]):
            assert 'report' not in item
            assert item['tournament_brief'] == 'Retain h1 uncertainty; compare independent controls.'
            assert hashlib.sha256((directory / item['full_response_file']).read_bytes()).hexdigest() == item['full_response_sha256']


def test_configured_output_budget_is_reserved_for_every_synthesis_request(tmp_path):
    cfg = make_cfg(tmp_path)
    goal = tmp_path / 'long-goal.md'
    goal.write_text('Scientific campaign context. ' * 1500)
    cfg.goal = str(goal)
    cfg.max_history_bytes = 128000
    cfg.models['reasoner'].settings['max_tokens'] = 32768
    fixed = size(instructions(cfg))
    seen = []

    class Agents:
        async def run(self, role, text):
            assert size(text) + fixed + 32768 <= CONTEXT_BYTES
            seen.append(text)
            return Brief(report='Derived evidence only.', tournament_brief='Uncertainty retained.')

    digest = [{'hypothesis_id': str(i), 'evidence': 'measured ' * 600} for i in range(12)]
    result = asyncio.run(synthesize(Agents(), cfg, digest, [], '', 1))
    assert json.loads(Path(result['synthesis_receipt']).read_text())['status'] == 'complete'
    assert len(seen) > 1


def test_unsafe_output_allowance_is_rejected_before_any_model_call(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.models['reasoner'].settings['max_tokens'] = 115000

    class Agents:
        async def run(self, role, text):
            raise AssertionError('Unsafe context reached model')

    with pytest.raises(ValueError, match='too little safe synthesis context'):
        asyncio.run(synthesize(Agents(), cfg, [{'hypothesis_id': 'h'}], [], '', 1))


def test_an_unset_output_allowance_reserves_8192(tmp_path):
    cfg = make_cfg(tmp_path)
    goal = tmp_path / 'goal.md'
    goal.write_text('Scientific campaign context. ' * 2500)
    cfg.goal = str(goal)
    cfg.max_history_bytes = 128000
    assert 'max_tokens' not in cfg.models['reasoner'].settings

    class Agents:
        async def run(self, role, text):
            return Brief(report='r', tournament_brief='b')

    result = asyncio.run(synthesize(Agents(), cfg, [{'hypothesis_id': 'h'}], [], '', 1))
    budget = json.loads(Path(result['synthesis_receipt']).read_text())['prompt_budget_bytes']
    assert budget == CONTEXT_BYTES - size(instructions(cfg)) - 8192


def test_instructions_leaving_too_little_context_are_rejected_before_any_model_call(tmp_path):
    cfg = make_cfg(tmp_path)
    goal = tmp_path / 'huge-goal.md'
    goal.write_text('Scientific campaign context. ' * 3500)
    cfg.goal = str(goal)

    class Agents:
        async def run(self, role, text):
            raise AssertionError('Unsafe context reached model')

    with pytest.raises(ValueError, match='too little safe synthesis context'):
        asyncio.run(synthesize(Agents(), cfg, [{'hypothesis_id': 'h'}], [], '', 1))


def test_a_failed_reduction_leaves_a_failed_receipt(tmp_path):
    cfg = make_cfg(tmp_path)

    class Agents:
        async def run(self, role, text):
            raise RuntimeError('endpoint down')

    with pytest.raises(RuntimeError, match='endpoint down'):
        asyncio.run(synthesize(Agents(), cfg, [{'hypothesis_id': 'h'}], [], '', 3))
    [manifest] = (cfg.state_dir / 'synthesis' / 'round-003').glob('*/manifest.json')
    receipt = json.loads(manifest.read_text())
    assert receipt['status'] == 'failed' and receipt['error_type'] == 'RuntimeError'
    assert receipt['calls'][0]['status'] == 'failed'


def test_unbounded_history_still_bounds_each_request_to_48000_bytes(tmp_path):
    cfg = make_cfg(tmp_path)
    assert cfg.max_history_bytes is None

    class Agents:
        async def run(self, role, text):
            return Brief(report='r', tournament_brief='b')

    result = asyncio.run(synthesize(Agents(), cfg, [{'hypothesis_id': 'h'}], [], '', 1))
    assert json.loads(Path(result['synthesis_receipt']).read_text())['prompt_budget_bytes'] == 48000
