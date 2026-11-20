"""Backlogged admission and first-request context safety under realistic evidence volume."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from loam.orchestrator import Campaign
from loam.schemas import Brief, Hypothesis, ValidationReport
from loam.synthesis import encoded, partition, prompt, size, synthesize
from test_round import make_cfg


def test_backlog_admits_next_thread_without_waiting_for_slowest(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.concurrency = 2
    hypotheses = [Hypothesis(id=str(i), summary=str(i), content='test', elo=1200) for i in range(5)]
    order = []
    active = 0
    peak = 0

    class Agents:
        async def run(self, role, text, deps=None):
            nonlocal active, peak
            assert role == 'validator'
            hid = deps.ws.hypothesis_id
            active += 1
            peak = max(peak, active)
            order.append(('start', hid))
            await asyncio.sleep(0.15 if hid == '0' else 0.005)
            active -= 1
            order.append(('finish', hid))
            return ValidationReport(outcome='inconclusive', evidence_score=0, ledger='', claims=[], next_step='')

    campaign = Campaign(cfg, agents=Agents(), tournament=SimpleNamespace())
    async def run():
        gate = asyncio.Semaphore(2)
        return await asyncio.gather(*(campaign.validate(1, h, gate) for h in hypotheses))
    threads = asyncio.run(run())
    assert peak == 2
    assert order.index(('start', '4')) < order.index(('finish', '0'))
    assert all(t['error'] is None for t in threads)
    events = [json.loads(line) for line in (cfg.state_dir / 'admission-timings.jsonl').read_text().splitlines()]
    assert len(events) == 15
    assert all(t['admission']['finished_at'] >= t['admission']['started_at'] for t in threads)


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
    active = 0
    peak = 0
    class Agents:
        async def run(self, role, text):
            nonlocal active,peak
            assert role == 'reasoner'
            seen.append(text)
            active += 1
            peak = max(peak,active)
            await asyncio.sleep(0.002)
            active -= 1
            return Brief(report='Derived scoped evidence; sources retained.', tournament_brief='Test discriminating neighbors.')
    digest = [{'hypothesis_id': str(i), 'accepted_claims': [{'claim': 'measured ' * 250,
               'source': f'batches/owned-{i}/result.json'}]} for i in range(616)]
    table = [{'hypothesis_id': str(i), 'priority': 1} for i in range(1232)]
    result = asyncio.run(synthesize(Agents(), cfg, digest, table, 'Prior uncertain evidence', 1))
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


def test_staging_and_preflight_preserve_independent_admission(tmp_path, monkeypatch):
    import sys
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    import stage_single_allocation as staging
    from check_single_allocation import check
    # The real frozen input/runtime copies validate integration with this machine.
    job = staging.stage(tmp_path, 20, 8, 'test-backlog', 'campaign', '06:00:00', '', 12,
                        top_k=48, initial_hypotheses=96, new_per_round=24,tournament_workers=16)
    proof = check(job)
    assert proof['concurrent_hypotheses'] == 12
    assert proof['selected_hypotheses_per_round'] == 48
    plan = json.loads((job / 'plan.json').read_text())
    assert plan['reservation'] == ''
    assert plan['tournament_workers'] == 16
    assert plan['synthesis_concurrency'] == 8
    from loam.config import load
    baseline=load(staging.REFERENCE/'campaign.yaml')
    staged=load(job/'campaign/campaign.yaml')
    for role in ['validator','code_generator','reviewer']:
        assert staged.models[role].settings==baseline.models[role].settings
    expected=dict(baseline.models['reasoner'].settings);expected['max_tokens']=32768
    assert staged.models['reasoner'].settings==expected
    assert 'bne_deepwing' not in (job/'campaign/campaign.yaml').read_text()
    assert plan['admission'] == dict(concurrency=12, top_k=48, initial_hypotheses=96, new_per_round=24)


def test_verbose_valid_reports_are_archived_and_compact_briefs_are_reduced(tmp_path):
    import hashlib
    cfg=make_cfg(tmp_path);cfg.max_history_bytes=96000
    long_report='Detailed evidence with source lineage. ' * 400
    class Agents:
        async def run(self,role,text):
            return Brief(report=long_report,tournament_brief='Retain h1 uncertainty; compare independent controls.')
    digest=[{'hypothesis_id':str(n),'accepted_claims':[{'claim':'measured '*180,'source':f'owned/{n}.json'}]} for n in range(60)]
    result=asyncio.run(synthesize(Agents(),cfg,digest,[], '',1))
    directory=Path(result['synthesis_receipt']).parent;manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['levels']>=2
    assert result['report']==long_report and len(long_report.encode())>8000
    for path in directory.glob('*-response.json'):
        assert json.loads(path.read_text())['report']==long_report
    parents=list(directory.glob('level-01-*-request.json'));assert parents
    for path in parents:
        request=json.loads(path.read_text())['prompt'];assert size(request)<=manifest['prompt_budget_bytes']
        for item in json.loads(request.split('\n\n',1)[1]):
            assert 'report' not in item
            assert item['tournament_brief']=='Retain h1 uncertainty; compare independent controls.'
            assert hashlib.sha256((directory/item['full_response_file']).read_bytes()).hexdigest()==item['full_response_sha256']


def test_dispatch_receipt_records_actual_row_buffer_acceptance(tmp_path, monkeypatch):
    import importlib
    import concurrent.futures
    scripts = Path(__file__).resolve().parents[1] / 'skills/hpc-deployment/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    agent = importlib.import_module('hpc_agent')
    script = tmp_path/'work.py'; script.write_text('pass\n')
    inputs = tmp_path/'inputs.jsonl'; inputs.write_text('{"id":"row1","seed":1}\n')
    monkeypatch.setattr(agent, '_MIN_BATCH_ROWS', 1)
    monkeypatch.setattr(agent, '_MIN_BATCH_DISTINCT', 1)
    submitted = []
    class Client:
        def submit_batch(self, tasks):
            submitted.extend(tasks)
            return [concurrent.futures.Future() for _ in tasks]
    client = SimpleNamespace(base_dir=str(tmp_path),inspect_roots=[str(tmp_path)],
                             _client=Client(),_batches={},science_workers_per_gpu=1)
    result = asyncio.run(agent.HpcClientAgent.run_batch(client,script_path=str(script),inputs_path=str(inputs)))
    batch = tmp_path/'batches'/result['batch_id']
    receipt = json.loads(next(batch.glob('dispatch-*.json')).read_text())
    assert receipt['task_ids'] == [submitted[0].task_id]
    assert receipt['run_ids'] == ['row1']
    assert receipt['requested_at'] <= receipt['prepared_at'] <= receipt['buffered_at']


def test_configured_output_budget_is_reserved_for_every_synthesis_request(tmp_path):
    cfg=make_cfg(tmp_path)
    goal=tmp_path/'long-goal.txt';goal.write_text('Scientific campaign context. '*1500)
    cfg.goal=str(goal);cfg.max_history_bytes=128000
    cfg.models['reasoner'].settings['max_tokens']=32768
    instructions=cfg.goal_text()+cfg.rules_text()+(Path(__file__).parents[1]/'loam/prompts/reasoner.md').read_text()
    seen=[]
    class Agents:
        async def run(self,role,text):
            assert size(text)+size(instructions)+32768<=120000
            seen.append(text)
            return Brief(report='Derived evidence only.',tournament_brief='Uncertainty retained.')
    digest=[{'hypothesis_id':str(i),'evidence':'measured '*600} for i in range(12)]
    result=asyncio.run(synthesize(Agents(),cfg,digest,[],None,1))
    manifest=json.loads(Path(result['synthesis_receipt']).read_text())
    assert manifest['status']=='complete' and len(seen)>1


def test_unsafe_output_allowance_is_rejected_before_any_model_call(tmp_path):
    import pytest
    cfg=make_cfg(tmp_path);cfg.models['reasoner'].settings['max_tokens']=115000
    class Agents:
        async def run(self,role,text):
            raise AssertionError('Unsafe context reached model')
    with pytest.raises(ValueError,match='too little safe synthesis context'):
        asyncio.run(synthesize(Agents(),cfg,[{'hypothesis_id':'h'}],[],None,1))
