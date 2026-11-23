"""The single-allocation coordinator: it checks what it was given before it starts anything.

Ported from the moeen branch's run_single_allocation.py owner test and controller GC tests,
importing loam.coordinator. No scheduler, GPU or inference service is used.
"""
import asyncio
import copy
import gc
import json
import subprocess
import sys
from contextlib import asynccontextmanager

import pytest
import yaml

from loam import coordinator, staging
from loam.allocation import read_allocation
from loam.staging import InferenceProfile
from loam.systems import JUPITER
from test_single_allocation_staging import stage


def forbid_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('startup reached scheduler or service work before rejecting the job')
    monkeypatch.setattr(coordinator.subprocess, 'run', forbidden)
    monkeypatch.setattr(coordinator.subprocess, 'Popen', forbidden)


@pytest.fixture
def job(tmp_path, monkeypatch):
    profile = tmp_path / 'inference.json'
    profile.write_text(json.dumps({'model': 'served-model', 'command': ['vllm', 'serve', '--port', '{port}']}))
    staged = stage(tmp_path, monkeypatch, profile)
    plan = json.loads((staged / 'plan.json').read_text())
    (staged / 'submission.json').write_text(json.dumps({
        'job_id': '123', 'job_name': plan['job_name'], 'script_sha256': staging.sha(staged / 'job.sh')}))
    return staged


def run(job, environ):
    runtime = json.loads((job / 'plan.json').read_text())['runtime_repo']
    return asyncio.run(coordinator.run(job, environ=environ, runtime=runtime))


def test_a_path_outside_the_job_stops_the_coordinator_before_any_service(job, monkeypatch, tmp_path):
    path = job / 'campaign' / 'campaign.yaml'
    raw = yaml.safe_load(path.read_text())
    raw['state_dir'] = str(tmp_path / 'historical' / 'state')
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    forbid_processes(monkeypatch)

    with pytest.raises(RuntimeError, match='state_dir'):
        run(job, {'SLURM_JOB_ID': '123'})


def test_a_job_that_is_not_the_submitted_one_starts_nothing(job, monkeypatch):
    forbid_processes(monkeypatch)

    with pytest.raises(RuntimeError, match='not the submitted job 123'):
        run(job, {'SLURM_JOB_ID': '456'})


@pytest.mark.parametrize(('field', 'value', 'problem'), [
    ('concurrency', 1, 'Admission settings differ'),
    ('tournament.workers', 1, 'Tournament workers differ'),
    ('synthesis_concurrency', 1, 'Synthesis concurrency differs'),
    ('execution.hpc.allocation.nodes', 1, 'not sized to the science nodes'),
    ('execution.hpc.allocation.queue', 'changed', 'Account or queue differ'),
    ('execution.hpc.science_workers_per_gpu', 1, 'Science workers per GPU differ'),
    ('models.validator.model', 'changed', 'no longer names the served model'),
])
def test_configuration_drift_is_rejected_before_any_service(job, monkeypatch, field, value, problem):
    path = job / 'campaign/campaign.yaml'
    raw = yaml.safe_load(path.read_text())
    target = raw
    *parents, name = field.split('.')
    for parent in parents:
        target = target[parent]
    target[name] = value
    path.write_text(yaml.safe_dump(raw))
    forbid_processes(monkeypatch)

    with pytest.raises(RuntimeError, match=problem):
        staging.check(job)
    with pytest.raises(RuntimeError, match=problem):
        run(job, {'SLURM_JOB_ID': '123'})


def test_the_coordinator_runs_only_from_the_frozen_runtime(job, monkeypatch, tmp_path):
    forbid_processes(monkeypatch)

    with pytest.raises(RuntimeError, match='not the frozen runtime'):
        asyncio.run(coordinator.run(job, environ={'SLURM_JOB_ID': '123'}, runtime=tmp_path))


def plan_of(job):
    return json.loads((job / 'plan.json').read_text())


def test_the_allocation_is_split_into_science_then_inference_hosts(job):
    hosts = [f'n{i:02d}' for i in range(20)]

    manifest = coordinator.plan_allocation(plan_of(job), hosts, job_id='123', hostname='n00.cluster')

    assert manifest['science_nodes'] == hosts[:12] and manifest['inference_nodes'] == hosts[12:]
    assert manifest['coordinator_host'] == 'n00' and manifest['job_id'] == '123'
    path = job / 'allocation.json'
    path.write_text(json.dumps(manifest))
    assert read_allocation(path)['all_nodes'] == hosts


@pytest.mark.parametrize(('hosts', 'hostname', 'problem'), [
    ([f'n{i:02d}' for i in range(19)], 'n00', '19 hosts'),
    ([f'n{i:02d}' for i in range(21)], 'n00', '21 hosts'),
    ([f'n{i:02d}' for i in range(20)], 'n19', 'not a science host'),
])
def test_an_allocation_that_does_not_match_the_plan_is_refused(job, hosts, hostname, problem):
    with pytest.raises(RuntimeError, match=problem):
        coordinator.plan_allocation(plan_of(job), hosts, job_id='123', hostname=hostname)


def test_each_inference_host_runs_one_replica_behind_one_pool():
    profile = InferenceProfile(model='served-model', command=['vllm', 'serve', '--host', '{host}', '--port', '{port}'],
                               port=8100, checkpoint='org/weights')

    steps = coordinator.replica_steps(JUPITER, profile, ['a', 'b'])
    pool = coordinator.pool_config(profile, ['a', 'b'], job_id='123')

    assert [step[-6:] for step in steps] == [['vllm', 'serve', '--host', 'a', '--port', '8100'],
                                             ['vllm', 'serve', '--host', 'b', '--port', '8100']]
    assert all('--nodelist=' + host in step and '--cpus-per-task=32' in step for step, host in zip(steps, 'ab'))
    assert pool['model'] == 'served-model' and pool['checkpoint'] == 'org/weights'
    assert [b['base_url'] for b in pool['backends']] == ['http://a:8100/v1', 'http://b:8100/v1']
    assert {b['job_id'] for b in pool['backends']} == {'123'}


def test_served_roles_are_pointed_at_the_pool_and_nothing_else(job):
    path = job / 'campaign' / 'campaign.yaml'
    raw = yaml.safe_load(path.read_text())
    raw['models']['reviewer']['model'] = 'other-model'
    path.write_text(yaml.safe_dump(raw))

    coordinator.point_models_at_pool(path, ['validator', 'code_generator', 'reasoner'], 'http://127.0.0.1:9/v1',
                                     mcp_port=4321)

    raw = yaml.safe_load(path.read_text())
    assert raw['models']['validator'] == {'model': 'served-model', 'base_url': 'http://127.0.0.1:9/v1',
                                          'api_key_env': 'LOAM_POOL_TOKEN'}
    assert raw['models']['reviewer']['model'] == 'other-model'
    assert raw['models']['reviewer']['base_url'] == 'http://127.0.0.1:1/v1'
    assert raw['execution']['hpc']['port'] == 4321


@pytest.fixture
def science_service(job, monkeypatch):
    """One submitted row can finish only while the controller's science service is alive."""
    from loam import launcher, orchestrator

    host = coordinator.socket.gethostname().split('.')[0]
    hosts = [host, *(f'worker-{i}' for i in range(19))]
    allocation = coordinator.plan_allocation(plan_of(job), hosts, job_id='123', hostname=host)
    staging.write(job / 'allocation.json', allocation)
    staging.write(job / 'status.json', allocation)
    batch = job / 'campaign/batches/pending'
    row = batch / 'run_1'
    row.mkdir(parents=True)
    state = {'alive': False, 'row': row}

    @asynccontextmanager
    async def service(cfg):
        state['alive'] = True
        try:
            yield
        finally:
            state['alive'] = False

    class CampaignWorkload:
        def __init__(self, cfg):
            pass

        async def run(self, rounds):
            staging.write(batch / 'batch.json', {'batch_id': 'pending', 'total': 1})

    monkeypatch.setattr(launcher, 'managed_hpc_server', service)
    monkeypatch.setattr(orchestrator, 'Campaign', CampaignWorkload)
    return state


def test_submitted_science_finishes_before_its_service_stops(job, monkeypatch, science_service):
    async def finish_pending_row(delay):
        if science_service['alive']:
            staging.write(science_service['row'] / 'status.json', {'returncode': 0})
            (science_service['row'] / 'done').touch()

    monkeypatch.setattr(coordinator.asyncio, 'sleep', finish_pending_row)

    asyncio.run(coordinator.control(job, environ={'SLURM_JOB_ID': '123'}))

    assert coordinator.pending_batches(job / 'campaign') == []
    assert not science_service['alive']
    status = staging.read(job / 'status.json')
    assert status['phase'] == 'science_drained' and status['pending_batches'] == []


def test_a_science_drain_timeout_still_stops_the_service(job, monkeypatch, science_service):
    monkeypatch.setattr(coordinator, 'DRAIN_TIMEOUT', 0)

    with pytest.raises(TimeoutError, match='Submitted science did not finish: pending'):
        asyncio.run(coordinator.control(job, environ={'SLURM_JOB_ID': '123'}))

    assert not science_service['alive']
    assert coordinator.pending_batches(job / 'campaign') == ['pending']


# --- Controller garbage collection, ported from the moeen branch's test_controller_gc.py ---

@pytest.fixture
def gc_policy():
    before = gc.get_threshold()
    enabled = gc.isenabled()
    gc.enable()
    gc.set_threshold(4321, 17, 23)
    # Some Python versions ignore threshold2; preserve the policy actually installed.
    yield gc.get_threshold()
    gc.set_threshold(*before)
    gc.enable() if enabled else gc.disable()


@pytest.mark.parametrize('plan', [{}, {'controller_gc_threshold0': None}])
def test_by_default_the_process_gc_policy_is_kept(gc_policy, plan):
    before = gc.get_threshold()
    assert coordinator.configure_controller_gc(plan) == {
        'requested_threshold0': None, 'thresholds_before': list(before), 'thresholds_applied': list(before),
        'automatic_gc_enabled': True}
    assert gc.get_threshold() == before


@pytest.mark.parametrize('threshold', [700, 10000, 100000])
def test_only_the_young_generation_changes_and_collection_stays_on(gc_policy, threshold):
    receipt = coordinator.configure_controller_gc({'controller_gc_threshold0': threshold})

    assert gc.get_threshold() == (threshold, *gc_policy[1:])
    assert receipt['thresholds_before'] == list(gc_policy)
    assert receipt['thresholds_applied'] == [threshold, *gc_policy[1:]]
    assert receipt['automatic_gc_enabled'] and gc.isenabled()


@pytest.mark.parametrize('invalid', [0, -1, 699, 100001, True, False, 10000.0, '10000', [], {}])
def test_an_invalid_policy_fails_without_side_effects(gc_policy, invalid):
    with pytest.raises(ValueError):
        coordinator.configure_controller_gc({'controller_gc_threshold0': invalid})
    assert gc.get_threshold() == gc_policy and gc.isenabled()


def test_tuning_refuses_a_disabled_collector_without_changing_it(gc_policy):
    gc.disable()
    with pytest.raises(ValueError):
        coordinator.configure_controller_gc({'controller_gc_threshold0': 10000})
    assert gc.get_threshold() == gc_policy and not gc.isenabled()


def test_the_default_leaves_a_disabled_collector_disabled(gc_policy):
    gc.disable()
    receipt = coordinator.configure_controller_gc({})
    assert not receipt['automatic_gc_enabled'] and not gc.isenabled()


def test_the_plan_is_not_mutated(gc_policy):
    plan = {'controller_gc_threshold0': 10000, 'admission': {'concurrency': 12}}
    original = copy.deepcopy(plan)
    coordinator.configure_controller_gc(plan)
    assert plan == original


def test_another_interpreter_keeps_its_own_policy(gc_policy):
    command = [sys.executable, '-c', 'import gc,json;print(json.dumps(gc.get_threshold()))']
    before = json.loads(subprocess.check_output(command, text=True))
    coordinator.configure_controller_gc({'controller_gc_threshold0': 10000})
    assert json.loads(subprocess.check_output(command, text=True)) == before
    assert gc.get_threshold() == (10000, *gc_policy[1:])


def test_a_telemetry_sample_satisfies_startup_readiness(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from loam import systems, telemetry
    from loam.allocation_startup import telemetry_ready_hosts
    rows = '\n'.join(f'{i}, GPU-{i}, 50, 10, 1000, 96000, 300, 40' for i in range(4))
    monkeypatch.setattr(systems.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=rows, returncode=0))
    allocation = {'job_id': '123', 'started_at': 100.0, 'all_nodes': ['a', 'b'], 'science_nodes': ['a'],
                  'inference_nodes': ['b']}

    for host, role in [('a', 'science'), ('b', 'inference')]:
        row, _ = telemetry.sample(JUPITER, host=host, job_id='123', role=role, now=200.0)
        folder = tmp_path / 'telemetry'
        folder.mkdir(exist_ok=True)
        (folder / f'123-{host}.jsonl').write_text(json.dumps(row) + '\n')

    assert telemetry_ready_hosts(tmp_path, allocation, system=JUPITER, now=201.0) == {'a', 'b'}
