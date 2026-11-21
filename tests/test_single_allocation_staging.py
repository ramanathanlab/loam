"""Staging and submitting a single-allocation campaign: one job directory that owns everything it runs.

Ported from the moeen branch's staging, path-binding, sealing and submission tests, importing
loam.staging instead of scripts/. The account and partition come from the Site profile; the
scheduler is never called for real.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from loam import staging
from loam.allocation import validate_campaign_paths
from loam.config import load
from loam.scaling import UnmeasuredScalingWarning
from test_round import hpc_config_values

SITE_PROFILE = {'allocation': {'account': 'profile-account', 'queue': 'profile-partition'},
                'software_root': '/soft/loam'}
INFERENCE_PROFILE = {'model': 'served-model', 'command': ['vllm', 'serve', '--host', '{host}', '--port', '{port}'],
                     'port': 8000, 'startup_timeout': 600}


def reference(tmp_path, monkeypatch, *, allocation=None, **settings):
    """A Jupiter campaign config to stage from; its account and partition come from a test Site profile."""
    profile = tmp_path / 'profiles' / 'jupiter-test.yaml'
    profile.parent.mkdir(parents=True)
    profile.write_text(yaml.safe_dump(SITE_PROFILE))
    monkeypatch.setenv('LOAM_SITE_PROFILE', str(profile))
    folder = tmp_path / 'reference'
    folder.mkdir()
    values, _ = hpc_config_values(folder)
    hpc = values['execution']['hpc']
    hpc.update(system='jupiter', launch_mode='globus')
    del hpc['allocation']['account'], hpc['allocation']['queue']
    hpc['allocation'].update(allocation or {})
    (folder / 'goal.md').write_text('# Goal\nFind binders.\n')
    values['goal'] = 'goal.md'
    values['state_dir'] = str(tmp_path / 'historical' / 'state')
    for role in values['models']:
        values['models'][role] = {'model': 'served-model', 'base_url': 'http://127.0.0.1:1/v1'}
    values.update(settings)
    path = folder / 'loam.yaml'
    path.write_text(yaml.safe_dump(values))
    return path


@pytest.fixture
def inference_profile(tmp_path):
    path = tmp_path / 'inference.json'
    path.write_text(json.dumps(INFERENCE_PROFILE))
    return path


@pytest.fixture
def inputs(tmp_path):
    folder = tmp_path / 'science-inputs'
    (folder / 'recipes').mkdir(parents=True)
    (folder / 'recipes' / 'screen.py').write_text('print("screen")\n')
    (folder / 'target.fasta').write_text('>t\nMKV\n')
    return folder


def stage(tmp_path, monkeypatch, inference_profile, *, root=None, reference_settings=None, **changes):
    values = dict(label='trial-20', nodes=20, inference_profile=inference_profile, walltime='06:00:00')
    values.update(changes)
    return staging.stage(reference(tmp_path, monkeypatch, **(reference_settings or {})),
                         root or tmp_path / 'experiment', **values)


def test_a_staged_campaign_is_sized_by_the_measured_recipe_and_owns_its_paths(tmp_path, monkeypatch, inference_profile):
    job = stage(tmp_path, monkeypatch, inference_profile)

    plan = json.loads((job / 'plan.json').read_text())
    campaign = job / 'campaign'
    cfg = load(campaign / 'campaign.yaml')
    # The measured 20-node Jupiter operating point (docs/scaling-recipe-book.md).
    assert (plan['nodes'], plan['inference_count'], plan['science_count']) == (20, 8, 12)
    assert plan['admission'] == {'concurrency': 48, 'top_k': 144, 'initial_hypotheses': 288, 'new_per_round': 72}
    assert (cfg.concurrency, cfg.tournament.workers, cfg.synthesis_concurrency) == (48, 16, 8)
    assert cfg.execution.hpc.science_workers_per_gpu == 2
    assert cfg.execution.hpc.allocation.nodes == 12 and cfg.execution.hpc.launch_mode == 'existing'
    assert validate_campaign_paths(cfg, campaign) == {
        'state_dir': str(campaign / 'state'), 'goal': str(campaign / 'goal.md'),
        'execution.hpc.base_dir': str(campaign), 'execution.workdir': str(campaign / 'work')}
    assert (campaign / 'goal.md').read_text() == '# Goal\nFind binders.\n'
    assert (job / '.gitignore').read_text() == '*\n'
    assert json.loads((tmp_path / 'experiment' / 'experiment.json').read_text())['owned_job_ids'] == []
    assert not (job / 'submission.json').exists() and not (job / 'submission-intent.json').exists()


def test_account_and_partition_come_from_the_site_profile(tmp_path, monkeypatch, inference_profile):
    job = stage(tmp_path, monkeypatch, inference_profile)

    plan = json.loads((job / 'plan.json').read_text())
    assert (plan['account'], plan['partition']) == ('profile-account', 'profile-partition')
    assert 'deepwing' not in (job / 'campaign' / 'campaign.yaml').read_text()


def test_a_campaign_config_value_wins_over_the_site_profile(tmp_path, monkeypatch, inference_profile):
    job = stage(tmp_path, monkeypatch, inference_profile, reference_settings={'allocation': {'queue': 'debug'}})

    plan = json.loads((job / 'plan.json').read_text())
    assert (plan['account'], plan['partition']) == ('profile-account', 'debug')


def test_the_runtime_is_frozen_inside_the_job_and_sealed(tmp_path, monkeypatch, inference_profile):
    job = stage(tmp_path, monkeypatch, inference_profile, controller_gc_threshold0=10000)

    plan = json.loads((job / 'plan.json').read_text())
    cfg = load(job / 'campaign' / 'campaign.yaml')
    runtime = Path(plan['runtime_repo'])
    assert runtime.is_relative_to(job) and cfg.execution.hpc.worker_code_dir == str(runtime)
    assert cfg.execution.hpc.server_script == runtime / 'skills/hpc-deployment/scripts/mcp_server.py'
    assert all(path.is_relative_to(runtime) for path in cfg.skills_dirs)
    seal = json.loads((job / 'prepared-control-source.json').read_text())
    for name, digest in seal['source_sha256'].items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == digest
    assert 'loam/coordinator.py' in seal['source_sha256']
    assert plan['controller_gc_threshold0'] == 10000
    script = (job / 'job.sh').read_text()
    assert f'-m loam.coordinator {job}' in script and str(runtime) in script


def test_changed_frozen_source_is_rejected_at_preflight(tmp_path, monkeypatch, inference_profile):
    job = stage(tmp_path, monkeypatch, inference_profile)
    source = Path(json.loads((job / 'plan.json').read_text())['runtime_repo']) / 'loam' / 'coordinator.py'
    source.write_text(source.read_text() + '\n# Changed after staging\n')

    with pytest.raises(RuntimeError, match='loam/coordinator.py'):
        staging.check(job)


def test_a_separate_input_bundle_is_copied_and_its_hashes_sealed(tmp_path, monkeypatch, inference_profile, inputs):
    job = stage(tmp_path, monkeypatch, inference_profile, inputs=inputs)

    plan = json.loads((job / 'plan.json').read_text())
    assert plan['input_bundle_source'] == str(inputs)
    staged = job / 'campaign' / 'inputs'
    assert (staged / 'recipes' / 'screen.py').read_text() == 'print("screen")\n'
    integrity = json.loads((job / 'inputs-integrity-preflight.json').read_text())['input_sha256']
    assert set(integrity) == {'recipes/screen.py', 'target.fasta'}
    (staged / 'target.fasta').write_text('changed while queued')
    with pytest.raises(RuntimeError, match='target.fasta'):
        staging.check(job)


def test_seed_hypotheses_get_campaign_scoped_ids(tmp_path, monkeypatch, inference_profile):
    seeds = tmp_path / 'seeds.json'
    seeds.write_text(json.dumps([{'hypothesis_id': str(i), 'summary': 's', 'content': 'c'} for i in range(24)]))

    job = stage(tmp_path, monkeypatch, inference_profile, seed_hypotheses=seeds)

    state = json.loads((job / 'campaign' / 'state' / 'jnana.json').read_text())
    assert len(state['hypotheses']) == 24
    assert {h['hypothesis_id'] for h in state['hypotheses']}.isdisjoint({str(i) for i in range(24)})
    assert json.loads((job / 'plan.json').read_text())['seed_hypotheses_count'] == 24
    assert not list((job / 'campaign' / 'state').glob('rounds/round-*.json'))


def test_explicit_admission_is_kept_through_preflight(tmp_path, monkeypatch, inference_profile):
    job = stage(tmp_path, monkeypatch, inference_profile, concurrency=12, top_k=48, initial_hypotheses=96,
                new_per_round=24, tournament_workers=16)

    proof = staging.check(job)
    assert proof['concurrent_hypotheses'] == 12 and proof['selected_hypotheses_per_round'] == 48
    assert proof['science_GPU_slots'] == 48
    plan = json.loads((job / 'plan.json').read_text())
    assert plan['admission'] == dict(concurrency=12, top_k=48, initial_hypotheses=96, new_per_round=24)
    assert plan['reservation'] == '' and plan['tournament_workers'] == 16 and plan['synthesis_concurrency'] == 8


@pytest.mark.parametrize('changes', [dict(concurrency=10, top_k=5), dict(label='Not_A_Label'),
                                     dict(walltime='6h'), dict(controller_gc_threshold0=10)])
def test_invalid_staging_requests_are_refused(tmp_path, monkeypatch, inference_profile, changes):
    with pytest.raises(ValueError):
        stage(tmp_path, monkeypatch, inference_profile, **changes)


def test_an_existing_job_is_never_overwritten(tmp_path, monkeypatch, inference_profile):
    path = reference(tmp_path, monkeypatch)
    staging.stage(path, tmp_path / 'experiment', label='once', nodes=20, inference_profile=inference_profile)

    with pytest.raises(RuntimeError, match='overwrite'):
        staging.stage(path, tmp_path / 'experiment', label='once', nodes=20, inference_profile=inference_profile)


def test_a_reference_reached_through_a_symlink_is_rebased_into_the_job(tmp_path, monkeypatch, inference_profile):
    path = reference(tmp_path, monkeypatch)
    alias = tmp_path / 'reference-alias'
    alias.symlink_to(path.parent, target_is_directory=True)

    job = staging.stage(alias / path.name, tmp_path / 'experiment', label='rebased', nodes=20,
                        inference_profile=inference_profile)

    campaign = job / 'campaign'
    bindings = validate_campaign_paths(load(campaign / 'campaign.yaml'), campaign)
    assert bindings['state_dir'] == str(campaign / 'state')
    assert staging.check(job)['campaign_path_bindings'] == bindings


def fake_runtime(tmp_path, monkeypatch):
    """An installed-looking inference executable and Python, so submission's runtime check passes."""
    executable = tmp_path / 'software' / 'vllm'
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o700)
    profile = tmp_path / 'installed-inference.json'
    profile.write_text(json.dumps({**INFERENCE_PROFILE, 'command': [str(executable), 'serve']}))
    monkeypatch.setenv('LOAM_PYTHON', sys.executable)
    return profile


def test_submission_sends_one_job_with_site_resources_and_records_it(tmp_path, monkeypatch):
    profile = fake_runtime(tmp_path, monkeypatch)
    job = stage(tmp_path, monkeypatch, profile, label='submission-test', nodes=256, reservation='test-reservation',
                walltime='09:00:00')
    calls = []

    def scheduler(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='123456;jupiter\n')
    monkeypatch.setattr(staging.subprocess, 'run', scheduler)

    staging.submit(tmp_path / 'experiment', job)

    assert len(calls) == 1 and calls[0][0] == 'sbatch' and calls[0][-1] == str(job / 'job.sh')
    assert {'--nodes=256', '--account=profile-account', '--partition=profile-partition', '--gres=gpu:4',
            '--cpus-per-task=288', '--reservation=test-reservation', '--time=09:00:00'} <= set(calls[0])
    receipt = json.loads((job / 'submission.json').read_text())
    assert receipt['job_id'] == '123456'
    assert receipt['script_sha256'] == hashlib.sha256((job / 'job.sh').read_bytes()).hexdigest()
    assert json.loads((tmp_path / 'experiment' / 'experiment.json').read_text())['owned_job_ids'] == ['123456']
    assert staging.check(job)['job_id'] == '123456'
    with pytest.raises(RuntimeError, match='already attempted'):
        staging.submit(tmp_path / 'experiment', job)
    assert len(calls) == 1


def test_a_missing_runtime_refuses_submission_before_the_scheduler(tmp_path, monkeypatch, inference_profile):
    with pytest.warns(UnmeasuredScalingWarning):  # two nodes is not a measured operating point
        job = stage(tmp_path, monkeypatch, inference_profile, label='no-runtime', nodes=2)

    def forbidden(*args, **kwargs):
        raise AssertionError('the scheduler was called without a runtime')
    monkeypatch.setattr(staging.subprocess, 'run', forbidden)
    with pytest.raises(RuntimeError, match='Required runtime executable is missing'):
        staging.submit(tmp_path / 'experiment', job)
    assert not (job / 'submission-intent.json').exists()


def test_a_dependency_must_be_owned_by_the_experiment(tmp_path, monkeypatch):
    profile = fake_runtime(tmp_path, monkeypatch)
    job = stage(tmp_path, monkeypatch, profile, after='999')
    monkeypatch.setattr(staging.subprocess, 'run', lambda *a, **k: pytest.fail('submitted'))

    with pytest.raises(RuntimeError, match='belong to this experiment'):
        staging.submit(tmp_path / 'experiment', job)


def test_the_command_stages_and_submits(tmp_path, monkeypatch):
    profile = fake_runtime(tmp_path, monkeypatch)
    path = reference(tmp_path, monkeypatch)
    monkeypatch.setattr(staging.subprocess, 'run', lambda args, **kwargs: SimpleNamespace(stdout='77\n'))

    staging.main([str(path), str(tmp_path / 'experiment'), '--label', 'cli', '--nodes', '20',
                  '--inference-profile', str(profile), '--submit'])

    job = tmp_path / 'experiment' / 'cli'
    assert json.loads((job / 'submission.json').read_text())['job_id'] == '77'
    assert json.loads((job / 'plan.json').read_text())['account'] == 'profile-account'


def test_the_command_has_no_account_or_partition_options():
    with pytest.raises(SystemExit):
        staging.main(['ref.yaml', 'root', '--label', 'x', '--nodes', '2', '--inference-profile', 'p.json',
                      '--account', 'deepwing'])


@pytest.mark.parametrize('field', ['state_dir', 'goal', 'execution.hpc.base_dir', 'execution.workdir'])
def test_any_path_outside_the_staged_campaign_is_rejected(tmp_path, field):
    campaign = tmp_path / 'new'
    cfg = SimpleNamespace(state_dir=campaign / 'state', goal=str(campaign / 'goal.md'),
                          execution=SimpleNamespace(workdir=str(campaign / 'work'),
                                                    hpc=SimpleNamespace(base_dir=str(campaign))))
    validate_campaign_paths(cfg, campaign)
    *owners, name = field.split('.')
    owner = cfg
    for step in owners:
        owner = getattr(owner, step)
    setattr(owner, name, str(tmp_path / 'historical' / name))

    with pytest.raises(RuntimeError, match=field.replace('.', r'\.')):
        validate_campaign_paths(cfg, campaign)


def test_a_state_symlink_cannot_escape_the_campaign(tmp_path):
    campaign = tmp_path / 'new'
    campaign.mkdir()
    historical = tmp_path / 'historical'
    historical.mkdir()
    (campaign / 'state').symlink_to(historical, target_is_directory=True)
    cfg = SimpleNamespace(state_dir=campaign / 'state', goal=str(campaign / 'goal.md'),
                          execution=SimpleNamespace(workdir=str(campaign / 'work'),
                                                    hpc=SimpleNamespace(base_dir=str(campaign))))

    with pytest.raises(RuntimeError, match='state_dir'):
        validate_campaign_paths(cfg, campaign)


def test_site_profile_roots_are_filled_into_the_replica_command_without_a_shell(tmp_path, monkeypatch):
    profile = tmp_path / 'rooted.json'
    profile.write_text(json.dumps({**INFERENCE_PROFILE, 'command': ['${LOAM_SOFTWARE_ROOT}/vllm', '$(reboot)',
                                                                    '${UNSET:-x}']}))

    job = stage(tmp_path, monkeypatch, profile)

    staged = json.loads((job / 'inference-profile.json').read_text())
    assert staged['command'] == ['/soft/loam/vllm', '$(reboot)', '${UNSET:-x}']
