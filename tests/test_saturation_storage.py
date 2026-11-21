"""Cross-filesystem staging must keep canonical paths and the one real ledger."""
import json

import pytest

from loam.saturation.storage import experiment_control_root, output_root


def control_root(tmp_path, owned=()):
    control = tmp_path / 'control'
    control.mkdir()
    (control / 'experiment.json').write_text(json.dumps({'owned_job_ids': list(owned)}))
    return control


def test_external_output_root_shares_the_ledger_and_resolves_back_to_it(tmp_path):
    control = control_root(tmp_path)
    ledger = control / 'experiment.json'
    assert output_root(control, None) == control.resolve()
    storage = output_root(control, tmp_path / 'other-filesystem')
    assert (storage / 'experiment.json').samefile(ledger)
    assert output_root(control, storage) == storage  # Idempotent once linked.
    job = storage / 'unified-256'
    job.mkdir()
    alias = control / job.name
    alias.symlink_to(job, target_is_directory=True)
    assert experiment_control_root(alias) == control.resolve()
    assert experiment_control_root(job) == control.resolve()
    assert json.loads(ledger.read_text()) == {'owned_job_ids': []}


@pytest.mark.parametrize('use_symlink', [False, True])
def test_other_ledger_is_never_replaced(tmp_path, use_symlink):
    control = control_root(tmp_path)
    storage = tmp_path / 'storage'
    storage.mkdir()
    other = tmp_path / 'other-ledger'
    other.write_text('{"owned_job_ids": ["unowned"]}')
    linked = storage / 'experiment.json'
    if use_symlink:
        linked.symlink_to(other)
    else:
        linked.write_text(other.read_text())
    with pytest.raises(RuntimeError, match='exact existing ownership ledger'):
        output_root(control, storage)
    assert json.loads(linked.read_text())['owned_job_ids'] == ['unowned']


def test_external_output_root_shares_ledger_and_stages_owned_paths(tmp_path, monkeypatch):
    from loam import staging
    from test_single_allocation_staging import reference
    profile = tmp_path / 'inference.json'
    profile.write_text(json.dumps({'model': 'served-model', 'command': ['vllm', 'serve']}))
    seeds = tmp_path / 'seeds.json'
    seeds.write_text(json.dumps([{'hypothesis_id': str(i), 'summary': 's', 'content': 'c'} for i in range(24)]))
    control = control_root(tmp_path)
    storage = output_root(control, tmp_path / 'other-filesystem')
    assert (storage / 'experiment.json').samefile(control / 'experiment.json')

    job = staging.stage(reference(tmp_path, monkeypatch), storage, label='storage-path-test', nodes=20,
                        inference_profile=profile, seed_hypotheses=seeds, concurrency=48, top_k=96,
                        initial_hypotheses=192, new_per_round=48, tournament_workers=16)
    alias = control / job.name
    alias.symlink_to(job, target_is_directory=True)

    assert experiment_control_root(alias) == control.resolve()
    proof = staging.check(alias)
    assert proof['job_root'] == str(job.resolve())
    assert proof['campaign_path_bindings']['state_dir'] == str((job / 'campaign/state').resolve())
    assert proof['campaign_path_bindings']['execution.hpc.base_dir'] == str((job / 'campaign').resolve())
    assert len(json.loads((job / 'campaign/state/jnana.json').read_text())['hypotheses']) == 24
    assert output_root(control, storage) == storage
    assert json.loads((control / 'experiment.json').read_text()) == {'owned_job_ids': []}


def test_pending_test_cannot_prepare_external_storage(tmp_path):
    from loam.saturation.scaleup import prepare
    root = tmp_path / 'control'
    job = root / 'test'
    job.mkdir(parents=True)
    (root / 'experiment.json').write_text('{"owned_job_ids": ["123"]}')
    (job / 'submission.json').write_text('{"job_id": "123"}')
    (job / 'plan.json').write_text('{"nodes": 20}')
    storage = tmp_path / 'not-created'

    with pytest.raises(RuntimeError, match='not available'):
        prepare(job, 256, 'unified-256', storage_root=storage)
    assert not storage.exists()


def test_a_trial_that_failed_the_principle_gate_cannot_prepare(tmp_path):
    from loam.saturation.scaleup import prepare
    root = tmp_path / 'control'
    job = root / 'test'
    job.mkdir(parents=True)
    (root / 'experiment.json').write_text('{"owned_job_ids": ["123"]}')
    (job / 'submission.json').write_text('{"job_id": "123"}')
    (job / 'plan.json').write_text('{"nodes": 20}')
    (job / 'saturation-audit.json').write_text(json.dumps(
        {'principle_gate_passed': False, 'principle_gate': {'screen_throughput_gain_at_least_1p5': False}}))

    with pytest.raises(RuntimeError, match='principle gate failed'):
        prepare(job, 256, 'unified-256')
    assert sorted(p.name for p in root.iterdir()) == ['experiment.json', 'test']


def test_an_unowned_trial_cannot_prepare(tmp_path):
    from loam.saturation.scaleup import prepare
    root = tmp_path / 'control'
    job = root / 'test'
    job.mkdir(parents=True)
    (root / 'experiment.json').write_text('{"owned_job_ids": []}')
    (job / 'submission.json').write_text('{"job_id": "123"}')
    (job / 'plan.json').write_text('{"nodes": 20}')

    with pytest.raises(RuntimeError, match='not owned'):
        prepare(job, 256, 'unified-256')


def qualified_trial(tmp_path, monkeypatch):
    """A staged, submitted 20-node trial whose saturation audit passed, with three clean synthesis rounds."""
    from loam import staging
    from test_single_allocation_staging import reference
    profile = tmp_path / 'inference.json'
    profile.write_text(json.dumps({'model': 'served-model', 'command': ['vllm', 'serve']}))
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    (inputs / 'target.fasta').write_text('>t\nMKV\n')
    control = control_root(tmp_path, owned=['123'])
    trial = staging.stage(reference(tmp_path, monkeypatch), control, label='trial-20', nodes=20,
                          inference_profile=profile, inputs=inputs)
    plan = json.loads((trial / 'plan.json').read_text())
    (trial / 'submission.json').write_text(json.dumps({'job_id': '123', 'job_name': plan['job_name'],
                                                       'script_sha256': staging.sha(trial / 'job.sh')}))
    (trial / 'saturation-audit.json').write_text(json.dumps({
        'principle_gate_passed': True, 'principle_gate': {'gate': True},
        'current': {'summary': {'job_id': '123', 'scientific_completion_passed': True, 'recorded_rounds': 3,
                                'clean_synthesis_rounds': 3, 'final_allocation_seconds': 3 * 3600}}}))
    campaign = trial / 'campaign'
    (campaign / 'scaling-audit.json').write_text(json.dumps({
        'model_coordinates_independently_checked': True, 'required_campaign_completion_checks_passed': True,
        'errors': [], 'unowned_or_missing_claim_sources': []}))
    rounds = campaign / 'state' / 'rounds'
    rounds.mkdir(parents=True)
    for n in range(1, 4):
        receipt = campaign / 'state' / 'synthesis' / f'round-{n}'
        receipt.mkdir(parents=True)
        (receipt / 'input.json').write_text('{}')
        (receipt / 'leaf-response.json').write_text('{}')
        (receipt / 'leaf-request.json').write_text(json.dumps({'prompt': 'ten bytes!'}))
        (receipt / 'manifest.json').write_text(json.dumps({
            'status': 'complete', 'thread_count': 1, 'input_sha256': staging.sha(receipt / 'input.json'),
            'prompt_budget_bytes': 64000, 'calls': [{'node': 'leaf', 'status': 'complete', 'prompt_bytes': 10,
                                                     'response_sha256': staging.sha(receipt / 'leaf-response.json')}]}))
        (rounds / f'round-{n:03d}.json').write_text(json.dumps(
            {'threads': [{}], 'brief': {'synthesis_receipt': str(receipt / 'manifest.json')}}))
    return control, trial


def test_a_qualified_trial_stages_a_successor_at_the_same_density(tmp_path, monkeypatch):
    from loam import staging
    from loam.saturation.scaleup import prepare
    control, trial = qualified_trial(tmp_path, monkeypatch)

    settings = prepare(trial, 256, 'unified-256', check_only=True)
    assert not (control / 'unified-256').exists()
    # The measured 256-node operating point (docs/scaling-recipe-book.md): 102 inference, 154 science.
    assert (settings['inference'], settings['concurrency'], settings['top_k']) == (102, 616, 1848)
    assert settings['tournament_workers'] == 204 and settings['walltime'] == '06:00:00'

    job = prepare(trial, 256, 'unified-256', storage_root=tmp_path / 'scratch')

    assert (control / 'unified-256').resolve() == job.resolve()
    plan = json.loads((job / 'plan.json').read_text())
    assert (plan['nodes'], plan['science_count'], plan['qualified_trial_job']) == (256, 154, '123')
    assert plan['account'] == 'profile-account'
    assert staging.check(job)['concurrent_hypotheses'] == 616
    assert (job / 'campaign' / 'inputs' / 'target.fasta').read_text() == '>t\nMKV\n'
    assert not (job / 'submission.json').exists()
