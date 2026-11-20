"""Cross-filesystem staging must keep canonical paths and the one real ledger."""
import json

import pytest

from stage_saturation_scaleup import experiment_control_root, output_root, prepare


def test_external_output_root_shares_ledger_and_stages_owned_paths(tmp_path):
    import stage_single_allocation as staging
    from check_single_allocation import check
    if not staging.REFERENCE.exists():pytest.skip('Jupiter reference assets unavailable')
    control=tmp_path/'control';control.mkdir();ledger=control/'experiment.json'
    ledger.write_text(json.dumps({'owned_job_ids':[]}))
    storage=output_root(control,tmp_path/'other-filesystem')
    assert (storage/'experiment.json').samefile(ledger)
    job=staging.stage(storage,20,8,'storage-path-test','campaign','06:00:00','',48,
        top_k=96,initial_hypotheses=192,new_per_round=48,tournament_workers=16,science_workers_per_gpu=2)
    alias=control/job.name;alias.symlink_to(job,target_is_directory=True)
    assert experiment_control_root(alias)==control
    proof=check(alias)
    assert proof['job_root']==str(job)
    assert proof['campaign_path_bindings']['state_dir']==str(job/'campaign/state')
    assert proof['campaign_path_bindings']['execution.hpc.base_dir']==str(job/'campaign')
    assert len(json.loads((job/'campaign/state/jnana.json').read_text())['hypotheses'])==24
    assert output_root(control,storage)==storage
    assert json.loads(ledger.read_text())=={'owned_job_ids':[]}


@pytest.mark.parametrize('use_symlink',[False,True])
def test_other_ledger_is_never_replaced(tmp_path,use_symlink):
    control=tmp_path/'control';control.mkdir();(control/'experiment.json').write_text('{"owned_job_ids": []}')
    storage=tmp_path/'storage';storage.mkdir();other=tmp_path/'other-ledger';other.write_text('{"owned_job_ids": ["unowned"]}')
    linked=storage/'experiment.json'
    if use_symlink:linked.symlink_to(other)
    else:linked.write_text(other.read_text())
    with pytest.raises(RuntimeError,match='exact existing ownership ledger'):
        output_root(control,storage)
    assert json.loads(linked.read_text())['owned_job_ids']==['unowned']


def test_pending_test_cannot_prepare_external_storage(tmp_path):
    root=tmp_path/'control';job=root/'test';job.mkdir(parents=True)
    (root/'experiment.json').write_text('{"owned_job_ids": ["123"]}')
    (job/'submission.json').write_text('{"job_id": "123"}')
    (job/'plan.json').write_text('{"nodes": 20}')
    storage=tmp_path/'not-created'
    with pytest.raises(RuntimeError,match='not available'):
        prepare(job,256,'unified-256',storage_root=storage)
    assert not storage.exists()
