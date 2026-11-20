"""A staged job must never execute against its historical reference state."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from loam.allocation import validate_campaign_paths


def config(base):
    return SimpleNamespace(state_dir=base/'state',goal=str(base/'goal.md'),workspace_inputs=base/'inputs',
        execution=SimpleNamespace(workdir=str(base/'work'),hpc=SimpleNamespace(base_dir=str(base))))


@pytest.mark.parametrize('field',['state_dir','goal','workspace_inputs','execution.hpc.base_dir','execution.workdir'])
def test_any_reference_path_is_rejected(tmp_path,field):
    campaign=tmp_path/'new';cfg=config(campaign)
    validate_campaign_paths(cfg,campaign)
    owner=cfg
    names=field.split('.')
    for name in names[:-1]:owner=getattr(owner,name)
    setattr(owner,names[-1],str(tmp_path/'historical'/names[-1]))
    with pytest.raises(RuntimeError,match=field.replace('.',r'\.')):
        validate_campaign_paths(cfg,campaign)


def test_state_symlink_cannot_escape_the_campaign(tmp_path):
    campaign=tmp_path/'new';campaign.mkdir();historical=tmp_path/'historical';historical.mkdir()
    (campaign/'state').symlink_to(historical,target_is_directory=True)
    with pytest.raises(RuntimeError,match='state_dir'):
        validate_campaign_paths(config(campaign),campaign)


def test_symlinked_reference_is_rebased_and_owner_fails_before_services(tmp_path,monkeypatch):
    from loam.config import load
    import stage_single_allocation as staging
    from check_single_allocation import check
    import run_single_allocation as owner
    if not staging.REFERENCE.exists():pytest.skip('Jupiter reference assets unavailable')
    alias=tmp_path/'reference-alias';alias.symlink_to(staging.REFERENCE.resolve(),target_is_directory=True)
    monkeypatch.setattr(staging,'REFERENCE',alias)
    job=staging.stage(tmp_path/'experiment',20,8,'test-rebased','campaign','06:00:00','',48,
        top_k=96,initial_hypotheses=192,new_per_round=48,tournament_workers=16,science_workers_per_gpu=2)
    campaign=job/'campaign';cfg=load(campaign/'campaign.yaml')
    bindings=validate_campaign_paths(cfg,campaign)
    assert bindings['state_dir']==str(campaign/'state')
    assert bindings['execution.hpc.base_dir']==str(campaign)
    assert len(json.loads((campaign/'state/jnana.json').read_text())['hypotheses'])==24
    assert not list((campaign/'state').glob('rounds/round-*.json'))
    assert check(job)['campaign_path_bindings']==bindings
    # Prove that a bad state path stops the owner before even querying Slurm,
    # let alone starting model services or touching the historical campaign.
    import yaml
    path=campaign/'campaign.yaml';raw=yaml.safe_load(path.read_text());raw['state_dir']=str(alias/'state')
    path.write_text(yaml.safe_dump(raw,sort_keys=False))
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setenv('SLURM_JOB_NAME',json.loads((job/'plan.json').read_text())['job_name'])
    def forbidden(*args,**kwargs):raise AssertionError('Startup reached Slurm/service work before rejecting the path')
    monkeypatch.setattr(owner.subprocess,'run',forbidden)
    with pytest.raises(RuntimeError,match='state_dir'):
        asyncio.run(owner.run(job))
