"""Node/GPU recipes, fresh snapshot staging, and scheduler submission boundaries."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from loam.config import expand_environment, load
from loam.scaling import dense_recipe
import stage_single_allocation as staging


@pytest.mark.parametrize('nodes,inference,science,active,k,initial,fresh,rank,synth', [
    (20, 8, 12, 48, 144, 288, 72, 16, 8),
    (256, 102, 154, 616, 1848, 3696, 924, 204, 102),
])
def test_measured_density(nodes,inference,science,active,k,initial,fresh,rank,synth):
    r=dense_recipe(nodes)
    assert (r.inference,r.science,r.concurrency,r.top_k,r.initial_hypotheses,
            r.new_per_round,r.tournament_workers,r.synthesis_concurrency)==(
            inference,science,active,k,initial,fresh,rank,synth)
    assert r.generation_batch_size==4 and r.swiss_rounds==1


def test_density_tracks_logical_gpu_capacity_and_explicit_ratio():
    r=dense_recipe(10,8,inference=4)
    assert r.science==6 and r.concurrency==48 and r.top_k==144
    assert dense_recipe(10,8,inference=4,science_workers_per_gpu=1).concurrency==24


@pytest.mark.parametrize('args',[(1,), (0,), (20,0), (20,-1), (True,)])
def test_invalid_budget_is_rejected(args):
    with pytest.raises(ValueError):dense_recipe(*args)


def test_site_placeholders_do_not_execute_shell(monkeypatch):
    monkeypatch.setenv('LOAM_SOFTWARE_ROOT','/shared/software')
    value=expand_environment({'path':'${LOAM_SOFTWARE_ROOT}/tool','shell':'${UNSET:-fallback}; $(exit 1)'})
    assert value['path']=='/shared/software/tool'
    assert value['shell']=='${UNSET:-fallback}; $(exit 1)'


def test_snapshot_and_input_seals_are_complete(tmp_path):
    job=staging.stage(tmp_path,20,8,'sealed','campaign','08:00:00','',48,
        top_k=144,initial_hypotheses=288,new_per_round=72,science_workers_per_gpu=2,
        software_root=tmp_path/'software',controller_gc_threshold0=10000)
    plan=json.loads((job/'plan.json').read_text());cfg=load(job/'campaign/campaign.yaml')
    runtime=Path(plan['runtime_repo'])
    assert runtime.is_relative_to(job) and cfg.execution.hpc.worker_code_dir==str(runtime)
    assert cfg.timeouts.mcp_init==60
    assert cfg.tournament.gen_batch_size==4 and cfg.tournament.max_seconds==2400
    seal=json.loads((job/'prepared-control-source.json').read_text())
    for name,digest in seal['source_sha256'].items():
        assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==digest
    assert 'loam/skills.py' in seal['source_sha256']
    assert 'scripts/jupiter_nemotron.py' in seal['source_sha256']
    assert not (job/'submission.json').exists() and not (job/'submission-intent.json').exists()
    assert cfg.models['reasoner'].settings['max_tokens']==32768
    assert all(m.settings['extra_body']['chat_template_kwargs']['enable_thinking'] is False
               for m in cfg.models.values())
    request=json.loads((job/'inference-request-template.json').read_text())
    assert request['max_model_len']==131072 and request['max_num_seqs']==16
    assert request['tensor_parallel_size']==4
    assert request['command']==[str(tmp_path/'software/nemotron-vllm/.venv/bin/vllm')]
    assert all('runs/jupiter-' not in p for p in seal['source_sha256'])
    assert plan['account']=='deepwing' and plan['controller_gc_threshold0']==10000


def test_missing_runtime_refuses_submission_before_sbatch(tmp_path,monkeypatch):
    job=staging.stage(tmp_path,2,1,'no-runtime','campaign','01:00:00','',1,
        software_root=tmp_path/'missing-software')
    def forbidden(*args,**kwargs):raise AssertionError('sbatch called without runtime')
    monkeypatch.setattr(staging.subprocess,'run',forbidden)
    with pytest.raises(RuntimeError,match='Required runtime executable is missing'):
        staging.submit(tmp_path,job)
    assert not (job/'submission-intent.json').exists()


def test_submit_emits_one_parent_with_site_resources_and_a_receipt(tmp_path,monkeypatch):
    import sys
    software=tmp_path/'software'
    for rel in ('nemotron-vllm/.venv/bin/vllm','redis/bin/redis-server'):
        p=software/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('#!/bin/sh\nexit 0\n');p.chmod(0o700)
    revision='fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe';weights=tmp_path/'models'/revision;weights.mkdir(parents=True)
    (weights/'config.json').write_text('{"model_type":"nemotron_h_omni"}')
    (weights/'model.safetensors.index.json').write_text('{"weight_map":{"fixture":"fixture.safetensors"}}')
    (weights/'fixture.safetensors').write_bytes(b'TEST_ONLY_NOT_MODEL_WEIGHTS')
    monkeypatch.setenv('LOAM_PYTHON',sys.executable)
    job=staging.stage(tmp_path/'experiment',256,102,'submission-test','campaign','09:00:00','test-reservation',616,
        top_k=1848,initial_hypotheses=3696,new_per_round=924,
        software_root=software,model_dir=weights,account='test-account',partition='test-partition',
        cpus_per_node=96,gpus_per_node=8,science_workers_per_gpu=2)
    calls=[]
    def scheduler(args,**kwargs):
        calls.append(args);return SimpleNamespace(stdout='123456;test-cluster\n')
    monkeypatch.setattr(staging.subprocess,'run',scheduler)
    staging.submit(tmp_path/'experiment',job)
    assert len(calls)==1 and calls[0][0]=='sbatch'
    assert {'--nodes=256','--account=test-account','--partition=test-partition',
            '--gres=gpu:8','--cpus-per-task=96','--reservation=test-reservation'}<=set(calls[0])
    receipt=json.loads((job/'submission.json').read_text())
    assert receipt['job_id']=='123456'
    assert json.loads((job.parent/'experiment.json').read_text())['owned_job_ids']==['123456']
    assert receipt['script_sha256']==hashlib.sha256((job/'job.sh').read_bytes()).hexdigest()
    with pytest.raises(RuntimeError,match='already attempted'):
        staging.submit(job.parent,job)
    assert len(calls)==1


def test_native_source_mutation_is_rejected_at_preflight(tmp_path):
    from check_single_allocation import check
    job=staging.stage(tmp_path,2,1,'tamper','smoke','01:00:00','',1)
    native=job/'infra/inference-runtime/scripts/jupiter_nemotron.py'
    native.write_text(native.read_text()+'\n# Changed after staging\n')
    with pytest.raises(AssertionError):check(job)
