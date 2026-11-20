#!/usr/bin/env python3
"""Read-only ownership/preflight summary; never print service credentials."""
import argparse, hashlib, json, subprocess
from pathlib import Path

from loam.config import load
from loam.allocation import validate_campaign_paths

REPO=Path(__file__).resolve().parents[1]

def check(job):
    job=job.resolve();plan=json.loads((job/'plan.json').read_text());campaign=Path(plan['campaign']);cfg=load(campaign/'campaign.yaml')
    path_bindings=validate_campaign_paths(cfg,campaign)
    proof=json.loads((campaign/'scientific-protocol.json').read_text())
    for file in proof['files']:
        assert hashlib.sha256((campaign/'inputs'/file['file']).read_bytes()).hexdigest()==file['sha256']
    integrity=job/'inputs-integrity-preflight.json'
    if integrity.exists():
        for name,digest in json.loads(integrity.read_text())['input_sha256'].items():
            assert hashlib.sha256((campaign/'inputs'/name).read_bytes()).hexdigest()==digest,name
    seal_path=job/'prepared-control-source.json'
    if seal_path.exists():
        seal=json.loads(seal_path.read_text());runtime=Path(seal['runtime_repo'])
        assert runtime.resolve()==Path(plan.get('runtime_repo',seal['runtime_repo'])).resolve()
        for name,digest in seal['source_sha256'].items():
            assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==digest,name
        for name,digest in seal.get('native_source_sha256',{}).items():
            assert hashlib.sha256((Path(plan['infra'])/name).read_bytes()).hexdigest()==digest,name
        if 'inference_request_template_sha256' in seal:
            assert hashlib.sha256(Path(plan['inference_request_template']).read_bytes()).hexdigest()==seal['inference_request_template_sha256']
    assert plan['mode']=='single-allocation' and plan['inference_count']+plan['science_count']==plan['nodes']
    assert cfg.execution.hpc.allocation.nodes==plan['science_count']
    assert cfg.execution.hpc.science_workers_per_gpu==plan.get('science_workers_per_gpu',1)
    for name,model in cfg.models.items():
        assert model.settings['extra_body']['chat_template_kwargs']['enable_thinking'] is False,name
    for model in [cfg.tournament.generation,cfg.tournament.ranking]:
        assert 'enable_thinking' not in str(model.settings)
    assert all(model.model=='nemotron-3.5-super-vl' for model in [*cfg.models.values(),cfg.tournament.generation,cfg.tournament.ranking]), 'The combined native pool requires its registered Nemotron model identity'
    assert 0<cfg.concurrency<=cfg.tournament.top_k
    if 'tournament_workers' in plan:assert cfg.tournament.workers==plan['tournament_workers']
    if 'synthesis_concurrency' in plan:assert cfg.synthesis_concurrency==plan['synthesis_concurrency']
    if plan.get('admission'):
        assert plan['admission']=={'concurrency':cfg.concurrency,'top_k':cfg.tournament.top_k,'initial_hypotheses':cfg.tournament.initial_hypotheses,'new_per_round':cfg.tournament.new_per_round}
    assert str(plan['science_count'])+' scientific nodes' in cfg.execution.setup
    physical_gpus=plan.get('gpus_per_node',4)
    assert (cfg.execution.hpc.gpus_per_node or 4)==physical_gpus
    result={'job_root':str(job),'nodes':plan['nodes'],'inference':plan['inference_count'],'science':plan['science_count'],'concurrent_hypotheses':cfg.concurrency,'selected_hypotheses_per_round':cfg.tournament.top_k,'science_GPU_slots':physical_gpus*plan['science_count'],'control_submitter_agents':cfg.execution.hpc.pool_size,'frozen_recipes_unchanged':True,'thinking_profiles_preserved':True,'reservation':plan['reservation']}
    receipt=job/'submission.json'
    result['campaign_path_bindings']=path_bindings
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['job_name']==plan['job_name'];assert hashlib.sha256((job/'job.sh').read_bytes()).hexdigest()==r['script_sha256'];result['job_id']=r['job_id']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);a=p.parse_args();print(json.dumps(check(a.job),indent=2))
