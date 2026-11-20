#!/usr/bin/env python3
"""Stage the large successor only after the measured 20-node gates pass.

No Slurm submission without --submit. Scaling preserves per-science-node
admission/workload density, generation batch size, sharing and node ratio.
"""
import argparse
import ast
import hashlib
import json
import math
import time
from pathlib import Path

import yaml

from check_single_allocation import check
from loam.config import load
from stage_single_allocation import REPO,stage,submit


def output_root(control_root, requested):
    """Keep one ownership ledger when outputs use another project filesystem."""
    control_root=control_root.resolve();destination=Path(requested).resolve() if requested else control_root
    ledger=control_root/'experiment.json'
    if destination==control_root:return destination
    destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    linked=destination/'experiment.json'
    if linked.exists() or linked.is_symlink():
        if not linked.is_symlink() or linked.resolve()!=ledger:
            raise RuntimeError('Output root must share the exact existing ownership ledger')
    else:linked.symlink_to(ledger)
    return destination


def experiment_control_root(test):
    """An external test may link to the original experiment ownership ledger."""
    ledger=(Path(test).resolve().parent/'experiment.json').resolve(strict=True)
    if ledger.name!='experiment.json':raise RuntimeError('Unexpected ownership ledger filename')
    return ledger.parent


def prepare(test,nodes,label,check_only=False,send=False,storage_root=None):
    test=test.resolve();root=experiment_control_root(test)
    read=lambda path:json.loads(path.read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    receipt=read(test/'submission.json');plan=read(test/'plan.json')
    assert receipt['job_id'] in read(root/'experiment.json')['owned_job_ids']
    assert plan['nodes']==20 and nodes==256
    audit_path=test/'saturation-audit.json'
    if not audit_path.exists():raise RuntimeError('Full 20-node saturation audit is not available; no large job staged or submitted')
    audit=read(audit_path)
    if not audit['principle_gate_passed'] or not all(audit['principle_gate'].values()):
        raise RuntimeError('The 20-node principle gate failed; no large job staged or submitted')
    check(test)
    summary=audit['current']['summary'];assert summary['job_id']==receipt['job_id']
    assert summary['scientific_completion_passed']
    assert summary['recorded_rounds']==summary['clean_synthesis_rounds']==3
    tournament_path=test/'tournament-audit.json';tournament=read(tournament_path)
    assert tournament['job_id']==receipt['job_id'] and tournament['meets_three_round_tournament_requirement']
    for name,digest in tournament['sources'].items():assert sha(Path(name))==digest
    scientific_path=test/'campaign/scaling-audit.json';science=read(scientific_path)
    assert science['model_coordinates_independently_checked'] and science['required_campaign_completion_checks_passed']
    assert not science['errors'] and not science['unowned_or_missing_claim_sources']
    for name in ['numeric_audit','claim_audit']:
        lineage=science.get('verification_lineage',{})
        if name in lineage:assert sha(Path(lineage[name]))==lineage[name+'_sha256']
    for n in range(1,4):
        record=read(test/f'campaign/state/rounds/round-{n:03d}.json')
        manifest_path=Path(record['brief']['synthesis_receipt']);manifest=read(manifest_path)
        assert manifest['status']=='complete' and manifest['thread_count']==len(record['threads'])
        assert sha(manifest_path.parent/'input.json')==manifest['input_sha256']
        for call in manifest['calls']:
            assert call['status']=='complete' and call['prompt_bytes']<=manifest['prompt_budget_bytes']<=64000
            request=read(manifest_path.parent/(call['node']+'-request.json'))['prompt']
            assert len(request.encode())==call['prompt_bytes']
            assert sha(manifest_path.parent/(call['node']+'-response.json'))==call['response_sha256']
    cfg=load(test/'campaign/campaign.yaml')
    inference=round(nodes*plan['inference_count']/plan['nodes']);science_nodes=nodes-inference
    factor=science_nodes/plan['science_count'];scaled=lambda value:math.ceil(value*factor)
    # Explicit headroom for larger service startup/dispatch; elapsed is measured,
    # while this walltime is a requested upper bound, not a throughput prediction.
    hours=max(6,math.ceil((summary['final_allocation_seconds']*1.5+1800)/3600))
    settings=dict(nodes=nodes,inference=inference,concurrency=scaled(cfg.concurrency),
        top_k=scaled(cfg.tournament.top_k),initial_hypotheses=scaled(cfg.tournament.initial_hypotheses),
        new_per_round=scaled(cfg.tournament.new_per_round),
        tournament_workers=math.ceil(cfg.tournament.workers*inference/plan['inference_count']),
        science_workers_per_gpu=cfg.execution.hpc.science_workers_per_gpu,
        generation_batch_size=cfg.tournament.gen_batch_size,walltime=f'{hours:02d}:00:00',
        gpus_per_node=plan.get('gpus_per_node',4),cpus_per_node=plan.get('cpus_per_node',288),
        account=plan.get('account','deepwing'),partition=plan.get('partition','booster'),
        software_root=plan.get('software_root'),controller_gc_threshold0=plan.get('controller_gc_threshold0'),
        inputs=str(test/'campaign/inputs'),reference=str(test/'campaign'))
    destination=Path(storage_root).resolve() if storage_root else test.parent
    print(json.dumps({'qualified_test_job':receipt['job_id'],'large_settings':settings,'check_only':check_only,
                      'output_root':str(destination),'ownership_ledger':str(root/'experiment.json')},indent=2),flush=True)
    if check_only:return settings
    alias=root/label
    if destination!=root and (alias.exists() or alias.is_symlink()):
        raise RuntimeError('Refuse to replace an existing job or output alias')
    destination=output_root(root,destination)
    args={k:v for k,v in settings.items() if k!='generation_batch_size'}
    job=stage(destination,label=label,mode='campaign',reservation='',**args)
    if destination!=root:alias.symlink_to(job,target_is_directory=True)
    path=job/'campaign/campaign.yaml';raw=yaml.safe_load(path.read_text())
    raw['tournament']['gen_batch_size']=settings['generation_batch_size']
    for role,spec in cfg.models.items():
        candidate=raw['models'][role].get('settings',{})
        assert candidate.get('max_tokens')==spec.settings.get('max_tokens'), 'Scale-up output budget differs from qualified test'
        assert candidate.get('extra_body',{}).get('chat_template_kwargs')==spec.settings.get('extra_body',{}).get('chat_template_kwargs'), 'Scale-up thinking profile differs from qualified test'
    path.write_text(yaml.safe_dump(raw,sort_keys=False))
    new_protocol={f['file']:f['sha256'] for f in read(job/'campaign/scientific-protocol.json')['files']}
    old_protocol={f['file']:f['sha256'] for f in read(test/'campaign/scientific-protocol.json')['files']}
    assert all(new_protocol.get(name)==digest for name,digest in old_protocol.items())
    newplan=read(job/'plan.json');newplan.update(generation_batch_size=settings['generation_batch_size'],
        qualified_test_job=receipt['job_id'],qualification_audit=str(audit_path),qualification_audit_sha256=sha(audit_path),
        qualification_scientific_audit_sha256=sha(scientific_path),qualification_tournament_audit_sha256=sha(tournament_path),
        measured_test_allocation_seconds=summary['final_allocation_seconds'],qualified_at=time.time(),
        ownership_ledger=str(root/'experiment.json'),control_alias=str(alias))
    (job/'plan.json').write_text(json.dumps(newplan,indent=2)+'\n')
    proof=check(job);assert load(path).tournament.gen_batch_size==cfg.tournament.gen_batch_size
    (job/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
    # Seal the input dataset and runtime before a potentially long queue wait.
    hashes={str(p.relative_to(job/'campaign/inputs')):sha(p) for p in (job/'campaign/inputs').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    assert hashes==read(test/'inputs-integrity-preflight.json')['input_sha256']
    (job/'inputs-integrity-preflight.json').write_text(json.dumps({'checked_at':time.time(),
        'source_campaign':str(test/'campaign'),'all_input_files_identical':True,'input_sha256':hashes},indent=2)+'\n')
    if send:submit(destination,job)
    return job


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('test',type=Path);parser.add_argument('--nodes',type=int,default=256)
    parser.add_argument('--label',default='unified-256');parser.add_argument('--check-only',action='store_true');parser.add_argument('--submit',action='store_true')
    parser.add_argument('--output-root',type=Path,help='Optional project output filesystem; shares the original experiment ownership ledger')
    args=parser.parse_args();assert not(args.check_only and args.submit)
    prepare(args.test,args.nodes,args.label,args.check_only,args.submit,args.output_root)
