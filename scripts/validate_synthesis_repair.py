#!/usr/bin/env python3
"""Replay an archived failed synthesis against this owned allocation's Nemotron.

Diagnostic output is isolated from campaign state and never changes a recorded
round or claims that the original failed checkpoint succeeded.
"""
import argparse
import asyncio
import hashlib
import json
import os
import socket
import time
import uuid
from pathlib import Path

from loam.allocation import validate_role
from loam.config import load
from loam.agents import build
from loam.synthesis import synthesize


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


async def replay(job, round_number=1, max_output_tokens=None):
    allocation=validate_role(job/'allocation.json','science')
    assert allocation['job_id']==read(job/'submission.json')['job_id']
    assert socket.gethostname().split('.')[0]==allocation['coordinator_host']
    old=next((job/'campaign/state/synthesis'/f'round-{round_number:03d}').glob('*/manifest.json'))
    before=read(old);assert before['status']=='failed'
    source=old.parent/'input.json';raw=read(source)
    directory=job/'diagnostics/synthesis-repair'/('live-'+uuid.uuid4().hex)
    directory.mkdir(parents=True)
    cfg=load(job/'campaign/campaign.yaml');cfg.state_dir=directory/'state'
    cfg.execution.mcp_url=read(job/'campaign/server-ready.json')['mcp_url']
    service=read(job/'infra/pool-service.json');token=Path(service['token_file']).resolve()
    assert token.is_relative_to((job/'infra').resolve())
    spec=cfg.models['reasoner'];assert spec.base_url.rstrip('/')==service['base_url'].rstrip('/')
    original_max_output=spec.settings.get('max_tokens')
    if max_output_tokens is not None:
        assert max_output_tokens>0
        spec.settings['max_tokens']=max_output_tokens
    assert spec.settings['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    os.environ[spec.api_key_env]=token.read_text().strip()
    started=time.time()
    result=await synthesize(build(cfg),cfg,raw['threads'],raw['steering'],raw['previous_brief'],round_number)
    manifest=read(Path(result['synthesis_receipt']))
    assert manifest['status']=='complete' and result['tournament_brief']
    assert manifest['input_sha256']==before['input_sha256']==sha(source)
    assert all(c['status']=='complete' and c['prompt_bytes']<=manifest['prompt_budget_bytes'] for c in manifest['calls'])
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    record={'passed':True,'round':round_number,'original_max_output_tokens':original_max_output,
            'requested_max_output_tokens':spec.settings.get('max_tokens'),'started_at':started,'finished_at':time.time(),'job_id':allocation['job_id'],
            'corpus_source':str(source),'corpus_sha256':sha(source),'input_identical_to_failed_checkpoint':True,
            'source_threads':len(raw['threads']),'source_steering_rows':len(raw['steering']),
            'model_calls':len(manifest['calls']),'reduction_levels':manifest['levels'],
            'largest_prompt_bytes':max(c['prompt_bytes'] for c in manifest['calls']),
            'largest_response_bytes':max(c['output_bytes'] for c in manifest['calls']),
            'all_requests_within_budget':True,'full_reports_retained_by_hash':True,
            'thinking_enabled':False,'synthesis_receipt':result['synthesis_receipt'],
            'reducer_sha256':sha(Path(__file__).resolve().parents[1]/'loam/synthesis.py'),
            'scope':'Diagnostic replay only. Original failed round and scientific evidence remain unchanged; this is not campaign completion.'}
    (job/'diagnostics/synthesis-repair/live-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=lambda p:Path(p).resolve())
    parser.add_argument('--round',type=int,default=1,dest='round_number')
    parser.add_argument('--max-output-tokens',type=int)
    args=parser.parse_args();assert args.round_number>0
    asyncio.run(asyncio.wait_for(replay(args.job,args.round_number,args.max_output_tokens),timeout=600))
