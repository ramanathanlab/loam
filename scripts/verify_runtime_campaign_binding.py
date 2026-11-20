#!/usr/bin/env python3
"""Read-only binding check inside an already owned running allocation.

Confirms the science server's actual directory and declared resource budget.
Runs no GPU kernels and creates no scientific evidence or additional Slurm job.
Only four explicitly selected, nonsecret process environment fields are read out.
"""
import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from loam.allocation import validate_campaign_paths,validate_role
from loam.config import load


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


async def worker(job):
    from fastmcp import Client
    allocation=validate_role(job/'allocation.json','science');receipt=read(job/'submission.json')
    assert allocation['job_id']==receipt['job_id']
    assert socket.gethostname().split('.')[0]==allocation['coordinator_host']
    campaign=job/'campaign';cfg=load(campaign/'campaign.yaml')
    paths=validate_campaign_paths(cfg,campaign)
    process=read(job/'science-mcp-process.json');assert process['parent_job']==receipt['job_id']
    expected={'SLURM_JOB_ID':receipt['job_id'],'HPC_BASE_DIR':str(campaign),
              'HPC_CAMPAIGN_ID':cfg.execution.hpc.campaign_id,
              'HPC_NNODES':str(len(allocation['science_nodes']))}
    selected={}
    for entry in (Path('/proc')/str(process['pid'])/'environ').read_bytes().split(b'\0'):
        for name in expected:
            prefix=name.encode()+b'='
            if entry.startswith(prefix):selected[name]=entry[len(prefix):].decode()
    for name,value in expected.items():
        if selected.get(name)!=value:raise RuntimeError('Live science server binding mismatch: '+name)
    captured=read(campaign/'control-plane-at-start.json');repo=Path(__file__).resolve().parents[1]
    assert captured['job_id']==receipt['job_id']
    for name,digest in captured['source_sha256'].items():assert sha(repo/name)==digest,name
    prepared=job/'prepared-control-source.json'
    if prepared.exists():assert read(prepared)['source_sha256']==captured['source_sha256']
    inputs=read(job/'inputs-integrity-preflight.json')
    for name,digest in inputs['input_sha256'].items():assert sha(campaign/'inputs'/name)==digest,name
    async with Client(f'http://127.0.0.1:{cfg.execution.hpc.port}/mcp',timeout=120) as client:
        response=await client.call_tool('pool_status',{})
        if response.is_error:raise RuntimeError('Live pool_status request failed')
        pool=json.loads(response.content[0].text)
    nodes=len(allocation['science_nodes']);factor=cfg.execution.hpc.science_workers_per_gpu
    assert pool['campaign_id']==cfg.execution.hpc.campaign_id
    assert pool['nodes']==nodes and pool['total_tiles']==nodes*4
    assert pool['science_workers_per_gpu']==factor and pool['concurrent_scientific_rows']==nodes*4*factor
    proof={'checked_at':time.time(),'job_id':receipt['job_id'],'passed':True,
           'campaign_path_bindings':paths,'live_server_environment':selected,
           'captured_control_sources_checked':len(captured['source_sha256']),
           'prepared_source_capture_matches':prepared.exists(),'input_files_checked':len(inputs['input_sha256']),
           'pool':{key:pool[key] for key in ['nodes','total_tiles','science_workers_per_gpu','concurrent_scientific_rows','campaign_id']},
           'scope':'Live directory bindings, sealed inputs and declared physical/row budgets only. No GPU computation; not proof of shared execution, scientific results or campaign completion.',
           'verifier_sha256':sha(Path(__file__))}
    (job/'diagnostics/live-campaign-binding.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2),flush=True)


def main(job):
    receipt=read(job/'submission.json');assert receipt['job_id'] in read(job.parent/'experiment.json')['owned_job_ids']
    response=subprocess.run(['scontrol','show','job',receipt['job_id'],'-o'],capture_output=True,text=True,check=True,timeout=30)
    fields=dict(part.split('=',1) for part in response.stdout.split() if '=' in part)
    assert fields['JobName']==receipt['job_name'] and fields['WorkDir']==str(job)
    if fields['JobState']!='RUNNING':raise RuntimeError('Job is '+fields['JobState']+'; no runtime check started')
    status=read(job/'status.json')
    if not status.get('science_ready'):raise RuntimeError('Science service is still starting; retry this read-only check after readiness')
    allocation=read(job/'allocation.json');log=job/'diagnostics/live-campaign-binding.log'
    with log.open('a') as stream:
        result=subprocess.run(['srun','--jobid='+receipt['job_id'],'--overlap','--exact','--mpi=none',
            '--nodes=1','--ntasks=1','--cpus-per-task=1','--gpus-per-node=0',
            '--nodelist='+allocation['coordinator_host'],sys.executable,str(Path(__file__).resolve()),str(job),'--worker'],
            stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Runtime binding check failed; inspect '+str(log))
    print(json.dumps(read(job/'diagnostics/live-campaign-binding.json'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=lambda p:Path(p).resolve());parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:asyncio.run(worker(args.job))
    else:main(args.job)
