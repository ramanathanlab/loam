#!/usr/bin/env python3
"""Drain and retire only recorded owned replicas, between scientific trials."""
import argparse, fcntl, json, subprocess, time
from pathlib import Path
import httpx


def read(p):return json.loads(p.read_text())
def main(root,keep):
    lock=(root/'.replica-management.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    experiment=read(root/'experiment.json');pool=read(root/'pool.json');active=[b for b in pool['backends'] if b.get('enabled',True)]
    assert 0<=keep<=len(active)
    for p in (root/'trials').glob('*/controller-watch.json'):
        if read(p).get('state')=='controller running':raise RuntimeError('A scientific trial is still running')
    inference_ids={b['job_id'] for b in pool['backends']}
    queue=subprocess.run(['squeue','-h','-j',','.join(experiment['owned_job_ids']),'-o','%i'],capture_output=True,text=True,check=True).stdout.split()
    assert not(set(queue)-inference_ids),'Wait until every old owned science job has fully left Slurm'
    service=read(root/'pool-service.json')
    with httpx.Client(trust_env=False,timeout=15,headers={'Authorization':'Bearer '+Path(service['token_file']).read_text().strip()}) as client:
        result=client.get(service['status_url']);result.raise_for_status();status=result.json()
        assert all(b['inflight']==0 for b in status['backends']),'Inference requests are still running'
    retire=active[keep:]
    for entry in retire:
        job=entry['job_id'];attempt=Path(entry['campaign'])/'state/nemotron'/entry['attempt']
        assert job in experiment['owned_job_ids'] and attempt.resolve().is_relative_to(root/'replicas')
        assert read(attempt/'request.json')['campaign']==entry['campaign'] and str(read(attempt/'submission.json')['job_id'])==job
        if job not in queue:continue
        identity=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,check=True).stdout
        assert ('JobName='+entry['attempt']+' ') in identity and ('WorkDir='+str(attempt)+' ') in identity
        assert 'Account=deepwing ' in identity and 'Reservation=bne_deepwing ' in identity
    for entry in retire:entry['enabled']=False
    pool['phase']='between-trials';(root/'pool.json').write_text(json.dumps(pool,indent=2)+'\n')
    directory=root/'retirements';directory.mkdir(exist_ok=True)
    for entry in retire:
        record={'job_id':entry['job_id'],'replica':entry['id'],'attempt':entry['attempt'],'campaign':entry['campaign'],'requested_at':time.time(),'reason':'Owned inference ratio experiment finished or reducing replica count between trials','cancel_requested':entry['job_id'] in queue}
        (directory/(entry['job_id']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        if entry['job_id'] in queue:subprocess.run(['scancel',entry['job_id']],capture_output=True,text=True,check=True)
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--keep',required=True,type=int);a=p.parse_args();main(a.root.resolve(),a.keep)
