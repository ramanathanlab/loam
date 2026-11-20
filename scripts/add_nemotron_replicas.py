#!/usr/bin/env python3
"""Add owned, identically configured replicas between scientific trials only."""
import argparse, fcntl, json, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace


def read(p):return json.loads(p.read_text())
def main(root,target):
    repo=Path(__file__).resolve().parents[1]
    lock=(root/'.replica-management.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    experiment=read(root/'experiment.json');pool=read(root/'pool.json')
    assert 1<=target<experiment['node_budget']==20
    for receipt in (root/'trials').glob('*/controller-watch.json'):
        watch=read(receipt)
        if watch.get('state')=='controller running':raise RuntimeError('Finish the active scientific trial before changing the ratio')
    sys.path.insert(0,str(root/'inference-runtime/scripts'));import jupiter_nemotron as native
    from jupiter_nemotron_api import atomic_json
    template=pool['backends'][0];source=Path(template['campaign'])/'state/nemotron'/template['attempt'];request=read(source/'request.json')
    assert request['campaign']==template['campaign'] and template['job_id'] in experiment['owned_job_ids']
    assert request['checkpoint_revision']==experiment['checkpoint_revision']
    assert len(request['command'])==1 and request['image_digest'] is None
    active=sum(b.get('enabled',True) for b in pool['backends'])
    first=max(int(b['id'].split('-')[-1]) for b in pool['backends'])+1
    for index in range(first,first+max(0,target-active)):
        experiment=read(root/'experiment.json')
        q=subprocess.run(['squeue','-h','-j',','.join(experiment['owned_job_ids']),'-o','%i %T %D'],capture_output=True,text=True,check=True)
        reserved=sum(int(line.split()[2]) for line in q.stdout.splitlines())
        assert reserved+1<=experiment['node_budget'],'Owned live/pending allocations would exceed twenty nodes'
        identifier=f'nemotron-{index:02d}';campaign=root/'replicas'/identifier
        if (campaign/'state/nemotron/current.json').exists():raise RuntimeError('Existing unregistered replica; reconcile its submission before retrying')
        args=SimpleNamespace(campaign=campaign,model_dir=Path(request['model_dir']),model_revision=request['checkpoint_revision'],login_address=request['login_address'],image=None,image_digest=None,vllm=Path(request['command'][0]),cc=Path(request['compiler']['cc']),cxx=Path(request['compiler']['cxx']),context=request['max_model_len'],sequences=request['max_num_seqs'],warmup_seconds=request['warmup_seconds'],account=request['account'],reservation=request['reservation'],walltime=request['walltime'],replace=False,standby=False)
        result=native.submit(args)
        entry={'id':identifier,'campaign':str(campaign),'attempt':result['attempt'],'job_id':str(result['job_id']),'enabled':True}
        for filename in ('experiment.json','scale-progress.json'):
            data=read(root/filename);data['owned_job_ids']=list(dict.fromkeys(data['owned_job_ids']+[entry['job_id']]))
            atomic_json(root/filename,data)
        pool['backends'].append(entry);atomic_json(root/'pool.json',pool)
        print(json.dumps({'replica':identifier,'submission':result,'created_at':time.time()}),flush=True)
    print(json.dumps({'active_replicas':sum(b.get('enabled',True) for b in pool['backends']),'registered_replica_attempts':len(pool['backends']),'target':target}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--target',required=True,type=int);a=p.parse_args();main(a.root.resolve(),a.target)
