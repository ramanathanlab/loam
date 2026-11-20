#!/usr/bin/env python3
"""Attach one trial after its owned allocation is ready; never cancel other jobs."""
import argparse, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path
import httpx


def read(path):return json.loads(path.read_text())
def write(path,data):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
def start(repo,script,args,log,cwd):
    with log.open('ab',buffering=0) as stream:
        return subprocess.Popen([str(repo/'.venv/bin/python'),'-u',str(repo/'scripts'/script),*map(str,args)],cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def main(trial):
    lock=(trial/'.trial-supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    repo=Path(__file__).resolve().parents[1];progress=read(trial/'scale-progress.json');root=Path(progress['inference_pool_experiment'])
    receipt=trial/'controller-watch.json'
    if receipt.exists():raise RuntimeError('Supervisor receipt exists; inspect it before any restart')
    watch={'watcher_pid':os.getpid(),'state':'waiting for owned science allocation','started_at':time.time()};write(receipt,watch)
    science=None
    while science is None:
        assert alive(read(trial/'server-launch.json')['pid']),'Science holder exited before allocation'
        q=subprocess.run(['squeue','-h','-u',os.environ.get('USER','meigooni1'),'-o','%i %T %D'],capture_output=True,text=True,check=True)
        for line in q.stdout.splitlines():
            job,state,nodes=line.split()
            if job in read(root/'experiment.json')['owned_job_ids']:continue
            script=subprocess.run(['scontrol','write','batch_script',job,'-'],capture_output=True,text=True)
            if script.returncode or str(trial) not in script.stdout or 'HPC_CAMPAIGN_ID='+progress['campaign_id'] not in script.stdout:continue
            assert int(nodes)==progress['scientific_allocation_nodes']
            science=job
            write(trial/'science-job-ownership.json',{'job_id':job,'trial':str(trial),'campaign_id':progress['campaign_id'],'nodes':int(nodes),'batch_script_sha256':hashlib.sha256(script.stdout.encode()).hexdigest(),'verified_at':time.time(),'basis':'unique absolute campaign directory and campaign ID in actual Slurm batch script'})
            for target in (root/'experiment.json',root/'scale-progress.json',trial/'scale-progress.json'):
                data=read(target);data['owned_job_ids']=list(dict.fromkeys(data['owned_job_ids']+[job]))
                if target.parent==trial:data.update(scientific_job=job,phase='waiting for science allocation/service readiness')
                write(target,data)
            break
        if science is None:time.sleep(15)
    telemetry=start(repo,'watch_science_telemetry.py',[trial],trial/'science-telemetry.log',repo)
    watch.update(scientific_job=science,telemetry_watcher_pid=telemetry.pid);write(receipt,watch)
    while not (trial/'server-ready.json').exists():
        assert alive(read(trial/'server-launch.json')['pid']),'Science server holder exited before readiness'
        time.sleep(15)
    service=read(root/'pool-service.json');expected=set(progress['active_inference_job_ids'])
    with httpx.Client(trust_env=False,timeout=15,headers={'Authorization':'Bearer '+Path(service['token_file']).read_text().strip()}) as client:
        while True:
            status=client.get(service['status_url']);status.raise_for_status();backends=status.json()['backends']
            ready={b['job_id'] for b in backends if b['enabled'] and b['available']}
            if ready==expected:break
            watch.update(state='waiting for every configured inference replica',ready_inference_job_ids=sorted(ready));write(receipt,watch)
            time.sleep(15)
    cfg=read(root/'pool.json');assert cfg['phase']=='campaign-ready'
    cfg['phase']=trial.name;write(root/'pool.json',cfg)
    controller=start(repo,'run_attached_campaign.py',[trial/'campaign.yaml'],trial/'controller.log',repo)
    watch.update(state='controller running',controller_pid=controller.pid,controller_started_at=time.time());write(receipt,watch)
    indexer=None
    while controller.poll() is None:
        if indexer is None and (trial/'campaign-tool-registrations.json').exists():
            indexer=start(repo,'update_hypothesis_artifact_index.py',[trial,'--watch'],trial/'artifact-index.log',repo)
            watch['artifact_index_watcher_pid']=indexer.pid;write(receipt,watch)
        time.sleep(15)
    watch.update(state='controller exited',returncode=controller.returncode,controller_finished_at=time.time());write(receipt,watch)
    if controller.returncode==0:
        deadline=time.time()+7200
        while True:
            pending=[]
            for metadata in trial.glob('batches/*/batch.json'):
                batch=read(metadata)
                terminal=sum((row/'done').exists() and (row/'status.json').exists() for row in metadata.parent.glob('run_*'))
                if terminal<batch['total']:pending.append({'batch_id':batch['batch_id'],'unfinished':batch['total']-terminal})
            watch['scientific_drain']={'checked_at':time.time(),'pending':pending,'deadline':deadline};write(receipt,watch)
            if not pending:break
            if time.time()>deadline:raise RuntimeError('Submitted science rows did not drain; preserve allocation for inspection')
            if not alive(read(trial/'server-launch.json')['pid']):raise RuntimeError('Science server exited while submitted rows remained pending')
            time.sleep(15)
        # Managed server cleanup addresses only the allocation owned by this trial.
        (trial/'stop-server').write_text('Campaign controller completed; release this trial managed science allocation.\n')
        shutdown_deadline=time.time()+600
        while alive(read(trial/'server-launch.json')['pid']):
            if time.time()>shutdown_deadline:raise RuntimeError('Managed science server did not finish cleanup; inspect before releasing allocation')
            time.sleep(5)
        from release_completed_science import release
        release(trial)
    print(json.dumps(watch),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trial',type=Path);a=p.parse_args();main(a.trial.resolve())
