#!/usr/bin/env python3
"""Renew only campaign-owned inference at a completed scientific checkpoint.

The replacement is dependency-bound to the old owned job, keeping the total
at or below twenty nodes. The science allocation and server are never stopped.
"""
import argparse,asyncio,json,os,signal,subprocess,sys,time
from pathlib import Path


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


async def main(root):
    repo=Path(__file__).resolve().parents[1];record=root/'inference-renewal.json'
    fd=os.open(record,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    state={'watcher_pid':os.getpid(),'state':'waiting for completed round2 checkpoint','started_at':time.time()}
    with os.fdopen(fd,'w') as stream:json.dump(state,stream)
    sys.path.insert(0,str(root/'inference-runtime/scripts'))
    import jupiter_nemotron as native
    from jupiter_nemotron_api import atomic_json
    def save():atomic_json(record,state)
    while not(root/'state/rounds/round-002.json').exists():
        watch=json.loads((root/'controller-watch.json').read_text())
        if not alive(watch['controller_pid']):raise RuntimeError('Controller ended before checkpoint; no inference job changed')
        await asyncio.sleep(1)
    checkpoint=json.loads((root/'state/rounds/round-002.json').read_text());assert checkpoint['round']==2
    progress=json.loads((root/'scale-progress.json').read_text());old=progress['inference_submission'];new=progress['standby_inference_submission']
    for entry in (old,new):
        assert entry['job_id'] in progress['owned_job_ids']
        directory=root/'state/nemotron'/entry['attempt']
        assert json.loads((directory/'request.json').read_text())['campaign']==str(root)
        assert str(json.loads((directory/'submission.json').read_text())['job_id'])==entry['job_id']
    pending=subprocess.check_output(['scontrol','show','job',new['job_id'],'-o'],text=True)
    assert ('Dependency=afterany:'+old['job_id']) in pending
    identity=subprocess.check_output(['squeue','-h','-j',old['job_id'],'-o','%i %j %u'],text=True).split()
    assert identity==[old['job_id'],old['attempt'],'meigooni1']
    assert subprocess.check_output(['squeue','-h','-j',progress['scientific_job'],'-o','%T %D'],text=True).strip()=='RUNNING 19'
    controller=json.loads((root/'controller-watch.json').read_text())['controller_pid']
    command=Path(f'/proc/{controller}/cmdline').read_bytes()
    assert str(root/'campaign.yaml').encode() in command and b'run_attached_campaign.py' in command
    os.kill(controller,signal.SIGSTOP);old_stopped=False
    try:
        state.update(state='renewing inference after completed round2',controller_pid=controller,old=old,replacement=new,checkpoint_time=time.time());save()
        subprocess.run(['scancel',old['job_id']],check=True,capture_output=True,text=True);old_stopped=True
        state.update(state='waiting for dependency-bound replacement allocation',old_retired_at=time.time());save()
        while True:
            q=subprocess.check_output(['squeue','-h','-j',new['job_id'],'-o','%T %D'],text=True).strip()
            if q=='RUNNING 1':break
            if not q:
                terminal=subprocess.check_output(['sacct','-n','-X','-j',new['job_id'],'--format=State','--parsable2'],text=True)
                if any(x in terminal for x in ('FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY')):raise RuntimeError('Owned replacement inference job became terminal')
            await asyncio.sleep(5)
        directory=root/'state/nemotron'/new['attempt']
        with(root/'renewal-inference-telemetry.log').open('a') as log:
            sampler=subprocess.Popen(['srun','--jobid='+new['job_id'],'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1',str(repo/'.venv/bin/python'),str(repo/'scripts/sample_jupiter_telemetry.py'),str(root),'--process',str(directory/'process.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(root/'renewal-inference-telemetry.json',{'pid':sampler.pid,'job_id':new['job_id'],'started_at':time.time()})
        state.update(state='replacement allocated; waiting for actual API completion',replacement_allocated_at=time.time());save()
        while True:
            try:
                activation=await native.activate_standby(root)
                break
            except (FileNotFoundError,RuntimeError,KeyError):
                q=subprocess.check_output(['squeue','-h','-j',new['job_id'],'-o','%T'],text=True).strip()
                if q not in ('RUNNING','CONFIGURING'):raise RuntimeError('Replacement ended before readiness')
                await asyncio.sleep(10)
        progress=json.loads((root/'scale-progress.json').read_text())
        progress.setdefault('retired_inference_submissions',[]).append({**old,'retired_at':state['old_retired_at'],'reason':'Owned inference walltime renewal at completed round2 checkpoint'})
        progress['inference_submission']=new;progress['standby_inference_submission']={**new,'activated':True}
        progress['verified_active_split']['inference_job']=new['job_id'];atomic_json(root/'scale-progress.json',progress)
        state.update(state='replacement API verified; controller resumed',activation=activation,resumed_at=time.time());save()
        os.kill(controller,signal.SIGCONT)
    except BaseException as exc:
        state.update(state='inference renewal requires recovery',error_type=type(exc).__name__,error=str(exc),failed_at=time.time());save()
        if not old_stopped:os.kill(controller,signal.SIGCONT)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('campaign',type=Path);args=parser.parse_args()
    asyncio.run(main(args.campaign.resolve()))
