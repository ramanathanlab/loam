#!/usr/bin/env python3
"""Release one proven owned idle allocation after its managed server exits."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path


def read(p):return json.loads(p.read_text())
def release(trial):
    progress=read(trial/'scale-progress.json');job=progress['scientific_job'];watch=read(trial/'controller-watch.json');ownership=read(trial/'science-job-ownership.json')
    assert watch['state']=='controller exited' and watch['returncode']==0
    assert job.isdigit() and job in progress['owned_job_ids'] and ownership['job_id']==job and ownership['trial']==str(trial)
    pid=read(trial/'server-launch.json')['pid']
    if Path(f'/proc/{pid}/stat').exists():assert Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z','Managed science holder is still running'
    for p in trial.glob('batches/*/batch.json'):
        batch=read(p);assert sum((q/'done').exists() and (q/'status.json').exists() for q in p.parent.glob('run_*'))==batch['total'],'Submitted rows are not terminal'
    state=subprocess.run(['squeue','-h','-j',job,'-o','%T %D'],capture_output=True,text=True,check=True).stdout.strip()
    record={'job_id':job,'timestamp':time.time(),'observed_scheduler_state':state,'basis':'Controller exited0; managed science holder exited; all rows terminal; Slurm script matches original ownership receipt, exact trial root and campaign ID','cancel_requested':False}
    if state and not state.startswith('COMPLETING'):
        script=subprocess.run(['scontrol','write','batch_script',job,'-'],capture_output=True,text=True,check=True).stdout
        digest=hashlib.sha256(script.encode()).hexdigest()
        assert str(trial) in script and 'HPC_CAMPAIGN_ID='+progress['campaign_id'] in script and digest==ownership['batch_script_sha256']
        assert int(state.split()[1])==progress['scientific_allocation_nodes']
        record.update(batch_script_sha256=digest,cancel_requested=True)
        (trial/'science-release.json').write_text(json.dumps(record,indent=2)+'\n')
        subprocess.run(['scancel',job],capture_output=True,text=True,check=True)
    else:(trial/'science-release.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trial',type=Path);a=p.parse_args();print(json.dumps(release(a.trial.resolve())))
