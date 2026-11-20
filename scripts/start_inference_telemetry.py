#!/usr/bin/env python3
"""Attach hardware samplers once to proven owned, running inference jobs."""
import argparse, fcntl, json, subprocess, time
from pathlib import Path


def main(root):
    lock=(root/'.inference-telemetry-launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    repo=Path(__file__).resolve().parents[1];pool=json.loads((root/'pool.json').read_text());progress=json.loads((root/'scale-progress.json').read_text());receipt=root/'inference-telemetry-steps.json'
    records=json.loads(receipt.read_text()) if receipt.exists() else []
    for entry in pool['backends']:
        job=entry['job_id']
        if any(r['job_id']==job for r in records):continue
        assert job in progress['owned_job_ids']
        campaign=Path(entry['campaign']);assert campaign.resolve().is_relative_to(root/'replicas')
        attempt=campaign/'state/nemotron'/entry['attempt'];request=json.loads((attempt/'request.json').read_text());submission=json.loads((attempt/'submission.json').read_text())
        assert request['campaign']==str(campaign) and str(submission['job_id'])==job
        state=subprocess.run(['squeue','-h','-j',job,'-o','%T %D %j'],capture_output=True,text=True,check=True).stdout.strip()
        if state!=f"RUNNING 1 {entry['attempt']}":print(json.dumps({'job':job,'state':'not yet a running inference allocation'}));continue
        # The native generic batch script receives its attempt as a Slurm
        # argument. Verify Slurm's launch directory/name as well as receipts.
        identity=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,check=True).stdout
        assert ('WorkDir='+str(attempt)+' ') in identity and ('JobName='+entry['attempt']+' ') in identity
        args=['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1',str(repo/'.venv/bin/python'),str(repo/'scripts/sample_jupiter_telemetry.py'),str(root),'--process',str(attempt/'process.json')]
        with (root/'inference-telemetry.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record={'pid':process.pid,'job_id':job,'started_at':time.time(),'attempt':entry['attempt']};records.append(record)
        receipt.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
