#!/usr/bin/env python3
"""Wait for the recorded science allocation and start one sampler per node.

Queue observation never resubmits or cancels jobs. Sampling starts at Slurm
RUNNING, including science-service bootstrap before the MCP server is ready.
"""
import argparse,json,os,subprocess,time
from pathlib import Path

def main(campaign):
    progress=json.loads((campaign/'scale-progress.json').read_text());job=progress['scientific_job'];assert job in progress['owned_job_ids'];nodes=int(progress['scientific_allocation_nodes']);record=campaign/'science-telemetry-step.json'
    if record.exists():raise RuntimeError('Sampler receipt exists; inspect its process before any restart')
    record.write_text(json.dumps({'watcher_pid':os.getpid(),'job_id':job,'state':'waiting for allocation'})+'\n')
    while True:
        q=subprocess.run(['squeue','-h','-j',job,'-o','%T %D'],capture_output=True,text=True,check=True);state=q.stdout.strip()
        if state==f'RUNNING {nodes}':break
        with (campaign/'queue-observations.jsonl').open('a') as f:f.write(json.dumps({'timestamp':time.time(),'job_id':job,'scheduler':state})+'\n')
        if not state:
            # A transient empty observation is not terminal. Consult accounting.
            a=subprocess.run(['sacct','-n','-X','-j',job,'--format=State','--parsable2'],capture_output=True,text=True,check=True)
            if any(s in a.stdout for s in ('CANCELLED','FAILED','COMPLETED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY')):
                record.write_text(json.dumps({'watcher_pid':os.getpid(),'job_id':job,'state':'allocation terminal before sampling','accounting':a.stdout.strip()})+'\n');return
        time.sleep(20)
    script=subprocess.run(['scontrol','write','batch_script',job,'-'],capture_output=True,text=True,check=True).stdout
    campaign_id=progress.get('campaign_id','jupiter-scale20-20261113')
    assert str(campaign) in script and ('HPC_CAMPAIGN_ID='+campaign_id) in script
    repo=Path(__file__).resolve().parents[1]
    args=['srun','--jobid='+job,'--overlap','--nodes='+str(nodes),'--ntasks='+str(nodes),'--ntasks-per-node=1','--cpus-per-task=1',str(repo/'.venv/bin/python'),str(repo/'scripts/sample_jupiter_telemetry.py'),str(campaign)]
    process=subprocess.Popen(args);record.write_text(json.dumps({'watcher_pid':os.getpid(),'sampler_pid':process.pid,'job_id':job,'nodes':nodes,'state':'sampling','started_at':time.time()})+'\n');code=process.wait()
    record.write_text(json.dumps({'watcher_pid':os.getpid(),'sampler_pid':process.pid,'job_id':job,'nodes':nodes,'state':'sampler ended','returncode':code,'ended_at':time.time()})+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args();main(a.campaign.resolve())
