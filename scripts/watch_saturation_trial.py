#!/usr/bin/env python3
"""Read-only runtime observer; never starts/cancels jobs or changes campaign state.

Derived snapshots/history live beside the owned trial, outside scientific state.
Use --once for a bounded snapshot. One observer lock prevents duplicate loops.
"""
import argparse
import collections
import fcntl
import json
import os
import time
from pathlib import Path

from snapshot_saturation import snapshot,tail_rows

TERMINAL=('COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','BOOT_FAIL','OUT_OF_MEMORY','DEADLINE','REVOKED')


def atomic(path,value):
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def inventory(campaign):
    counts=collections.Counter()
    for path in campaign.glob('batches/*/batch.json'):
        counts['submitted_batches']+=1
        counts['raw_screen_summaries']+=sum(1 for _ in path.parent.glob('run_*/summary.json'))
        counts['raw_full_context_summaries']+=sum(1 for _ in path.parent.glob('run_*/followup-summary.json'))
    return dict(counts)


def watch(job,interval=60,inventory_interval=300,once=False):
    lock=(job/'runtime-observer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(job/'runtime-observer-process.json',{'pid':os.getpid(),'started_at':time.time(),
           'job_id':json.loads((job/'submission.json').read_text())['job_id'],'read_only_campaign_state':True})
    previous=None;last_inventory=0;counts={}
    while True:
        begun=time.monotonic()
        try:
            value=snapshot(job)
            if time.monotonic()-last_inventory>=inventory_interval or once:
                counts=inventory(job/'campaign');last_inventory=time.monotonic()
            value['raw_science_inventory']=counts
            requests=tail_rows(job/'infra/pool-request-timings.jsonl',1024*1024)
            since=value['at']-120
            recent=[q for q in requests if q.get('finished_at',0)>=since]
            value['recent_inference_HTTP_status_counts']=dict(collections.Counter(str(q.get('status')) for q in recent))
            value['last_inference_finished_at']=max((q.get('finished_at',0) for q in requests),default=None)
            phase=value.get('controller',{}).get('phase')
            state=value.get('slurm',{}).get('JobState')
            if state is None:
                records=[line.split('|') for line in value.get('accounting',[]) if line.startswith(value['job_id']+'|')]
                state=records[-1][1] if records else 'ACCOUNTING_UNAVAILABLE'
            summary={k:value.get(k) for k in ['at','job_id','controller','admission','completed_rounds',
                'science_observed_recent_GPU_busy_percent','inference_observed_recent_GPU_busy_percent',
                'raw_science_inventory','recent_inference_HTTP_status_counts','last_inference_finished_at']}
            summary['slurm_state']=state
            summary['inventory_scope']='Preliminary raw summaries, not independently audited/deduplicated scientific output'
            atomic(job/'runtime-observer-latest.json',value)
            with (job/'runtime-observations.jsonl').open('a') as stream:stream.write(json.dumps(summary)+'\n')
            fingerprint=json.dumps({'state':state,'phase':phase,'admission':value.get('admission'),
                'rounds':value.get('completed_rounds'),'inventory':counts,'HTTP':summary['recent_inference_HTTP_status_counts']},sort_keys=True)
            if fingerprint!=previous or once:
                print(json.dumps(summary),flush=True);previous=fingerprint
            if state.startswith(TERMINAL) or once:return
        except Exception as exc:
            # An observation error does not terminate/restart the real job.
            event={'at':time.time(),'observer_error':type(exc).__name__,'job_root':str(job)}
            print(json.dumps(event),flush=True)
            with (job/'runtime-observations.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            if once:raise
        until=begun+interval
        while time.monotonic()<until:
            time.sleep(min(60,max(.1,until-time.monotonic())))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=lambda p:Path(p).resolve())
    parser.add_argument('--interval',type=float,default=60);parser.add_argument('--inventory-interval',type=float,default=300)
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    if args.interval<=0 or args.inventory_interval<=0:parser.error('Intervals must be positive')
    watch(args.job,args.interval,args.inventory_interval,args.once)
