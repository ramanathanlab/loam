"""Credential-free operational timing exports for completed saturation trials."""
import collections
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_inference_atlas import write_csv


def timing_plots(run,prefix,save,output,sources,sha):
    base=run['root'];start=run['start']
    admission=base/'state/admission-timings.jsonl'
    if admission.exists():
        rows=[json.loads(line) for line in admission.read_text().splitlines()]
        sources[str(admission)]=sha(admission)
        write_csv(output/(prefix+'-admission-timings.csv'),rows)
        waiting=active=0;trace=[];starts={};waits=[];lifetimes=[]
        for row in sorted(rows,key=lambda r:r['timestamp']):
            key=(row['round'],row['hypothesis_id'])
            if row['event']=='queued':waiting+=1
            elif row['event']=='started':
                waiting-=1;active+=1;starts[key]=row['timestamp'];waits.append(row['wait_seconds'])
            elif row['event']=='finished':
                active-=1
                if key in starts:lifetimes.append((row['timestamp']-starts[key])/60)
            trace.append(((row['timestamp']-start)/60,waiting,active))
        fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
        axes[0].step([t[0] for t in trace],[t[1] for t in trace],where='post',label='Selected, awaiting admission',color='#c6812f')
        axes[0].step([t[0] for t in trace],[t[2] for t in trace],where='post',label='Active validation/review thread',color='#147f78')
        axes[0].set(xlabel='Minutes since allocation start',ylabel='Hypothesis threads');axes[0].legend(fontsize=9)
        for values,label,color in [(np.asarray(waits)/60,'Admission wait','#c6812f'),(lifetimes,'Active thread duration','#147f78')]:
            if len(values):
                ordered=np.sort(values);axes[1].plot(ordered,np.arange(1,len(ordered)+1)/len(ordered),label=label,color=color)
        axes[1].set(xlabel='Minutes',ylabel='Cumulative fraction');axes[1].legend(fontsize=9)
        for ax in axes:ax.grid(alpha=.2)
        save(fig,prefix+'-admission','Hypothesis backlog and round tails: '+run['label'],
             'Direct queued/start/finish events from the bounded hypothesis admission gate. A selected thread is not necessarily ready GPU science; validator/reviewer duration includes nested tools and inference.')

    path=Path(run['plan']['infra'])/'pool-request-timings.jsonl'
    if path.exists():
        allowed=['request_id','backend','status','started_at','finished_at','total_request_seconds',
                 'queue_seconds','time_to_first_token_seconds','thinking_enabled','prompt_tokens','completion_tokens']
        attempts=[{key:value.get(key) for key in allowed} for line in path.read_text().splitlines() if (value:=json.loads(line))]
        sources[str(path)]=sha(path)
        write_csv(output/(prefix+'-inference-attempts.csv'),attempts)
        successful=[r for r in attempts if r['status']==200 and isinstance(r['total_request_seconds'],(int,float))]
        if successful:
            fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
            for thinking,label,color in [(False,'Thinking off','#147f78'),(True,'Thinking on','#c6812f'),(None,'Unspecified','#8763a2')]:
                rows=[r for r in successful if r['thinking_enabled'] is thinking]
                if not rows:continue
                values=np.sort([r['total_request_seconds'] for r in rows])
                axes[0].plot(values,np.arange(1,len(values)+1)/len(values),label=f'{label} (n={len(values)})',color=color)
            axes[0].set(xlabel='Successful request wall time (seconds)',ylabel='Cumulative fraction',xscale='symlog');axes[0].legend(fontsize=8)
            groups=collections.defaultdict(list)
            for row in successful:groups[int((row['finished_at']-start)//60)].append(row)
            bins=sorted(groups)
            axes[1].plot(bins,[len(groups[n]) for n in bins],color='#147f78',label='Successful attempts/minute')
            axes[1].set(xlabel='Minutes since allocation start',ylabel='Successful HTTP attempts/minute')
            other=axes[1].twinx();other.plot(bins,[np.percentile([r['total_request_seconds'] for r in groups[n]],95) for n in bins],color='#c6812f',alpha=.7)
            other.set_ylabel('P95 request wall time (seconds)',color='#c6812f')
            for ax in axes:ax.grid(alpha=.2)
            errors=sum(r['status']!=200 for r in attempts)
            save(fig,prefix+'-inference-latency','Inference latency and request flow: '+run['label'],
                 f'All {len(attempts)} HTTP attempts retained in CSV, including {errors} non-200 attempts. Latency curves use successful attempts; retries are not independent scientific outputs. Thinking profiles are unchanged.')

    batches={};dispatches={};operational=[]
    for row in run['audit']['row_timings']:
        if not row.get('started_at') or float(row.get('gpus') or 0)<=0:continue
        batch=row['batch_id'];folder=base/'batches'/batch
        if batch not in batches:
            path=folder/'batch.json';batches[batch]=json.loads(path.read_text());sources[str(path)]=sha(path)
            dispatches[batch]=[]
            for path in folder.glob('dispatch-*.json'):
                data=json.loads(path.read_text());dispatches[batch].append(data);sources[str(path)]=sha(path)
        lease=float(row['started_at']);run_id=row['row'].removeprefix('run_')
        candidates=[d for d in dispatches[batch] if run_id in d['run_ids'] and d['requested_at']<=lease]
        lower=None
        if candidates:
            d=max(candidates,key=lambda d:d['requested_at'])
            upper=max(0,lease-d['requested_at']);lower=max(0,lease-d['buffered_at'])
            scope='Final lease since latest matching dispatch request; buffer-to-lease lower bound also recorded'
        else:
            stamps=[datetime.fromisoformat(s['at']).timestamp() for s in batches[batch].get('submissions',[]) if s.get('submitted')]
            eligible=[t for t in stamps if t<=lease+1]
            upper=max(0,lease-max(eligible)) if eligible else None
            scope='Historical submission-to-final-lease proxy; direct dispatch timestamps unavailable'
        operational.append({'batch_id':batch,'row':row['row'],'tool':row.get('tool') or 'auxiliary',
            'round':row.get('round'),'dispatch_request_to_final_lease_seconds':upper,
            'buffer_to_final_lease_lower_bound_seconds':lower,'process_seconds':row.get('seconds'),
            'returncode':row.get('returncode'),'scope':scope})
    write_csv(output/(prefix+'-science-dispatch.csv'),operational)
    if operational:
        fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
        for tool,label,color in [('scale20_complete_binder_pipeline','Screen','#147f78'),('scale20_intact_egfr_followup','Full context','#c6812f')]:
            selected=[r for r in operational if r['tool']==tool]
            for ax,key in zip(axes,['dispatch_request_to_final_lease_seconds','process_seconds']):
                values=sorted(r[key] for r in selected if r.get(key) is not None)
                if values:ax.plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color)
        axes[0].set(xlabel='Dispatch request/submission to final lease (s)',ylabel='Cumulative fraction',xscale='symlog')
        axes[1].set(xlabel='Recorded process duration (s)',ylabel='Cumulative fraction',xscale='symlog')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        save(fig,prefix+'-science-dispatch','Scientific dispatch and process latency: '+run['label'],
             'All recorded GPU-row attempts, including failed or scientifically excluded rows. New runs bracket dispatch timing with request and buffer acceptance; historical runs use a submission-to-final-lease proxy. Neither is a direct EL master ready-queue measurement.')
