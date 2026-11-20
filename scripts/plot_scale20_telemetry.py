#!/usr/bin/env python3
"""Plot measured, campaign-owned GPU telemetry; never fill missing data with zero."""
import argparse,csv,io,json,math,time
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def numeric(s):
    try:return float(s)
    except (ValueError,TypeError):return float('nan')
def finite_mean(v):
    v=[x for x in v if math.isfinite(x)];return sum(v)/len(v) if v else float('nan')
def main():
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=a.campaign.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    progress=json.loads((r/'scale-progress.json').read_text());audit=json.loads((r/'scaling-audit.json').read_text());pooled=bool(progress.get('inference_pool_experiment'));sci=progress['scientific_job']
    inf_ids=set(progress['active_inference_job_ids']) if pooled else {progress['inference_submission']['job_id'],*[x['job_id'] for x in progress.get('retired_inference_submissions',[])]}
    if not pooled and progress.get('standby_inference_submission'):inf_ids.add(progress['standby_inference_submission']['job_id'])
    inf=sorted(inf_ids)[0];allowed={sci,*inf_ids};observations=[];metrics=[]
    directories={r/'telemetry'}
    if pooled:directories.add(Path(progress['inference_pool_experiment'])/'telemetry')
    for f in sorted(f for directory in directories for f in directory.glob('*.jsonl')):
        if not any(f.name.startswith(j+'-') for j in allowed):continue
        for line in f.read_text().splitlines():
            try:d=json.loads(line)
            except ValueError:continue
            if d.get('job_id') not in allowed:continue
            records=list(csv.reader(io.StringIO(d.get('gpu_csv',''))));records=[x for x in records if len(x)>=8]
            if records:
                observations.append({'time':d['timestamp'],'host':d['host'].split('.')[0],'job':d['job_id'],'gpu_count':len(records),'busy':finite_mean([numeric(x[2]) for x in records]),'memory_percent':finite_mean([100*numeric(x[4])/numeric(x[5]) for x in records if numeric(x[5])>0]),'gpu_watts':sum(numeric(x[6]) for x in records),'gpu_ids':[x[1].strip() for x in records],'cpu_busy':numeric(d.get('cpu_busy_percent')),'cpu_iowait':numeric(d.get('cpu_iowait_percent')),'host_memory_percent':numeric(d.get('host_memory_used_percent'))})
            if d.get('vllm_metrics'):
                values=defaultdict(float)
                for text in d['vllm_metrics']:
                    name=text.split('{',1)[0].split()[0];value=numeric(text.rsplit(' ',1)[-1])
                    if math.isfinite(value):values[name]+=value
                metrics.append({'time':d['timestamp'],'job_id':d['job_id'],**dict(values)})
    assert observations,'No measured GPU telemetry available'
    sj=audit['jobs'].get(sci,{});ij=audit['jobs'].get(inf,{});start=sj.get('start_epoch') or ij.get('start_epoch') or min(d['time'] for d in observations);end=sj.get('end_epoch') or max(d['time'] for d in observations)
    scope='science allocation, including bootstrap' if sj.get('start_epoch') else 'owned inference service; science allocation still pending in this accounting snapshot'
    nodes=sorted(set(sj.get('hostnames',[]))|{x['host'] for x in observations if x['job']==sci});inodes=sorted(set().union(*(set(audit['jobs'].get(job,{}).get('hostnames',[])) for job in inf_ids))|{x['host'] for x in observations if x['job'] in inf_ids});hosts=inodes+sorted(set(nodes)-set(inodes));step=60;nb=max(1,int(math.ceil((end-start)/step)));bins=defaultdict(list);mbins=defaultdict(list)
    for d in observations:
        if start<=d['time']<=end:bins[(d['host'],min(nb-1,int((d['time']-start)//step)))].append(d)
    for d in metrics:
        if start<=d['time']<=end:mbins[min(nb-1,int((d['time']-start)//step))].append(d)
    arrays={key:np.full((len(hosts),nb),np.nan) for key in ('busy','memory_percent','gpu_watts','cpu_busy','cpu_iowait','host_memory_percent')}
    for (host,index),data in bins.items():
        for key in arrays:arrays[key][hosts.index(host),index]=finite_mean([d[key] for d in data])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.right':False,'axes.spines.top':False})
    files=[]
    def save(fig,name,tight=True):
        if tight:fig.tight_layout()
        fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig);files.append(name)
    labels=[h+(' [inference]' if h in inodes else '') for h in hosts];extent=[0,nb*step/60,-.5,len(hosts)-.5];start_label=datetime.fromtimestamp(start,ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds')
    for key,name,title in [('busy','node-gpu-utilization','GPU busy time (%)'),('memory_percent','node-gpu-memory','GPU memory capacity used (%)')]:
        fig,ax=plt.subplots(figsize=(11,max(2.8,len(hosts)*.28+1.5)));cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#dddddd');im=ax.imshow(arrays[key],aspect='auto',origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=100);ax.set_yticks(range(len(hosts)),labels);ax.set(xlabel='Minutes since '+start_label,title=title+' — mean of GPUs per node');fig.colorbar(im,ax=ax,label=title);save(fig,name)
    for key,name,title in [('cpu_busy','node-cpu-utilization','Host CPU busy time (%)'),('host_memory_percent','node-host-memory','Host memory used (%)')]:
        if np.isfinite(arrays[key]).any():
            fig,ax=plt.subplots(figsize=(11,max(2.8,len(hosts)*.28+1.5)));cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#dddddd');im=ax.imshow(arrays[key],aspect='auto',origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=100);ax.set_yticks(range(len(hosts)),labels);ax.set(xlabel='Minutes since '+start_label,title=title+' — node aggregate');fig.colorbar(im,ax=ax,label=title);save(fig,name)
    x=(np.arange(nb)+.5)*step/60
    def aggregate(arr,indices):
        return np.array([finite_mean([arr[i,j] for i in indices]) for j in range(nb)])
    fig,ax=plt.subplots(figsize=(11,3.3))
    for label,hh in [('Scientific tool nodes',nodes),('Nemotron nodes',inodes)]:
        indices=[hosts.index(h) for h in hh if h in hosts]
        if indices:ax.plot(x,aggregate(arrays['busy'],indices),label=label)
    ax.set(xlabel='Minutes since '+start_label,ylabel='Mean GPU busy time (%)',ylim=(0,105),title='Measured GPU activity; gray heatmap cells indicate missing observations');ax.legend();save(fig,'aggregate-gpu-activity')
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
    def pool_minute(key,j):
        values=[finite_mean([d.get(key,float('nan')) for d in mbins.get(j,[]) if d['job_id']==job]) for job in sorted(inf_ids)]
        if pooled and sum(math.isfinite(v) for v in values)!=len(inf_ids):return float('nan')
        return sum(v for v in values if math.isfinite(v)) if any(math.isfinite(v) for v in values) else float('nan')
    for key,label in [('vllm:num_requests_running','Running'),('vllm:num_requests_waiting','Waiting')]:axes[0].plot(x,[pool_minute(key,j) for j in range(nb)],label=label)
    axes[0].set(ylabel='Inference requests',title='Owned vLLM pool; per-replica minute means summed');axes[0].legend()
    kv=[finite_mean([d.get('vllm:kv_cache_usage_perc',float('nan')) for d in mbins.get(j,[])]) for j in range(nb)];axes[1].plot(x,kv,color='#187f70');axes[1].set(ylabel='KV-cache fraction (0–1)')
    for label,hh in [('Scientific nodes',nodes),('Nemotron nodes',inodes)]:
        indices=[hosts.index(h) for h in hh if h in hosts]
        if indices:axes[2].plot(x,aggregate(arrays['gpu_watts'],indices),label=label)
    axes[2].set(ylabel='GPU watts per node',xlabel='Minutes since '+start_label);axes[2].legend();save(fig,'inference-queue-and-gpu-power')
    lease_stats={}
    if nodes and end>start:
        grid=np.arange(start,end,1.0)+.5;leased=np.zeros((len(nodes),len(grid)));events=[];leased_seconds=0
        for row in audit['row_timings']:
            if not row.get('gpus') or row['host'] not in nodes or row.get('started_at') is None:continue
            lo=max(start,row['started_at']);hi=min(end,row['started_at']+row['seconds'])
            if hi<=lo:continue
            count=row['gpus'];leased[nodes.index(row['host']),(grid>=lo)&(grid<hi)]+=count;events.extend([(lo,count),(hi,-count)]);leased_seconds+=(hi-lo)*count
        active=peak=0
        for _,change in sorted(events,key=lambda item:(item[0],item[1])):active+=change;peak=max(peak,active)
        lease_stats={'leased_GPU_hours':leased_seconds/3600,'allocated_science_GPU_hours':len(nodes)*4*(end-start)/3600,'mean_leased_slot_fraction':leased_seconds/(len(nodes)*4*(end-start)),'peak_leased_slots':peak,'zero_leased_fraction_of_sampled_seconds':float((leased.sum(0)==0).mean()),'method':'Terminal GPU-row lease start plus recorded wall seconds; all sources and failed attempts included. Includes imports, CPU stages and I/O inside a leased row; not hardware utilization.'}
        fig,axes=plt.subplots(2,1,figsize=(11,8),gridspec_kw={'height_ratios':[3,1]},sharex=True,layout='constrained')
        minute_lease=np.array([[values[(grid>=start+j*step)&(grid<start+(j+1)*step)].mean() for j in range(nb)] for values in leased])
        im=axes[0].imshow(minute_lease,origin='lower',aspect='auto',extent=[0,nb,-.5,len(nodes)-.5],vmin=0,vmax=4,cmap='YlGnBu');axes[0].set_yticks(range(len(nodes)),nodes);axes[0].set(title='Scientific GPU-slot leases reconstructed from terminal row receipts');fig.colorbar(im,ax=axes,label='Mean leased GPU slots / node')
        axes[1].plot((grid-start)/60,leased.sum(0),color='#187f70',lw=.8);axes[1].axhline(4*len(nodes),color='gray',ls='--');axes[1].set(xlabel='Minutes since '+start_label,ylabel='Leased GPU slots',ylim=(0,4*len(nodes)+2));save(fig,'science-slot-leases',tight=False)
    summary={'generated_at':time.time(),'scope':scope,'start_epoch':start,'end_epoch':end,'minute_bins':nb,'sampling':'nvidia-smi utilization.gpu, memory.used/total and power.draw; four GPUs per node expected','interpretation':'GPU busy time is not SM occupancy or FLOPS efficiency. Memory residency is not computation. GPU power excludes CPU/system/facility power. Missing observations remain NaN/gray; gaps are never filled with zero.','nodes':{},'plots':files,'science_slot_leases':lease_stats}
    for host in hosts:
        rr=sorted([d for d in observations if d['host']==host and start<=d['time']<=end],key=lambda d:d['time']);busy=[d['busy'] for d in rr if math.isfinite(d['busy'])]
        summary['nodes'][host]={'kind':'inference' if host in inodes else 'science','samples':len(rr),'covered_minute_fraction':float(np.isfinite(arrays['busy'][hosts.index(host)]).mean()),'mean_sampled_GPU_busy_percent':statistics_mean(busy),'p95_sampled_GPU_busy_percent':float(np.percentile(busy,95)) if busy else None,'maximum_observation_gap_seconds':max((b['time']-a['time'] for a,b in zip(rr,rr[1:])),default=None),'gpu_counts_observed':sorted({d['gpu_count'] for d in rr}),'mean_sampled_CPU_busy_percent':statistics_mean([d['cpu_busy'] for d in rr if math.isfinite(d['cpu_busy'])]),'mean_sampled_host_memory_used_percent':statistics_mean([d['host_memory_percent'] for d in rr if math.isfinite(d['host_memory_percent'])])}
    summary['role_observed_node_minutes']={}
    for role,hh in [('science',nodes),('inference',inodes)]:
        indices=[hosts.index(h) for h in hh if h in hosts];values=arrays['busy'][indices].ravel();observed=values[np.isfinite(values)]
        summary['role_observed_node_minutes'][role]={'available_node_minutes':len(observed),'expected_node_minutes':len(values),'mean_GPU_busy_percent':float(observed.mean()) if len(observed) else None,'observed_node_minutes_at_least_80_percent_busy_fraction':float((observed>=80).mean()) if len(observed) else None}
    summary['inference_observed_requests']={}
    for key in ['vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc']:
        values=[d[key] for d in metrics if start<=d['time']<=end and key in d and math.isfinite(d[key])]
        summary['inference_observed_requests'][key]={'samples':len(values),'mean':statistics_mean(values),'p95':float(np.percentile(values,95)) if values else None,'max':max(values) if values else None,'positive_sample_fraction':sum(x>0 for x in values)/len(values) if values else None}
    summary['inference_metric_scope']='Individual replica samples; pool queue plots sum per-replica minute means. Missing pooled replica minutes remain missing.'
    summary['pool_minute_requests']={key:[None if not math.isfinite(v) else v for j in range(nb) for v in [pool_minute(key,j)]] for key in ['vllm:num_requests_running','vllm:num_requests_waiting']}
    all_busy=arrays['busy'].ravel();all_busy=all_busy[np.isfinite(all_busy)]
    summary['overall_observed_node_minute_GPU_busy_percent']=float(all_busy.mean()) if len(all_busy) else None
    (out/'telemetry-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'scope':scope,'nodes':len(hosts),'plots':files,'summary':str(out/'telemetry-summary.json')}))
def statistics_mean(v):return sum(v)/len(v) if v else None
if __name__=='__main__':main()
