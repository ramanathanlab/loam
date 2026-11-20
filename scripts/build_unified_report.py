#!/usr/bin/env python3
"""Offline report and editable deck for single-allocation scale-up evidence.

Use the scientific Python with PYTHONPATH including the repository and the
presentation dependencies. This reads only whitelisted scientific/telemetry
artifacts; private service directories are never copied to the report.
"""
from __future__ import annotations
import argparse, base64, collections, csv, hashlib, html, io, json, math, statistics, subprocess, time, zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from build_inference_atlas import deduplicate, json_safe, write_csv, write_json, finite_mean
from audit_scale20_campaign import accounting
from scaleup_slides import Slide, export as export_slides

REPO=Path(__file__).resolve().parents[1]
COLORS=['#4a79a0','#208d85','#a768a2','#d3963c']
ARMS={'dual':'#168b80','one_region':'#d98d39','ridge':'#8c6bab'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v,d=1):return 'unavailable' if v is None or not math.isfinite(v) else f'{v:,.{d}f}'
def esc(s):return html.escape(str(s))
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


from report_production_events import production_events


def load_run(job):
    plan=read(job/'plan.json');root=Path(plan['campaign']);audit=read(root/'scaling-audit.json');manifest=read(job/'allocation.json');receipt=read(job/'submission.json');state=read(job/'status.json')
    assert receipt['job_id']==manifest['job_id'] and audit['allocation_mode']=='single-allocation'
    assert audit['checks']['independent_model_checks_requested_and_passed']
    if plan['run_mode']=='campaign':
        assert state['phase']=='completed' and audit['required_campaign_completion_checks_passed'],audit['required_campaign_completion_checks']
    parent=audit['jobs'][receipt['job_id']]
    if plan['run_mode']=='campaign':assert parent['State']=='COMPLETED',parent
    screens,full=deduplicate(audit);start,end=parent['start_epoch'],parent['end_epoch'];seconds=end-start
    events,event_time_resolutions=production_events(audit,screens,full)
    rounds={int(p.stem.split('-')[1]):read(p) for p in (root/'state/rounds').glob('round-*.json')}
    r={'job':job,'root':root,'plan':plan,'audit':audit,'manifest':manifest,'receipt':receipt,'state':state,'parent':parent,'screens':screens,'full':full,'start':start,'end':end,'seconds':seconds,'events':sorted(events,key=lambda e:e['time']),'rounds':rounds,'label':f"{plan['nodes']} nodes ({plan['inference_count']}:{plan['science_count']})",'formal':plan['run_mode']=='campaign'}
    r['event_time_resolutions']=event_time_resolutions
    r['summary']={'label':r['label'],'mode':plan['run_mode'],'job_id':receipt['job_id'],'nodes':plan['nodes'],'inference':plan['inference_count'],'science':plan['science_count'],'allocation_seconds':seconds,'queue_seconds':parent['queue_seconds'],'parent_state':parent['State'],'controller_phase':state['phase'],'GPU_hours':4*plan['nodes']*seconds/3600,'screen_trials':len(screens),'expanded_trials':len(full),'verified_complex_files':5*(len(screens)+len(full)),'files_per_hour':5*(len(screens)+len(full))*3600/seconds,'screen_trials_per_hour':len(screens)*3600/seconds,'expanded_trials_per_hour':len(full)*3600/seconds,'hypotheses_tested':audit['unique_tested_hypotheses'],'rounds':len(rounds),'thread_errors':len(audit['thread_errors']),'failed_rows':sum(b['failed'] for b in audit['batch_states']),'completion_checks_passed':audit['required_campaign_completion_checks_passed']}
    r['summary']['files_per_GPU_hour']=r['summary']['verified_complex_files']/r['summary']['GPU_hours']
    r['summary']['reasoner_failures']=sum(str(d.get('brief',{}).get('report','')).startswith('reasoner failed:') for d in rounds.values())
    r['summary']['rounds_with_nonempty_steering_brief']=sum(bool(d.get('brief',{}).get('tournament_brief','')) for d in rounds.values())
    r['summary']['orchestration_error_threads']=sum(bool(t.get('error')) for d in rounds.values() for t in d['threads'])
    r['summary']['complete_reviewed_threads']=sum(bool(t.get('report') is not None and t.get('review') is not None and not t.get('error')) for d in rounds.values() for t in d['threads'])
    ready=read(root/'server-ready.json');watch=read(root/'controller-watch.json')
    r['summary']['services_startup_seconds']=max(0,ready['ready_at']-start)
    r['summary']['controller_wall_seconds']=watch['controller_finished_at']-watch['controller_started_at']
    r['summary']['handoff_drain_cleanup_seconds']=max(0,seconds-r['summary']['services_startup_seconds']-r['summary']['controller_wall_seconds'])
    rejected={c['batch_id'] for c in audit.get('nonstandard_source_classifications',[]) if not c['countable_source']}
    r['summary']['excluded_source_success_rows']=sum(b['terminal']-b['failed'] for b in audit['batch_states'] if b['batch_id'] in rejected)
    r['summary']['excluded_input_rows']=len(audit.get('nonstandard_input_classifications',[]))
    return r


def telemetry(run):
    hosts=run['manifest']['inference_nodes']+run['manifest']['science_nodes'];roles=['inference']*len(run['manifest']['inference_nodes'])+['science']*len(run['manifest']['science_nodes']);index={h:i for i,h in enumerate(hosts)};n=math.ceil(run['seconds']/60)
    values=collections.defaultdict(list);metrics=collections.defaultdict(list);paths=[]
    for p in sorted((run['root']/'telemetry').glob('*.jsonl')):
        paths.append(p)
        for line in p.read_text().splitlines():
            try:d=json.loads(line)
            except ValueError:continue
            host=d['host'].split('.')[0];t=d['timestamp']
            if not run['start']<=t<=run['end']:continue
            assert d['job_id']==run['receipt']['job_id'] and host in index and d['role']==roles[index[host]]
            minute=min(n-1,int((t-run['start'])//60));gpu=list(csv.reader(io.StringIO(d.get('gpu_csv',''))))
            if len(gpu)==4 and all(len(row)>=8 for row in gpu):
                assert {int(row[0]) for row in gpu}=={0,1,2,3}
                sample={'gpu':[float(row[2]) for row in sorted(gpu,key=lambda row:int(row[0]))],'cpu':d.get('cpu_busy_percent'),'ram':d.get('host_memory_used_percent'),'gpu_memory':finite_mean([100*float(row[4])/float(row[5]) for row in gpu]),'power':sum(float(row[6]) for row in gpu)}
                values[(host,minute)].append(sample)
            if d.get('vllm_metrics'):
                m=collections.defaultdict(float)
                for line in d['vllm_metrics']:
                    try:m[line.split('{')[0].split()[0]]+=float(line.rsplit(' ',1)[-1])
                    except (ValueError,IndexError):continue
                metrics[(host,minute)].append(dict(m))
    arrays={k:np.full((len(hosts),n),np.nan) for k in ['busy','cpu','ram','gpu_memory','power']};per_gpu=np.full((len(hosts),4,n),np.nan)
    for (host,minute),samples in values.items():
        i=index[host];per_gpu[i,:,minute]=[finite_mean([s['gpu'][g] for s in samples]) for g in range(4)];arrays['busy'][i,minute]=finite_mean(per_gpu[i,:,minute])
        for key in ['cpu','ram','gpu_memory','power']:arrays[key][i,minute]=finite_mean([s[key] for s in samples])
    coverage=np.isfinite(arrays['busy']);assert coverage.any()
    run.update(hosts=hosts,roles=roles,arrays=arrays,per_gpu=per_gpu,metrics=metrics,telemetry_paths=paths,minutes=np.arange(n)+.5)
    for role in ['inference','science']:
        ii=[i for i,r in enumerate(roles) if r==role];run['summary'][role+'_GPU_busy_percent']=float(np.nanmean(arrays['busy'][ii]))
    run['summary'].update(overall_GPU_busy_percent=float(np.nanmean(arrays['busy'])),telemetry_coverage_percent=100*float(coverage.mean()),science_hosts_with_rows=len(set(run['manifest']['science_nodes'])&set(run['audit']['scientific_execution_hosts'])))
    req=Path(run['plan']['infra'])/'pool-request-timings.jsonl';requests=[json.loads(s) for s in req.read_text().splitlines()] if req.exists() else []
    final={q['request_id']:q for q in sorted(requests,key=lambda q:q['finished_at'])};run['requests']=list(final.values());run['request_attempts']=requests
    run['summary']['inference_request_attempts']=len(requests);run['summary']['non200_inference_attempts']=sum(q.get('status')!=200 for q in requests)


class Figures:
    def __init__(self,out):
        self.out=out;self.figures=[];(out/'figures').mkdir(exist_ok=True)
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False,'svg.fonttype':'none','savefig.facecolor':'white'})
    def save(self,fig,name,title,caption,group='overview'):
        for ext in ['svg','png']:fig.savefig(self.out/'figures'/f'{name}.{ext}',bbox_inches='tight',dpi=150)
        plt.close(fig);self.figures.append({'id':name,'title':title,'caption':caption,'group':group});return name


def plots(figures,runs,baseline):
    formal=[r for r in runs if r['formal']];labels=[r['label'] for r in formal]
    # Each parent allocation contributes exactly once to the denominator.
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,key,title in zip(axes,['files_per_hour','expanded_trials_per_hour','files_per_GPU_hour'],['Coordinate files / allocation hour','Expanded trials / allocation hour','Coordinate files / allocated GPU-hour']):
        ax.bar(range(len(formal)),[r['summary'][key] for r in formal],color=COLORS[:len(formal)]);ax.set_xticks(range(len(formal)),labels,rotation=15);ax.set_title(title);ax.grid(axis='y',alpha=.25)
    figures.save(fig,'throughput','Scientific throughput and efficiency','Five coordinate models per retained crop or expanded-context trial. Earliest successful identical inputs are counted once. Full parent-job elapsed time includes cold startup and drain.')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');xx=np.arange(len(formal))
    axes[0].bar(xx,[r['parent']['queue_seconds']/3600 for r in formal],color=COLORS[:len(formal)]);axes[0].set_xticks(xx,labels,rotation=15);axes[0].set_ylabel('Queue hours');axes[0].grid(axis='y',alpha=.25)
    bottom=np.zeros(len(formal))
    for key,title,color in [('services_startup_seconds','Services startup','#d3963c'),('controller_wall_seconds','Tournament/science/review','#168b80'),('handoff_drain_cleanup_seconds','Handoff, drain and cleanup','#8799a2')]:
        values=np.array([r['summary'][key]/60 for r in formal]);axes[1].bar(xx,values,bottom=bottom,label=title,color=color);bottom+=values
    axes[1].set_xticks(xx,labels,rotation=15);axes[1].set_ylabel('Allocated minutes');axes[1].legend();axes[1].grid(axis='y',alpha=.25)
    figures.save(fig,'queue-and-startup','Queue waiting versus allocated execution','Queue time consumes no allocated GPU-hours. Startup includes the private services, all Nemotron replicas and science worker registration; scientific models also load within individual rows. The controller interval includes generation, ranking, scientific tools, review and steering.')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for i,r in enumerate(formal):
        events=r['events'];times=[(e['time']-r['start'])/60 for e in events];axes[0].step([0,*times],[0,*np.cumsum([e['files'] for e in events])],where='post',label=r['label'],color=COLORS[i])
        n=math.ceil(r['seconds']/300);counts=np.histogram([e['time']-r['start'] for e in events],bins=np.arange(n+1)*300,weights=[e['files'] for e in events])[0];axes[1].plot(np.arange(n)*5+2.5,counts*12,label=r['label'],color=COLORS[i])
    axes[0].set_ylabel('Retained coordinate files');axes[1].set_ylabel('Files/hour, five-minute bins');axes[1].set_title('Final partial bin retains a full-bin denominator')
    for ax in axes:ax.set_xlabel('Minutes since parent allocation start');ax.grid(alpha=.25);ax.legend()
    figures.save(fig,'production-timeline','Production over elapsed allocation time','Includes startup and round boundaries. Crop and expanded models remain distinct outputs of the same candidate; model files are not independent experimental binders.')
    if len(formal)>=2:
        small=min(formal,key=lambda r:r['plan']['nodes']);large=max(formal,key=lambda r:r['plan']['nodes'])
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');xx=np.array([r['plan']['nodes'] for r in formal]);rate=np.array([r['summary']['files_per_hour'] for r in formal]);ideal=small['summary']['files_per_hour']*xx/small['plan']['nodes']
        axes[0].plot(xx,ideal,'--',color='#aaa',label='Linear reference from smaller job');axes[0].plot(xx,rate,'o-',color='#168b80',label='Observed whole-job throughput');axes[0].set(xlabel='Total nodes in one allocation',ylabel='Retained files / allocation hour');axes[0].legend();axes[0].grid(alpha=.25)
        efficiencies=100*rate/ideal;axes[1].bar([str(v) for v in xx],efficiencies,color=COLORS[:len(formal)]);axes[1].axhline(100,color='#aaa',linestyle='--');axes[1].set(xlabel='Total nodes',ylabel='Throughput scaling efficiency (%)');axes[1].grid(axis='y',alpha=.25)
        figures.save(fig,'weak-scaling','Throughput scaling at the selected inference/science split','Workload concurrency and hypothesis-pool size grow with science capacity. These are adaptive weak-scaling runs with one observation per size, not a controlled strong-scaling benchmark. Full parent elapsed time includes cold startup and drain.')
    if baseline:
        trials=baseline['trials'];fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        for i,t in enumerate(trials):
            v=t['distinct_trial_complex_files_per_science_allocation_hour'];ax.scatter(t['split']['inference'],v,s=70,color=COLORS[i%4]);ax.annotate(fmt(v,0),(t['split']['inference'],v),xytext=(5,6),textcoords='offset points')
        ax.set(xlabel='Nemotron nodes within a total of 20 nodes',ylabel='Retained files / science-allocation hour',title='Previous ratio search: source of the 8:12 split');ax.grid(alpha=.25)
        figures.save(fig,'ratio-search','Why use 40% inference nodes?','Historical separate-allocation trials, including the repeat. Their science-allocation denominator differs from the new whole-parent-job denominator; these points establish ratio selection, not a controlled speedup claim.')
    for j,r in enumerate(runs):
        prefix='n'+str(r['plan']['nodes']);arrays=r['arrays'];minutes=r['minutes'];ninf=r['plan']['inference_count'];role_colors={'inference':'#9466a5','science':'#168b80'}
        fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
        for role,sl in [('inference',slice(0,ninf)),('science',slice(ninf,None))]:
            for ax,key in zip(axes.ravel(),['busy','gpu_memory','cpu','ram']):
                values=arrays[key][sl];ax.plot(minutes,np.nanmean(values,axis=0),label=role,color=role_colors[role]);ax.fill_between(minutes,np.nanpercentile(values,10,axis=0),np.nanpercentile(values,90,axis=0),color=role_colors[role],alpha=.15)
        for ax,title in zip(axes.ravel(),['GPU busy (%)','GPU memory used (%)','CPU busy (%)','Host memory used (%)']):ax.set_title(title);ax.set_xlabel('Allocation minute');ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend()
        figures.save(fig,prefix+'-hardware','Hardware balance: '+r['label'],'Role means and 10–90th percentile node bands. GPU busy is sampled nvidia-smi utilization, not model FLOP utilization or GPU lease occupancy.')
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');lease=np.zeros(len(minutes))
        for row in r['audit']['row_timings']:
            if row.get('started_at') is None or row.get('seconds') is None:continue
            lo=row['started_at'];hi=lo+row['seconds']
            for minute in range(max(0,int((lo-r['start'])//60)),min(len(minutes),int((hi-r['start'])//60)+1)):
                duration=min(r['end'],r['start']+(minute+1)*60)-(r['start']+minute*60)
                overlap=max(0,min(hi,r['start']+(minute+1)*60,r['end'])-max(lo,r['start']+minute*60));lease[minute]+=100*row['gpus']*overlap/(4*r['plan']['science_count']*duration)
        axes[0].plot(minutes,lease,label='Final recorded row leases',color='#d3963c');axes[0].plot(minutes,np.nanmean(arrays['busy'][ninf:],axis=0),label='Observed science GPU busy',color='#168b80');axes[0].set(xlabel='Allocation minute',ylabel='Percent of science GPU capacity');axes[0].legend();axes[0].grid(alpha=.2)
        for key,color,label in [('vllm:num_requests_running','#9466a5','Running inference requests'),('vllm:num_requests_waiting','#c47b3c','Waiting inference requests')]:
            yy=[]
            for minute in range(len(minutes)):
                samples=[finite_mean([v.get(key) for v in r['metrics'][(host,minute)]]) for host in r['hosts'][:ninf]];yy.append(sum(samples) if all(math.isfinite(v) for v in samples) else np.nan)
            axes[1].plot(minutes,yy,label=label,color=color)
        axes[1].set(xlabel='Allocation minute',ylabel='Requests summed across inference replicas');axes[1].legend();axes[1].grid(alpha=.2)
        figures.save(fig,prefix+'-work-and-busy','GPU leases versus device activity: '+r['label'],'A held GPU slot includes CPU preparation and I/O. Lease curves use final terminal row records; earlier retry leases are not reconstructed. Full allocated GPU-hours still include every attempt. Inference counters remain missing until every replica has a sample.')
        fig,ax=plt.subplots(figsize=(14,max(5,len(r['hosts'])*.12)),layout='constrained');cm=plt.colormaps['viridis'].copy();cm.set_bad('#e3e6e8');im=ax.imshow(arrays['busy'],aspect='auto',vmin=0,vmax=100,cmap=cm,extent=(0,len(minutes),len(r['hosts'])-.5,-.5));ax.axhline(ninf-.5,color='#e7b64b',linewidth=1.5);ax.set_yticks(range(len(r['hosts'])),[('N ' if role=='inference' else 'S ')+host for host,role in zip(r['hosts'],r['roles'])],fontsize=max(4,9-len(r['hosts'])/100));ax.set_xlabel('Allocation minute');fig.colorbar(im,ax=ax,label='Mean of four GPU busy samples (%)',shrink=.65)
        figures.save(fig,prefix+'-node-heatmap','Every node over time: '+r['label'],'Inference hosts precede science hosts; the gold line separates roles. Grey cells are missing samples and remain excluded from observed means.')
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for role,sl in [('inference',slice(0,ninf)),('science',slice(ninf,None))]:
            v=arrays['busy'][sl];node=np.nanmean(v,axis=1);axes[0].hist(node,bins=np.arange(0,105,5),alpha=.5,label=role,color=role_colors[role]);valid=v[np.isfinite(v)];axes[1].step(np.sort(valid),np.arange(1,len(valid)+1)/len(valid),where='post',label=role,color=role_colors[role])
        axes[0].set(xlabel='Time-mean node GPU busy (%)',ylabel='Nodes');axes[1].set(xlabel='Node-minute GPU busy (%)',ylabel='Empirical cumulative fraction')
        for ax in axes:ax.grid(alpha=.2);ax.legend()
        figures.save(fig,prefix+'-utilization-distribution','Saturation distribution: '+r['label'],'Left: per-node time means. Right: all observed node-minute means. Missing samples are omitted; startup and drain are retained.')
        if r['plan']['nodes']>=20:
            fig,axes=plt.subplots(2,3,figsize=(14,6),layout='constrained')
            for row,(role,indices) in enumerate([('inference',list(range(ninf))),('science',list(range(ninf,len(r['hosts']))))]):
                ordered=sorted(indices,key=lambda i:float(np.nanmean(arrays['busy'][i])))
                for col,(position,description) in enumerate([(0,'Least busy'),(len(ordered)//2,'Median'),(-1,'Busiest')]):
                    i=ordered[position];ax=axes[row,col]
                    for gpu in range(4):ax.plot(minutes,r['per_gpu'][i,gpu],label='GPU '+str(gpu),linewidth=.8,alpha=.8)
                    ax.set(title=description+' '+role+' · '+r['hosts'][i],xlabel='Allocation minute',ylabel='GPU busy (%)',ylim=(0,100));ax.grid(alpha=.2)
            axes[0,0].legend(ncol=4,fontsize=7)
            figures.save(fig,prefix+'-node-examples','Individual-node traces across the saturation range: '+r['label'],'Deterministic examples: minimum, upper median and maximum time-mean node GPU busy within each role. All four GPUs are shown; the HTML selector includes every node, not only these examples.')
        # Individual four-GPU plots provide a readable selector for large jobs.
        for i,(host,role) in enumerate(zip(r['hosts'],r['roles'])):
            fig,ax=plt.subplots(figsize=(10,3.2),layout='constrained')
            for gpu in range(4):ax.plot(minutes,r['per_gpu'][i,gpu],label='GPU '+str(gpu),linewidth=1,alpha=.85)
            ax.set(xlabel='Allocation minute',ylabel='GPU busy (%)',ylim=(0,100),title=f"{host} · {role} · {r['receipt']['job_id']}");ax.legend(ncol=4,loc='upper right');ax.grid(alpha=.2)
            figures.save(fig,prefix+'-'+host,host+' / '+role,'One-minute means of ten-second nvidia-smi samples; all four physical GPUs shown. Gaps represent missing data.','node')
        if r['requests']:
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for thinking,color in [(True,'#9466a5'),(False,'#168b80')]:
                req=[q for q in r['requests'] if q.get('thinking_enabled') is thinking and q.get('status')==200];x=sorted(q['total_request_seconds'] for q in req)
                if x:axes[0].step(x,np.arange(1,len(x)+1)/len(x),where='post',label='thinking '+str(thinking),color=color)
            counts=collections.Counter(q.get('backend') for q in r['requests']);ids=sorted(counts,key=lambda x:str(x));axes[1].bar(range(len(ids)),[counts[k] for k in ids],color='#9466a5');axes[1].set(xlabel='Inference replica index',ylabel='Final request attempts');axes[0].set(xlabel='End-to-end request seconds',ylabel='Cumulative fraction');axes[0].legend();axes[0].grid(alpha=.2)
            figures.save(fig,prefix+'-inference','Inference latency and work distribution: '+r['label'],'Latency includes gateway waiting. Retries are retained in raw request-attempt counts and costs; final attempts are used for the latency distribution. Role calls keep thinking off; tournament calls retain the prior default.')
        for metric,title in [('best_iptm','Best crop ipTM'),('median_iptm','Median crop ipTM'),('dual_contact_models','Crop models with both-region contacts')]:
            fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
            for arm,color in ARMS.items():
                groups=[[c[metric] for c in r['screens'] if c['condition']==arm and c['round']==n] for n in sorted({c['round'] for c in r['screens']})];ns=sorted({c['round'] for c in r['screens']})
                means=[finite_mean(x) for x in groups];ax.plot(ns,means,marker='o',label=arm,color=color)
                for n,x in zip(ns,groups):
                    if x:ax.scatter(np.full(len(x),n)+np.random.default_rng(17).uniform(-.12,.12,len(x)),x,color=color,s=9,alpha=.18)
            ax.set(xlabel='Tournament round (0 = infrastructure pilot)',ylabel=title);ax.legend();ax.grid(alpha=.25)
            figures.save(fig,prefix+'-'+metric,title+': '+r['label'],'Points are retained input trials; lines connect arm means. Arms and rounds are adaptive, not randomized independent biological replicates.','science')
        if r['audit'].get('agent_timings'):
            fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained');roles=['code_generator','validator','reviewer','reasoner']
            for pos,role in enumerate(roles):
                times=[a['seconds'] for a in r['audit']['agent_timings'] if a['role']==role]
                if times:
                    axes[0].boxplot([times],positions=[pos],widths=.55,manage_ticks=False);axes[0].scatter(np.full(len(times),pos)+np.random.default_rng(pos).uniform(-.1,.1,len(times)),times,s=8,alpha=.2,color='#168b80')
            axes[0].set_xticks(range(4),roles,rotation=18);axes[0].set(ylabel='Wall seconds per agent call',title='Tools and nested agents included');axes[0].grid(axis='y',alpha=.2)
            rounds=sorted(r['rounds'])
            for role,color in [('validator','#168b80'),('reviewer','#9466a5')]:
                groups=[[a['seconds'] for a in r['audit']['agent_timings'] if a['role']==role and a.get('round')==n] for n in rounds]
                axes[1].plot(rounds,[finite_mean(g) for g in groups],marker='o',label=role+' mean',color=color);axes[1].plot(rounds,[max(g) if g else np.nan for g in groups],linestyle='--',label=role+' maximum',color=color)
            axes[1].set(xlabel='Tournament round',ylabel='Agent wall seconds',title='Mean versus straggler duration');axes[1].legend();axes[1].grid(alpha=.2)
            figures.save(fig,prefix+'-agent-timing','Agent timing and workflow tails: '+r['label'],'Agent wall time includes tools and nested code-generator calls; role durations must not be added as disjoint costs. Maximum durations expose round-barrier stragglers. These are operational calls, not independent biological replicates.')
        if r['full']:
            fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
            for arm,color in ARMS.items():
                rows=[c for c in r['full'] if c['condition']==arm]
                if rows:
                    axes[0].scatter([read(c['source_screen_summary'])['best_iptm'] for c in rows],[c['intact_iptm_best'] for c in rows],label=arm,color=color,alpha=.6,s=22)
                    axes[1].scatter([c['intact_iptm_best'] for c in rows],[c['static_MMGBSA_kcal_mol'] for c in rows],color=color,alpha=.6,s=22)
                    axes[2].scatter([c['minimized_pose_geometry']['target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A'] for c in rows],[c['intact_models_with_dual_region_contact'] for c in rows],color=color,alpha=.6,s=22)
            axes[0].plot([0,1],[0,1],color='#aaa',linestyle='--');axes[0].set(xlabel='Crop best ipTM',ylabel='Expanded best ipTM',xlim=(0,1),ylim=(0,1));axes[1].set(xlabel='Expanded best ipTM',ylabel='Static MM/GBSA (kcal/mol)');axes[2].set(xlabel='Minimized binder RMSD to crop (Å)',ylabel='Expanded models with both-region contacts',ylim=(-.2,5.2));axes[0].legend()
            for ax in axes:ax.grid(alpha=.2)
            figures.save(fig,prefix+'-expanded','Expanded-context retention and energy: '+r['label'],'622-residue EGFR context, five new Chai poses, a minimized energy frame, and 25 annotated target disulfides. Static MM/GBSA and ipTM are computational metrics, not measured affinity.','science')


def exports(out,runs,root):
    screens=[];binders=[];hypotheses=[];nodes=[]
    for r in runs:
        for c in r['screens']:
            screens.append({k:c.get(k) for k in ['round','hypothesis','condition','candidate_id','seed','binder_sequence','best_iptm','median_iptm','dual_contact_models','summary_path']}|{'run':r['label']})
        for c in r['full']:
            g=c['minimized_pose_geometry'];binders.append({k:c.get(k) for k in ['round','hypothesis','condition','screen_candidate_id','seed','binder_sequence','intact_iptm_best','intact_iptm_median','intact_models_with_dual_region_contact','static_MMGBSA_kcal_mol','summary_path']}|{'run':r['label'],'minimized_dual_contact':g['dual_region_contact'],'minimized_binder_RMSD_A':g['target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A'],'close_atom_pairs_lt2A':g['close_atom_pairs_lt2A'],'minimized_coordinate_sha256':g['sha256']})
        for n,record in sorted(r['rounds'].items()):
            for thread in record['threads']:
                h=thread['hypothesis'];hypotheses.append({'run':r['label'],'round':n,'id':h['id'],'summary':h['summary'],'elo':h['elo'],'validator_outcome':(thread.get('report') or {}).get('outcome','untested'),'reviewed':thread.get('report') is not None and thread.get('review') is not None and not thread.get('error'),'error':thread.get('error'),'accepted_claims':len(thread.get('accepted',[])),'complete_screen_trials':sum(c['round']==n and c['hypothesis']==h['id'] for c in r['screens']),'expanded_trials':sum(c['round']==n and c['hypothesis']==h['id'] for c in r['full'])})
        for i,(host,role) in enumerate(zip(r['hosts'],r['roles'])):
            nodes.append({'run':r['label'],'job_id':r['receipt']['job_id'],'host':host,'role':role,'GPU_busy_percent':float(np.nanmean(r['arrays']['busy'][i])),'coverage_percent':100*float(np.isfinite(r['arrays']['busy'][i]).mean()),'GPU_memory_used_percent':float(np.nanmean(r['arrays']['gpu_memory'][i])),'CPU_busy_percent':float(np.nanmean(r['arrays']['cpu'][i]))})
    jobs=accounting(read(root/'experiment.json')['owned_job_ids']);cost=sum(int(j['NNodes'])*4*(j['end_epoch']-j['start_epoch'])/3600 for j in jobs.values() if j.get('start_epoch') and j.get('end_epoch'))
    result={'generated_at':time.time(),'experiment':str(root),'runs':[r['summary'] for r in runs],'all_owned_jobs':jobs,'all_attempt_GPU_hours':cost,'cost_definition':'Unique parent Slurm jobs only, including failed/cancelled startup tests, model warmup, campaign, drain and cleanup. Individual inference steps are not added again.','hypotheses':hypotheses,'screen_trials':screens,'expanded_trials':binders,'nodes':nodes,'incidents':read(root/'incidents.json') if (root/'incidents.json').exists() else {},'methods':{'thinking':'Code generator, validator, reviewer, reasoner: off; tournament generation/ranking: prior default on.','selection':'Same frozen science recipes and per-hypothesis nominal budget as the previous 20-node run. 256-node hypothesis concurrency is 154; 64 interchangeable control submitter agents are not a scientific row limit. The 256-node run also rejects substitute scientific tools; the 20-node test allowed new tool registrations, including a failed placeholder batch retained in costs.','deduplication':'Earliest successful identical screen condition/input SHA/seed/target sequence; expanded binder sequence/seed/target length. No best-score selection among duplicate executions.','telemetry':'Utilization summaries average observed node-minute bins; coverage is the fraction of bins with observations, including partial boundary minutes. GPU busy is not FLOP utilization or scientific GPU lease occupancy.','interpretation':'Computational designs; no measured binding affinity or experimental validation. ipTM is model confidence; static minimized MM/GBSA is not a measured free energy. Adaptive tournament hypotheses are not independent biological replicates.'}}
    steps={}
    for r in runs:
        job_id=r['receipt']['job_id'];assert job_id.isdigit()
        fields=['JobIDRaw','JobName','State','NNodes','NodeList','ElapsedRaw','ExitCode']
        q=subprocess.run(['sacct','-n','-j',job_id,'--parsable2','--format='+','.join(fields)],capture_output=True,text=True,check=True)
        steps[job_id]=[dict(zip(fields,line.split('|'))) for line in q.stdout.splitlines() if line.strip()]
        receipts=[]
        for p in sorted((Path(r['plan']['infra'])/'replicas').glob('*/state/nemotron/*/step.json')):
            d=read(p);assert d['job_id']==job_id and d['host'].split('.')[0] in r['manifest']['inference_nodes'];receipts.append(d)
        assert len(receipts)==r['plan']['inference_count']
        write_json(out/(r['job'].name+'-inference-steps.json'),receipts)
    write_json(out/'slurm-parent-and-steps.json',steps)
    write_json(out/'data.json',result)
    ownership_proof=root/'postrun-ownership-and-protocol.json'
    if ownership_proof.exists():write_json(out/'execution-verification.json',read(ownership_proof))
    for r in runs:
        handoff=r['job']/'monitor-status.json'
        if handoff.exists():write_json(out/(r['job'].name+'-monitor-handoff.json'),read(handoff))
    write_json(out/'exclusions-and-failures.json',[{'run':r['label'],'source_classifications':r['audit'].get('nonstandard_source_classifications',[]),'ineligible_inputs':r['audit'].get('nonstandard_input_classifications',[]),'thread_errors':r['audit']['thread_errors'],'reasoner_failures':[{'round':n,'brief':d['brief']} for n,d in r['rounds'].items() if str(d.get('brief',{}).get('report','')).startswith('reasoner failed:')]} for r in runs])
    for name,rows in [('throughput',[r['summary'] for r in runs]),('hypotheses',hypotheses),('screen-trials',screens),('expanded-trials',binders),('node-utilization',nodes)]:write_csv(out/(name+'.csv'),rows)
    write_csv(out/'production-events.csv',[{'run':r['label'],'allocation_start_epoch':r['start'],**event} for r in runs for event in r['events']])
    write_csv(out/'scientific-row-timings.csv',[{'run':r['label'],**row} for r in runs for row in r['audit']['row_timings']])
    timing_keys=['request_id','backend','status','started_at','finished_at','total_request_seconds','queue_seconds','time_to_first_token_seconds','thinking_enabled','prompt_tokens','completion_tokens']
    write_csv(out/'inference-request-timings.csv',[{'run':r['label'],**{k:q.get(k) for k in timing_keys}} for r in runs for q in r['request_attempts']])
    sequences={c['binder_sequence'] for r in runs for c in r['screens']};(out/'candidate-sequences.fasta').write_text(''.join('>binder-'+hashlib.sha256(s.encode()).hexdigest()[:16]+'\n'+s+'\n' for s in sorted(sequences)))
    for r in runs:
        label=r['job'].name;write_json(out/(label+'-checks.json'),{'job_id':r['receipt']['job_id'],'allocation':r['manifest'],'checks':r['audit']['checks'],'completion_checks':r['audit']['required_campaign_completion_checks'],'explicit_claim_source_resolutions':r['audit'].get('explicit_claim_source_resolutions',[]),'verification_lineage':r['audit'].get('verification_lineage'),'scientific_recipe_proof':read(r['root']/'scientific-protocol.json'),'control_plane_at_start':read(r['root']/'control-plane-at-start.json') if (r['root']/'control-plane-at-start.json').exists() else None})
        if (r['root']/'claim-ownership-audit.json').exists():write_json(out/(label+'-claim-ownership-audit.json'),read(r['root']/'claim-ownership-audit.json'))
    return result


def slides(out,runs,data,figures):
    formal=[r for r in runs if r['formal']];largest=max(formal,key=lambda r:r['plan']['nodes']);total=largest['summary'];scenes=[]
    s=Slide('One allocation for inference and binder science','Jupiter · loam scale-up',f"{total['nodes']} nodes · {total['inference']} Nemotron + {total['science']} science · three tournament rounds",notes=data['methods']['interpretation'],dark=True)
    for i,(value,label) in enumerate([(total['nodes'],'nodes in one Slurm job'),(total['verified_complex_files'],'retained coordinate files'),(fmt(total['files_per_hour'],0),'files per allocation hour')]):s.card(40+i*300,176,280,value,label)
    s.text(40,292,880,'All results and resource costs come from completed parent jobs, scientific receipts, independent coordinate checks and per-node GPU samples.',19,'#d5e6e9',max_height=105);scenes.append(s)
    s=Slide('How the single job is organized','Execution architecture','No coordination between separately queued high-node jobs is required.',notes='One sbatch request; internal srun steps only. Redis, the gateway, scientific MCP and the tournament controller run on the first science host. Inference and science use disjoint host lists from the parent allocation.')
    for x,title,text in [(40,'Parent Slurm allocation',f"{total['nodes']} exclusive nodes\nOne scheduler start and end"),(345,'Inference pool',f"{total['inference']} nodes, TP4 each\nLeast-loaded routing + affinity"),(650,'Science workers',f"{total['science']} nodes, four GPUs each\nFrozen MCP recipes via srun")]:
        s.rect(x,158,270,170,'#eef5f5');s.text(x+14,175,242,title,18,kind='bold',max_height=55);s.text(x+14,238,242,text,15,max_height=76)
    s.text(40,366,880,'Private job-local Redis and TLS services start after allocation. Credentials are created after queue waiting. Telemetry samples every node; normal job exit cleans up owned child steps.',16,max_height=90);scenes.append(s)
    s=Slide('Debug small, validate the tournament, then scale','Scale-up sequence','The five-node pilot exercises full production tools but is excluded from formal tournament throughput comparisons.',notes=json.dumps(data['incidents']))
    for i,r in enumerate(runs):
        sm=r['summary'];s.text(40,146+i*79,880,f"{sm['nodes']} nodes · job {sm['job_id']} · {sm['screen_trials']} screens + {sm['expanded_trials']} expanded trials · {sm['rounds']} tournament rounds",17,max_height=65)
    scenes.append(s)
    if len(formal)>=2:
        small=min(formal,key=lambda r:r['plan']['nodes'])['summary'];speedup=total['files_per_hour']/small['files_per_hour'];node_factor=total['nodes']/small['nodes']
        s=Slide('Throughput and resource efficiency','Scale-up findings','Whole parent-job elapsed time includes startup, tournament work, scientific tools and cleanup.')
        s.text(40,145,880,f"Retained coordinate throughput increased {speedup:.2f}× as the allocation grew {node_factor:.1f}×, from {small['nodes']} to {total['nodes']} nodes. Output per allocated GPU-hour reached {100*speedup/node_factor:.1f}% of the smaller job's rate.",20,max_height=112)
        s.text(40,280,880,f"Observed overall GPU busy fell from {small['overall_GPU_busy_percent']:.1f}% to {total['overall_GPU_busy_percent']:.1f}%. The ratio selected at 20 nodes did not preserve saturation at 256 nodes. Workflow tails and context-limit failures remain visible in the timing plots.",18,max_height=111)
        s.text(40,419,880,'These adaptive campaigns change concurrency and hypothesis-pool size. The measurements establish achieved throughput and cost; they do not isolate a causal scaling limit.',15,max_height=72);scenes.append(s)
    selected=[f for f in figures.figures if f['group']!='node' and not f['id'].endswith('-node-heatmap')]
    for f in selected:
        s=Slide(f['title'],'Measured results',notes=f['caption']);s.image(out/'figures'/(f['id']+'.png'),36,117,888,338);s.text(40,462,880,f['caption'],10.5,max_height=47);scenes.append(s)
    for r in formal:
        for n in sorted(r['rounds']):
            rows=[h for h in data['hypotheses'] if h['run']==r['label'] and h['round']==n];rows=sorted(rows,key=lambda h:(h['reviewed'],h['elo']),reverse=True)
            s=Slide(f"Round {n}: leading hypotheses · {r['plan']['nodes']} nodes",'Tournament evidence','Ordering: reviewed first, then advisory tournament Elo. Scientific output counts are context; Elo is not binding evidence.',notes='\n'.join(str(h) for h in rows))
            for i,h in enumerate(rows[:3]):s.text(40,145+i*104,880,f"{i+1}. {h['summary']}\nElo {h['elo']:.0f} · validator: {h['validator_outcome']} · {h['complete_screen_trials']} screens · {h['expanded_trials']} expanded trials · {h['accepted_claims']} accepted claims",15,max_height=96)
            scenes.append(s)
            rows=[b for b in data['expanded_trials'] if b['run']==r['label'] and b['round']==n];unique={}
            for b in rows:unique.setdefault(b['binder_sequence'],b)
            rows=sorted(unique.values(),key=lambda b:(b['minimized_dual_contact'],b['intact_models_with_dual_region_contact'],b['intact_iptm_best']),reverse=True)
            s=Slide(f"Round {n}: computational binder candidates",f"{r['plan']['nodes']} nodes · expanded EGFR context",'Unique sequences, earliest trial each; order: minimized dual contact, dual-contact poses, expanded ipTM.',notes='\n'.join(str(b) for b in rows))
            for i,b in enumerate(rows[:3]):
                seq=b['binder_sequence'];s.text(40,144+i*110,880,f"{b['condition']} · ipTM {b['intact_iptm_best']:.3f} · both-region contacts {b['intact_models_with_dual_region_contact']}/5 · MM/GBSA {b['static_MMGBSA_kcal_mol']:.1f} kcal/mol",15,max_height=45);s.text(40,189+i*110,880,seq,12,kind='mono',max_height=45)
            scenes.append(s)
    s=Slide('Operational limitations at scale','Failures remain in the evidence',notes=json.dumps(data['incidents']))
    s.text(40,145,880,f"Across three rounds, {total['orchestration_error_threads']} hypothesis threads raised errors; {total['thread_errors']} lacked a complete successful review. Their records remain visible. Independently checked scientific outputs and reviewed thread successes are separate counts.",18,max_height=110)
    s.text(40,270,880,f"The optional reasoner failed in {total['reasoner_failures']} rounds on the context limit; {total['rounds_with_nonempty_steering_brief']} rounds had a nonempty steering brief. The repository retained the previous brief. Tournament generation, science, review and reconciliation continued.",17,max_height=125)
    s.text(40,420,880,'A missing signal import in metadata-process timeout cleanup was fixed after this run and tested with a real subprocess and descendant. The fix was not present during these measurements.',15,max_height=70);scenes.append(s)
    s=Slide('Costs, checks and interpretation','Evidence limits',notes=json.dumps(data['methods']))
    s.text(40,145,880,f"All submitted attempts: {data['all_attempt_GPU_hours']:,.1f} allocated GPU-hours, counting each parent job once. Failed startup jobs remain in the accounting export. Missing utilization samples remain missing.",18,max_height=106)
    s.text(40,268,880,data['methods']['interpretation'],17,max_height=127);s.text(40,415,880,'See the HTML report for every-node plots, full hypothesis/binder tables, CSV and FASTA exports, job receipts and provenance hashes.',16,max_height=76);scenes.append(s)
    return export_slides(scenes,out)


def render(out,runs,data,figures):
    def figure(f):
        svg=base64.b64encode((out/'figures'/(f['id']+'.svg')).read_bytes()).decode();return '<figure><h3>'+esc(f['title'])+'</h3><a href="figures/'+esc(f['id'])+'.svg"><img loading="lazy" src="data:image/svg+xml;base64,'+svg+'" alt="'+esc(f['title'])+'"></a><figcaption>'+esc(f['caption'])+'</figcaption></figure>'
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Loamlite · Jupiter single-allocation scale-up</title><style>body{margin:0;color:#193542;background:#f5f8fa;font:16px system-ui;line-height:1.55}main{max-width:1400px;margin:auto;padding:30px}h1{font-size:2.4rem}h2{margin-top:45px}figure{margin:24px 0;padding:22px;background:white;border:1px solid #dbe5e8;border-radius:9px}figure img{width:100%;max-height:900px;object-fit:contain}figcaption{color:#506572;font-size:14px}table{border-collapse:collapse;background:white;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dbe5e8;text-align:left;max-width:600px;overflow-wrap:anywhere}th{background:#e6f1f1;position:sticky;top:0}.scroll{overflow:auto;max-height:650px}nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#087b78}input,select{font:inherit;padding:10px;max-width:100%}.node{display:none}.node.active{display:block}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{padding:16px 24px;background:#e6f1f1;border-radius:8px}.card b{font-size:28px;display:block}.note{padding:16px;background:#fff4dd;border-left:4px solid #c68a32}</style><main><h1>One Slurm job, inference and science together</h1><p>Jupiter · Nemotron-driven binder-design tournaments · measured single-allocation scale-up.</p><nav><a href="scaleup.pptx">Editable PowerPoint</a><a href="scaleup.pdf">PDF slides</a><a href="slides.html">Browser slides</a><a href="data.json">Evidence data</a><a href="throughput.csv">Throughput CSV</a><a href="node-utilization.csv">Per-node utilization</a><a href="production-events.csv">Production events</a><a href="scientific-row-timings.csv">Scientific row timings</a><a href="inference-request-timings.csv">Inference timings</a><a href="candidate-sequences.fasta">Candidate FASTA</a></nav>']
    largest=max((r for r in runs if r['formal']),key=lambda r:r['plan']['nodes']);sm=largest['summary'];parts.append('<div class="cards">'+''.join('<p class="card"><b>'+esc(value)+'</b>'+esc(label)+'</p>' for value,label in [(sm['nodes'],'nodes in one completed job'),(sm['verified_complex_files'],'retained coordinate files'),(fmt(sm['files_per_hour'],0),'files / allocation hour'),(fmt(sm['overall_GPU_busy_percent'])+'%','observed overall GPU busy')])+'</div>')
    formal=[r for r in runs if r['formal']]
    if len(formal)>=2:
        small=min(formal,key=lambda r:r['plan']['nodes'])['summary'];speedup=sm['files_per_hour']/small['files_per_hour'];node_factor=sm['nodes']/small['nodes']
        parts.append('<p>Throughput increased <b>'+fmt(speedup,2)+'×</b> as the allocation grew '+fmt(node_factor,1)+'×. Output per allocated GPU-hour reached '+fmt(100*speedup/node_factor)+'% of the smaller job\'s rate. Observed GPU busy fell from '+fmt(small['overall_GPU_busy_percent'])+'% to '+fmt(sm['overall_GPU_busy_percent'])+'%. The ratio selected at 20 nodes did not preserve saturation at 256 nodes. These adaptive campaigns are not a controlled strong-scaling benchmark.</p>')
    parts.append('<p class="note">The largest campaign recorded '+str(sm['orchestration_error_threads'])+' hypothesis-thread errors; '+str(sm['thread_errors'])+' threads lacked a complete successful review. Its optional reasoner failed in '+str(sm['reasoner_failures'])+' rounds when the aggregated prompt exceeded the context window; '+str(sm['rounds_with_nonempty_steering_brief'])+' rounds had a nonempty steering brief. The previous brief was retained. Tournament generation, scientific validation, review and reconciliation continued. Full failures remain in the export.</p>')
    parts.append('<p>'+esc(data['methods']['interpretation'])+'</p><h2>Job layout and completion</h2><p>A single sbatch request grants every node. The first science host runs the tournament, gateway, private Redis and scientific MCP control services. Internal srun steps place Nemotron and scientific workers on disjoint host subsets. No additional queued allocations are launched. The queue can wait without consuming API-token lifetime because credentials are minted after allocation starts.</p>')
    parts.append(table(['Run','Job','Parent state','Nemo : science','Rounds','Screens','Expanded','Files / hour','Files / GPU-hour','GPU busy','Coverage'],[[r['label'],r['receipt']['job_id'],r['parent']['State'],str(r['plan']['inference_count'])+' : '+str(r['plan']['science_count']),r['summary']['rounds'],len(r['screens']),len(r['full']),fmt(r['summary']['files_per_hour']),fmt(r['summary']['files_per_GPU_hour'],3),fmt(r['summary']['overall_GPU_busy_percent'])+'%',fmt(r['summary']['telemetry_coverage_percent'])+'%'] for r in runs]))
    parts.append('<p class="note">Five-node operator pilot: the one-region energy follow-up was rejected for a cysteine-containing binder; successful full production stages in the other arms are retained, and this run is excluded from formal tournament speedup comparisons. Its GPU cost remains included. A coordinate model file is not an independent binder or an experiment.</p><h2>Throughput, inference and GPU saturation</h2>')
    for f in figures.figures:
        if f['group']=='overview':parts.append(figure(f))
    parts.append('<h2>Inspect every node and all four GPUs</h2><p>Select a node to view its complete allocation timeline.</p><select id="node-select" aria-label="Select node">')
    nodefigs=[f for f in figures.figures if f['group']=='node']
    for i,f in enumerate(nodefigs):parts.append('<option value="node-'+str(i)+'">'+esc(f['id'])+'</option>')
    parts.append('</select>')
    for i,f in enumerate(nodefigs):parts.append('<div id="node-'+str(i)+'" class="node'+(' active' if i==0 else '')+'">'+figure(f)+'</div>')
    parts.append('<h2>Binder confidence, geometry and energy</h2>')
    for f in figures.figures:
        if f['group']=='science':parts.append(figure(f))
    parts.append('<p>Tested-hypothesis counts mean at least one eligible, independently checked scientific screen result, including hypotheses with a later agent failure. A complete reviewed thread requires a validation report and review with no orchestration error; these counts are kept separate.</p>')
    parts.append('<h2>Hypotheses from each tournament iteration</h2><p>Order is run, round, then advisory Elo. Validator outcome is a self-reported classification; accepted claims can support or contradict a proposal. Artifact counts alone do not demonstrate every proposed mechanism. Failed hypotheses remain visible. <a href="hypotheses.csv">Download all hypotheses</a>.</p>')
    hs=sorted(data['hypotheses'],key=lambda h:(h['run'],h['round'],-h['elo']));parts.append(table(['Run','Round','Hypothesis','Elo','Validator outcome','Reviewed','Screens','Expanded','Error'],[[h['run'],h['round'],h['summary'],fmt(h['elo'],0),h['validator_outcome'],h['reviewed'],h['complete_screen_trials'],h['expanded_trials'],h['error'] or ''] for h in hs]))
    parts.append('<h2>Expanded-context candidates by iteration</h2><p>Search by sequence, condition, run or hypothesis. Ordering prioritizes minimized dual contact, number of dual-contact poses, then expanded ipTM; this is a computational display order. <a href="expanded-trials.csv">Download full metrics</a> · <a href="screen-trials.csv">All screened trials</a>.</p><input id="binder-search" placeholder="Filter candidate rows" aria-label="Filter candidate rows"><div id="binders">')
    bs=sorted(data['expanded_trials'],key=lambda b:(b['run'],b['round'],-int(b['minimized_dual_contact']),-b['intact_models_with_dual_region_contact'],-b['intact_iptm_best']));parts.append(table(['Run','Round','Condition','Sequence','Best ipTM','Dual-contact poses','Minimized dual contact','RMSD Å','Static MM/GBSA'],[[b['run'],b['round'],b['condition'],b['binder_sequence'],fmt(b['intact_iptm_best'],3),b['intact_models_with_dual_region_contact'],b['minimized_dual_contact'],fmt(b['minimized_binder_RMSD_A']),fmt(b['static_MMGBSA_kcal_mol'])] for b in bs]));parts.append('</div><h2>Accounting and reproducibility</h2><p>'+esc(data['cost_definition'])+' Total: <b>'+fmt(data['all_attempt_GPU_hours'])+' allocated GPU-hours</b>.</p>')
    parts.append(table(['Job','Name','State','Nodes','Elapsed seconds','Queue seconds'],[[j['JobIDRaw'],j['JobName'],j['State'],j['NNodes'],j['ElapsedRaw'],fmt(j['queue_seconds'],0)] for j in data['all_owned_jobs'].values()]))
    incidents=data.get('incidents',{}).get('incidents',[])
    if incidents:parts.append('<h3>Operational incidents</h3>'+table(['Job / run','Cause','Effect or outcome','Fix'],[[str(i['job_id'])+' / '+i.get('label',''),i.get('cause',''),i.get('effect',i.get('outcome','')),i.get('fix','No runtime fix applied')] for i in incidents]))
    parts.append('<p><a href="exclusions-and-failures.json">Scientific exclusions and failed threads</a>. Unapproved sources, noncanonical or mislabeled inputs are excluded from positive output counts; their original records and full allocated costs are retained.</p><p>'+esc(data['methods']['deduplication'])+'</p><p>'+esc(data['methods']['telemetry'])+'</p><p>'+esc(data['methods']['thinking'])+'</p><p>'+esc(data['methods']['selection'])+'</p>')
    if (out/'execution-verification.json').exists():parts.append('<p><a href="execution-verification.json">Independent ownership, recipe, model-setting and postrun source comparison</a>. The postrun timeout fix is distinguished from the measured runtime source.</p>')
    for r in runs:parts.append('<p><a href="'+esc(r['job'].name)+'-checks.json">'+esc(r['label'])+' ownership and independent scientific checks</a></p>')
    parts.append('<p><a href="slurm-parent-and-steps.json">Slurm parent and step accounting</a>. Service steps are stopped when the campaign ends; service-step cancellation during cleanup is distinct from the parent job outcome. Parent allocations alone determine node counts and GPU-hour costs.</p><p><a href="build-manifest.json">Source and export hashes</a></p><script>document.getElementById("node-select").onchange=e=>{document.querySelectorAll(".node").forEach(n=>n.classList.toggle("active",n.id===e.target.value))};document.getElementById("binder-search").oninput=e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("#binders tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))};</script></main></html>')
    (out/'index.html').write_text(''.join(parts))


def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);p.add_argument('output',type=Path);p.add_argument('--labels',nargs='+',default=['smoke-05c','unified-20','unified-256']);a=p.parse_args();root=a.experiment.resolve();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
    runs=[load_run(root/label) for label in a.labels]
    for r in runs:telemetry(r)
    baseline_path=REPO/'reports/inference-balance-20261113/comparison.json';baseline=read(baseline_path) if baseline_path.exists() else None
    figures=Figures(out);plots(figures,runs,baseline);data=exports(out,runs,root);count=slides(out,runs,data,figures);render(out,runs,data,figures)
    sources={str(p):sha(p) for r in runs for p in [r['job']/'submission.json',r['job']/'allocation.json',r['root']/'scaling-audit.json',r['root']/'scientific-protocol.json',r['root']/'campaign.yaml',r['root']/'server-ready.json',r['root']/'controller-watch.json',*sorted((r['root']/'state/rounds').glob('round-*.json')),*[r['root']/name for name in ['claim-source-resolutions.json','operator-reviewed-source-aliases.json','control-plane-at-start.json'] if (r['root']/name).exists()],*r['telemetry_paths']]}

    if baseline_path.exists():sources[str(baseline_path)]=sha(baseline_path)
    for p in [root/name for name in ['experiment.json','incidents.json','postrun-ownership-and-protocol.json','postrun-pytest.stdout','telemetry-quickcheck.json']]+[r['job']/'monitor-status.json' for r in runs]:
        if p.exists():sources[str(p)]=sha(p)
    for r in runs:
        for key in ['numeric_audit','claim_audit']:
            name=r['audit'].get('verification_lineage',{}).get(key)
            if name:sources[name]=sha(Path(name))
    for name in ['audit_scale20_campaign.py','audit_claim_resolutions.py','resolve_campaign_claim_sources.py','scaleup_slides.py']:
        p=Path(__file__).with_name(name);sources[str(p)]=sha(p)
    manifest={'generated_at':time.time(),'sources':sources,'builder_sha256':sha(Path(__file__)),'slide_count':count,'figure_count':len(figures.figures),'exports':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='build-manifest.json'}};write_json(out/'build-manifest.json',manifest)
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,Path(out.name)/f.relative_to(out))
    print(json.dumps({'report':str(out/'index.html'),'slides':str(out/'scaleup.pptx'),'slide_count':count,'figures':len(figures.figures)},indent=2))

if __name__=='__main__':main()
