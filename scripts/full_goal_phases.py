"""Attribute final allocation time to observed admission/finish boundaries."""
import collections
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_inference_atlas import write_csv


def phase_plots(runs,save,output,sources,sha):
    rows=[];issues=[]
    for run in runs:
        path=run['root']/'state/admission-timings.jsonl'
        sources[str(path)]=sha(path)
        grouped=collections.defaultdict(lambda:{'started':{},'finished':{},'queued':set()})
        for line in path.read_text().splitlines():
            event=json.loads(line);t=event['timestamp']
            if not run['start']<=t<=run['end']:continue
            bucket=grouped[event['round']];kind=event['event'];h=event['hypothesis_id']
            if kind=='queued':bucket['queued'].add(h)
            elif kind=='started':bucket['started'].setdefault(h,t)
            elif kind=='finished':bucket['finished'][h]=t
        phases=[];cursor=run['start']
        def add(kind,name,lo,hi):
            if hi>lo:phases.append((kind,name,lo,hi))
        for number,bucket in sorted(grouped.items()):
            if not bucket['started']:continue
            begin=min(bucket['started'].values())
            if begin<cursor-.1:
                issues.append({'run':run['label'],'reason':'Overlapping/retried round boundaries; no clean phase partition asserted'})
                phases=[];break
            add('startup' if cursor==run['start'] else 'between',
                'Startup and initial tournament' if cursor==run['start'] else 'Synthesis and next tournament',cursor,begin)
            ends=sorted(bucket['finished'].values())
            complete=bool(bucket['queued']) and set(bucket['finished'])==bucket['queued']
            if complete:
                p90=ends[int(.9*(len(ends)-1))]
                add('bulk',f'Round {number}: first 90% complete',begin,p90)
                add('tail',f'Round {number}: final 10% drain',p90,ends[-1])
                cursor=ends[-1]
            else:
                add('incomplete',f'Round {number}: incomplete at retirement',begin,run['end'])
                cursor=run['end'];break
        if not phases:
            continue
        add('final','Final synthesis/science drain/cleanup',cursor,run['end'])
        assert abs(sum(hi-lo for kind,name,lo,hi in phases)-run['seconds'])<.1
        for kind,name,lo,hi in phases:
            result={'run':run['label'],'job_id':run['receipt']['job_id'],'kind':kind,'phase':name,
                    'start':lo,'end':hi,'seconds':hi-lo}
            for role in ('overall','science','inference'):
                nodes=[n for n in run['metric']['per_node'] if role=='overall' or n['role']==role]
                integral=0.0
                for node in nodes:
                    for point in node['trace']:
                        seconds=max(0,min(hi,point['timestamp']+point['duration'])-max(lo,point['timestamp']))
                        integral+=seconds*sum(point['gpu_busy'])
                result[role+'_GPU_busy_percent']=integral/(4*len(nodes)*(hi-lo))
            rows.append(result)
    write_csv(output/'allocation-phase-metrics.csv',rows)
    if not rows:return {'rows':0,'issues':issues}
    labels=[r['label'] for r in runs if any(x['run']==r['label'] for x in rows)]
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained');bottom=np.zeros(len(labels))
    for kind,title,color in [('startup','Startup/initial tournament','#8799a2'),('bulk','Round bulk','#168b80'),
                              ('tail','Last 10% of threads','#cb8c35'),('between','Synthesis/next tournament','#946aa5'),
                              ('final','Final synthesis/drain/cleanup','#5c79a0'),('incomplete','Incomplete work at retirement','#c66f6c')]:
        values=np.array([sum(x['seconds'] for x in rows if x['run']==label and x['kind']==kind)/60 for label in labels])
        ax.bar(range(len(labels)),values,bottom=bottom,label=title,color=color);bottom+=values
    ax.set_xticks(range(len(labels)),labels,rotation=20,ha='right');ax.set(ylabel='Final assigned allocation minutes')
    ax.legend(ncol=3,fontsize=8);ax.grid(axis='y',alpha=.2)
    save(fig,'allocation-phase-budget','Where the allocated wall time went',
         'Boundaries come from recorded thread admissions and completions. The last 10% category describes thread completion, not a biological quality threshold. Every minute of each depicted final allocation is retained. Unobserved prior requeue attempts remain in whole-parent cost and utilization; they are not assigned invented phases.','runtime-phases')
    latest=next(r for r in reversed(runs) if any(x['run']==r['label'] for x in rows))
    selected=[x for x in rows if x['run']==latest['label']]
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained',gridspec_kw={'width_ratios':[1,2]})
    y=np.arange(len(selected))
    axes[0].barh(y,[x['seconds']/60 for x in selected],color='#697b87')
    for offset,role,color in [(-.23,'science','#168b80'),(0,'inference','#cb8c35'),(.23,'overall','#5c79a0')]:
        axes[1].barh(y+offset,[x[role+'_GPU_busy_percent'] for x in selected],height=.22,color=color,label=role)
    axes[0].set_yticks(y,[x['phase'] for x in selected],fontsize=9);axes[0].invert_yaxis();axes[0].set(xlabel='Minutes')
    axes[1].set_yticks(y,[]);axes[1].invert_yaxis();axes[1].set(xlabel='Sampled GPU busy (%)',xlim=(0,100));axes[1].legend()
    for ax in axes:ax.grid(axis='x',alpha=.2)
    save(fig,'latest-phase-utilization','GPU activity by observed phase: '+latest['label'],
         'Native telemetry is integrated over every physical GPU-second in each observed phase; missing samples contribute zero. Science and inference activity are separate. Global barriers and long hypothesis tails are included, rather than selecting a favorable steady interval.','runtime-phases')
    return {'rows':len(rows),'issues':issues,'scope':'Final assigned allocation phases; all parent attempts remain in aggregate GPU-hour accounting.'}
