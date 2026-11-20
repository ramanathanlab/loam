#!/usr/bin/env python3
"""Export scientific throughput and paired inference latency comparisons."""
import argparse,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root,output):
    data=json.loads((output/'comparison.json').read_text());trials=[t for t in data['trials'] if t.get('completion_checks_passed')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout();fig.savefig(output/(name+'.svg'),bbox_inches='tight');fig.savefig(output/(name+'.png'),bbox_inches='tight',dpi=160);plt.close(fig)
    if trials:
        labels=[f"{t['split']['inference']} inference / {t['split']['science']} science"+('\nconfirmation' if 'confirm' in Path(t['trial']).name else '') for t in trials]
        x=list(range(len(trials)));crop=[t['unique_screen_input_trials']*5/(t['science_allocation_seconds']/3600) for t in trials];context=[t['unique_complete_intact_input_trials']*5/(t['science_allocation_seconds']/3600) for t in trials]
        fig,ax=plt.subplots(figsize=(9,4.3));ax.bar(x,crop,label='Crop complexes, distinct input/seed trials',color='#24638c');ax.bar(x,context,bottom=crop,label='Expanded-EGFR complexes, distinct input/seed trials',color='#258273')
        for i,t in enumerate(trials):ax.text(i,crop[i]+context[i],f" {t['distinct_trial_complex_files_per_science_allocation_hour']:.0f}/h",ha='center',va='bottom')
        ax.set(xticks=x,xticklabels=labels,ylabel='Verified complex files / science-allocation hour',title='Useful scientific throughput at twenty total owned nodes');ax.set_ylim(0,max(a+b for a,b in zip(crop,context))*1.3);ax.legend(loc='upper left',fontsize=8);save(fig,'ratio-scientific-throughput')
        fig,axes=plt.subplots(1,2,figsize=(10,4.1));axes[0].bar(x,[t['distinct_sequences_per_science_allocation_hour'] for t in trials],color='#24638c');axes[1].bar(x,[t['distinct_intact_trials_per_science_allocation_hour'] for t in trials],color='#258273')
        for ax,title,label in [(axes[0],'New binder sequence throughput','Distinct sequences / hour'),(axes[1],'Complete expanded-context energy trials','Distinct complete trials / hour')]:ax.set(xticks=x,xticklabels=labels,ylabel=label,title=title);ax.tick_params(axis='x',labelrotation=15)
        save(fig,'ratio-output-categories')
        fig,ax=plt.subplots(figsize=(9,4.3))
        for offset,role,color in [(-.24,'science','#24638c'),(0,'inference','#258273')]:
            values=[t.get('telemetry',{}).get('role_observed_node_minutes',{}).get(role,{}).get('mean_GPU_busy_percent',float('nan')) for t in trials]
            ax.bar([i+offset for i in x],values,width=.23,label=role.capitalize()+' nodes',color=color)
        overall=[t['overall_observed_GPU_busy_percent'] if t.get('overall_observed_GPU_busy_percent') is not None else float('nan') for t in trials]
        ax.bar([i+.24 for i in x],overall,width=.23,label='All observed nodes',color='#dd964b')
        for i,value in enumerate(overall):
            if math.isfinite(value):ax.text(i+.24,value+1,f'{value:.1f}%',ha='center',fontsize=8)
        ax.set(xticks=x,xticklabels=labels,ylabel='Mean sampled NVIDIA GPU busy (%)',ylim=(0,105),title='Measured GPU activity during each science allocation')
        ax.legend(loc='upper right',fontsize=8);save(fig,'ratio-gpu-utilization')
    profile=root/'inference-profile.json'
    if profile.exists():
        results=json.loads(profile.read_text())['results'];rows=[r for r in results if r['label']=='one-replica-warm' or 'affinity-warm' in r['label']]
        rows=sorted(rows,key=lambda r:len(r['replicas']))
        fig,axes=plt.subplots(1,2,figsize=(10,4.1));n=[len(r['replicas']) for r in rows]
        axes[0].plot(n,[r['median_request_seconds'] for r in rows],'o-',label='Median');axes[0].plot(n,[r['p95_request_seconds'] for r in rows],'o-',label='P95');axes[0].set(xlabel='Ready Nemotron replicas',ylabel='Request seconds',title='Identical frozen-context reviews; concurrency12',xticks=n);axes[0].legend()
        axes[1].bar(list(range(len(rows))),[r['successful_reviews_per_second'] for r in rows],color='#24638c');axes[1].set(xticks=list(range(len(rows))),xticklabels=n,xlabel='Ready Nemotron replicas',ylabel='Strictly validated reviews / second',title='Validation failures retained')
        for i,r in enumerate(rows):axes[1].text(i,r['successful_reviews_per_second'],f" {r['passed']}/{r['requests']} passed",ha='center',va='bottom',fontsize=8)
        axes[1].set_ylim(0,max((r['successful_reviews_per_second'] for r in rows),default=1)*1.25);save(fig,'ratio-inference-latency')
    print(json.dumps({'completed_trials_plotted':len(trials),'output':str(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.root.resolve(),a.output.resolve())
