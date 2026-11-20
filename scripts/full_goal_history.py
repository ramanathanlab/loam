"""Whole-parent GPU history for every receipt-owned job, including aborts."""
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt

from audit_saturation_campaign import hardware
import audit_saturation_campaign as native
from build_inference_atlas import write_csv


def history_plots(runs,recipes,costs,experiment,save,output,sources,sha):
    byid={r['receipt']['job_id']:r for r in runs}
    settings={r['job_id']:r for r in recipes}
    rows=[]
    for cost in costs['jobs']:
        jid=cost['job_id'];recipe=settings[jid];accounting=cost['accounting']
        row={'job_id':jid,'trial':recipe['trial'],'nodes':recipe['nodes'],
             'inference':recipe['inference'],'science':recipe['science'],
             'allocated_GPU_hours':cost['allocated_GPU_hours'],
             'allocation_attempts':cost['allocation_attempts'],
             'qualified_campaign':cost['included_as_successful_trial'],
             'scientific_output_analysis_available':jid in byid}
        if not accounting['allocations']:
            row.update(parent_state='No allocation',overall_GPU_busy_percent=None,
                       telemetry_coverage_percent=None,scope='No allocated interval available')
            rows.append(row);continue
        final=accounting['allocations'][-1]
        row['parent_state']=final['State']
        job=Path(recipe['job_script']).parent.resolve(strict=True)
        status_path=job/'status.json'
        if status_path.exists():
            status=json.loads(status_path.read_text());sources[str(status_path)]=sha(status_path)
            row['controller_phase']=status['phase']
        reasons=[]
        for name in ['cancellation-intent.json','diagnostic-retirement-intent.json']:
            path=job/name
            if path.exists():
                intent=json.loads(path.read_text());assert intent['job_id']==jid
                sources[str(path)]=sha(path)
                reasons.append(intent.get('reason') or intent.get('action'))
        row['retirement_reason']='; '.join(x for x in reasons if x)
        if not row['retirement_reason']:
            row['retirement_reason']='Native accounting '+final['State']+'; controller phase '+str(row.get('controller_phase','unavailable'))
        if jid in byid:
            metric=byid[jid]['metric']['summary']
            row.update(overall_GPU_busy_percent=metric['overall_GPU_busy_percent'],
                       telemetry_coverage_percent=metric['overall_telemetry_coverage_percent'],
                       scope='Main numerical-output comparison; native whole-parent metric')
        else:
            job=Path(recipe['job_script']).parent.resolve(strict=True)
            cache=experiment/'diagnostics'/f'report-history-hardware-{jid}.json'
            if not cache.exists():
                try:
                    start,end=final['start_epoch'],final['end_epoch']
                    summary,nodes=hardware(job,start,end)
                    ratio=(end-start)/accounting['total_allocated_seconds']
                    summary['overall_GPU_busy_percent']*=ratio
                    summary['overall_telemetry_coverage_percent']*=ratio
                    inputs={str(job/'allocation.json'):sha(job/'allocation.json'),
                            str(Path(native.__file__)):sha(Path(native.__file__))}
                    for path in (job/'campaign/telemetry').glob('*.jsonl'):
                        inputs[str(path)]=sha(path)
                    value={'job_id':jid,'summary':summary,'input_sha256':inputs,
                           'accounting':accounting,'node_count':len(nodes),
                           'scope':'Native all-four-GPU integration on final assigned nodes. Prior allocated attempts without surviving roles/telemetry contribute zero busy and retain their full cost.'}
                except (AssertionError,OSError,KeyError,ValueError) as exc:
                    value={'job_id':jid,'error':repr(exc),'input_sha256':{},'accounting':accounting}
                cache.write_text(json.dumps(value,indent=2)+'\n')
            value=json.loads(cache.read_text());assert value['job_id']==jid
            sources[str(cache)]=sha(cache)
            for path,digest in value['input_sha256'].items():
                assert sha(Path(path))==digest,path
                sources[path]=digest
            if 'summary' in value:
                row.update(overall_GPU_busy_percent=value['summary']['overall_GPU_busy_percent'],
                           telemetry_coverage_percent=value['summary']['overall_telemetry_coverage_percent'],scope=value['scope'])
            else:
                row.update(overall_GPU_busy_percent=None,telemetry_coverage_percent=None,
                           scope='Metric unavailable; original issue retained: '+value['error'])
        rows.append(row)
    assert {r['job_id'] for r in rows}==set(settings)
    write_csv(output/'all-owned-job-history.csv',rows)
    fig,axes=plt.subplots(2,1,figsize=(16,8),layout='constrained',sharex=True)
    for i,row in enumerate(rows):
        color='#168b80' if row['nodes']==20 else '#5c79a0'
        value=row['overall_GPU_busy_percent']
        axes[0].bar(i,value or 0,color=color,hatch='' if row['qualified_campaign'] else '//',edgecolor='white')
        if value is None or (row['telemetry_coverage_percent'] or 0)<1:
            axes[0].text(i,3,'N/A',ha='center',rotation=90,fontsize=8)
        axes[1].bar(i,row['allocated_GPU_hours'],color=color)
    axes[0].axhline(50,color='#555',ls='--',label='50% target');axes[0].legend()
    axes[0].set(ylabel='Whole-parent sampled GPU busy (%)',ylim=(0,100))
    axes[1].set(ylabel='Allocated GPU-hours (all attempts)')
    axes[1].set_xticks(range(len(rows)),[r['trial']+'\n'+r['job_id'] for r in rows],rotation=25,ha='right',fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    save(fig,'all-owned-job-history','Every owned job: utilization, failures and resource cost',
         'All 13 receipt-owned jobs are included. Teal denotes 20 nodes and blue 256. Hatching denotes a campaign not counted as a qualified success; it does not imply its numerical outputs are absent. Missing telemetry contributes zero to the conservative busy bound; unavailable/near-zero-coverage cases are marked N/A and coverage is exported. Parent allocation attempts, including requeues, are charged once; internal steps are not added again.','full-history')
    return rows
