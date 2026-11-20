#!/usr/bin/env python3
"""Offline report/deck for audited saturation trials; historical references stay immutable.

Scientific Python, PYTHONPATH=.:scripts:/tmp/loam-presentation-deps-20261113.
Supply --trial LABEL JOB for every completed trial and --metric-cache for saved
audit_saturation_campaign outputs. Missing caches are recomputed, never invented.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import json
import math
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_saturation_campaign import measure, reviewed_science_threads, completed_assessment_threads
from build_unified_report import load_run
from build_inference_atlas import write_csv
from scaleup_slides import Slide, export
from saturation_timing_plots import timing_plots
from saturation_accounting import allocation_attempts
from saturation_resource_plots import resource_plots

ARMS = {'dual':'#147f78','one_region':'#c6812f','ridge':'#8763a2'}
COLORS = ['#168b80','#cb8c35','#5c79a0','#946aa5','#697b87']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(value):
    return html.escape(str(value))


def table(headers, rows):
    return ('<div class="scroll"><table><thead><tr>'
            + ''.join('<th>'+text(h)+'</th>' for h in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>'+''.join('<td>'+text(v)+'</td>' for v in row)+'</tr>' for row in rows)
            + '</tbody></table></div>')


def minute_arrays(metric, start, duration):
    count = math.ceil(duration/60)
    busy = np.zeros((len(metric['per_node']),4,count))
    covered = np.zeros((len(metric['per_node']),count))
    widths = np.minimum(60, duration-np.arange(count)*60)
    for n,node in enumerate(metric['per_node']):
        for sample in node['trace']:
            lo=max(0,sample['timestamp']-start)
            hi=min(duration,lo+sample['duration'])
            for b in range(max(0,int(lo//60)),min(count,math.ceil(hi/60))):
                overlap=max(0,min(hi,(b+1)*60)-max(lo,b*60))
                busy[n,:,b] += np.asarray(sample['gpu_busy'])*overlap/widths[b]
                covered[n,b] += overlap/widths[b]
    return busy,covered


def experiment_costs(experiment, successful_ids, sources):
    """Retain costs of unsuccessful owned trials outside the success comparison."""
    if experiment is None:
        return None
    experiment=Path(experiment).resolve()
    ledger=experiment/'experiment.json'
    owned=read(ledger)['owned_job_ids']
    assert len(owned)==len(set(owned))
    sources[str(ledger)]=sha(ledger)
    receipts={}
    for path in experiment.glob('*/submission.json'):
        receipt=read(path)
        if receipt['job_id'] in owned:
            assert receipt['job_id'] not in receipts
            receipts[receipt['job_id']]=(path,receipt)
    assert set(receipts)==set(owned),'An owned submission receipt is missing'
    rows=[]
    for job_id in owned:
        path,receipt=receipts[job_id]
        plan=path.parent/'plan.json'
        assert read(plan)['job_name']==receipt['job_name']
        for source in [path,plan]:sources[str(source)]=sha(source)
        accounting=allocation_attempts(job_id)
        cancellation=path.parent/'cancellation-intent.json'
        note='Audited comparison trial' if job_id in successful_ids else 'Awaiting or excluded from audited comparison'
        if cancellation.exists():
            sources[str(cancellation)]=sha(cancellation)
            intent=read(cancellation)
            note=intent.get('action') or intent.get('reason') or 'Cancellation intent recorded; original metadata retained'
        rows.append({'job_id':job_id,'trial':path.parent.name,
                     'included_as_successful_trial':job_id in successful_ids,
                     'note':note,
                     'allocation_attempts':accounting['allocation_attempts'],
                     'allocated_GPU_hours':accounting['total_allocated_GPU_seconds']/3600,
                     'accounting':accounting})
    return {'snapshot_at':time.time(),'jobs':rows,
            'total_allocated_GPU_hours':sum(row['allocated_GPU_hours'] for row in rows),
            'scope':'All receipt-owned jobs in this experiment, including failed/cancelled diagnostics and requeues. Historical reference costs are separate. Pending time contributes no GPU-hours; live-job costs are a snapshot.'}


def plan_change_reviews(runs, output, sources):
    """Export exactly hash-matched, receipt-owned declared-change reviews.

    This records a declared change's scope; numerical science verification is
    separate. Existing campaign files and earlier report sources stay intact.
    """
    records = []
    for run in runs:
        plan = run['plan']
        sections = [(key, plan[key]) for key in
                    ['postqualification_control_plane_guidance', 'independent_verifier_compatibility', 'qualified_registry_index', 'controller_optimization_review', 'qualified_controller_optimization']
                    if key in plan]
        if not sections:
            continue
        ledger = Path(plan['ownership_ledger']).resolve(strict=True)
        job_id = run['receipt']['job_id']
        assert job_id.isdigit() and job_id in read(ledger)['owned_job_ids']
        for kind, section in sections:
            path = Path(section['review']).resolve(strict=True)
            assert path.is_relative_to(ledger.parent) and path.suffix == '.json'
            assert sha(path) == section['review_sha256'], 'Postqualification review hash changed'
            read(path)  # The copied artifact must itself be valid JSON.
            scope = section['scope']
            assert isinstance(scope, str) and scope
            artifact = f'reviews/job-{job_id}-{kind}.json'
            destination = output / artifact
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(path.read_bytes())
            sources[str(path)] = section['review_sha256']
            sources[str(ledger)] = sha(ledger)
            records.append({'job_id': job_id, 'kind': kind, 'scope': scope,
                            'artifact': artifact, 'source_sha256': section['review_sha256'],
                            'verification_scope': 'Review identity and owned directory boundary; scientific checks remain separate.'})
    return records


from saturation_control_plane_figures import import_verified_control_plane_figures


def canonical_campaign_root(job, campaign):
    physical_job=Path(job).resolve(strict=True)
    physical_campaign=Path(campaign).resolve(strict=True)
    assert physical_campaign==physical_job/'campaign', 'Report campaign is outside the exact owned job directory'
    return physical_campaign


def build(trials, output, cache_files, node_plots=True, experiment=None):
    output.mkdir(parents=True,exist_ok=True)
    figures_dir=output/'figures';figures_dir.mkdir(exist_ok=True)
    cache={};gates={}
    for path in cache_files:
        data=read(path)
        for key in ('current','baseline'):
            if key in data:cache[data[key]['summary']['job_id']]=data[key]
        if 'current' in data and 'principle_gate' in data:
            gates[data['current']['summary']['job_id']]={k:data[k] for k in ['principle_gate','principle_gate_passed','final_gate','final_gate_passed']}
    sources={str(path.resolve()):sha(path) for path in cache_files}
    runs=[]
    for label,folder in trials:
        job=Path(folder).resolve()
        run=load_run(job)
        run['root']=canonical_campaign_root(job,run['root'])
        job_id=run['receipt']['job_id']
        if (job/'saturation-audit.json').exists():
            path=job/'saturation-audit.json';gate=read(path);assert gate['current']['summary']['job_id']==job_id
            cache[job_id]=gate['current'];gates[job_id]={k:gate[k] for k in ['principle_gate','principle_gate_passed','final_gate','final_gate_passed']}
            sources[str(path)]=sha(path)
        metric=cache[job_id] if job_id in cache else measure(job)
        assert metric['summary']['nodes']==run['plan']['nodes']
        assert metric['summary']['screen_trials']==len(run['screens'])
        assert metric['summary']['full_context_trials']==len(run['full'])
        assert abs(metric['summary'].get('final_allocation_seconds',metric['summary']['allocated_seconds'])-run['seconds'])<.01
        tested=reviewed_science_threads(run['audit'],run['rounds'])
        assessed=completed_assessment_threads(run['rounds'],tested)
        run.update(label=label,metric=metric,reviewed_science_threads=tested,completed_assessment_threads=assessed)
        run['summary'].update(metric['summary'],label=label,
                              top_k=run['plan'].get('admission',{}).get('top_k'),
                              concurrency=run['plan'].get('admission',{}).get('concurrency'))
        sm=run['summary'];sm['GPU_hours']=sm['allocated_GPU_hours'];sm['allocation_seconds']=sm['allocated_seconds']
        sm['hypotheses_with_verified_models']=len({row['hypothesis'] for key in ['candidate_rows','intact_followups'] for row in run['audit'].get(key,[])})
        sm['hypotheses_tested']=len({h for n,h in assessed})
        sm['reviewed_scientific_test_threads']=len(assessed)
        sm['reviewed_scientific_test_hypotheses']=sm['hypotheses_tested']
        sm['reviewed_fresh_science_threads']=len(tested)
        sm['hypotheses_with_reviewed_fresh_science']=len({h for n,h in tested})
        sm['completed_scientific_assessment_threads']=len(assessed)
        sm['files_per_hour']=5*(len(run['screens'])+len(run['full']))*3600/sm['allocated_seconds']
        sm['files_per_GPU_hour']=5*(len(run['screens'])+len(run['full']))/sm['allocated_GPU_hours']
        runs.append(run)
        for path in [job/'plan.json',job/'submission.json',job/'allocation.json',job/'status.json',
                     run['root']/'campaign.yaml',run['root']/'scientific-protocol.json',
                     run['root']/'server-ready.json',run['root']/'controller-watch.json',
                     run['root']/'scaling-audit.json',*sorted((run['root']/'state/rounds').glob('round-*.json'))]:
            sources[str(path)]=sha(path)
        for name in ['control-plane-at-start.json','claim-source-resolutions.json','operator-reviewed-source-aliases.json','independent-claim-review.json']:
            path=run['root']/name
            if path.exists():sources[str(path)]=sha(path)
        lineage=run['audit'].get('verification_lineage',{})
        for name in ['numeric_audit','claim_audit']:
            if name in lineage:
                path=Path(lineage[name]).resolve()
                assert path.is_relative_to(run['root']) and sha(path)==lineage[name+'_sha256']
                sources[str(path)]=sha(path)
        for record in run['rounds'].values():
            receipt=record.get('brief',{}).get('synthesis_receipt')
            if receipt:
                directory=Path(receipt).parent.resolve()
                assert directory.is_relative_to(run['root'])
                for path in directory.glob('*.json'):sources[str(path)]=sha(path)
        for batch in run['audit']['batch_states']:
            if batch.get('tool')=='scale20_summarize_binder_pipeline':
                path=run['root']/'batches'/batch['batch_id']/'inputs.jsonl'
                sources[str(path)]=sha(path)
    gates={r['receipt']['job_id']:gates[r['receipt']['job_id']] for r in runs if r['receipt']['job_id'] in gates}
    costs=experiment_costs(experiment,{r['receipt']['job_id'] for r in runs},sources)
    change_reviews=plan_change_reviews(runs,output,sources)
    figures=[]
    def save(fig,slug,title,caption,group='overview'):
        fig.savefig(figures_dir/(slug+'.png'),dpi=145,bbox_inches='tight')
        fig.savefig(figures_dir/(slug+'.svg'),bbox_inches='tight')
        plt.close(fig)
        figures.append({'id':slug,'title':title,'caption':caption,'group':group})

    labels=[r['label'] for r in runs];x=np.arange(len(runs))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for offset,key,name,color in [(-.18,'screen_trials_per_allocated_GPU_hour','Distinct screens','#168b80'),
                                 (.18,'full_context_trials_per_allocated_GPU_hour','Full-context trials','#cb8c35')]:
        axes[0].bar(x+offset,[r['summary'][key] for r in runs],.36,label=name,color=color)
    axes[0].set(ylabel='Verified trials / allocated GPU-hour',title='Scientific resource efficiency')
    axes[0].legend()
    axes[1].bar(x,[r['summary']['files_per_hour'] for r in runs],color=COLORS[:len(runs)])
    axes[1].set(ylabel='Retained coordinate files / allocation hour',title='Whole-job output throughput')
    for ax in axes:
        ax.set_xticks(x,labels,rotation=12);ax.grid(axis='y',alpha=.2)
    save(fig,'throughput','Scientific throughput and GPU-hour efficiency',
         'Earliest successful distinct-input trials only. Five retained model files per trial are correlated predictions. Full parent-job time includes startup, tournament, drain and cleanup.')

    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for i,(role,color) in enumerate([('science','#168b80'),('inference','#cb8c35'),('overall','#5c79a0')]):
        axes[0].bar(x+(i-1)*.25,[r['summary'][role+'_GPU_busy_percent'] for r in runs],.25,label=role,color=color)
    axes[0].axhline(50,color='#555',ls='--',label='50% final target')
    axes[0].set(ylabel='GPU busy (%) over full allocated time',ylim=(0,100));axes[0].legend()
    axes[1].bar(x,[100*r['summary']['fraction_time_below_10percent_science_slot_occupancy'] for r in runs],color=COLORS[:len(runs)])
    axes[1].axhline(30,color='#555',ls='--',label='20-node gate: at most 30%');axes[1].legend()
    axes[1].set(ylabel='Time below 10% science-slot occupancy (%)',ylim=(0,100))
    for ax in axes:ax.set_xticks(x,labels,rotation=12);ax.grid(axis='y',alpha=.2)
    save(fig,'utilization','GPU busy and science starvation',
         'Overall busy includes every parent allocation attempt, including requeues; unobserved time contributes zero. Science/inference role bars describe the final assigned interval. Coverage is explicit. GPU busy is distinct from row occupancy and FLOP utilization.')

    nodes=[];binders=[];hypotheses=[];screens=[];events=[]
    for index,run in enumerate(runs):
        prefix=f'trial-{index+1:02d}'
        timing_plots(run,prefix,save,output,sources,sha)
        metric=run['metric'];busy,coverage=minute_arrays(metric,run['start'],run['seconds'])
        role_nodes=metric['per_node'];bound=sum(n['role']=='science' for n in role_nodes)
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained',gridspec_kw={'width_ratios':[3,1]})
        matrix=np.ma.masked_where(coverage==0,busy.mean(axis=1))
        palette=plt.get_cmap('viridis').copy();palette.set_bad('#ddd')
        im=axes[0].imshow(matrix,aspect='auto',origin='upper',vmin=0,vmax=100,cmap=palette,
                          extent=[0,run['seconds']/60,len(role_nodes)-.5,-.5])
        axes[0].axhline(bound-.5,color='white',lw=1)
        axes[0].set(xlabel='Minutes since allocation start',ylabel='Node index (science first, inference last)')
        fig.colorbar(im,ax=axes[0],label='Mean of four GPUs (%)')
        for role,color in [('science','#168b80'),('inference','#cb8c35')]:
            nn=[n for n in role_nodes if n['role']==role]
            axes[1].scatter([n['whole_allocation_GPU_busy_lower_bound_percent'] for n in nn],
                            [i for i,n in enumerate(role_nodes) if n['role']==role],s=12,label=role,color=color)
        axes[1].set(xlabel='Final allocation GPU busy (%)',ylim=(len(role_nodes)-.5,-.5),xlim=(0,100));axes[1].legend()
        save(fig,prefix+'-nodes','Every-node utilization: '+run['label'],
             'Final assigned allocation interval for every node; earlier requeued allocations remain in aggregate cost. Gray cells lack telemetry. The right panel includes unobserved time within this interval in its denominator.')

        fig,axes=plt.subplots(2,1,figsize=(13,6),sharex=True,layout='constrained')
        minutes=(np.arange(busy.shape[2])+.5)
        for role,color in [('science','#168b80'),('inference','#cb8c35')]:
            subset=[i for i,n in enumerate(role_nodes) if n['role']==role]
            axes[0].plot(minutes,busy[subset].mean(axis=(0,1)),label=role,color=color)
        axes[0].set(ylabel='GPU busy (%)',ylim=(0,100));axes[0].legend()
        trace=metric['occupancy_trace']
        axes[1].step([(p['timestamp']-run['start'])/60 for p in trace],
                     [100*p.get('active_physical_GPUs',p['active_GPU_rows'])/(4*run['plan']['science_count']) for p in trace],where='post',color='#168b80')
        axes[1].set(ylabel='Science slots with a recorded row (%)',xlabel='Minutes since allocation start',ylim=(0,105))
        for ax in axes:ax.grid(alpha=.2)
        save(fig,prefix+'-timeline','GPU activity and occupied science slots: '+run['label'],
             'The full timeline includes startup and final tails. Row occupancy is reconstructed from final leases and process durations; it is not a direct EL master ready-queue measurement.')

        for n,node in enumerate(role_nodes):
            nodes.append({'run':run['label'],**{k:v for k,v in node.items() if k!='trace'}})
            if node_plots:
                fig,ax=plt.subplots(figsize=(11,3.5),layout='constrained')
                for gpu in range(4):ax.plot(minutes,busy[n,gpu],label=f'GPU {gpu}',lw=.9)
                ax.set(xlabel='Minutes since allocation start',ylabel='GPU busy (%)',ylim=(0,100));ax.legend(ncol=4);ax.grid(alpha=.2)
                save(fig,prefix+'-node-'+node['host'],run['label']+' / '+node['host']+' / '+node['role'],
                     f"All four GPUs. Final allocation busy {node['whole_allocation_GPU_busy_lower_bound_percent']:.1f}%; telemetry coverage {node['coverage_percent']:.1f}%.",'node')

        for candidate in run['screens']:
            screens.append({'run':run['label'],**{k:candidate.get(k) for k in ['round','hypothesis','condition','seed','binder_sequence','best_iptm','summary_path']}})
        for candidate in run['full']:
            pose=candidate['minimized_pose_geometry']
            binders.append({'run':run['label'],**{k:candidate.get(k) for k in ['round','hypothesis','condition','seed','binder_sequence','intact_iptm_best','intact_models_with_dual_region_contact','static_MMGBSA_kcal_mol','summary_path']},
                'minimized_dual_contact':pose['dual_region_contact'],'minimized_binder_RMSD_A':pose['target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A'],
                'close_pairs_lt2A':pose['close_atom_pairs_lt2A']})
        for number,record in sorted(run['rounds'].items()):
            for thread in record['threads']:
                h=thread['hypothesis'];report=thread.get('report') or {}
                hypotheses.append({'run':run['label'],'round':number,'id':h['id'],'summary':h['summary'],'elo':h['elo'],
                    'outcome':report.get('outcome','untested'),'accepted_claims':len(thread.get('accepted',[])),
                    'reviewed':thread.get('review') is not None and not thread.get('error'),'error':thread.get('error')})
                hypotheses[-1]['reviewed_fresh_science']=(number,h['id']) in run['reviewed_science_threads']
                hypotheses[-1]['completed_scientific_assessment']=(number,h['id']) in run['completed_assessment_threads']
        events.extend({'run':run['label'],**e} for e in run['events'])
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for arm,color in ARMS.items():
            selected=[b for b in binders if b['run']==run['label'] and b['condition']==arm]
            if not selected:continue
            values=[b['intact_iptm_best'] for b in selected]
            axes[0].hist(values,bins=np.linspace(0,1,26),histtype='step',density=True,color=color,label=f'{arm} (n={len(selected)})',lw=1.6)
            axes[1].scatter(values,[b['static_MMGBSA_kcal_mol'] for b in selected],color=color,s=15,alpha=.45)
            axes[2].scatter([b['minimized_binder_RMSD_A'] for b in selected],
                            [b['intact_models_with_dual_region_contact'] for b in selected],color=color,s=15,alpha=.45)
        axes[0].set(xlabel='Best full-context ipTM',ylabel='Density',xlim=(0,1));axes[0].legend(fontsize=8)
        axes[1].set(xlabel='Best full-context ipTM',ylabel='Static MM/GBSA (kcal/mol)')
        axes[2].set(xlabel='Minimized binder RMSD to screen (Å)',ylabel='Poses contacting both regions (of 5)',ylim=(-.2,5.2))
        for ax in axes:ax.grid(alpha=.2)
        save(fig,prefix+'-science','Binder confidence, energy and geometry: '+run['label'],
             'EGFR residues 25–646 (622 residues) cofolded with binders of 60 residues, followed by minimized-pose checks. ipTM is prediction confidence; static MM/GBSA is a computational score, neither is measured binding affinity. Adaptive hypotheses and five poses are not independent biological replicates.','science')

    diagnostics=resource_plots(experiment,save,output,sources,sha)
    control_figures,control_records=import_verified_control_plane_figures(experiment,output,sources)
    figures.extend(control_figures)
    data={'generated_at':time.time(),'operator_diagnostics':diagnostics,'runs':[r['summary'] for r in runs],'gates':gates,'nodes':nodes,'screen_trials':screens,
          'full_context_trials':binders,'hypotheses':hypotheses,'production_events':events,
          'methods':{'cost':'Every parent-allocated GPU-hour, including requeued boot attempts and startup/drain; internal job steps are not counted twice. Per-node/role plots describe the final assigned interval, while overall utilization and throughput include all attempts.',
                     'deduplication':'Earliest successful screen condition/input SHA/seed/target sequence; full-context binder sequence/seed/target length.',
                     'utilization':'Time-weighted sampled GPU busy over the full allocation; missing intervals contribute zero and coverage is explicit. GPU busy is not FLOP utilization.',
                     'science':'Target context: EGFR UniProt P00533 residues 25–646 (622 residues), with binders of 60 residues. Full-context candidates are selected adaptively from cropped screens. Confidence, geometry and static energy are computational metrics, not experimental affinity. Five poses and adaptive hypotheses are correlated. Accepted claims and hypothesis outcomes are agent judgments; citation ownership does not independently validate every narrative number or mechanism. Binder distributions use numerical artifacts rather than prose claims.',
                     'hypothesis_counts':'Reviewed fresh-science threads have no orchestration error and contain an accepted claim on an independently verified scientific batch from that round. Completed scientific assessments additionally have supported, contradicted or inconclusive outcomes; blocked outcomes remain visible but are excluded from completed test counts. Reused prior-round evidence alone does not establish a new test. Valid numerical binder outputs from failed, blocked or unreviewed threads remain in model-output throughput.'}}
    data['control_plane_diagnostics']=control_records
    data['production_event_time_resolutions']=[{'run':run['label'],**item} for run in runs for item in run.get('event_time_resolutions',[])]
    data['methods']['production_timing']='Canonical production timelines use audited row timing. A missing screen timestamp may resolve only to the timed same-hypothesis, same-round, same-input/seed/binder production of five byte-identical coordinates; every such resolution is retained in data.json. Copied outputs do not add a scientific trial or extra production event.'
    from report_production_events import __file__ as event_helper_source
    sources[str(Path(event_helper_source).resolve())]=sha(Path(event_helper_source))
    if costs is not None:
        data['experiment_costs']=costs
        write_csv(output/'experiment-costs.csv', [{k:v for k,v in row.items() if k!='accounting'} for row in costs['jobs']])
    data['postqualification_reviews']=change_reviews
    for review in change_reviews:
        data['methods'][review['job_id']+'_'+review['kind']]=f"Job {review['job_id']}: {review['scope']}"
    (output/'control-source-reviews.json').write_text(json.dumps(change_reviews,indent=2)+'\n')
    (output/'data.json').write_text(json.dumps(data,indent=2)+'\n')
    for filename,rows in [('throughput',data['runs']),('node-utilization',nodes),('screen-trials',screens),
                          ('full-context-trials',binders),('hypotheses',hypotheses),('production-events',events)]:
        write_csv(output/(filename+'.csv'),rows)
    sequences=sorted({s['binder_sequence'] for s in screens})
    (output/'candidate-sequences.fasta').write_text(''.join('>binder-'+hashlib.sha256(s.encode()).hexdigest()[:16]+'\n'+s+'\n' for s in sequences))

    scenes=[]
    latest=runs[-1]['summary']
    s=Slide('Scientific work supply and GPU saturation','Loamlite / Jupiter',
            f"Latest included allocation: {latest['nodes']} nodes, job {latest['job_id']}",notes=json.dumps(data['methods']),dark=True)
    for i,(value,label) in enumerate([(f"{latest['overall_GPU_busy_percent']:.1f}%",'whole-job GPU busy'),
                                     (latest['screen_trials'],'distinct screens'),(latest['full_context_trials'],'full-context trials')]):
        s.card(40+300*i,165,280,value,label)
    s.text(40,280,880,'One parent Slurm job contains disjoint Nemotron and science node pools. Throughput uses verified distinct-input science and the full allocated GPU-hour cost.',20,'#d5e6e9',max_height=125)
    scenes.append(s)
    if costs is not None:
        s=Slide('Total experiment resource cost','All owned trials',notes=json.dumps(costs))
        s.card(40,145,420,f"{costs['total_allocated_GPU_hours']:.1f}",'allocated GPU-hours, including diagnostics')
        s.text(40,245,880,'\n'.join(f"Job {row['job_id']} / {row['trial']}: {row['allocated_GPU_hours']:.1f} GPUh; {row['note']}" for row in costs['jobs']),14,max_height=230)
        scenes.append(s)
    for figure in figures:
        if figure['group']=='node':continue
        s=Slide(figure['title'],{'diagnostic':'Paired resource diagnostic','control-plane':'Operator diagnostic'}.get(figure['group'],'Measured results'),notes=figure['caption'])
        s.image(figures_dir/(figure['id']+'.png'),36,118,888,332)
        s.text(40,458,880,figure['caption'],10.5,max_height=50);scenes.append(s)
    for run in runs:
        for number in sorted(run['rounds']):
            hh=sorted([h for h in hypotheses if h['run']==run['label'] and h['round']==number and h['completed_scientific_assessment']],
                      key=lambda h:(h['reviewed_fresh_science'],h['accepted_claims'],h['elo']),reverse=True)[:3]
            earliest={}
            for b in binders:
                if b['run']==run['label'] and b['round']==number:earliest.setdefault(b['binder_sequence'],b)
            unique=sorted(earliest.values(),key=lambda b:(b['minimized_dual_contact'],b['intact_models_with_dual_region_contact'],b['intact_iptm_best']),reverse=True)
            s=Slide(f'Round {number}: hypotheses and binders',run['label'],
                    'Hypotheses: completed assessments with reviewed fresh science, claims, then Elo. Binders: earliest retained trial per sequence, contacts, then ipTM.',
                    notes=json.dumps({'hypotheses':hh,'binders':unique[:3]}))
            s.text(40,143,425,'Leading hypothesis threads',16,kind='bold')
            s.text(490,143,425,'Full-context binder candidates',16,kind='bold')
            for i,h in enumerate(hh):
                s.text(40,180+100*i,425,h['summary'],13,max_height=65)
                s.text(40,247+100*i,425,f"Elo {h['elo']:.0f}; {h['accepted_claims']} accepted claims; {h['outcome']}",10,max_height=28)
            for i,b in enumerate(unique[:3]):
                s.text(490,180+100*i,425,f"{b['condition']}: ipTM {b['intact_iptm_best']:.3f}, both regions {b['intact_models_with_dual_region_contact']}/5; MM/GBSA {b['static_MMGBSA_kcal_mol']:.1f}",12,max_height=44)
                s.text(490,226+100*i,425,b['binder_sequence'],10,kind='mono',max_height=44)
            scenes.append(s)
    count=export(scenes,output)

    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Loamlite / work supply and saturation</title><style>body{margin:0;background:#f3f6f7;color:#183345;font:16px system-ui}main{max-width:1280px;margin:auto;padding:28px}a{color:#087e78}nav{display:flex;gap:20px;flex-wrap:wrap}figure{background:white;padding:18px;margin:22px 0;border-radius:10px}figure img{width:100%;height:auto}figcaption{color:#526777;font-size:14px}.scroll{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left;max-width:600px;overflow-wrap:anywhere}th{background:#e4f1ef;position:sticky;top:0}select,input{font:inherit;padding:9px;max-width:100%}.node{display:none}.node.active{display:block}</style><main><h1>Loamlite: scientific work supply and GPU saturation</h1>',
           '<nav><a href="scaleup.pptx">PowerPoint</a><a href="scaleup.pdf">PDF</a><a href="slides.html">Browser slides</a><a href="throughput.csv">Throughput CSV</a><a href="node-utilization.csv">Node metrics</a><a href="data.json">Evidence data</a><a href="candidate-sequences.fasta">Binder sequences</a></nav>',
           '<p>Campaign comparison tables use completed allocations and independently verified scientific outputs. Historical references and new trials retain their own labels and job IDs. Metrics include startup and drain. Separately labelled resource diagnostics contain paired replays.</p>',
           table(['Trial','Job','Nodes N:S','Screens','Full-context','Screens/GPUh','Full/GPUh','Science GPU busy','Overall GPU busy','Min node coverage'],
                 [[r['label'],r['job_id'],f"{r['inference']}:{r['science']}",r['screen_trials'],r['full_context_trials'],f"{r['screen_trials_per_allocated_GPU_hour']:.3f}",f"{r['full_context_trials_per_allocated_GPU_hour']:.3f}",f"{r['science_GPU_busy_percent']:.1f}%",f"{r['overall_GPU_busy_percent']:.1f}%",f"{r['minimum_node_telemetry_coverage_percent']:.1f}%"] for r in data['runs']])]
    parts.append('<h2>Hypothesis testing</h2><p>Verified model production and completed, reviewed hypothesis tests are counted separately.</p>')
    parts.append(table(['Trial','Unique hypotheses with verified models','Unique hypotheses with reviewed fresh science','Unique completed assessments','Completed current-round assessment threads'],
                       [[r['label'],r['hypotheses_with_verified_models'],r['hypotheses_with_reviewed_fresh_science'],r['hypotheses_tested'],r['completed_scientific_assessment_threads']] for r in data['runs']]))
    if costs is not None:
        parts.extend(['<h2>Total experiment cost, including unsuccessful trials</h2><p>'+text(costs['scope'])+'</p>',
                      table(['Job','Trial','Included in success comparison','Allocated attempts','GPU-hours','Note'],
                            [[row['job_id'],row['trial'],row['included_as_successful_trial'],row['allocation_attempts'],f"{row['allocated_GPU_hours']:.3f}",row['note']] for row in costs['jobs']]),
                      '<p>Total: '+f"{costs['total_allocated_GPU_hours']:.3f}"+' allocated GPU-hours. <a href="experiment-costs.csv">Cost CSV</a></p>'])
    def figure_tag(f):
        return '<figure><h3>'+text(f['title'])+'</h3><a href="figures/'+text(f['id'])+'.svg"><img loading="lazy" src="figures/'+text(f['id'])+'.png" alt="'+text(f['title'])+'"></a><figcaption>'+text(f['caption'])+'</figcaption></figure>'
    if gates:
        parts.append('<h2>Predeclared scale-up gates</h2>')
        parts.append(table(['Job','Gate','Check','Passed'],[[job,kind,key,value] for job,g in gates.items() for kind in ['principle_gate','final_gate'] for key,value in g[kind].items()]))
    diagnostic_heading=False;control_plane_heading=False
    for f in figures:
        if f['group']=='node':continue
        if f['group']=='diagnostic' and not diagnostic_heading:
            parts.append('<h2>Paired resource diagnostics</h2><p>These replay experiments explain the launch changes. Their duplicate designs do not enter campaign throughput or tested-hypothesis counts.</p>')
            diagnostic_heading=True
        if f['group']=='control-plane' and not control_plane_heading:
            parts.append('<h2>Registry and retired-allocation diagnostics</h2><p>The static metadata benchmark and retired incomplete allocation retain their own scopes and do not enter completed-campaign throughput.</p>')
            control_plane_heading=True
        parts.append(figure_tag(f))
    node_figures=[f for f in figures if f['group']=='node']
    if node_figures:
        parts.append('<h2>Inspect each node and its four GPUs</h2><select id="node-select" aria-label="Select node">')
        parts.extend('<option value="node-'+str(i)+'">'+text(f['title'])+'</option>' for i,f in enumerate(node_figures));parts.append('</select>')
        parts.extend('<div class="node'+(' active' if i==0 else '')+'" id="node-'+str(i)+'">'+figure_tag(f)+'</div>' for i,f in enumerate(node_figures))
    parts.append('<h2>Hypothesis results by iteration</h2><p>Tournament rank and hypothesis outcomes are agent judgments. Accepted-claim counts include supporting, contradicting and contextual evidence; file ownership does not attest every narrative number or mechanism. Binder metrics below come from numerical artifacts.</p>')
    parts.append(table(['Trial','Round','Hypothesis','Elo','Outcome','Accepted claims','Reviewed','Reviewed fresh science','Completed assessment','Error'],
                       [[h[k] for k in ['run','round','summary','elo','outcome','accepted_claims','reviewed','reviewed_fresh_science','completed_scientific_assessment','error']] for h in hypotheses]))
    parts.append('<h2>Full-context binder metrics</h2><p>Display order favors retained minimized contacts and five-pose contact consistency before ipTM. Full results remain in the <a href="full-context-trials.csv">CSV</a>.</p><input id="binder-filter" aria-label="Filter binders" placeholder="Filter sequence, condition, trial or hypothesis"><div id="binders">')
    parts.append(table(['Trial','Round','Condition','Sequence','ipTM','Both-region poses','Minimized both regions','RMSD Å','Static MM/GBSA'],
                       [[b['run'],b['round'],b['condition'],b['binder_sequence'],f"{b['intact_iptm_best']:.3f}",b['intact_models_with_dual_region_contact'],b['minimized_dual_contact'],f"{b['minimized_binder_RMSD_A']:.2f}",f"{b['static_MMGBSA_kcal_mol']:.1f}"] for b in sorted(binders,key=lambda b:(b['run'],b['round'],-int(b['minimized_dual_contact']),-b['intact_models_with_dual_region_contact'],-b['intact_iptm_best']))]))
    parts.append('</div>')
    if change_reviews:
        parts.append('<h2>Changes recorded after qualification</h2><p>These records disclose the control guidance and verifier compatibility used for each large run. Review identity checks do not replace scientific verification.</p>')
        parts.extend('<p><a href="'+text(review['artifact'])+'">Job '+text(review['job_id'])+' / '+text(review['kind'])+'</a>: '+text(review['scope'])+'</p>' for review in change_reviews)
    parts.append('<h2>Methods and interpretation</h2>'+''.join('<p>'+text(v)+'</p>' for v in data['methods'].values())+'<p><a href="build-manifest.json">Source and export hashes</a></p><script>const nodeSelect=document.getElementById("node-select");if(nodeSelect)nodeSelect.onchange=e=>document.querySelectorAll(".node").forEach(n=>n.classList.toggle("active",n.id===e.target.value));document.getElementById("binder-filter").oninput=e=>document.querySelectorAll("#binders tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()));</script></main></html>')
    (output/'index.html').write_text(''.join(parts))
    for source in [Path(__file__),*[Path(__file__).with_name(name) for name in ['audit_saturation_campaign.py','saturation_accounting.py','scaleup_slides.py','build_unified_report.py','audit_scale20_campaign.py','build_inference_atlas.py','saturation_timing_plots.py']]]:
        sources[str(source.resolve())]=sha(source)
    manifest={'built_at':time.time(),'sources':sources,'figures':figures,'slide_count':count,
              'exports':{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='build-manifest.json'}}
    (output/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():archive.write(path,Path(output.name)/path.relative_to(output))
    return {'report':str(output/'index.html'),'slides':str(output/'scaleup.pptx'),
            'slide_count':count,'figures':len(figures),'node_plots':len(node_figures)}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--trial',nargs=2,action='append',required=True,metavar=('LABEL','JOB'))
    parser.add_argument('--metric-cache',type=Path,action='append',default=[])
    parser.add_argument('--no-node-plots',action='store_true')
    parser.add_argument('--experiment',type=Path,help='Include costs of every owned trial, including unsuccessful diagnostics')
    args=parser.parse_args()
    print(json.dumps(build(args.trial,args.output.resolve(),args.metric_cache,not args.no_node_plots,args.experiment),indent=2))
