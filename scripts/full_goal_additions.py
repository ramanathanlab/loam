"""Scientific stage throughput and source-backed operating recipes for the report."""
from __future__ import annotations
import collections
import html
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np
import yaml

from build_inference_atlas import write_csv
from scaleup_slides import Slide
from full_goal_phases import phase_plots
from full_goal_history import history_plots

STAGES = [('rfd_backbones','RFD backbones'),('mpnn_sequences','MPNN sampled sequences'),
          ('crop_folding_calls','Crop cofold input sets'),('crop_fold_models','Crop fold models'),
          ('full_folding_calls','Full-context cofold input sets'),('full_fold_models','Full-context fold models'),
          ('minimizations','Minimizations'),('mmgbsa','Single-frame MM/GBSA'),
          ('audited_md_trajectories','Audited MD trajectories')]
COLORS = ['#168b80','#cb8c35','#5c79a0','#946aa5','#697b87','#c66f6c','#42728b','#8d8b42','#875438','#555b84','#b16587','#5f8970','#a28450']


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    e=lambda x:html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def recipe_inventory(experiment, sources, sha):
    owned=set(read(experiment/'experiment.json')['owned_job_ids'])
    records=[]
    for path in sorted(experiment.glob('*/submission.json')):
        receipt=read(path)
        if receipt['job_id'] not in owned:
            continue
        job=path.parent
        plan=read(job/'plan.json')
        cfgpath=Path(plan['campaign'])/'campaign.yaml'
        cfg=yaml.safe_load(cfgpath.read_text())
        for p in (path,job/'plan.json',cfgpath):sources[str(p)]=sha(p)
        t=cfg['tournament']
        hpc=cfg.get('execution',{}).get('hpc',{})
        roles={role:{'model':spec['model'],'max_tokens':spec.get('settings',{}).get('max_tokens'),
                     'thinking':spec.get('settings',{}).get('extra_body',{}).get('chat_template_kwargs',{}).get('enable_thinking')}
               for role,spec in cfg['models'].items()}
        records.append({'trial':job.name,'job_id':receipt['job_id'],'nodes':plan['nodes'],
                        'account':plan.get('account'),'partition':plan.get('partition'),
                        'reservation':plan.get('reservation'),
                        'job_script':str(job/'job.sh'),'job_script_sha256':sha(job/'job.sh'),
                        'inference':plan['inference_count'],'science':plan['science_count'],
                        'concurrency':cfg['concurrency'],'top_k':t['top_k'],
                        'initial_hypotheses':t['initial_hypotheses'],'new_per_round':t['new_per_round'],
                        'generation_batch_size':t['gen_batch_size'],'scheme':t['scheme'],
                        'swiss_rounds':t['swiss_rounds'],'tournament_workers':t['workers'],
                        'synthesis_concurrency':cfg.get('synthesis_concurrency',plan.get('synthesis_concurrency')),
                        'science_workers_per_gpu':hpc.get('science_workers_per_gpu',1),
                        'physical_science_GPUs':4*plan['science_count'],
                        'logical_science_worker_slots':4*plan['science_count']*hpc.get('science_workers_per_gpu',1),
                        'hypotheses_per_science_node':cfg['concurrency']/plan['science_count'],
                        'admission_waves':t['top_k']/cfg['concurrency'],
                        'tournament_max_seconds':t['max_seconds'],'timeouts':cfg['timeouts'],
                        'model_startup_timeout':plan['model_startup_timeout'],
                        'hpc_startup_timeout':hpc.get('startup_timeout'),
                        'mcp_init_timeout_seconds':plan.get('mcp_init_timeout_seconds'),
                        'controller_gc_threshold0':plan.get('controller_gc_threshold0'),
                        'model_retries':cfg.get('model_retries'),'max_requests':cfg.get('max_requests'),
                        'max_code_calls':cfg.get('max_code_calls'),'roles':roles,
                        'walltime':plan['walltime'],'config_source':str(cfgpath),
                        'config_sha256':sha(cfgpath)})
        sources[str(job/'job.sh')]=sha(job/'job.sh')
    assert {x['job_id'] for x in records}==owned
    return records


def additions(runs,experiment,save,output,sources,sha,costs):
    experiment=Path(experiment).resolve(strict=True)
    recipes=recipe_inventory(experiment,sources,sha)
    budget_path=experiment/'diagnostics/report-batch-budgets.json'
    budget_inventory=read(budget_path)
    sources[str(budget_path)]=sha(budget_path)
    for path,digest in budget_inventory['source_sha256'].items():
        assert sha(Path(path))==digest,path
        sources[path]=digest
    write_csv(output/'science-row-budgets.csv',budget_inventory['rows'])
    stages=[];events=[];node_rows=[];inventories=[]
    for run in runs:
        suffix=run['job'].name.removeprefix('unified-')
        inventory_path=experiment/'diagnostics'/f'report-stage-inventory-{suffix}.json'
        inv=read(inventory_path)
        assert inv['job_id']==run['receipt']['job_id'] and inv['audit_sha256']==sha(run['audit_path'])
        sources[str(inventory_path)]=sha(inventory_path)
        witness=Path(inv['witnesses']['path'])
        if not witness.is_absolute():
            # Earlier direct collectors were invoked from the workspace,
            # whereas the report can be invoked from the repository.
            witness=experiment.parents[2]/witness
        witness=witness.resolve(strict=True)
        assert witness==inventory_path.with_suffix('.witnesses.jsonl').resolve(strict=True)
        assert sha(witness)==inv['witnesses']['sha256']
        sources[str(witness)]=sha(witness)
        for path,digest in inv['sources'].items():
            assert sha(Path(path))==digest,path
            sources[path]=digest
        data=[json.loads(line) for line in witness.read_text().splitlines()]
        assert len(data)==inv['witnesses']['rows']
        counted=collections.Counter()
        timing={(x['batch_id'],x['row']):x for x in run['audit']['row_timings']}
        host_resolutions={(x['canonical_batch_id'],x['canonical_row']):x['source_recorded_timing']
                          for x in inv['event_time_resolutions']}
        bynode=collections.defaultdict(collections.Counter)
        for row in data:
            counted.update(row['counts'])
            key=(row['batch_id'],Path(row['summary_path']).parent.name)
            host=timing[key]['host'].split('.')[0]
            host_source='row_timing'
            if not host and key in host_resolutions:
                host=host_resolutions[key]['host'].split('.')[0]
                host_source='native_exact_identity_resolution'
            bynode[host].update(row['counts'])
            events.append({'run':run['label'],'job_id':inv['job_id'],'host':host,
                           'host_source':host_source,'round':row['round'],'time':row['completion_time'],
                           **{k:row['counts'].get(k,0) for k,title in STAGES}})
        assert all(counted[k]==v for k,v in inv['stage_counts'].items())
        assert inv['canonical_crop_input_sets']==len(run['screens'])
        assert inv['canonical_full_input_sets']==len(run['full'])
        gpu_hours=run['metric']['summary']['allocated_GPU_hours']
        parent_hours=run['metric']['summary']['allocated_seconds']/3600
        science_node_hours=run['seconds']*run['plan']['science_count']/3600
        for key,title in STAGES:
            count=inv['stage_counts'][key]
            durations=inv['stage_seconds'].get(key,[])
            stages.append({'run':run['label'],'job_id':inv['job_id'],'stage':key,'unit':title,
                           'count':count,'per_parent_allocation_hour':count/parent_hours,
                           'per_allocated_GPU_hour':count/gpu_hours,
                           'per_assigned_science_node_hour':count/science_node_hours,
                           'per_assigned_science_GPU_hour':count/(4*science_node_hours),
                           'timing_samples':len(durations),
                           'median_stage_wall_seconds':statistics.median(durations) if durations else None,
                           'p95_stage_wall_seconds':float(np.quantile(durations,.95)) if durations else None,
                           'stage_witness_issues':sum(x['stage']==key for x in inv['issues']),
                           'campaign_qualified':run['summary']['qualified_campaign']})
        for node in run['metric']['per_node']:
            if node['role']!='science':continue
            counts=bynode[node['host']]
            node_rows.append({'run':run['label'],'job_id':inv['job_id'],'host':node['host'],
                              'GPU_busy_percent':node['whole_allocation_GPU_busy_lower_bound_percent'],
                              **{k:counts[k] for k,title in STAGES},
                              'rfd_per_node_hour':counts['rfd_backbones']/(run['seconds']/3600),
                              'mmgbsa_per_node_hour':counts['mmgbsa']/(run['seconds']/3600)})
        inventories.append({'run':run['label'],'job_id':inv['job_id'],'path':str(inventory_path),
                            'sha256':sha(inventory_path),'issues':inv['issues'],
                            'excluded':inv['excluded'],'scope':inv['scope'],
                            'md_classification':inv['md_classification']})
    write_csv(output/'science-stage-throughput.csv',stages)
    write_csv(output/'science-stage-events.csv',events)
    write_csv(output/'science-node-throughput.csv',node_rows)
    write_csv(output/'run-recipes.csv',[{k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in row.items()} for row in recipes])
    labels=[r['label'] for r in runs]
    selected=[('rfd_backbones','RFD backbones'),('mpnn_sequences','MPNN sequences'),
              ('crop_fold_models','Crop fold models'),('full_fold_models','Full-context fold models'),
              ('mmgbsa','Single-frame MM/GBSA'),('audited_md_trajectories','Audited MD trajectories')]
    for metric,slug,title,axis in [
        ('count','stage-counts','Scientific output by stage','Distinct-input witnessed outputs'),
        ('per_parent_allocation_hour','stage-rate-parent','Scientific throughput across node counts','Outputs / whole-parent allocation hour'),
        ('per_assigned_science_node_hour','stage-rate-science-node','Productivity of the science-node allocation','Outputs / assigned science-node hour')]:
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for ax,(key,name) in zip(axes.flat,selected):
            rows=[next(s for s in stages if s['run']==label and s['stage']==key) for label in labels]
            ax.bar(range(len(labels)),[s[metric] for s in rows],color=COLORS[:len(labels)])
            ax.set(title=name,ylabel=axis);ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right',fontsize=8)
            ax.grid(axis='y',alpha=.2)
            if key=='audited_md_trajectories':ax.set_ylim(0,1);ax.text(.5,.6,'No production MD in the audited route',transform=ax.transAxes,ha='center',fontsize=10)
        save(fig,slug,title,'Counts require artifact witnesses on canonical distinct inputs. Native MPNN input sequences, duplicate summaries, and filesystem aliases do not add designs. Five correlated fold poses are models, not five binders. The energy route minimizes one pose and evaluates one MM/GBSA frame; it does not run production MD. Partial/unqualified cases retain status labels. Science-node rates use the final assigned science-node interval; whole-parent GPU-hour costs retain all attempts.','scientific-throughput')
    latency_keys=['rfd_backbones','mpnn_sequences','crop_folding_calls','full_folding_calls','mmgbsa']
    fig,axes=plt.subplots(1,5,figsize=(17,5),layout='constrained')
    for ax,key in zip(axes,latency_keys):
        for index,label in enumerate(labels):
            row=next(x for x in stages if x['run']==label and x['stage']==key)
            if not row['timing_samples']:continue
            median=row['median_stage_wall_seconds'];p95=row['p95_stage_wall_seconds']
            ax.plot([median,p95],[index,index],color=COLORS[index],lw=2)
            ax.scatter([median],[index],color=COLORS[index],s=24)
            ax.scatter([p95],[index],color=COLORS[index],s=20,marker='|')
        ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.invert_yaxis()
        ax.set(title=dict(STAGES)[key],xlabel='Recorded stage wall seconds');ax.grid(axis='x',alpha=.2)
    save(fig,'stage-latency','Scientific stage latency: median and 95th percentile',
         'Points are medians and right endpoints the 95th percentile over canonical rows with recorded stage durations. These are wall-clock stage times including software overhead and contention, not GPU kernel times. Missing timings are not imputed. The single-frame MM/GBSA stage timing includes its minimization route. Samples and exact quantiles are exported in science-stage-throughput.csv.','scientific-throughput')
    round_rows=[]
    for run in runs:
        numbers=sorted({x['round'] for x in events if x['run']==run['label']})
        for number in numbers:
            for key,title in STAGES:
                count=sum(x[key] for x in events if x['run']==run['label'] and x['round']==number)
                round_rows.append({'run':run['label'],'job_id':run['receipt']['job_id'],'round':number,
                                   'stage':key,'count':count,'campaign_qualified':run['summary']['qualified_campaign']})
    write_csv(output/'science-stage-by-round.csv',round_rows)
    round_cases=[runs[-1]]
    qualified=[r for r in runs if r['summary']['qualified_campaign']]
    if qualified:
        best=max(qualified,key=lambda r:r['metric']['summary']['overall_GPU_busy_percent'])
        if best not in round_cases:round_cases.insert(0,best)
    for run in round_cases:
        subset=[x for x in round_rows if x['run']==run['label']]
        numbers=sorted({x['round'] for x in subset})
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for ax,key in zip(axes,['rfd_backbones','crop_fold_models','mmgbsa']):
            values=[next(x['count'] for x in subset if x['round']==n and x['stage']==key) for n in numbers]
            ax.bar(range(len(numbers)),values,color='#168b80')
            ax.set_xticks(range(len(numbers)),['Round '+str(n) for n in numbers])
            ax.set(title=dict(STAGES)[key],ylabel='Witnessed distinct-input outputs');ax.grid(axis='y',alpha=.2)
        save(fig,'stage-rounds-'+run['receipt']['job_id'],'Scientific output in each iteration: '+run['label'],
             'Each canonical output is assigned to its retained owning hypothesis round. This is production per iteration, not a randomized comparison of round quality or proof that reused evidence is a new hypothesis test. The stage-by-round CSV contains all stages for every included run.','scientific-throughput')
    for run in runs:
        rr=[x for x in events if x['run']==run['label'] and x['time'] is not None]
        fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained',sharex=True)
        bins=np.append(np.arange(run['start'],run['end'],300),run['end'])
        widths=np.diff(bins);minutes=(bins[:-1]-run['start'])/60
        for index,key in enumerate(['rfd_backbones','mpnn_sequences','crop_fold_models','full_fold_models','mmgbsa']):
            values=np.histogram([x['time'] for x in rr],bins=bins,weights=[x.get(key,0) for x in rr])[0]
            axes[0].step(minutes,np.cumsum(values),where='post',label=dict(STAGES)[key],color=COLORS[index])
            axes[1].step(minutes,values*3600/widths,where='post',label=dict(STAGES)[key],color=COLORS[index])
        axes[0].set(ylabel='Cumulative witnessed outputs');axes[0].legend(fontsize=8,ncol=2)
        axes[1].set(ylabel='Outputs / hour (5-minute bins)',xlabel='Minutes since final allocation start')
        for ax in axes:ax.grid(alpha=.2)
        total_events=sum(x['run']==run['label'] for x in events)
        save(fig,'stage-timeline-'+run['receipt']['job_id'],'Scientific production over time: '+run['label'],
             f'Production timestamps are available for {len(rr)}/{total_events} canonical rows; rows without timing remain unplotted. Stage outputs are credited at the enclosing scientific row completion; exact per-stage completion timestamps were not recorded. The full allocated timeline includes startup, tournament boundaries and drain. Empty periods remain visible. Counts use the native identity-checked timing resolution for copied summaries.','scientific-throughput')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for run_index,run in enumerate(runs):
        rows=[x for x in node_rows if x['run']==run['label']]
        axes[0].scatter([x['GPU_busy_percent'] for x in rows],[x['rfd_per_node_hour'] for x in rows],s=17,alpha=.6,color=COLORS[run_index],label=run['label'])
        axes[1].scatter([x['GPU_busy_percent'] for x in rows],[x['mmgbsa_per_node_hour'] for x in rows],s=17,alpha=.6,color=COLORS[run_index])
    for ax,title in zip(axes,['RFD backbones / science-node hour','MM/GBSA evaluations / science-node hour']):
        ax.set(xlabel='Science-node whole-allocation sampled GPU busy (%)',ylabel=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig,'science-node-productivity','Per-node scientific productivity versus GPU activity',
         'Each point is one assigned science node, including nodes with zero verified production. Counts are attributed by the owning batch lease/status host. Stage wall time and sampled GPU busy measure different things; shared worker slots do not multiply physical GPU capacity.','scientific-throughput')
    for run in {r['receipt']['job_id']:r for r in (runs[0],runs[-1])}.values():
        rows=[x for x in node_rows if x['run']==run['label']]
        keys=['rfd_backbones','mpnn_sequences','crop_fold_models','full_fold_models','mmgbsa']
        matrix=np.array([[x[k] for k in keys] for x in rows],float)/(run['seconds']/3600)
        fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
        image=ax.imshow(np.log1p(matrix),aspect='auto',cmap='viridis')
        ax.set_xticks(range(len(keys)),[dict(STAGES)[k] for k in keys],rotation=15,ha='right')
        ax.set(ylabel='Every assigned science node',title=run['label']+' / per-node scientific throughput')
        fig.colorbar(image,ax=ax,label='log(1 + outputs / assigned node-hour)')
        save(fig,'science-node-stages-'+run['receipt']['job_id'],'Scientific throughput on every science node: '+run['label'],
             'All assigned science nodes are present, including zero-output nodes. Color is log-scaled because stages have different output multiplicities; exact untransformed counts and host names are in science-node-throughput.csv.','scientific-throughput')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for i,run in enumerate(runs):
        recipe=next(x for x in recipes if x['job_id']==run['receipt']['job_id'])
        axes[0].scatter(recipe['hypotheses_per_science_node'],run['metric']['summary']['overall_GPU_busy_percent'],s=65,color=COLORS[i],label=run['label'])
        axes[1].scatter(recipe['admission_waves'],run['metric']['summary']['full_context_trials_per_allocated_GPU_hour'],s=65,color=COLORS[i])
    axes[0].set(xlabel='Active validators / science node',ylabel='Whole-parent sampled GPU busy (%)')
    axes[1].set(xlabel='Selected hypotheses / active validator slots',ylabel='Full-context trials / total allocated GPU-hour')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(ncol=2,fontsize=8,loc='upper left')
    save(fig,'recipe-observations','Observed recipes and their measured outcomes',
         'These are successive adaptive campaigns, not randomized controlled ablations. Settings, controller changes, failures and scientific work differ. The plot describes measured outcomes without assigning every gain to a single parameter.','recipes')
    ratio_context=[]
    ratio_path=experiment.parent.parent/'reports/inference-balance-20261113/comparison.json'
    if ratio_path.exists():
        prior=read(ratio_path)
        sources[str(ratio_path)]=sha(ratio_path)
        fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
        for index,trial in enumerate(prior['trials']):
            assert trial['completion_checks_passed'] and trial['independent_coordinates_checked']
            split=trial['split'];rate=trial['distinct_trial_complex_files_per_science_allocation_hour']
            ax.scatter(split['inference'],rate,s=75,color=COLORS[index])
            ax.annotate(f"{split['inference']}:{split['science']}  {rate:.0f}",
                        (split['inference'],rate),xytext=(6,7),textcoords='offset points',fontsize=10)
            ratio_context.append({'trial':trial['trial'],'inference':split['inference'],'science':split['science'],
                                  'distinct_model_files_per_science_allocation_hour':rate})
        ax.set(xlabel='Nemotron inference nodes within 20 total nodes',
               ylabel='Distinct-input coordinate files / science-allocation hour',
               title='Historical ratio study underlying the 8:12 starting split')
        ax.grid(alpha=.2)
        save(fig,'historical-ratio-study','Historical context: inference/science ratio selection',
             'Five earlier, independently checked separate-allocation trials, including an 8:12 repeat. The denominator is science-allocation time, unlike the current whole-parent GPU-hour metric. These observations informed the 40% inference starting split; they are not a controlled speedup comparison with the single-job campaigns or part of this goal’s 13-job cost ledger. Five model files are correlated poses, not five independent binders.','historical-context')
    winners=[]
    for nodes in sorted({r['plan']['nodes'] for r in runs}):
        candidates=[r for r in runs if r['plan']['nodes']==nodes]
        qualified=[r for r in candidates if r['summary']['qualified_campaign']]
        eligible=qualified or candidates
        best=max(eligible,key=lambda r:r['metric']['summary']['overall_GPU_busy_percent'])
        efficiency=max(eligible,key=lambda r:r['metric']['summary']['full_context_trials_per_allocated_GPU_hour'])
        winners.append({'nodes':nodes,'busy_winner':best['label'],'job_id':best['receipt']['job_id'],
                        'qualified':best['summary']['qualified_campaign'],
                        'overall_GPU_busy_percent':best['metric']['summary']['overall_GPU_busy_percent'],
                        'full_context_per_GPU_hour':best['metric']['summary']['full_context_trials_per_allocated_GPU_hour'],
                        'efficiency_winner':efficiency['label'],
                        'best_full_context_per_GPU_hour':efficiency['metric']['summary']['full_context_trials_per_allocated_GPU_hour']})
    sequence_summary=[{'run':r['label'],'job_id':r['receipt']['job_id'],
                       'canonical_screen_input_sets':len(r['screens']),
                       'unique_screen_binder_sequences':len({x['binder_sequence'] for x in r['screens']}),
                       'canonical_full_input_sets':len(r['full']),
                       'unique_full_binder_sequences':len({x['binder_sequence'] for x in r['full']})} for r in runs]
    write_csv(output/'sequence-production.csv',sequence_summary)
    quality=[]
    quality_rounds=[]
    for run in runs:
        for number in [None]+sorted({x['round'] for x in run['full']}):
            full=[x for x in run['full'] if number is None or x['round']==number]
            crop=[x for x in run['screens'] if number is None or x['round']==number]
            def quantile(values,q):return float(np.quantile(values,q)) if values else None
            result={'run':run['label'],'job_id':run['receipt']['job_id'],'round':number,
                    'full_context_trials':len(full),'canonical_crop_trials':len(crop),
                    'crop_iptm_median':quantile([x['best_iptm'] for x in crop],.5),
                    'full_iptm_p10':quantile([x['intact_iptm_best'] for x in full],.1),
                    'full_iptm_median':quantile([x['intact_iptm_best'] for x in full],.5),
                    'full_iptm_p90':quantile([x['intact_iptm_best'] for x in full],.9),
                    'full_iptm_max':max([x['intact_iptm_best'] for x in full],default=None),
                    'all_five_poses_dual_contact':sum(x['intact_models_with_dual_region_contact']==5 for x in full),
                    'minimized_dual_contact':sum(x['minimized_pose_geometry']['dual_region_contact'] for x in full),
                    'minimized_RMSD_median_A':quantile([x['minimized_pose_geometry']['target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A'] for x in full],.5),
                    'MMGBSA_p10_kcal_mol':quantile([x['static_MMGBSA_kcal_mol'] for x in full],.1),
                    'MMGBSA_median_kcal_mol':quantile([x['static_MMGBSA_kcal_mol'] for x in full],.5),
                    'MMGBSA_p90_kcal_mol':quantile([x['static_MMGBSA_kcal_mol'] for x in full],.9),
                    'campaign_qualified':run['summary']['qualified_campaign']}
            (quality if number is None else quality_rounds).append(result)
    write_csv(output/'binder-metric-summary.csv',quality)
    write_csv(output/'binder-metrics-by-round.csv',quality_rounds)
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    for i,row in enumerate(quality):
        if not row['full_context_trials']:continue
        color=COLORS[i]
        axes[0].plot([row['full_iptm_p10'],row['full_iptm_p90']],[i,i],color=color,lw=2)
        axes[0].scatter([row['full_iptm_median']],[i],color=color,s=28)
        axes[1].barh(i-.15,100*row['all_five_poses_dual_contact']/row['full_context_trials'],height=.28,color='#168b80',label='All 5 predicted poses' if i==0 else None)
        axes[1].barh(i+.15,100*row['minimized_dual_contact']/row['full_context_trials'],height=.28,color='#cb8c35',label='Minimized pose' if i==0 else None)
        axes[2].plot([row['MMGBSA_p10_kcal_mol'],row['MMGBSA_p90_kcal_mol']],[i,i],color=color,lw=2)
        axes[2].scatter([row['MMGBSA_median_kcal_mol']],[i],color=color,s=28)
    for ax in axes:ax.set_yticks(range(len(labels)),labels,fontsize=9);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].set(xlabel='Best full-context ipTM',xlim=(0,1),title='Confidence: median and P10–P90')
    axes[1].set(xlabel='Full-context trials contacting both regions (%)',xlim=(0,100),title='Geometric persistence');axes[1].legend(fontsize=8)
    axes[2].set(xlabel='Static MM/GBSA (kcal/mol)',title='Energy: median and P10–P90')
    save(fig,'binder-quality-summary','Binder confidence, geometric persistence and energy across runs',
         'Full-context trials are adaptively selected and may reuse a sequence with different seeds. The five predictions are correlated poses. Contact persistence requires a heavy-atom distance of at most 4 Å to at least one residue of each predeclared region. More negative static energy is not measured affinity, and none of these computational summaries establishes biological binding. Failed/provenance-flagged campaign outputs remain diagnostic.','science')
    latest_recipe=next(x for x in recipes if x['job_id']==runs[-1]['receipt']['job_id'])
    phases=phase_plots(runs,save,output,sources,sha)
    history=history_plots(runs,recipes,costs,experiment,save,output,sources,sha)
    return {'recipes':recipes,'latest_recipe':latest_recipe,'stage_throughput':stages,'stage_inventories':inventories,'best_observed_recipes':winners,
            'all_owned_job_history':history,
            'allocation_phases':phases,
            'science_row_budgets':budget_inventory['rows'],
            'missing_science_row_budget_metadata':budget_inventory['missing_timeout_metadata'],
            'sequence_production':sequence_summary,
            'binder_metric_summaries':quality,
            'binder_metrics_by_round':quality_rounds,
            'historical_ratio_context':ratio_context,
            'stage_node_rows':len(node_rows),'stage_event_rows':len(events),
            'recipe_scope':'20-node and256-node settings are measured. Interpolation to other node counts is an unqualified starting point, not a demonstrated saturation guarantee.'}


def status_html(runs):
    return '<h2>Campaign qualification and retained limitations</h2><p>Native unresolved-source counts include missing paths and unresolved aliases; they are citation-proof failures, not a measured count of false scientific statements. Numerical coordinate checks and workflow completion are reported separately.</p>'+table(
        ['Trial','Parent state','Recorded rounds','Campaign status','Numeric errors retained','Unresolved native claim sources'],
        [[r['label'],r['parent']['State'],len(r['rounds']),r['summary']['case_status'],
          r['summary']['numeric_errors_retained'],r['summary']['unresolved_claim_sources_retained']] for r in runs])


def stage_summary_html(extra):
    return '<h2>Scientific throughput by stage</h2><p>Counts use witnessed distinct inputs. Cofold input sets and correlated coordinate models are separate units. Energy minimization and single-frame MM/GBSA are separate from production MD. A stage witness issue makes that stage count a conservative witnessed subset; no missing stage is inferred. These are retained, independently checked outputs, not every attempted computation in failed or retried rows; all allocated costs remain.</p>'+table(
        ['Trial','Canonical screen inputs','Unique screen binder sequences','Canonical full-context inputs','Unique full-context binder sequences'],
        [[x['run'],x['canonical_screen_input_sets'],x['unique_screen_binder_sequences'],x['canonical_full_input_sets'],x['unique_full_binder_sequences']] for x in extra['sequence_production']])+table(
        ['Trial','Stage / unit','Count','Per parent hour','Per science-node hour','Per total GPUh','Stage witness issues'],
        [[x['run'],x['unit'],x['count'],f"{x['per_parent_allocation_hour']:.2f}",
          f"{x['per_assigned_science_node_hour']:.3f}",f"{x['per_allocated_GPU_hour']:.3f}",
          x['stage_witness_issues']] for x in extra['stage_throughput']])


def history_html(extra):
    return '<h2>All owned jobs, including aborts and retired trials</h2>'+table(
        ['Trial / job','Nodes I:S','Parent state','Qualified','Whole-parent GPU busy','Telemetry coverage','GPU-hours','Numerical-output analysis'],
        [[x['trial']+'/'+x['job_id'],f"{x['inference']}:{x['science']}",x['parent_state'],x['qualified_campaign'],
          f"{x['overall_GPU_busy_percent']:.2f}%" if x['overall_GPU_busy_percent'] is not None else 'Unavailable',
          f"{x['telemetry_coverage_percent']:.2f}%" if x['telemetry_coverage_percent'] is not None else 'Unavailable',
          f"{x['allocated_GPU_hours']:.2f}",x['scientific_output_analysis_available']] for x in extra['all_owned_job_history']])+table(['Trial/job','Retained retirement or completion record'],[[x['trial']+'/'+x['job_id'],x['retirement_reason']] for x in extra['all_owned_job_history']])


def status_scenes(runs):
    scenes=[]
    for start in range(0,len(runs),8):
        s=Slide('Qualification and retained audit findings','Evidence scope',
                'Numerical model production, workflow completion and claim provenance are separate checks.')
        s.table(40,135,[220,225,90,145,200],
                ['Trial','Campaign status','Rounds','Numeric errors','Unresolved claim sources'],
                [[r['label'],'Qualified' if r['summary']['qualified_campaign'] else 'Diagnostic / not qualified',
                  str(len(r['rounds'])),str(r['summary']['numeric_errors_retained']),
                  str(r['summary']['unresolved_claim_sources_retained'])] for r in runs[start:start+8]],size=11)
        s.text(40,457,880,'Numerically checked outputs from partial or provenance-flagged cases are labeled throughout. Original failed gates and costs remain.',12,max_height=45)
        scenes.append(s)
    return scenes


def recipe_html(extra):
    r=extra['recipes']
    best=table(['Nodes','Best observed whole-job busy recipe','Qualified campaign','Whole-job busy','Full-context/GPUh','Scientific-efficiency winner'],
               [[x['nodes'],x['busy_winner'],x['qualified'],f"{x['overall_GPU_busy_percent']:.2f}%",f"{x['full_context_per_GPU_hour']:.3f}",x['efficiency_winner']] for x in extra['best_observed_recipes']])
    budgets=table(['Trial/job','Tool','Row timeout s','GPU request / row','Submitted batches'],[[x['trial']+'/'+x['job_id'],x['tool'],x['timeout_each_seconds'],x['gpus_per_row'],x['submitted_batches']] for x in extra['science_row_budgets']])
    return '<h2>Recipe book: operating Loam from 20 to 256 nodes</h2><p>'+html.escape(extra['recipe_scope'])+'</p><h3>Most successful observed recipes by scale</h3>'+best+'<p>If no campaign at a scale qualified, the table explicitly labels its best numerical-output observation as unqualified. A measured winner is not a guarantee for another target, model or workload.</p>'+table(
        ['Trial/job','N:S','Active','Top k','Initial','New/round','Gen batch','Rank workers','Synthesis','Workers/GPU','Tournament timeout s'],
        [[x['trial']+'/'+x['job_id'],f"{x['inference']}:{x['science']}",x['concurrency'],x['top_k'],x['initial_hypotheses'],x['new_per_round'],x['generation_batch_size'],x['tournament_workers'],x['synthesis_concurrency'],x['science_workers_per_gpu'],x['tournament_max_seconds']] for x in r])+'''
<h3>A measured dense starting recipe</h3><p>Let I be inference nodes and S science nodes. The tested split uses about 40% inference nodes: 8:12 at 20 total nodes and 102:154 at 256. Use C=4S active validators, K=3C=12S selected hypotheses per round, an initial pool of 2K=24S, and K/2=6S fresh hypotheses in each later round. Generate four hypotheses per inference call; use one Swiss round, 2I ranking workers and I synthesis workers. Keep two logical science workers per physical GPU. Each Jupiter science node has four physical GPUs, so eight logical worker slots do not represent eight GPUs. A validator may submit multiple scientific rows.</p>
<h3>Timeouts: change the relevant boundary, not every timer</h3><p>The production role/request values stayed fixed across the dense 20-node and 256-node routes: model request 600 s, tool/MCP read 6000 s, validator 10800 s, delegated code generator 6600 s, reviewer 1200 s and reasoner 1800 s. The tournament generation/ranking budget rose from 2400 to 10800 s for the larger hypothesis pool. The later controller explicitly used a 60 s MCP initialization timeout after 5 s default initialization failures under high fan-out. Model-service startup allowed 5400 s and the HPC startup setting 7200 s. Queue waiting is a separate lifecycle phase; it must not consume an allocated-node model-loading deadline.</p>
<p>Scientific batch metadata usually requested 5000 s per screen or full-context row, with a few 6000 s exceptions in the final campaign. Summary rows normally requested 300 s, with one 900 s exception. The full actual distribution, including row GPU requests, appears below and in science-row-budgets.csv. These are submitted-row deadlines, not guaranteed tool service durations.</p>
<p>Do not multiply every timeout by the node count. Bound per-request work from model latency and retries, delegated science from measured cold/warm row durations, and the enclosing validator from the entire delegated plan. Size generation/ranking deadlines from pool size, serving capacity and checkpoint overhead. Use measured tail percentiles and a documented margin; requalify changes. Long global tails still occur with these generous limits and reduce whole-job utilization.</p>
<p>The dense execution capsules configured 160 model requests per agent run, at most six code-generator delegations from a validator, ten model-request retries, and three structured-output retries. These are separate boundaries; a delegation has its own request budget. An entire hypothesis may therefore consume much more work than one agent request limit suggests. See run-recipes.csv for each retained configuration rather than substituting package defaults.</p>
<h3>What helped, what did not, and what remains</h3><p>The sparse 20-node route used 12 active validators and 48 selected hypotheses. Later routes raised active admission to 48 and supplied 144 selected hypotheses, while batching generation and preserving three tournament/evidence/synthesis iterations. Bounded hierarchical synthesis preserved every thread once while fitting the prompt budget. The reasoner output budget increased from 8192 to 32768 after truncation failures; validator, code-generator and reviewer budgets stayed at 8192. Four agent roles retained thinking=false. Generation/ranking settings and the native serving profile were kept separately pinned.</p>
<p>A single parent Slurm allocation starts disjoint inference and science pools on a private network. The inference service balances requests across local Nemotron replicas and cools unhealthy or throttled backends. Shared endpoint cooldown, jitter and staggered retries avoid retry bursts without reducing the scientific campaign. A queue monitor hands control back after allocation starts. Source/input manifests and readiness checks prevent starting against the wrong campaign directory.</p>
<p>Metadata registry indexing removed repeated worker-backed metadata lookups. Cached skill discovery, enabled GC with threshold 10000, and longer MCP initialization addressed measured controller overhead and connection errors. CPU replay gains are not GPU-throughput gains: the 20j campaign did not outperform the 20i campaign on whole-job GPU busy or normalized scientific throughput. The best audited 20-node result was 20i: 51.06% whole-job GPU busy, 12.574 distinct screens and 2.120 full-context trials per allocated GPU-hour.</p>
<p>At 256 nodes, compare the final measured campaign against both retired predecessors; retain requeues, startup, idle barriers, drain, failed tools and all parent GPU-hours. A completed workflow, valid numerical binders, claim provenance and a 50% saturation target are different checks. The report preserves each result instead of treating one passing metric as overall success. The provisional empty-output citation guard was CPU-tested but was not deployed into the running campaign. An offline prior-duration priority replay, holding recorded service durations fixed, did not justify deployment; it is not a measured scheduling or GPU improvement.</p>
<p>For Jupiter with at least one dedicated node for each role, the dense formula also reads: W=8S logical science workers, C=W/2 active validators, K=1.5W selected hypotheses, 3W initially generated hypotheses and 0.75W fresh hypotheses per later round. The multiplier depends on four physical GPUs and two logical workers per GPU. It is an empirical starting point for this target and tool chain. One-node functional debugging requires a different placement or external inference arrangement and is outside the two tested density scales.</p>
<p>Preserve the recorded execution capsule, scientific input checksums and current model/thinking settings when reproducing a recipe. Regenerate fresh campaign paths and job names rather than resubmitting into old evidence. The staging entry point is scripts/stage_saturation_scaleup.py QUALIFIED_20_NODE_JOB --nodes 256 --label NEW_LABEL --output-root NEW_OUTPUT_ROOT --check-only. That command is a non-submitting preflight; the reviewed controller capsule, sufficient storage and all-role allocation plan still need to match the selected recipe before submission. Actual scheduler/account settings, job-script hashes, role budgets and retained configuration hashes are exported in run-recipes.csv.</p>
<p>For an untested node count, use the density formula as an initial qualification candidate, then measure distinct scientific outputs per total GPU-hour, science-node productivity, every-node GPU busy, latency and queue/backlog traces. Preserve matched controls and independent audit gates. Global tournament barriers and long hypothesis tails remain candidates for future work; overlapping groups or larger scientific batches were considered but not deployed in this campaign. No interpolation is a demonstrated optimum.</p>
<h3>Recorded scientific row deadlines</h3><p>These are actual retained batch metadata values, including unsuccessful or repeated submissions. Submitted-batch counts are not scientific output counts. Distinguish each row deadline from the enclosing tool-read, delegated-agent and Slurm time budgets.</p>'''+budgets+'''
<p><a href="science-row-budgets.csv">Scientific row budget inventory</a> · <a href="run-recipes.csv">Exact recipe settings</a> · <a href="science-stage-throughput.csv">Stage counts and rate denominators</a> · <a href="science-node-throughput.csv">Every science node</a> · <a href="science-stage-events.csv">Stage production events</a></p>'''


def recipe_scenes(extra):
    scenes=[]
    s=Slide('Best observed recipes at each tested scale','Recipe book',notes=json.dumps(extra['best_observed_recipes']))
    s.table(40,145,[100,250,145,160,225],['Nodes','Best busy recipe','Qualified','Whole-job busy','Full-context / GPUh'],
            [[str(x['nodes']),x['busy_winner'],'Yes' if x['qualified'] else 'No',f"{x['overall_GPU_busy_percent']:.2f}%",f"{x['full_context_per_GPU_hour']:.3f}"] for x in extra['best_observed_recipes']],size=14)
    s.text(40,325,880,'The 20-node qualification results and the 256-node diagnostic/production results retain their separate audit status. These adaptive campaigns do not establish a universal optimum.',18,max_height=130)
    scenes.append(s)
    latest=extra['latest_recipe']
    s=Slide('LLM budgets are separate from wall-clock timers','Recipe book',notes=json.dumps(latest))
    s.table(40,135,[340,200,340],['Boundary','Configured value','Meaning'],[
        ['Per-agent model requests',str(latest['max_requests']),'Each agent run has its own budget'],
        ['Code-generator delegations',str(latest['max_code_calls']),'Bounded within a validator run'],
        ['Model-request retry limit',str(latest['model_retries']),'Transport/model retry policy'],
        ['Structured-output retries','3','Separate from model-request retries'],
        ['Validator / codegen / reviewer output','8192 tokens','Thinking=false for these agent roles'],
        ['Reasoner output','32768 tokens','Raised after structured-output truncation']],size=12.5)
    s.text(40,458,880,'Generation/ranking profiles remain separately pinned. A bounded delegation is not a single LLM call.',12,max_height=45)
    scenes.append(s)
    s=Slide('A reproducible dense starting recipe','Recipe book',notes=recipe_html(extra))
    s.table(40,130,[260,260,360],['Setting','20 nodes','256 nodes'],[
        ['Inference : science','8 : 12','102 : 154'],['Active / selected','48 / 144','616 / 1848'],
        ['Initial / new per round','288 / 72','3696 / 924'],['Generation batch / Swiss rounds','4 / 1','4 / 1'],
        ['Ranking / synthesis workers','16 / 8','204 / 102'],['Science workers per GPU','2','2']],size=14)
    s.text(40,455,880,'C=4S; K=3C; initial=2K; fresh=K/2. This is a tested density at two scales, not a guarantee for other node counts.',13,max_height=48)
    scenes.append(s)
    s=Slide('Timeouts follow work and contention','Recipe book',notes=recipe_html(extra))
    s.table(40,130,[300,190,390],['Boundary','Seconds','Scope'],[
        ['Model request / tool read','600 / 6000','Unchanged in dense runs'],
        ['Validator / code generator','10800 / 6600','Full delegated plan / one delegation'],
        ['Reviewer / reasoner','1200 / 1800','Unchanged'],
        ['Tournament at20 / at256','2400 / 10800','Larger generation/ranking pool'],
        ['MCP initialization','5 default → 60','Observed high-fan-out initialization failures'],
        ['Model / HPC startup','5400 / 7200','Separate from scheduler queue waiting']],size=12.5)
    s.text(40,458,880,'Measure tail latency and checkpoint/metadata overhead. Do not scale every timeout linearly with node count.',13,max_height=45)
    scenes.append(s)
    actual=[x for x in extra['science_row_budgets'] if x['job_id']==latest['job_id']]
    grouped=collections.defaultdict(collections.Counter)
    for row in actual:grouped[row['tool']][row['timeout_each_seconds']]+=row['submitted_batches']
    s=Slide('Actual scientific row deadlines','Recipe book',
            'Recorded batch metadata, including unsuccessful and repeated submissions.',notes=json.dumps(actual))
    s.table(40,140,[390,230,260],['Scientific tool','Row deadline seconds','Submitted batches'],[
        [tool.removeprefix('scale20_'),str(timeout),str(count)]
        for tool,values in sorted(grouped.items()) for timeout,count in sorted(values.items())],size=12)
    s.text(40,449,880,'A submitted batch is not an output count. Row, tool-read, delegated-agent and Slurm deadlines protect different boundaries.',13,max_height=50)
    scenes.append(s)
    s=Slide('Scaling lessons that survived measurement','Recipe book',notes=recipe_html(extra))
    s.text(40,133,880,'Supply enough admitted hypotheses and retain several waves of work. Batch hypothesis generation, bound hierarchical synthesis, and raise only the truncated reasoner output budget.\n\nPackage inference and science in one parent allocation. Pin source/input identities, inspect readiness, balance local replicas, and stagger throttled retries.\n\nCache static metadata and skill catalogs. Treat CPU microbenchmarks separately from measured GPU or scientific-throughput gains.\n\nCount all allocated GPU-seconds, failed attempts and idle tails. Publish per-stage outputs and per-node productivity alongside saturation.',19,max_height=330)
    scenes.append(s)
    s=Slide('What to run next at a new scale','Recipe book',notes=recipe_html(extra))
    s.text(40,135,880,'Start from the nearest qualified recipe and preserve model/thinking/scientific settings. Use about 40% inference nodes as a starting split, C=4S active validators and three admission waves.\n\nRequalify at small scale, then measure full-job GPU busy, distinct outputs/GPU-hour, science-node productivity and per-stage latency at the target scale.\n\nChoose the best observed recipe by both scientific efficiency and whole-job utilization. More nodes, longer timeouts or faster CPU replay alone do not establish improvement.\n\nBarrier overlap and better tail control remain untested follow-ups; they were not deployed during this run.',18,max_height=330)
    scenes.append(s)
    return scenes
