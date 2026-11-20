#!/usr/bin/env python3
"""Compare independently audited science and measured inference/telemetry.

Reads existing audits only. Shared inference leases are charged once in the
experiment ledger; per-trial overlap cost is separately labeled, never a claim
that cold startup or idle time was free.
"""
import argparse, collections, hashlib, html, json, statistics, time
from pathlib import Path
from audit_scale20_campaign import accounting


def read(p):return json.loads(p.read_text())
def lines(p):
    if not p.exists():return []
    rows=[]
    for line in p.read_text().splitlines():
        try:rows.append(json.loads(line))
        except ValueError:pass  # A live append may not yet have its trailing bytes.
    return rows
def percentile(values,p):
    values=sorted(values)
    if not values:return None
    x=(len(values)-1)*p;lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)
def stats(values):
    return {'n':len(values),'median':statistics.median(values) if values else None,'p95':percentile(values,.95),'mean':statistics.mean(values) if values else None}
def concurrency_stats(requests,start,end,replicas):
    """Integrate actual attempt overlap, including allocation time with no calls."""
    if end<=start:return None
    events=[(start,0),(end,0)]
    for r in requests:
        lo=max(start,r['started_at']);hi=min(end,r['finished_at'])
        if hi>lo:events.extend([(lo,1),(hi,-1)])
    active=peak=0;last=start;seconds=collections.Counter()
    for timestamp,change in sorted(events,key=lambda x:(x[0],x[1])):
        seconds[active]+=timestamp-last;active+=change;peak=max(peak,active);last=timestamp
    duration=end-start
    return {'peak':peak,'time_weighted_mean':sum(n*s for n,s in seconds.items())/duration,'fraction_of_allocation_at_least_one_attempt_per_replica':sum(s for n,s in seconds.items() if n>=replicas)/duration,'seconds_by_concurrency':dict(seconds),'scope':'Pool dispatch attempts, including network/queue time; not a count of actively decoding GPU kernels. Allocation intervals with no requests are included.'}
def number(v,d=2):return 'pending' if v is None else f'{v:.{d}f}'
def table(headers,rows):
    e=lambda s:html.escape(str(s))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def collect_trial(trial,requests,output):
    progress=read(trial/'scale-progress.json');pooled=bool(progress.get('inference_pool_experiment'))
    result={'trial':str(trial),'split':{'inference':len(progress['active_inference_job_ids']) if pooled else 1,'science':progress.get('scientific_allocation_nodes',19)},'phase':progress.get('phase'),'audit_available':False}
    watch=read(trial/'controller-watch.json') if (trial/'controller-watch.json').exists() else {}
    result['controller']=watch
    if not (trial/'scaling-audit.json').exists():return result
    audit=read(trial/'scaling-audit.json');result.update(audit_available=True,independent_coordinates_checked=audit['model_coordinates_independently_checked'],completion_checks_passed=audit.get('required_campaign_completion_checks_passed',False))
    result['excluded_nonstandard_inputs']=audit.get('nonstandard_input_classifications',[])
    result['unique_tested_hypotheses']=audit['unique_tested_hypotheses']
    sj=audit['jobs'].get(progress.get('scientific_job'),{});seconds=(sj.get('end_epoch') or 0)-(sj.get('start_epoch') or 0)
    hours=seconds/3600 if seconds>0 else None
    crop={m['sha256'] for m in audit['verified_models'] if m['kind']=='crop'}
    sequences={c['binder_sequence'] for c in audit['candidate_rows']}
    result.update(science_allocation_seconds=seconds,verified_crop_files=len(crop),distinct_screen_sequences=len(sequences),successful_screen_rows=len(audit['candidate_rows']),intact_followup_rows=len(audit['intact_followups']),rounds=audit['rounds'],reviewed_hypothesis_rounds=sum(r['reviewed'] for r in audit['rounds'].values()),thread_failures=audit['thread_errors'],failed_scientific_rows=sum(b['failed'] for b in audit['batch_states']),unfinished_scientific_rows=audit['unfinished_main_rows'],provenance_errors=audit['errors'],coordinate_files_per_science_allocation_hour=len(crop)/hours if hours else None,distinct_sequences_per_science_allocation_hour=len(sequences)/hours if hours else None)
    result.update(verified_total_complex_files=audit['unique_verified_coordinate_files'],verified_total_complex_files_per_science_allocation_hour=audit['unique_verified_coordinate_files']/hours if hours else None,complete_energy_followups_per_science_allocation_hour=len(audit['intact_followups'])/hours if hours else None,minimized_intact_followups_retaining_both_regions=sum(bool(c.get('minimized_pose_geometry',{}).get('dual_region_contact')) for c in audit['intact_followups']))
    starts={(row['batch_id'],row['row']):row.get('started_at') or float('inf') for row in audit['row_timings']}
    screen_trials={};intact_trials={};repeated_screen=[];repeated_intact=[]
    order=lambda c:(starts.get((c['batch_id'],Path(c['summary_path']).parent.name),float('inf')),c['summary_path'])
    for c in sorted(audit['candidate_rows'],key=order):
        key=(c['condition'],c['input_sha256'],c['seed'],c['target_sequence'])
        if key in screen_trials:repeated_screen.append({'source':c['summary_path'],'first_successful_source':screen_trials[key]['summary_path'],'seed':c['seed']})
        else:screen_trials[key]=c
    for c in sorted(audit['intact_followups'],key=order):
        row=read(Path(c['summary_path']).parent/'input.json');key=(c['binder_sequence'],row['seed'],c['target_residues'])
        if key in intact_trials:repeated_intact.append({'source':c['summary_path'],'first_successful_source':intact_trials[key]['summary_path'],'seed':row['seed']})
        else:intact_trials[key]=c
    selected_screens=list(screen_trials.values());result['raw_distinct_screen_sequences']=len(sequences)
    sequences={c['binder_sequence'] for c in selected_screens}
    best={sequence:max(c['best_iptm'] for c in selected_screens if c['binder_sequence']==sequence) for sequence in sequences}
    result.update(distinct_screen_sequences=len(sequences),distinct_sequences_per_science_allocation_hour=len(sequences)/hours if hours else None,median_sequence_best_crop_iptm=statistics.median(best.values()) if best else None,distinct_sequences_crop_iptm_at_least_0_8=sum(v>=.8 for v in best.values()),quality_trial_method='Sequence/confidence comparisons and top binders use earliest successful complete distinct-input trials, matching throughput deduplication. Never choose the best score among repeated identical input/seed submissions. Raw inventory remains available.')
    selected_folders={Path(c['summary_path']).parent for c in [*screen_trials.values(),*intact_trials.values()]}
    distinct_files={v['sha256'] for v in audit['verified_models'] if selected_folders.intersection(Path(v['path']).parents)}
    result.update(unique_screen_input_trials=len(screen_trials),unique_complete_intact_input_trials=len(intact_trials),repeated_screen_input_rows=len(repeated_screen),repeated_intact_input_rows=len(repeated_intact),verified_files_on_distinct_input_trials=len(distinct_files),distinct_trial_complex_files_per_science_allocation_hour=len(distinct_files)/hours if hours else None,distinct_intact_trials_per_science_allocation_hour=len(intact_trials)/hours if hours else None,duplicate_trial_sources={'screen':repeated_screen,'intact':repeated_intact},distinct_trial_method='Keep earliest successfully verified complete trial per (condition,input coordinate SHA256,seed,target sequence) for screens; per (binder sequence,seed,canonical target length) for frozen expanded-context recipes. All repeated submissions stay in raw file counts and resource costs. Independent seeds on the same sequence remain separate trials.')
    retained=sum(bool(c.get('minimized_pose_geometry',{}).get('dual_region_contact')) for c in intact_trials.values())
    result.update(distinct_intact_trials_retaining_both_regions=retained,retained_intact_trials_per_science_allocation_hour=retained/hours if hours else None,high_confidence_sequences_per_science_allocation_hour=sum(v>=.8 for v in best.values())/hours if hours else None)
    result['controller_seconds']=(watch.get('controller_finished_at') or time.time())-watch['controller_started_at'] if watch.get('controller_started_at') else None
    science_gpu_h=(int(sj.get('ElapsedRaw',0))*4*result['split']['science'])/3600
    infer_ids=progress.get('active_inference_job_ids') or [j['JobIDRaw'] for j in audit.get('inference_jobs',[])]
    infer_overlap_h=sum(max(0,min(j.get('end_epoch') or 0,sj.get('end_epoch') or 0)-max(j.get('start_epoch') or 0,sj.get('start_epoch') or 0))*4/3600 for job in infer_ids if (j:=audit['jobs'].get(job)))
    result['trial_window_allocated_GPU_hours']=science_gpu_h+infer_overlap_h
    result['trial_window_crop_files_per_GPU_hour']=len(crop)/(science_gpu_h+infer_overlap_h) if science_gpu_h+infer_overlap_h else None
    result['trial_window_distinct_trial_files_per_GPU_hour']=len(distinct_files)/(science_gpu_h+infer_overlap_h) if science_gpu_h+infer_overlap_h else None
    # Preserve baseline whole-job cost; shared long inference jobs are not double charged across new trials.
    if not pooled:
        result['all_own_job_allocated_GPU_hours']=sum(int(j.get('ElapsedRaw',0))*4*int(j.get('NNodes',0))/3600 for j in audit['jobs'].values())
        result['whole_job_crop_files_per_GPU_hour']=len(crop)/result['all_own_job_allocated_GPU_hours'] if result['all_own_job_allocated_GPU_hours'] else None
    else:result['shared_inference_cost_note']='Full shared inference allocations are charged once in the experiment ledger, not independently to every trial.'
    phase=[r for r in requests if r.get('phase')==trial.name]
    final={r['request_id']:r for r in sorted(phase,key=lambda r:r.get('finished_at',0))}
    result['inference']={'attempts':len(phase),'requests':len(final),'status':dict(collections.Counter(str(r['status']) for r in final.values())),'nonthinking_seconds':stats([r['total_request_seconds'] for r in final.values() if not r['thinking_enabled'] and r['status']==200]),'thinking_seconds':stats([r['total_request_seconds'] for r in final.values() if r['thinking_enabled'] and r['status']==200]),'queue_seconds':stats([r['queue_seconds'] for r in phase if r.get('queue_seconds') is not None]),'by_replica':{backend:{'attempts':len(rr),'prompt_tokens':sum(r.get('prompt_tokens') or 0 for r in rr),'completion_tokens':sum(r.get('completion_tokens') or 0 for r in rr),'failed_attempts':sum(r['status']!=200 for r in rr)} for backend in sorted({r.get('backend') for r in phase if r.get('backend')}) if (rr:=[r for r in phase if r.get('backend')==backend])}}
    result['inference']['all_request_seconds']=stats([r['total_request_seconds'] for r in final.values() if r['status']==200])
    result['inference']['attempt_concurrency']=concurrency_stats(phase,sj.get('start_epoch',0),sj.get('end_epoch',0),result['split']['inference']) if pooled else None
    if not pooled:
        legacy=[r for r in lines(trial/'state/nemotron/request-timings.jsonl') if sj.get('start_epoch',float('inf'))<=r.get('start',0)<=(sj.get('end_epoch') or 0)]
        result['inference'].update(attempts=len(legacy),requests=len(legacy),status=dict(collections.Counter(str(r['status']) for r in legacy)),all_request_seconds=stats([r['seconds'] for r in legacy if r['status']==200]),limitation='Legacy server logs lack request thinking flags, token counts and separate queue latency; no per-mode inference comparison is claimed.')
    telemetry_path=output/trial.name/'telemetry-summary.json'
    if telemetry_path.exists():result['telemetry']=read(telemetry_path)
    elif not pooled:
        baseline_telemetry=Path(__file__).resolve().parents[1]/'reports/scale20-20261113/telemetry-summary.json'
        if baseline_telemetry.exists():result['telemetry']=read(baseline_telemetry)
    roles=result.get('telemetry',{}).get('role_observed_node_minutes',{})
    available=sum(r.get('available_node_minutes',0) for r in roles.values())
    expected=sum(r.get('expected_node_minutes',0) for r in roles.values())
    result['overall_observed_GPU_busy_percent']=sum(r['mean_GPU_busy_percent']*r['available_node_minutes'] for r in roles.values())/available if available else None
    result['telemetry_node_minute_coverage']={'observed':available,'expected':expected,'fraction':available/expected if expected else None,'method':'Weight role means by observed node-minutes. Every node has four GPUs; missing observations are not replaced with zero.'}
    result['top_binders_by_round']={}
    for n in audit['rounds']:
        candidates=sorted([c for c in selected_screens if str(c['round'])==n],key=lambda c:c['best_iptm'],reverse=True);seen=set();top=[]
        for c in candidates:
            if c['binder_sequence'] in seen:continue
            seen.add(c['binder_sequence']);top.append({'binder_id':'binder_'+hashlib.sha256(c['binder_sequence'].encode()).hexdigest()[:12],'sequence':c['binder_sequence'],'hypothesis':c['hypothesis'],'condition':c['condition'],'best_iptm':c['best_iptm'],'dual_contact_models':c['dual_contact_models'],'pose_RMSD_A':c['mean_pairwise_target_aligned_binder_CA_RMSD_A'],'source':c['summary_path']})
            if len(top)==5:break
        result['top_binders_by_round'][n]=top
    return result


def main(root,output):
    output.mkdir(parents=True,exist_ok=True);experiment=read(root/'experiment.json');requests=lines(root/'pool-request-timings.jsonl')
    trials=[collect_trial(Path(experiment['baseline']),[],output),*[collect_trial(t,requests,output) for t in sorted((root/'trials').glob('*')) if (t/'scale-progress.json').exists()]]
    jobs=accounting(experiment['owned_job_ids']);gpu_h=sum(int(j.get('ElapsedRaw',0))*4*int(j.get('NNodes',0))/3600 for j in jobs.values());events=[]
    for j in jobs.values():
        if j.get('start_epoch') and j.get('end_epoch'):events.extend([(j['start_epoch'],int(j['NNodes'])),(j['end_epoch'],-int(j['NNodes']))])
    active=peak=0
    for _,change in sorted(events,key=lambda row:(row[0],row[1])):active+=change;peak=max(peak,active)
    data={'generated_at':time.time(),'experiment':experiment,'trials':trials,'owned_jobs':jobs,'unique_job_allocated_GPU_hours':gpu_h,'peak_owned_nodes':peak,'cost_note':'Whole experiment GPU-hours sum each unique owned allocation once, including model load, queue-associated allocated idle time, repairs and failed science. Per-trial window cost charges inference overlap only and excludes shared pretrial startup; it must not be presented as full experiment cost.'}
    if (root/'inference-profile.json').exists():data['frozen_request_profiles']=read(root/'inference-profile.json')
    for name,key in [('selection-before-confirmation.json','selection_before_confirmation'),('recommendation.json','final_recommendation')]:
        if (root/name).exists():data[key]=read(root/name)
    (output/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    rows=[];details=[];quality_rows=[]
    for t in trials:
        split=t['split'];label=f"{split['inference']} inference / {split['science']} science"+(' (confirmation)' if 'confirm' in Path(t['trial']).name else '')
        if not t['audit_available']:rows.append([label,t['phase'],'pending','pending','pending','pending','pending','pending','pending']);continue
        tel=t.get('telemetry',{});roles=tel.get('role_observed_node_minutes',{})
        rows.append([label,'complete' if t['completion_checks_passed'] else 'running / verification pending',len(t['rounds']),t['distinct_screen_sequences'],t['verified_files_on_distinct_input_trials'],number(t['distinct_trial_complex_files_per_science_allocation_hour']),number(roles.get('science',{}).get('mean_GPU_busy_percent')),number(roles.get('inference',{}).get('mean_GPU_busy_percent')),number(t['overall_observed_GPU_busy_percent'])])
        quality_rows.append([label,number(t['median_sequence_best_crop_iptm'],3),f"{t['distinct_sequences_crop_iptm_at_least_0_8']}/{t['distinct_screen_sequences']}",number(t['high_confidence_sequences_per_science_allocation_hour']),f"{t['distinct_intact_trials_retaining_both_regions']}/{t['unique_complete_intact_input_trials']}",number(t['retained_intact_trials_per_science_allocation_hour'])])
        req=t['inference'];toprows=[[n,c['hypothesis'][:8],c['binder_id'],c['condition'],number(c['best_iptm'],3),c['dual_contact_models'],number(c['pose_RMSD_A'])] for n,cc in t['top_binders_by_round'].items() for c in cc]
        tpath=Path(t['trial']).name+'/node-gpu-utilization.svg'
        plots='<img src="'+html.escape(tpath)+'" alt="Measured node GPU utilization">' if (output/tpath).exists() else ''
        supplementary=[]
        for name,caption in [('node-cpu-utilization','CPU activity'),('node-host-memory','Host memory'),('node-gpu-memory','GPU memory residency'),('science-slot-leases','Scientific worker GPU leases'),('inference-queue-and-gpu-power','Inference queues and GPU power')]:
            relative=Path(t['trial']).name+'/'+name+'.svg'
            if (output/relative).exists():supplementary.append('<figure><figcaption>'+caption+'</figcaption><img src="'+html.escape(relative)+'" alt="'+caption+'"></figure>')
        if supplementary:plots+='<details><summary>CPU, memory, worker leases and inference queues</summary>'+''.join(supplementary)+'</details>'
        details.append('<section><h2>'+html.escape(label)+'</h2><p>Failed submitted rows (metadata and science): '+str(t['failed_scientific_rows'])+'; unfinished rows: '+str(t['unfinished_scientific_rows'])+'; fully reviewed hypothesis evaluations: '+str(t['reviewed_hypothesis_rounds'])+'; incomplete hypothesis threads: '+str(len(t['thread_failures']))+'. Nonthinking request median / p95: '+number(req['nonthinking_seconds']['median'])+' / '+number(req['nonthinking_seconds']['p95'])+' s. Thinking request median / p95: '+number(req['thinking_seconds']['median'])+' / '+number(req['thinking_seconds']['p95'])+' s.</p>'+plots+table(['Round','Hypothesis','Binder','Condition','Best crop ipTM','Models contacting both regions / 5','Pairwise pose RMSD Å'],toprows)+'</section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Jupiter inference balance</title><style>body{font:16px system-ui;color:#183037;background:#f5f8fa;max-width:1280px;margin:3rem auto;padding:0 1.5rem;line-height:1.5}section{background:white;padding:1.5rem;margin:2rem 0;border-radius:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.65rem;border-bottom:1px solid #dae2e8}th{background:#e6eff3}img{max-width:100%}</style><h1>Balancing Nemotron and scientific compute</h1>'''
    recommendation=data.get('final_recommendation')
    if recommendation:
        page+='<section><h2>Measured recommendation</h2>'+''.join('<p>'+html.escape(recommendation[key])+'</p>' for key in ('summary','scope','quality_tradeoff','resource_status') if recommendation.get(key))+'</section>'
    elif data.get('selection_before_confirmation'):
        split=data['selection_before_confirmation']['split']
        page+='<p>Selected for confirmation: '+str(split['inference'])+' inference / '+str(split['science'])+' science nodes. The repeat is pending final completion and audit.</p>'
    page+='<p>Twenty total owned Jupiter nodes; unchanged scientific settings and thinking behavior. The primary throughput counts verified crop/expanded-EGFR files from distinct scientific input trials, excluding repeated submissions of identical inputs and seeds. The JSON retains raw output, duplicates, stage categories and all costs. In-progress counts and sampled utilization are provisional.</p>'+table(['Split','Status','Recorded rounds','Distinct binder sequences','Files on distinct trials','Distinct-trial files / h','Science GPU busy %','Inference GPU busy %','Overall observed GPU busy %'],rows)
    page+='<p>The historical 1/19 baseline includes controller repairs and inference renewal. The initial 2/18 trial includes metadata-ID and native-filename compatibility repairs. Incorrect control labels and changed delegated recipes are excluded from positive counts; their original outputs and costs remain in the JSON ledger. These are operational measurements of adaptive scientific campaigns with matched protocols and initial proposal texts, rather than identical fixed scientific jobs. The separate frozen-request benchmark controls request content. Missing or failed hypothesis reviews remain visible; a completed campaign does not imply every hypothesis succeeded.</p>'
    page+='<h2>Scientific quality alongside throughput</h2><p>Quality and top-binder comparisons use the earliest successful complete trial for repeated identical inputs/seeds, matching throughput deduplication. They do not choose the best score among repeat submissions. These are descriptive computational screens. The ipTM0.80 column is not a new selection gate; the frozen workflow confidence/dispersion rules are unchanged. Stochastic designs and adaptive output mixes differ between runs. No statistical superiority in binding or measured affinity is claimed.</p>'+table(['Split','Median sequence-best crop ipTM','Sequences with crop ipTM ≥0.80','High-confidence sequences / h','Distinct minimized follow-ups retaining both regions','Retained-context trials / h'],quality_rows)
    page+='<p>The separate frozen-context inference benchmark repeats identical review requests at concurrency 12. Every measured replica count passed 15 of 24 strict review checks; the same arithmetic/gate-consistency failures remain in the results. Its latency and validated-review rates isolate inference behavior without treating all HTTP successes as correct scientific reasoning.</p>'
    page+='<p>Experiment allocated GPU-hours so far: '+number(gpu_h)+'; peak owned nodes: '+str(peak)+'. '+html.escape(data['cost_note'])+'</p><p>GPU busy is sampled NVIDIA utilization.gpu, not SM occupancy or FLOPS efficiency. Missing observations remain missing. Confidence, contact geometry and static MMGBSA are computational screens, not measured affinity. EGFR follow-ups use residues 25–646, without glycans or a membrane.</p>'+''.join(details)+'<p><a href="comparison.json">Full measurements, provenance, per-replica token counts and failure inventory</a></p></html>'
    charts=''.join('<figure><img src="'+name+'.svg" alt="'+name.replace('-',' ')+'"></figure>' for name in ('ratio-scientific-throughput','ratio-output-categories','ratio-gpu-utilization','ratio-inference-latency') if (output/(name+'.svg')).exists())
    page=page.replace('<p>Experiment allocated GPU-hours',charts+'<p>Experiment allocated GPU-hours')
    (output/'index.html').write_text(page);print(json.dumps({'output':str(output),'trials':len(trials),'peak_owned_nodes':peak,'experiment_GPU_hours':gpu_h}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.root.resolve(),a.output.resolve())
