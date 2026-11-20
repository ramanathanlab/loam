"""Report terminal numerical outputs without certifying failed campaign gates.

The native success-only loader and auditors stay unchanged. This adapter gives
partial and provenance-flagged cases an explicit diagnostic presentation scope.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import audit_saturation_campaign as native_auditor
FROZEN = Path(native_auditor.__file__).resolve().parent
from audit_saturation_campaign import (hardware, occupancy, reviewed_science_threads,
                                       completed_assessment_threads)
from build_inference_atlas import deduplicate
from report_production_events import production_events
from saturation_accounting import allocation_attempts


def read(path):
    return json.loads(path.read_text())


def load_run(job):
    job = job.resolve(strict=True)
    plan = read(job/'plan.json')
    root = Path(plan['campaign']).resolve(strict=True)
    assert root == job/'campaign'
    audit_path = root/'scaling-audit.json'
    if not audit_path.exists():
        audit_path = root/'scaling-audit.numeric.json'
    if not audit_path.exists():
        job_id=read(job/'submission.json')['job_id']
        audit_path=(job.parent/'experiment.json').resolve(strict=True).parent/'diagnostics'/f'report-historical-numeric-{job_id}.json'
    audit = read(audit_path)
    assert Path(audit['campaign']).resolve(strict=True)==root
    assert audit['model_coordinates_independently_checked'], 'No independent coordinate audit'
    manifest = read(job/'allocation.json')
    receipt = read(job/'submission.json')
    state = read(job/'status.json')
    assert receipt['job_id'] == manifest['job_id']
    parent = audit['jobs'][receipt['job_id']]
    assert parent['State'].split()[0].rstrip('+') in {
        'COMPLETED','CANCELLED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY'}, parent
    screens, full = deduplicate(audit)
    timings = {(x['batch_id'],x['row']):x for x in audit['row_timings']}
    def owned(row):
        return timings[(row['batch_id'],Path(row['summary_path']).parent.name)].get('recorded_owner') is True
    screens, full = [x for x in screens if owned(x)], [x for x in full if owned(x)]
    events, event_resolutions = production_events(audit,screens,full)
    rounds = {int(p.stem.split('-')[1]):read(p) for p in (root/'state/rounds').glob('round-*.json')}
    start,end = parent['start_epoch'],parent['end_epoch']
    qualified = bool(parent['State']=='COMPLETED' and state['phase']=='completed'
                     and audit.get('required_campaign_completion_checks_passed'))
    summary = {
        'job_id':receipt['job_id'],'nodes':plan['nodes'],'inference':plan['inference_count'],
        'science':plan['science_count'],'mode':plan['run_mode'],
        'parent_state':parent['State'],'controller_phase':state['phase'],
        'allocation_seconds':end-start,'queue_seconds':parent.get('queue_seconds'),
        'screen_trials':len(screens),'expanded_trials':len(full),
        'distinct_screen_binder_sequences':len({x['binder_sequence'] for x in screens}),
        'distinct_full_binder_sequences':len({x['binder_sequence'] for x in full}),
        'verified_complex_files':5*(len(screens)+len(full)),
        'screen_trials_per_hour':len(screens)*3600/(end-start),
        'expanded_trials_per_hour':len(full)*3600/(end-start),
        'rounds':len(rounds),'thread_errors':len(audit.get('thread_errors',[])),
        'reasoner_failures':sum(str(x.get('brief',{}).get('report','')).startswith('reasoner failed:') for x in rounds.values()),
        'orchestration_error_threads':sum(bool(t.get('error')) for x in rounds.values() for t in x['threads']),
        'failed_rows':sum(b.get('failed',0) for b in audit['batch_states']),
        'completion_checks_passed':qualified,'qualified_campaign':qualified,
        'numeric_errors_retained':len(audit.get('errors',[])),
        'unresolved_claim_sources_retained':len(audit.get('unowned_or_missing_claim_sources',[])),
        'scientific_output_scope':'Distinct numerically checked, receipt-owned outputs; campaign completion and narrative claims are separate gates.',
        'case_status':'Qualified completed campaign' if qualified else 'Diagnostic numerical outputs; campaign not qualified',
    }
    ready = read(root/'server-ready.json')
    watch = read(root/'controller-watch.json')
    summary['services_startup_seconds'] = max(0,ready['ready_at']-start)
    summary['controller_wall_seconds'] = (watch.get('controller_finished_at') or end)-watch['controller_started_at']
    summary['handoff_drain_cleanup_seconds'] = max(0,end-start-summary['services_startup_seconds']-summary['controller_wall_seconds'])
    return {'job':job,'root':root,'plan':plan,'audit':audit,'audit_path':audit_path,
            'manifest':manifest,'receipt':receipt,'state':state,'parent':parent,
            'screens':screens,'full':full,'start':start,'end':end,'seconds':end-start,
            'events':events,'event_time_resolutions':event_resolutions,'rounds':rounds,
            'summary':summary,'formal':qualified}


def measure(job):
    """Use unchanged native integrators and retain every parent allocation cost."""
    run = load_run(job)
    plan, audit, receipt = run['plan'],run['audit'],run['receipt']
    attempts = allocation_attempts(receipt['job_id'])
    assert attempts['allocations']
    assert all(int(x['NNodes'])==plan['nodes'] and x['JobName']==receipt['job_name']
               for x in attempts['allocations'])
    final = run['seconds']
    total = attempts['total_allocated_seconds']
    assert total >= final
    assert abs(attempts['allocations'][-1]['start_epoch']-run['start'])<1
    prior = total-final
    summary,nodes = hardware(job,run['start'],run['end'])
    slots = occupancy(job,audit,run['start'],run['end'])
    summary['final_allocation_overall_GPU_busy_percent'] = summary['overall_GPU_busy_percent']
    summary['overall_GPU_busy_percent'] *= final/total
    summary['overall_telemetry_coverage_percent'] *= final/total
    slots['science_slot_occupancy_percent'] *= final/total
    slots['fraction_time_below_10percent_science_slot_occupancy'] = (
        slots['fraction_time_below_10percent_science_slot_occupancy']*final+prior)/total
    gpu_hours = attempts['total_allocated_GPU_seconds']/3600
    synthesis = []
    for record in run['rounds'].values():
        brief=record['brief']; path=brief.get('synthesis_receipt')
        synthesis.append(bool(brief.get('tournament_brief')) and not brief.get('report','').startswith('reasoner failed:')
                         and bool(path) and read(Path(path))['status']=='complete')
    summary.update(job_id=receipt['job_id'],nodes=plan['nodes'],inference=plan['inference_count'],
                   science=plan['science_count'],allocated_seconds=total,
                   final_allocation_seconds=final,prior_allocation_seconds=prior,
                   allocation_attempt_count=attempts['allocation_attempts'],
                   prior_attempt_GPU_hours=4*plan['nodes']*prior/3600,
                   allocation_accounting=attempts,allocated_GPU_hours=gpu_hours,
                   queue_seconds=max(0,run['start']-receipt['submitted_at']-prior),
                   screen_trials=len(run['screens']),full_context_trials=len(run['full']),
                   screen_trials_per_allocated_GPU_hour=len(run['screens'])/gpu_hours,
                   full_context_trials_per_allocated_GPU_hour=len(run['full'])/gpu_hours,
                   clean_synthesis_rounds=sum(synthesis),recorded_rounds=len(run['rounds']),
                   scientific_completion_checks=audit.get('required_campaign_completion_checks',{}),
                   scientific_completion_passed=run['summary']['qualified_campaign'],
                   **{k:v for k,v in slots.items() if k!='trace'})
    tested=reviewed_science_threads(audit,list(run['rounds'].values()))
    assessed=completed_assessment_threads(list(run['rounds'].values()),tested)
    summary.update(reviewed_scientific_test_threads=len(assessed),
                   reviewed_scientific_test_hypotheses=len({h for n,h in assessed}),
                   reviewed_fresh_science_threads=len(tested),
                   hypotheses_with_reviewed_fresh_science=len({h for n,h in tested}),
                   completed_scientific_assessment_threads=len(assessed))
    return {'summary':summary,'per_node':nodes,'occupancy_trace':slots['trace']}
