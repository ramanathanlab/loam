#!/usr/bin/env python3
"""Evaluate the predeclared saturation gates against independently audited science.

Run the unchanged audit_scale20_campaign.py --verify-models first, using the
scientific Python. This evaluator never submits jobs or changes old reports.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import re
from pathlib import Path

from audit_scale20_campaign import accounting
from saturation_accounting import allocation_attempts

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / 'runs/jupiter-unified-20261114T172148Z/unified-20'


def read(path):
    return json.loads(path.read_text())


def reviewed_science_threads(audit, rounds):
    """Conservative tested count: reviewed claims on verified current-round batches.

    Valid coordinates from errored threads remain science outputs. They do not
    establish a completed hypothesis test. Earlier-round evidence alone also
    does not constitute a new test of that hypothesis in the current round.
    """
    valid={(row['round'],row['hypothesis'],row['batch_id'])
           for key in ['candidate_rows','intact_followups'] for row in audit.get(key,[])}
    batches={row['batch_id']:row for row in audit.get('batch_states',[])}
    aliases={(row['round'],row['hypothesis'],row['original_source']):row.get('files',[])
             for row in audit.get('explicit_claim_source_resolutions',[])}
    upstream={};qualified=set()
    records=rounds.values() if isinstance(rounds,dict) else rounds
    for record in records:
        n=record['round']
        for thread in record.get('threads',[]):
            h=thread['hypothesis']['id']
            if thread.get('error') or thread.get('report') is None or thread.get('review') is None:continue
            for claim in thread.get('accepted',[]):
                source=claim.get('source','')
                paths=[source]+[item['path'] for item in aliases.get((n,h,source),[])]
                for path in paths:
                    if path.startswith(('http://','https://')):continue
                    match=re.search(r'(?:^|/)batches/(b-[0-9a-f]+)(?:/|$)',path.strip(' `"\''))
                    if not match:continue
                    batch=match[1]
                    if batches.get(batch,{}).get('tool')=='scale20_summarize_binder_pipeline':
                        if batch not in upstream:
                            inputs=Path(audit['campaign'])/'batches'/batch/'inputs.jsonl'
                            rows=[json.loads(line) for line in inputs.read_text().splitlines()]
                            upstream[batch]=Path(rows[0]['batch_dir']).name if len(rows)==1 else None
                        batch=upstream[batch]
                    if (n,h,batch) in valid:qualified.add((n,h))
    return qualified


def completed_assessment_threads(rounds, fresh_science_threads):
    """Keep blocked/untested outcomes out of completed scientific test counts."""
    records=rounds.values() if isinstance(rounds,dict) else rounds
    completed=set()
    for record in records:
        for thread in record.get('threads',[]):
            key=(record['round'],thread['hypothesis']['id'])
            report=thread.get('report')
            if (key in fresh_science_threads and not thread.get('error')
                    and report is not None and thread.get('review') is not None
                    and report.get('outcome') in {'supported','contradicted','inconclusive'}):
                completed.add(key)
    return completed


def hardware(job, start, end):
    plan_path = job / 'plan.json'
    if plan_path.exists() and read(plan_path).get('gpus_per_node', 4) != 4:
        raise ValueError('Historical saturation integrators require four GPUs per node; qualify a new hardware integrator first')
    allocation = read(job / 'allocation.json')
    hosts = allocation['all_nodes']
    samples = collections.defaultdict(dict)
    for path in (job / 'campaign/telemetry').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # An in-progress final append is never positive evidence.
            host = row['host'].split('.')[0]
            assert row['job_id'] == allocation['job_id'] and host in hosts
            expected = 'science' if host in allocation['science_nodes'] else 'inference'
            assert row['role'] == expected
            t = float(row['timestamp'])
            gpu = list(csv.reader(io.StringIO(row.get('gpu_csv', ''))))
            if start <= t <= end and len(gpu) == 4 and all(len(g) >= 8 for g in gpu):
                assert {int(g[0]) for g in gpu} == {0, 1, 2, 3}
                values = [float(g[2]) for g in sorted(gpu, key=lambda g: int(g[0]))]
                assert all(0 <= value <= 100 for value in values)
                samples[host][t] = values
    nodes = []
    for host in hosts:
        points = sorted(samples[host].items())
        covered = 0.0
        integral = [0.0] * 4
        trace = []
        for i, (t, values) in enumerate(points):
            # Hold for at most 25s; outages remain explicitly unobserved. The
            # final sample represents at most one normal sampling interval.
            stop = min(end, points[i + 1][0] if i + 1 < len(points) else t + 10, t + 25)
            duration = max(0, stop - t)
            covered += duration
            for g, value in enumerate(values):
                integral[g] += duration * value
            trace.append({'timestamp': t, 'duration': duration, 'gpu_busy': values})
        nodes.append({'host': host, 'role': 'science' if host in allocation['science_nodes'] else 'inference',
                      'coverage_percent': 100 * covered / (end - start),
                      'observed_GPU_busy_percent': sum(integral) / (4 * covered) if covered else None,
                      'whole_allocation_GPU_busy_lower_bound_percent': sum(integral) / (4 * (end - start)),
                      'per_GPU_whole_allocation_busy_lower_bound_percent': [v / (end - start) for v in integral],
                      'trace': trace})
    summary = {}
    for role in ('overall', 'science', 'inference'):
        subset = [n for n in nodes if role == 'overall' or n['role'] == role]
        summary[role + '_GPU_busy_percent'] = sum(n['whole_allocation_GPU_busy_lower_bound_percent'] for n in subset) / len(subset)
        summary[role + '_telemetry_coverage_percent'] = sum(n['coverage_percent'] for n in subset) / len(subset)
    summary['minimum_node_telemetry_coverage_percent'] = min(n['coverage_percent'] for n in nodes)
    return summary, nodes


def occupancy(job, audit, start, end):
    capacity = 4 * read(job / 'plan.json')['science_count']
    events = collections.defaultdict(list)
    events[start] = []; events[end] = []
    for row in audit['row_timings']:
        if not row.get('started_at') or float(row.get('gpus') or 0) <= 0:
            continue
        folder = job / 'campaign/batches' / row['batch_id'] / row['row']
        status = read(folder / 'status.json')
        lo = max(start, float(row['started_at']))
        hi = min(end, status.get('process_finished_at') or float(row['started_at']) + float(row.get('seconds') or 0))
        host = status.get('host', '').split('.')[0]
        gpu = status.get('gpu', '')
        devices = [(host, g) for g in gpu.split(',') if g] if host and gpu else [
            ('legacy-'+row['batch_id']+'-'+row['row'], str(n)) for n in range(int(float(row['gpus']))) ]
        if hi > lo:
            for device in devices:
                events[lo].append((device, 1)); events[hi].append((device, -1))
    references = collections.Counter()
    integral = low = 0.0
    trace = []
    times = sorted(events)
    for i, t in enumerate(times[:-1]):
        for device, delta in events[t]: references[device] += delta
        active_devices = sum(n > 0 for n in references.values())
        active_rows = sum(max(0, n) for n in references.values())
        duration = times[i + 1] - t
        integral += min(capacity, active_devices) * duration
        if active_devices < 0.1 * capacity: low += duration
        trace.append({'timestamp': t, 'duration': duration,
                      'active_GPU_rows': active_devices,
                      'active_physical_GPUs': active_devices,
                      'active_scientific_rows': active_rows})
    return {'science_slot_occupancy_percent': 100 * integral / (capacity * (end - start)),
            'fraction_time_below_10percent_science_slot_occupancy': low / (end - start),
            'scope': 'Unique physical host/GPU devices with at least one recorded scientific row lease, including failed/excluded rows; shared rows never multiply physical occupancy. This is not kernel busy or a direct EL ready-queue measurement.',
            'trace': trace}


def measure(job):
    from build_inference_atlas import deduplicate
    plan = read(job / 'plan.json')
    receipt=read(job/'submission.json')
    job_id=receipt['job_id']
    parent = accounting([job_id])[job_id]
    assert parent['State'] == 'COMPLETED', parent
    attempts = allocation_attempts(job_id)
    assert attempts['allocations'], 'Missing parent allocation accounting'
    assert all(int(row['NNodes']) == plan['nodes'] and row['JobName'] == receipt['job_name']
               for row in attempts['allocations']), 'Allocation attempts disagree with the owned receipt'
    audit = read(job / 'campaign/scaling-audit.json')
    start, end = parent['start_epoch'], parent['end_epoch']
    screens, full = deduplicate(audit)
    final_seconds=end-start
    total_seconds=attempts['total_allocated_seconds']
    assert total_seconds>=final_seconds
    assert attempts['allocations'][-1]['State']=='COMPLETED'
    assert abs(attempts['allocations'][-1]['start_epoch']-start)<1
    gpu_hours=attempts['total_allocated_GPU_seconds']/3600
    prior_seconds=total_seconds-final_seconds
    original_submit=receipt.get('submitted_at')
    time_to_final_start=max(0,start-original_submit) if original_submit is not None else None
    rounds = [read(p) for p in sorted((job / 'campaign/state/rounds').glob('round-*.json'))]
    synthesis = []
    for record in rounds:
        brief = record['brief']
        path = brief.get('synthesis_receipt')
        synthesis.append(bool(brief.get('tournament_brief')) and not brief.get('report', '').startswith('reasoner failed:')
                         and bool(path) and read(Path(path))['status'] == 'complete')
    summary, nodes = hardware(job, start, end)
    slots = occupancy(job, audit, start, end)
    # Earlier scheduler allocations have no surviving telemetry/role manifest
    # in this trial. Their full cost remains; unobserved GPU time contributes
    # zero to the conservative overall estimate. Role/node plots describe the
    # final assigned allocation, rather than assigning fictitious old hosts.
    summary['final_allocation_overall_GPU_busy_percent']=summary['overall_GPU_busy_percent']
    summary['overall_GPU_busy_percent']*=final_seconds/total_seconds
    summary['overall_telemetry_coverage_percent']*=final_seconds/total_seconds
    slots['science_slot_occupancy_percent']*=final_seconds/total_seconds
    slots['fraction_time_below_10percent_science_slot_occupancy']=(
        slots['fraction_time_below_10percent_science_slot_occupancy']*final_seconds+prior_seconds)/total_seconds
    summary.update(job_id=job_id, nodes=plan['nodes'], inference=plan['inference_count'], science=plan['science_count'],
                   allocated_seconds=total_seconds, final_allocation_seconds=final_seconds,
                   prior_allocation_seconds=prior_seconds, allocation_attempt_count=attempts['allocation_attempts'],
                   prior_attempt_GPU_hours=4*plan['nodes']*prior_seconds/3600,
                   allocation_accounting=attempts, allocated_GPU_hours=gpu_hours,
                   time_from_submission_to_final_start_seconds=time_to_final_start,
                   queue_seconds=max(0,time_to_final_start-prior_seconds) if time_to_final_start is not None else parent.get('queue_seconds'),
                   final_attempt_queue_seconds=parent.get('queue_seconds'),
                   screen_trials=len(screens), full_context_trials=len(full),
                   screen_trials_per_allocated_GPU_hour=len(screens)/gpu_hours,
                   full_context_trials_per_allocated_GPU_hour=len(full)/gpu_hours,
                   clean_synthesis_rounds=sum(synthesis), recorded_rounds=len(rounds),
                   scientific_completion_checks=audit['required_campaign_completion_checks'],
                   scientific_completion_passed=audit['required_campaign_completion_checks_passed'],
                   **{k:v for k,v in slots.items() if k!='trace'})
    summary['role_and_node_utilization_scope']='Final assigned allocation interval. Overall busy, occupancy and throughput include all parent allocation attempts; prior unobserved intervals contribute zero busy/occupied rows, while retaining their GPU-hour cost.'
    tested=reviewed_science_threads(audit,rounds)
    assessed=completed_assessment_threads(rounds,tested)
    summary.update(reviewed_scientific_test_threads=len(assessed),
                   reviewed_scientific_test_hypotheses=len({h for n,h in assessed}),
                   reviewed_fresh_science_threads=len(tested),
                   hypotheses_with_reviewed_fresh_science=len({h for n,h in tested}),
                   completed_scientific_assessment_threads=len(assessed))
    return {'summary': summary, 'per_node': nodes, 'occupancy_trace': slots['trace']}


def evaluate(job, baseline=BASELINE):
    current = measure(job)
    old = measure(baseline)
    a, b = current['summary'], old['summary']
    gains = {key: a[key] / b[key] for key in ('screen_trials_per_allocated_GPU_hour',
             'full_context_trials_per_allocated_GPU_hour', 'science_GPU_busy_percent')}
    checks = {'scientific_campaign_completed_and_independently_verified': a['scientific_completion_passed'],
              'all_three_synthesis_rounds_succeeded': a['recorded_rounds'] == a['clean_synthesis_rounds'] == 3,
              'screen_throughput_gain_at_least_1p5': gains['screen_trials_per_allocated_GPU_hour'] >= 1.5,
              'full_context_throughput_gain_at_least_1p5': gains['full_context_trials_per_allocated_GPU_hour'] >= 1.5,
              'science_GPU_busy_gain_at_least_2': gains['science_GPU_busy_percent'] >= 2,
              'low_science_occupancy_fraction_at_most_0p30': a['fraction_time_below_10percent_science_slot_occupancy'] <= .30,
              'every_node_telemetry_coverage_at_least_95percent': a['minimum_node_telemetry_coverage_percent'] >= 95}
    final = {**checks, 'whole_allocation_GPU_busy_at_least_50percent': a['overall_GPU_busy_percent'] >= 50,
             'single_256_node_allocation': a['nodes'] == 256}
    return {'current': current, 'baseline': old, 'gains': gains, 'principle_gate': checks,
            'principle_gate_passed': all(checks.values()), 'final_gate': final,
            'final_gate_passed': all(final.values()),
            'GPU_busy_definition': 'Time-weighted GPU busy over every allocated GPU-second, including startup/drain. Unobserved intervals contribute zero to this conservative lower bound and are reported as missing coverage. Not FLOP utilization.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.job.resolve(), args.baseline.resolve())
    output = args.output or args.job / 'saturation-audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('gains', 'principle_gate', 'principle_gate_passed', 'final_gate', 'final_gate_passed')}, indent=2))
