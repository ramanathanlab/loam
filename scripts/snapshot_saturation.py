#!/usr/bin/env python3
"""Bounded, credential-free runtime snapshot for a receipt-owned allocation."""
import argparse
import collections
import csv
import io
import json
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def tail_rows(path, limit=262144):
    if not path.exists():
        return []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        length = stream.tell()
        stream.seek(max(0, length-limit))
        content = stream.read().decode(errors='replace').splitlines()
    if length > limit:
        content = content[1:]  # The first record may have begun before the tail.
    rows = []
    for line in content:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def snapshot(job, window=120):
    plan, receipt = read(job/'plan.json'), read(job/'submission.json')
    assert receipt['job_id'] in read(job.parent/'experiment.json')['owned_job_ids']
    assert receipt['job_name'] == plan['job_name']
    result = {'job_id': receipt['job_id'], 'job_root': str(job), 'at': time.time()}
    query = subprocess.run(['scontrol','show','job',receipt['job_id'],'-o'],
                           capture_output=True,text=True,timeout=30)
    if query.returncode == 0:
        fields = dict(part.split('=',1) for part in query.stdout.split() if '=' in part)
        assert fields['JobName'] == receipt['job_name'] and fields['WorkDir'] == str(job)
        result['slurm'] = {k:fields.get(k) for k in ['JobState','Reason','RunTime','StartTime','EndTime','NumNodes']}
    else:
        query = subprocess.run(['sacct','-n','-X','-j',receipt['job_id'],'-P',
                                '--format=JobIDRaw,State,Start,End,ElapsedRaw,NNodes'],
                               capture_output=True,text=True,check=True,timeout=30)
        result['accounting'] = query.stdout.strip().splitlines()
    if (job/'status.json').exists():
        status = read(job/'status.json')
        result['controller'] = {k:status[k] for k in ['phase','updated_at','inference_ready',
              'inference_total','science_ready','controller_returncode','error_type'] if k in status}
    if not (job/'allocation.json').exists():
        return result
    allocation = read(job/'allocation.json')
    nodes = []
    now = result['at']
    for role in ('science','inference'):
        for host in allocation[role+'_nodes']:
            values, timestamps = [], []
            for path in (job/'campaign/telemetry').glob(receipt['job_id']+'-'+host+'*.jsonl'):
                for row in tail_rows(path):
                    assert row['job_id'] == receipt['job_id'] and row['host'].split('.')[0] == host
                    if not now-window <= row['timestamp'] <= now:
                        continue
                    gpu = list(csv.reader(io.StringIO(row.get('gpu_csv',''))))
                    if len(gpu) == 4 and all(len(g) >= 8 for g in gpu):
                        values.extend(float(g[2]) for g in gpu)
                        timestamps.append(row['timestamp'])
            nodes.append({'host':host,'role':role,'samples':len(timestamps),
                          'latest_age_seconds':now-max(timestamps) if timestamps else None,
                          'observed_GPU_busy_percent':sum(values)/len(values) if values else None})
    result['recent_GPU_window_seconds'] = window
    result['per_node'] = nodes
    for role in ('science','inference'):
        subset = [n['observed_GPU_busy_percent'] for n in nodes if n['role']==role and n['samples']]
        result[role+'_observed_recent_GPU_busy_percent'] = sum(subset)/len(subset) if subset else None
    admissions = tail_rows(job/'campaign/state/admission-timings.jsonl',limit=16*1024*1024)
    by_round = collections.defaultdict(collections.Counter)
    for row in admissions:
        by_round[row['round']][row['event']] += 1
    result['admission'] = {n:{**counts,'active':counts['started']-counts['finished'],
                             'waiting':counts['queued']-counts['started']} for n,counts in by_round.items()}
    result['completed_rounds'] = []
    for path in sorted((job/'campaign/state/rounds').glob('round-*.json')):
        record = read(path)
        result['completed_rounds'].append({'round':record['round'],'selected':len(record['threads']),
            'thread_errors':sum(bool(t.get('error')) for t in record['threads']),
            'accepted_claims':sum(len(t.get('accepted',[])) for t in record['threads']),
            'reasoner_failed':record['brief']['report'].startswith('reasoner failed:')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job',type=lambda p:Path(p).resolve())
    parser.add_argument('--window',type=float,default=120)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = snapshot(args.job,args.window)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
