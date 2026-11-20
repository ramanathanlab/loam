#!/usr/bin/env python3
"""Read-only phase-duration and sampled-GPU profile of audited scientific rows.

Phase boundaries are inferred from the ordered subprocess durations in the
frozen recipes. They are approximate, not profiler traces: GPU samples have
about ten-second spacing and unknown per-sample internal driver windows.
"""
import argparse
import bisect
import collections
import csv
import hashlib
import io
import json
import statistics
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def quantile(values,p):
    ordered=sorted(values);position=(len(ordered)-1)*p;lo=int(position);hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo) if ordered else None


def profile(campaign,output):
    output.mkdir(parents=True,exist_ok=True)
    audit=read(campaign/'scaling-audit.json')
    assert audit['model_coordinates_independently_checked']
    telemetry=collections.defaultdict(list)
    for path in (campaign/'telemetry').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row.get('role')!='science':continue
            host=row['host'].split('.')[0]
            for gpu in csv.reader(io.StringIO(row.get('gpu_csv',''))):
                if len(gpu)>=8:
                    telemetry[(host,str(int(gpu[0])))].append((float(row['timestamp']),float(gpu[2])))
    for key in telemetry:telemetry[key].sort()
    row_times={(r['batch_id'],r['row']):r for r in audit['row_timings']}
    phases=collections.defaultdict(lambda:{'durations':[],'row_seconds':0,'samples':[],'rows':0})
    assignments=collections.defaultdict(list);row_records=[]
    for kind,candidates,order in [('screen',audit['candidate_rows'],['RFD4-Proteina','proteinmpnn','chai-lab']),
                                  ('full_context',audit['intact_followups'],['cofold','mmgbsa'])]:
        for candidate in candidates:
            path=Path(candidate['summary_path']);folder=path.parent;summary=read(path)
            row=row_times[(candidate['batch_id'],folder.name)];status=read(folder/'status.json')
            pin=str(status.get('gpu',''))
            if pin not in {'0','1','2','3'} or not row.get('started_at'):continue
            elapsed=float(summary['elapsed_seconds']);stages=summary['stage_seconds']
            # Python imports/wrapper setup precede main(); local filesystem and
            # geometry bookkeeping between stages are not separately timestamped.
            front=max(0,float(row['seconds'])-elapsed)
            origin=float(row['started_at'])+front
            points=telemetry[(row['host'].split('.')[0],pin)]
            timestamps=[point[0] for point in points]
            cumulative=0
            for stage in order:
                seconds=float(stages[stage]);phase=kind+':'+stage
                lo=origin+cumulative;hi=lo+seconds;cumulative+=seconds
                phases[phase]['durations'].append(seconds);phases[phase]['row_seconds']+=float(row['seconds']);phases[phase]['rows']+=1
                for t,value in points[bisect.bisect_left(timestamps,lo):bisect.bisect_left(timestamps,hi)]:
                    assignments[(row['host'].split('.')[0],pin,t)].append((phase,value))
            row_records.append({'kind':kind,'summary_path':str(path),'host':row['host'],'gpu':pin,
                'process_seconds':row['seconds'],'summary_elapsed_seconds':elapsed,
                'ordered_stage_seconds':stages,'inferred_main_start':origin,
                'unassigned_in_main_seconds':elapsed-cumulative})
    ambiguous=0
    for observations in assignments.values():
        if len(observations)!=1:
            ambiguous+=1;continue
        phase,value=observations[0];phases[phase]['samples'].append(value)
    summaries=[]
    for phase,values in phases.items():
        samples=values['samples'];durations=values['durations']
        summaries.append({'phase':phase,'recorded_successful_rows':values['rows'],
            'median_stage_wall_seconds':statistics.median(durations),'p95_stage_wall_seconds':quantile(durations,.95),
            'fraction_recorded_row_wall_time_in_stage':sum(durations)/values['row_seconds'],
            'GPU_samples':len(samples),'mean_sampled_GPU_busy_percent':statistics.mean(samples) if samples else None,
            'median_sampled_GPU_busy_percent':statistics.median(samples) if samples else None,
            'fraction_samples_at_least_10percent_busy':sum(v>=10 for v in samples)/len(samples) if samples else None})
    result={'generated_at':time.time(),'campaign':str(campaign),'audit_sha256':hashlib.sha256((campaign/'scaling-audit.json').read_bytes()).hexdigest(),
        'scope':'Eligible successful audited rows; includes repeated input trials as operational observations, not independent scientific output. Approximate phase alignment from sequential subprocess durations; not direct CUDA/kernel traces or proof that a model load caused an idle sample.',
        'boundary_uncertainty':'At least the approximately10-second telemetry interval near boundaries, plus untimestamped within-main bookkeeping. No samples assigned to multiple row/phase intervals are counted.',
        'row_count':len(row_records),'ambiguous_samples_excluded':ambiguous,'phases':summaries,'rows':row_records}
    (output/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'phases.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]) if summaries else ['phase']);writer.writeheader();writer.writerows(summaries)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
        labels=[s['phase'].replace('full_context:','intact / ').replace('screen:','screen / ') for s in summaries]
        axes[0].barh(labels,[s['median_stage_wall_seconds'] for s in summaries],color='#147f78')
        axes[0].set(xlabel='Median subprocess wall time (s)',title='Time inside recorded scientific rows')
        axes[1].barh(labels,[s['mean_sampled_GPU_busy_percent'] or 0 for s in summaries],color='#c6812f')
        axes[1].set(xlabel='Mean sampled GPU busy (%)',xlim=(0,100),title='Approximate phase-aligned GPU activity')
        for ax in axes:ax.grid(axis='x',alpha=.2)
        fig.savefig(output/'phase-profile.png',dpi=150);fig.savefig(output/'phase-profile.svg');plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('campaign',type=lambda p:Path(p).resolve());parser.add_argument('output',type=lambda p:Path(p).resolve());args=parser.parse_args();profile(args.campaign,args.output)
