#!/usr/bin/env python3
"""Check real pairwise tournaments before each recorded scientific round.

Rankings are advisory; this verifies workflow execution, not scientific truth.
Incomplete diagnostics remain explicitly separate from three-round completion.
"""
import argparse
import datetime
import hashlib
import json
import re
import time
from pathlib import Path


def verify(job):
    job=job.resolve();campaign=job/'campaign'
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=read(job/'submission.json')
    assert receipt['job_id'] in read(job.parent/'experiment.json')['owned_job_ids']
    from loam.config import load
    cfg=load(campaign/'campaign.yaml')
    state_path=campaign/'state/jnana.json';state=read(state_path)
    hypotheses={h['hypothesis_id']:h for h in state['hypotheses']}
    events=campaign/'state/admission-timings.jsonl'
    queued={}
    for line in events.read_text().splitlines():
        row=json.loads(line)
        if row['event']=='queued':queued[row['round']]=min(row['timestamp'],queued.get(row['round'],float('inf')))
    paths=sorted((campaign/'state/rounds').glob('round-*.json'))
    lower=read(job/'allocation.json')['started_at'];rows=[];issues=[]
    sources={str(p):sha(p) for p in [state_path,events,job/'allocation.json',campaign/'campaign.yaml',*paths]}
    matches=[]
    for match in state.get('tournament_state',{}).get('matches',[]):
        timestamp=float(match['match_id'])
        assert match['hypothesis1_id'] in hypotheses and match['hypothesis2_id'] in hypotheses
        assert match['hypothesis1_id']!=match['hypothesis2_id'] and match.get('overall_winner')
        matches.append((timestamp,match))
    for path in paths:
        record=read(path);n=record['round'];upper=queued[n]
        actual=[m for timestamp,m in matches if lower<=timestamp<=upper]
        pool=[h for h in hypotheses.values() if float(h['created_at'])<=upper]
        selected=len(record['threads'])
        if not actual:issues.append(f'Round{n} has no persisted pairwise competitions in its tournament window')
        if selected!=cfg.tournament.top_k:issues.append(f'Round{n} selected {selected}, configured {cfg.tournament.top_k}')
        rows.append({'round':n,'window_start':lower,'validation_admission_at':upper,
                     'pool_at_admission':len(pool),'selected':selected,
                     'completed_pairwise_comparisons':len(actual),
                     'match_ids':[m['match_id'] for m in actual]})
        lower=upper
    log=job/'controller.log';fallbacks=[]
    if log.exists():
        sources[str(log)]=sha(log)
        for line in log.read_text(errors='replace').splitlines():
            if 'tournament failed; falling back to the current pool ranking' not in line:continue
            match=re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[,.](\d+)',line)
            if match:
                at=datetime.datetime.fromisoformat(match[1]).astimezone().timestamp()+float('0.'+match[2])
                if rows and at<=rows[-1]['validation_admission_at']:fallbacks.append({'at':at})
            else:issues.append('A tournament fallback log could not be assigned to a completed-round window')
    if fallbacks:issues.append('A recorded round used the exception fallback instead of a completed tournament')
    result={'checked_at':time.time(),'job_id':receipt['job_id'],'rounds':rows,'issues':issues,
            'passed_for_recorded_rounds':bool(rows) and not issues,
            'meets_three_round_tournament_requirement':len(rows)>=3 and not issues,
            'fallbacks_before_final_recorded_admission':fallbacks,'sources':sources,
            'scope':'Pairwise match execution, generation pool and selected counts only. Elo is not evidence, and this does not replace science, synthesis or saturation gates.'}
    (job/'tournament-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path)
    verify(parser.parse_args().job)
