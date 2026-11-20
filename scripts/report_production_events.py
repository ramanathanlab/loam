"""Report-only event timing for independently audited canonical scientific rows.
A copied successful summary without timing may borrow ONLY the actual timed
production of byte-identical five poses with the same owner, round and inputs.
Raw audit data, scientific counts, resource costs and gate integrals stay intact.
"""
import math
from pathlib import Path

def finish(row):
 start,seconds=row.get('started_at'),row.get('seconds')
 if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in [start,seconds]):return None
 if seconds<0:return None
 return start+seconds

def signature(candidate):
 keys=('round','hypothesis','condition','seed','input_sha256','target_sequence','binder_sequence')
 if any(candidate.get(k) is None for k in keys):return None
 models=candidate.get('models',[])
 if len(models)!=5 or not all(isinstance(m.get('sha256'),str) and len(m['sha256'])==64 for m in models):return None
 return tuple(candidate[k] for k in keys)+(tuple(m['sha256'] for m in models),)

def production_events(audit,screens,full):
 rows={(r['batch_id'],r['row']):r for r in audit['row_timings']};events=[];resolutions=[]
 for kind,values in [('screen',screens),('expanded',full)]:
  for candidate in values:
   key=(candidate['batch_id'],Path(candidate['summary_path']).parent.name);row=rows[key];end=finish(row)
   if end is None:
    assert kind=='screen','Full-context production has no authoritative timing'
    wanted=signature(candidate);assert wanted is not None and row.get('recorded_owner') and row.get('returncode')==0
    origins=[]
    for source in audit['candidate_rows']:
     if signature(source)!=wanted:continue
     source_key=(source['batch_id'],Path(source['summary_path']).parent.name);native=rows[source_key];at=finish(native)
     if at is not None and native.get('recorded_owner') and native.get('returncode')==0 and native.get('tool')=='scale20_complete_binder_pipeline' and native.get('gpus',0)>0:origins.append((at,source_key,native))
    assert origins,'No timed byte-identical five-pose same-owner/same-round/same-input production exists; do not invent a timestamp'
    end,origin,native=min(origins,key=lambda x:(x[0],x[1]))
    resolutions.append({'canonical_batch_id':key[0],'canonical_row':key[1],'source_batch_id':origin[0],'source_row':origin[1],'hypothesis':candidate['hypothesis'],'round':candidate['round'],'condition':candidate['condition'],'seed':candidate['seed'],'input_sha256':candidate['input_sha256'],'binder_sequence':candidate['binder_sequence'],'five_coordinate_sha256':[m['sha256'] for m in candidate['models']],'original_canonical_timing':dict(row),'source_recorded_timing':dict(native),'production_time':end,'scope':'Report timeline only. Exact five-coordinate identity and matching owning hypothesis, round, condition, seed, target input/sequence and binder required. One canonical scientific trial counted once; raw copied status and both attempts retained. No science, GPU-hour cost or saturation gate changed.'})
   events.append({'time':end,'round':candidate['round'],'kind':kind,'files':5})
 return sorted(events,key=lambda e:e['time']),resolutions
