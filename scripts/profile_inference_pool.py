#!/usr/bin/env python3
"""Paired pool profiles using real frozen scientific context; no science jobs.

These are inference/structured-review probes, not new binder evidence. Both
profiles use identical requests and the successful campaign's thinking modes.
The first pass is retained as cold-prefix cost; timed warm profiles follow.
"""
import argparse,asyncio,collections,fcntl,hashlib,json,math,os,statistics,time
from pathlib import Path
import httpx

REPO=Path(__file__).resolve().parents[1]
BASELINE=REPO/'runs/jupiter-scale20-20261113T042053Z'


def percentile(values,p):
    ordered=sorted(values);position=(len(ordered)-1)*p;lower=math.floor(position);upper=math.ceil(position)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def atomic(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def cases():
    audit=json.loads((BASELINE/'scaling-audit.json').read_text());groups=collections.defaultdict(list)
    for row in audit['candidate_rows']:groups[row['batch_id']].append(row)
    cohorts=[]
    for bid,rows in sorted(groups.items()):
        if collections.Counter(c['condition'] for c in rows)!=collections.Counter({'dual':4,'one_region':4,'ridge':4}):continue
        means={arm:statistics.mean(c['best_iptm'] for c in rows if c['condition']==arm) for arm in ['dual','one_region','ridge']}
        dual=statistics.mean(c['mean_pairwise_target_aligned_binder_CA_RMSD_A'] for c in rows if c['condition']=='dual')
        control=[c['mean_pairwise_target_aligned_binder_CA_RMSD_A'] for c in rows if c['condition']!='dual'];threshold=statistics.mean(control)-statistics.stdev(control)
        cohorts.append({'batch_id':bid,'dual_pose_RMSD_mean_A':dual,'pooled_control_pose_RMSD_mean_A':statistics.mean(control),'pooled_control_pose_RMSD_sample_SD_A':statistics.stdev(control),'condition_mean_best_iptm':means,'expected':{'gate_pass':dual<=threshold,'threshold_A':threshold,'highest_confidence_arm':max(means,key=means.get),'measured_affinity':False}})
    selected=cohorts[:4]+[c for c in cohorts if c['expected']['gate_pass']][:2]
    assert len(selected)==6
    context_rows=[{k:c[k] for k in ['candidate_id','condition','binder_sequence','best_iptm','dual_contact_models','mean_pairwise_target_aligned_binder_CA_RMSD_A','stage_seconds']} for c in audit['candidate_rows']]
    setup=(BASELINE/'goal.md').read_text()+'\n'+(REPO/'skills/diffusion-models/rfd4-proteina/SKILL.md').read_text()
    schema={'type':'object','properties':{'gate_pass':{'type':'boolean'},'threshold_A':{'type':'number'},'highest_confidence_arm':{'type':'string','enum':['dual','one_region','ridge']},'measured_affinity':{'type':'boolean'},'analysis':{'type':'string'}},'required':['gate_pass','threshold_A','highest_confidence_arm','measured_affinity','analysis'],'additionalProperties':False}
    result=[]
    for i in range(8):
        sample=selected[i%6];count=[12,96,320][i%3]
        frozen=json.dumps(context_rows[:count],separators=(',',':'))
        instruction=('This is a read-only inference benchmark on previously audited frozen binder evidence. Do not submit jobs, run tools, invent new evidence or infer measured affinity. The recipe context and earlier inventory follow for realistic context size.\n'+setup+'\nFROZEN INVENTORY:\n'+frozen)
        task={k:v for k,v in sample.items() if k!='expected'}
        user='Evaluate this specific cohort. Gate: dual mean pairwise RMSD <= pooled control mean minus control candidate sample SD. Identify the highest mean-ipTM arm. Report no measured affinity. Explain arithmetic, control limitations and what the metrics do and do not support in 200-400 words. Return submit_review.\n'+json.dumps(task)
        body={'model':'nemotron-3.5-super-vl','messages':[{'role':'system','content':instruction},{'role':'user','content':user}],
              'tools':[{'type':'function','function':{'name':'submit_review','description':'Submit the bounded review of the supplied audited cohort. This does not execute scientific work.','parameters':schema}}],
              'tool_choice':{'type':'function','function':{'name':'submit_review'}},'max_tokens':32768 if i>=6 else 8192,'temperature':0}
        if i<6:body['chat_template_kwargs']={'enable_thinking':False}
        result.append({'id':f'frozen-review-{i}','thinking_enabled':i>=6,'source_batch_id':sample['batch_id'],'expected':sample['expected'],'body':body})
    return result


def validate(data,expected):
    try:
        choice=data['choices'][0];calls=choice['message']['tool_calls'];assert len(calls)==1 and calls[0]['function']['name']=='submit_review'
        answer=json.loads(calls[0]['function']['arguments']);assert answer['gate_pass'] is expected['gate_pass']
        assert math.isfinite(answer['threshold_A']) and abs(answer['threshold_A']-expected['threshold_A'])<.005
        assert answer['highest_confidence_arm']==expected['highest_confidence_arm'] and answer['measured_affinity'] is False
        assert choice['finish_reason']!='length' and isinstance(answer['analysis'],str) and len(answer['analysis'])>60
        return True,None
    except (KeyError,IndexError,TypeError,ValueError,AssertionError):return False,'structured_review_check_failed'


async def run(root,repeat,replicas=None):
    lock=(root/'inference-profile.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for receipt in (root/'trials').glob('*/controller-watch.json'):
        if json.loads(receipt.read_text()).get('state')=='controller running':raise RuntimeError('Inference profiles must run between scientific trials')
    pool_config=json.loads((root/'pool.json').read_text());active=[b['id'] for b in pool_config['backends'] if b.get('enabled',True)]
    if replicas is not None:assert replicas==len(active)>=2
    progress=root/'inference-profile-progress.json';atomic(progress,{'pid':os.getpid(),'phase':f'waiting for {replicas or 2} owned model APIs','started_at':time.time()})
    deadline=time.time()+2400
    while not(root/'pool-service.json').exists():
        if time.time()>deadline:raise TimeoutError('Pool gateway did not start')
        await asyncio.sleep(2)
    service=json.loads((root/'pool-service.json').read_text());token=Path(service['token_file']).read_text().strip();headers={'Authorization':'Bearer '+token}
    requests=cases();request_file=root/'inference-profile-requests.jsonl';frozen=''.join(json.dumps(c)+'\n' for c in requests)
    if request_file.exists():assert request_file.read_text()==frozen,'Frozen requests changed; do not overwrite the paired comparison'
    else:request_file.write_text(frozen)
    results=json.loads((root/'inference-profile.json').read_text())['results'] if replicas is not None and (root/'inference-profile.json').exists() else []
    async with httpx.AsyncClient(trust_env=False,timeout=600,limits=httpx.Limits(max_connections=64,max_keepalive_connections=32),headers=headers) as client:
        async def status():
            until=time.time()+60
            while True:
                try:
                    response=await client.get(service['status_url']);response.raise_for_status();return response.json()
                except (httpx.TransportError,httpx.HTTPStatusError):
                    if time.time()>until:raise
                    await asyncio.sleep(2)
        while (await status())['ready_replicas']<(replicas or 2):
            if time.time()>deadline:raise TimeoutError(f'{replicas or 2} owned inference APIs were not ready')
            await asyncio.sleep(5)
        async def configure(ids,policy,phase):
            config=json.loads((root/'pool.json').read_text());config.update(policy=policy,phase=phase)
            for b in config['backends']:b['enabled']=b['id'] in ids
            atomic(root/'pool.json',config)
            until=time.time()+30
            while True:
                current=await status()
                if current['phase']==phase and {b['id'] for b in current['backends'] if b['enabled']}==set(ids):return
                if time.time()>until:raise TimeoutError('Pool did not apply profile configuration')
                await asyncio.sleep(.5)
        async def phase(ids,policy,label,items,concurrency):
            await configure(ids,policy,label);atomic(progress,{'pid':os.getpid(),'phase':label,'started_at':time.time(),'replicas':ids})
            gate=asyncio.Semaphore(concurrency);started=time.time()
            async def one(i,case):
                async with gate:
                    now=time.time();row={'case':case['id'],'repeat_index':i,'thinking_enabled':case['thinking_enabled']}
                    try:
                        response=await client.post(service['base_url']+'/chat/completions',json=case['body']);row.update(status=response.status_code,backend=response.headers.get('x-loam-replica'))
                        if response.status_code==200:
                            data=response.json();ok,error=validate(data,case['expected']);row.update(ok=ok,error=error,usage=data.get('usage'),finish_reason=data['choices'][0].get('finish_reason'))
                        else:row.update(ok=False,error='HTTP_'+str(response.status_code))
                    except Exception as error:row.update(ok=False,error=type(error).__name__)
                    row['seconds']=time.time()-now;return row
            rows=await asyncio.gather(*(one(i,case) for i,case in enumerate(items)));elapsed=time.time()-started;latencies=sorted(x['seconds'] for x in rows)
            result={'label':label,'replicas':ids,'policy':policy,'concurrency':concurrency,'requests':len(rows),'started_at':started,'finished_at':time.time(),'wall_seconds':elapsed,'passed':sum(x['ok'] for x in rows),'successful_reviews_per_second':sum(x['ok'] for x in rows)/elapsed,'median_request_seconds':statistics.median(latencies),'p95_request_seconds':percentile(latencies,.95),'completion_tokens':sum((x.get('usage') or {}).get('completion_tokens',0) for x in rows),'rows':rows};results.append(result);atomic(root/'inference-profile.json',{'scope':'Frozen scientific-context structured reviews, not new binder evidence or end-to-end throughput','results':results});print(json.dumps({k:result[k] for k in ['label','replicas','passed','requests','wall_seconds','median_request_seconds','p95_request_seconds','successful_reviews_per_second']}),flush=True)
        if replicas is not None:
            known={r['label'] for r in results}
            for replica in active:
                if f'cold-prefix-{replica}' not in known:await phase([replica],'least_loaded',f'cold-prefix-{replica}',requests,4)
            await phase(active,'round_robin',f'{replicas}-replica-round-robin-warm',requests*repeat,12)
            await phase(active,'least_loaded_affinity',f'{replicas}-replica-affinity-warm',requests*repeat,12)
            await configure(active,'least_loaded_affinity','campaign-ready')
            atomic(progress,{'pid':os.getpid(),'phase':f'{replicas}-replica profiles complete','finished_at':time.time(),'results':'inference-profile.json'})
            return
        for replica in ['nemotron-01','nemotron-02']:
            await phase([replica],'least_loaded',f'cold-prefix-{replica}',requests,4)
        repeated=requests*repeat
        await phase(['nemotron-01'],'least_loaded','one-replica-warm',repeated,12)
        await phase(['nemotron-01','nemotron-02'],'round_robin','two-replica-round-robin-warm',repeated,12)
        await phase(['nemotron-01','nemotron-02'],'least_loaded_affinity','two-replica-affinity-warm',repeated,12)
        await configure(['nemotron-01','nemotron-02'],'least_loaded_affinity','campaign-ready')
    atomic(progress,{'pid':os.getpid(),'phase':'initial one/two-replica profiles complete','finished_at':time.time(),'results':'inference-profile.json'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',type=lambda x:Path(x).resolve());p.add_argument('--repeat',type=int,default=3);p.add_argument('--replicas',type=int);a=p.parse_args();asyncio.run(run(a.experiment,a.repeat,a.replicas))
