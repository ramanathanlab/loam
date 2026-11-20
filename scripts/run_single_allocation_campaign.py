#!/usr/bin/env python3
"""Run frozen scientific tools/tournaments against services in this one job."""
import argparse,asyncio,hashlib,json,logging,os,time,uuid
from pathlib import Path
from fastmcp import Client
from loam.config import load
from loam.orchestrator import Campaign
from loam.tournament import Tournament
from loam.allocation import validate_role,validate_campaign_paths

SOFTWARE=Path(os.environ.get('LOAM_SOFTWARE_ROOT',Path(__file__).resolve().parents[2]/'software'))
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
async def call(client,name,args):
    r=await client.call_tool(name,args)
    if r.is_error:raise RuntimeError('MCP '+name+' failed: '+str(r.content)[:2000])
    return json.loads(r.content[0].text)

async def register(client,base):
    inputs=base/'inputs';venv=str(SOFTWARE/'RFD4-Proteina/.venv');stages=inputs/'intact-egfr-energy-pathsafe';annotation=inputs/'intact-egfr/P00533.json'
    example={'id':'registration-example-only','condition':'dual','seed':100,'input_cif':str(inputs/'verified-egfr-inputs/dual.cif'),'input_sha256':sha(inputs/'verified-egfr-inputs/dual.cif'),'target_fasta':str(inputs/'verified-target.fasta'),'generation_script':str(inputs/'run_binder_generation.py'),'generation_script_sha256':sha(inputs/'run_binder_generation.py')}
    follow={'id':'registration-example-only','seed':100,'screen_summary':str(base/'batches/REPLACE/summary.json'),'screen_summary_sha256':'REPLACE','target_annotation':str(annotation),'target_annotation_sha256':sha(annotation),'stages_directory':str(stages),'cofold_script_sha256':sha(stages/'cofold.py'),'mmgbsa_script_sha256':sha(stages/'mmgbsa.py')}
    summarize={'id':'registration-example-only','batch_dir':str(base/'batches/REPLACE'),'expected_per_condition':4}
    definitions=[('scale20_complete_binder_pipeline','run_binder_pipeline.py',example,['summary.json','generation_stage/generation_manifest.json','mpnn/seqs/*.fa','fold/pred.model_idx_*.cif','fold/scores.model_idx_*.npz']),('scale20_summarize_binder_pipeline','summarize_binder_pipeline.py',summarize,['analysis.json','full_analysis.json']),('scale20_intact_egfr_followup','run_intact_egfr_followup.py',follow,['followup-summary.json','validation-report.json','intact-geometry.json','mmgbsa/report.json','mmgbsa/FINAL_RESULTS_MMPBSA.dat'])]
    registrations=[]
    for name,script,row,expects in definitions:
        registrations.append(await call(client,'register_tool',{'name':name,'script_path':str(inputs/script),'params':{k:'int' if k in ('seed','expected_per_condition') else 'str' for k in row if k!='id'},'description':'Operator-frozen production Jupiter science recipe; preserve scientific parameters; version1 only.','example':row,'skip_smoke':True,'venv_path':venv,'expects':expects}))
    (base/'campaign-tool-registrations.json').write_text(json.dumps(registrations,indent=2)+'\n')
    return example,follow

async def wait_batch(client,base,batch,deadline=5000):
    start=time.monotonic()
    while time.monotonic()-start<deadline:
        d=await call(client,'batch_status',{'batch_id':batch['batch_id']})
        folder=base/'batches'/batch['batch_id'];metadata=read(folder/'batch.json');terminal=sum((q/'done').exists() and (q/'status.json').exists() for q in folder.glob('run_*'))
        if terminal==metadata['total']:return d
        await asyncio.sleep(15)
    raise TimeoutError('Scientific batch did not finish')

async def smoke(client,base,example,follow):
    hypothesis=str(uuid.uuid4());out=base/'smoke';out.mkdir();rows=[]
    for arm in ['dual','one_region','ridge']:
        source=base/'inputs/verified-egfr-inputs'/f'{arm}.cif'
        for i in range(1,5):
            seed=int.from_bytes(hashlib.sha256(f'{hypothesis}:{arm}:{i}'.encode()).digest()[:4],'big')&0x7fffffff
            rows.append({**example,'id':f'{arm}-{i:02}','condition':arm,'seed':seed,'input_cif':str(source),'input_sha256':sha(source)})
    manifest=out/'screen.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows));labels={'hypothesis_id':hypothesis,'run_id':'r000-01'}
    batch=await call(client,'run_batch',{'tool':'scale20_complete_binder_pipeline','tool_version':1,'inputs_path':str(manifest),'gpus':1,'timeout_each':5000,'venv_path':str(SOFTWARE/'RFD4-Proteina/.venv'),'labels':labels});print('smoke screen batch',batch['batch_id'],flush=True)
    record=base/'state/hypotheses'/hypothesis/'runs/r000-01.json';record.parent.mkdir(parents=True);record.write_text(json.dumps({'run_id':'r000-01','batches':[batch['batch_id']],'state':'submitted','operator_pilot':True})+'\n')
    await wait_batch(client,base,batch);summaries=[]
    for p in (base/'batches'/batch['batch_id']).glob('run_*/summary.json'):
        s=read(p)
        if s.get('status')=='passed':summaries.append((p,s))
    assert len(summaries)>=10,'Too few complete frozen screen trials for infrastructure validation'
    selected=[]
    for arm in ['dual','one_region','ridge']:
        choices=[(p,s) for p,s in summaries if s['condition']==arm and 'C' not in s['binder_sequence']];assert choices,'No eligible cysteine-free pilot candidate for '+arm
        selected.append(max(choices,key=lambda x:x[1]['best_iptm']))
    follow_rows=[]
    for p,s in selected:
        follow_rows.append({**follow,'id':s['condition']+'-'+sha(p)[:16],'seed':int.from_bytes(hashlib.sha256((hypothesis+sha(p)).encode()).digest()[:4],'big')&0x7fffffff,'screen_summary':str(p),'screen_summary_sha256':sha(p)})
    manifest=out/'followups.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in follow_rows));batch2=await call(client,'run_batch',{'tool':'scale20_intact_egfr_followup','tool_version':1,'inputs_path':str(manifest),'gpus':1,'timeout_each':5000,'venv_path':str(SOFTWARE/'RFD4-Proteina/.venv'),'labels':{'hypothesis_id':hypothesis,'run_id':'r000-02'}});print('smoke followup batch',batch2['batch_id'],flush=True)
    record=record.with_name('r000-02.json');record.write_text(json.dumps({'run_id':'r000-02','batches':[batch2['batch_id']],'state':'submitted','operator_pilot':True})+'\n');await wait_batch(client,base,batch2)
    reports=[read(p) for p in (base/'batches'/batch2['batch_id']).glob('run_*/followup-summary.json')];assert len(reports)==3 and all(r['status']=='passed' and r['target_disulfides']==25 for r in reports)
    result={'passed':True,'screen_complete':len(summaries),'full_context_complete':len(reports),'batches':[batch['batch_id'],batch2['batch_id']],'selection_policy':'Infrastructure validation advances best-ipTM candidate per arm solely to exercise all full production stages; no biological hypothesis claim, excluded from formal tournament comparisons','single_parent_job':os.environ['SLURM_JOB_ID'],'completed_at':time.time()};(base/'smoke-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

def configure_controller_gc(plan):
    """Optional process-local collector frequency; never disable collection.

    Only the campaign-controller interpreter calls this hook. Inference and
    scientific workers run in separate processes with their original policy.
    """
    import gc
    before = gc.get_threshold()
    requested = plan.get('controller_gc_threshold0')
    if requested is not None:
        if type(requested) is not int or not 700 <= requested <= 100000:
            raise ValueError('controller_gc_threshold0 must be an integer in [700, 100000]')
        if not gc.isenabled():
            raise ValueError('Controller GC tuning requires automatic collection to be enabled')
        gc.set_threshold(requested, *before[1:])
    return {'requested_threshold0': requested, 'thresholds_before': list(before),
            'thresholds_applied': list(gc.get_threshold()), 'automatic_gc_enabled': gc.isenabled()}


async def main(job):
    allocation=validate_role(job/'allocation.json','science');plan=read(job/'plan.json');base=Path(plan['campaign']);cfg=load(base/'campaign.yaml');cfg.execution.mcp_url=read(base/'server-ready.json')['mcp_url'];assert cfg.execution.hpc.allocation.nodes==len(allocation['science_nodes'])
    validate_campaign_paths(cfg,base)
    policy=configure_controller_gc(plan)
    diagnostic=job/'diagnostics';diagnostic.mkdir(exist_ok=True)
    (diagnostic/'controller-gc-policy.json').write_text(json.dumps(
        {'at':time.time(),'controller_pid':os.getpid(),'source_sha256':sha(Path(__file__)),**policy},indent=2)+'\n')
    async with Client(cfg.execution.mcp_url,timeout=600) as client:
        pool=await call(client,'pool_status',{});assert pool['total_tiles']==len(allocation['science_nodes'])*plan.get('gpus_per_node',4)
        example,follow=await register(client,base)
        if plan['run_mode']=='smoke':await smoke(client,base,example,follow);return
    tournament=Tournament(cfg.tournament,cfg.state_dir,model_retries=cfg.model_retries,request_timeout=cfg.timeouts.model_request)
    await Campaign(cfg,tournament=tournament).run(cfg.rounds)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s');asyncio.run(main(Path(__import__('sys').argv[1]).resolve()))
