#!/usr/bin/env python3
"""Run against this campaign's already managed science server; no new jobs.

This entry point separates a long scheduler wait from inference readiness.
It never falls back across model routes or cancels scheduler resources.
"""
import argparse, asyncio, hashlib, json, logging, os, subprocess
from pathlib import Path
from fastmcp import Client
import httpx
from loam.config import load
from loam.orchestrator import Campaign
from loam.schemas import Hypothesis
from loam.tournament import Tournament

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
async def main(config):
    cfg=load(config);base=Path(cfg.execution.hpc.base_dir);progress=json.loads((base/'scale-progress.json').read_text())
    science_nodes=cfg.execution.hpc.allocation.nodes
    assert science_nodes==progress.get('scientific_allocation_nodes',science_nodes)
    for key,nodes in [('scientific_job',science_nodes)]:
        job=progress[key];assert job in progress['owned_job_ids']
        result=subprocess.run(['squeue','-h','-j',job,'-o','%T %D'],capture_output=True,text=True,check=True)
        assert result.stdout.strip()==f'RUNNING {nodes}',f'Owned scientific allocation not ready: {result.stdout.strip()}'
    if progress.get('inference_pool_experiment'):
        experiment=Path(progress['inference_pool_experiment']).resolve()
        pool_config=json.loads((experiment/'pool.json').read_text())
        inference_entries=[b for b in pool_config['backends'] if b.get('enabled',True)]
        assert {b['job_id'] for b in inference_entries}==set(progress['active_inference_job_ids'])
        relay=json.loads((experiment/'pool-service.json').read_text())
        for entry in inference_entries:
            assert entry['job_id'] in progress['owned_job_ids']
            result=subprocess.run(['squeue','-h','-j',entry['job_id'],'-o','%T %D'],capture_output=True,text=True,check=True)
            assert result.stdout.strip()=='RUNNING 1','Owned inference replica allocation is not ready'
        assert science_nodes+len(inference_entries)==progress['target_total_nodes']<=20
    else:
        infer=progress['inference_submission']['job_id'];assert infer in progress['owned_job_ids']
        result=subprocess.run(['squeue','-h','-j',infer,'-o','%T %D'],capture_output=True,text=True,check=True)
        assert result.stdout.strip()=='RUNNING 1','Owned inference allocation is not ready'
        inference_entries=[{'job_id':infer}]
        relay=json.loads((base/'state/nemotron/relay.json').read_text())
    os.environ['LOAM_NEMOTRON_TOKEN']=Path(relay['token_file']).read_text().strip()
    # Health checking is allowed while a gate is pending; campaign stages start
    # only after every configured route actually responds. Never print tokens.
    async with httpx.AsyncClient(trust_env=False,timeout=90) as client:
        for url,model,key in sorted({(s.base_url,s.model,s.api_key_env or '') for s in [*cfg.models.values(),cfg.tournament.generation,cfg.tournament.ranking]}):
            if not url:continue
            headers={'Authorization':'Bearer '+os.environ[key]} if key else {}
            body={'model':model,'messages':[{'role':'user','content':'Reply exactly LOAM_MODEL_READY'}],'max_tokens':128}
            if model=='nemotron-3.5-super-vl':body['chat_template_kwargs']={'enable_thinking':False}
            response=await client.post(url.rstrip('/')+'/chat/completions',json=body,headers=headers);response.raise_for_status();assert 'LOAM_MODEL_READY' in response.text,f'Model readiness marker absent for {model}'
    server=json.loads((base/'server-ready.json').read_text());cfg.execution.mcp_url=server['mcp_url']
    inputs=base/'inputs';venv='/e/project1/deepwing/meigooni1/agent/software/RFD4-Proteina/.venv';source=inputs/'verified-egfr-inputs/dual.cif';helper=inputs/'run_binder_generation.py';stages=inputs/'intact-egfr-energy';annotation=inputs/'intact-egfr/P00533.json'
    example={'id':'registration-example-only','condition':'dual','seed':100,'input_cif':str(source),'input_sha256':sha(source),'target_fasta':str(inputs/'verified-target.fasta'),'generation_script':str(helper),'generation_script_sha256':sha(helper)}
    follow={'id':'registration-example-only','seed':100,'screen_summary':str(base/'work/REPLACE_WITH_OWNED_BATCH/summary.json'),'screen_summary_sha256':'REPLACE_WITH_ACTUAL_HASH','target_annotation':str(annotation),'target_annotation_sha256':sha(annotation),'stages_directory':str(stages),'cofold_script_sha256':sha(stages/'cofold.py'),'mmgbsa_script_sha256':sha(stages/'mmgbsa.py')}
    summarize={'id':'registration-example-only','batch_dir':str(base/'batches/REPLACE_WITH_OWNED_BATCH'),'expected_per_condition':4}
    definitions=[('scale20_complete_binder_pipeline','run_binder_pipeline.py',example,['summary.json','generation_stage/generation_manifest.json','mpnn/seqs/*.fa','fold/pred.model_idx_*.cif','fold/scores.model_idx_*.npz']),('scale20_summarize_binder_pipeline','summarize_binder_pipeline.py',summarize,['analysis.json','full_analysis.json']),('scale20_intact_egfr_followup','run_intact_egfr_followup.py',follow,['followup-summary.json','validation-report.json','intact-geometry.json','mmgbsa/report.json','mmgbsa/FINAL_RESULTS_MMPBSA.dat'])]
    registrations=[]
    async with Client(cfg.execution.mcp_url,timeout=600) as client:
        pool=json.loads((await client.call_tool('pool_status',{})).content[0].text)
        assert pool['campaign_id']==cfg.execution.hpc.campaign_id and pool['total_tiles']==4*science_nodes
        for name,script,row,expects in definitions:
            result=await client.call_tool('register_tool',{'name':name,'script_path':str(inputs/script),'params':{k:('int' if k in ('seed','expected_per_condition') else 'str') for k in row if k!='id'},'description':'Operator-frozen, separately pilot-validated Jupiter science recipe. Read the rfd4-proteina skill for required fields and limitations. Registration examples are placeholders, not scientific rows.','example':row,'skip_smoke':True,'venv_path':venv,'expects':expects})
            registrations.append(json.loads(result.content[0].text))
    (base/'campaign-tool-registrations.json').write_text(json.dumps(registrations,indent=2)+'\n')
    progress['phase']='Owned scientific and inference allocations verified; campaign running';progress['verified_active_split']={'scientific_job':progress['scientific_job'],'scientific_nodes':science_nodes,'inference_job_ids':[b['job_id'] for b in inference_entries],'inference_nodes':len(inference_entries)}
    if not any(s.model.startswith('argo:') for s in [*cfg.models.values(),cfg.tournament.generation,cfg.tournament.ranking]):
        progress['unused_service_gates']={**progress.get('unused_service_gates',{}),**progress.get('gates',{})}
    progress['gates']={};(base/'scale-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    class PreparedTournament(Tournament):
        def run(self, goal, *, initial, brief=''):
            resumed=base/'resume-round-tournament-cache.json'
            completed=[int(p.stem.split('-')[1]) for p in(cfg.state_dir/'rounds').glob('round-*.json')]
            current=max(completed,default=0)+1
            if resumed.exists():
                cache=json.loads(resumed.read_text());state=cfg.state_dir/'jnana.json'
                core=goal.split('## Runtime execution observations (scientific goal unchanged)')[0].removesuffix('\n\n')
                if (cache['round']==current and sha(state)==cache['state_sha256']
                        and hashlib.sha256(core.encode()).hexdigest()==cache['scientific_goal_sha256']
                        and cfg.tournament.model_dump(mode='json')==cache['tournament_config']):
                    ranked=[Hypothesis.model_validate(h) for h in cache['rankings']]
                    assert len(ranked)==cfg.tournament.top_k and all(h.id in self.pool() for h in ranked)
                    logging.info('Reusing the completed, hash-verified round%d tournament; preserving owned scientific batches',current)
                    return ranked
            proof=base/'initial-tournament-cache.json'
            if initial and proof.exists():
                cache=json.loads(proof.read_text());state=cfg.state_dir/'jnana.json'
                if (sha(state)==cache['state_sha256']
                        and hashlib.sha256(goal.encode()).hexdigest()==cache['goal_sha256']
                        and cfg.tournament.model_dump(mode='json')==cache['tournament_config']):
                    prepared=json.loads((base/'initial-tournament-preparation.json').read_text())
                    ranked=[Hypothesis.model_validate(h) for h in prepared['rankings']]
                    assert len(ranked)==cfg.tournament.top_k and len({h.id for h in ranked})==len(ranked)
                    assert all(h.id in self.pool() for h in ranked)
                    logging.info('Using completed, hash-verified initial tournament prepared during queue wait')
                    return ranked
            return super().run(goal,initial=initial,brief=brief)
    tournament=PreparedTournament(cfg.tournament,cfg.state_dir,model_retries=cfg.model_retries,request_timeout=cfg.timeouts.model_request)
    await Campaign(cfg,tournament=tournament).run(cfg.rounds)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s');asyncio.run(main(a.config.resolve()))
