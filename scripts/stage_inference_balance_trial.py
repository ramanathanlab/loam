#!/usr/bin/env python3
"""Stage an independent, matched scientific trial; never submit scheduler jobs."""
import argparse, copy, hashlib, json, re, shutil, socket, time, uuid
from pathlib import Path
import yaml
from loam.config import load
from prepare_pathsafe_energy_stage import prepare


def stage(root, inference, science, label=None):
    experiment=json.loads((root/'experiment.json').read_text())
    assert inference+science==experiment['node_budget']==20
    for receipt in (root/'trials').glob('*/controller-watch.json'):
        if json.loads(receipt.read_text()).get('state')=='controller running':raise RuntimeError('Finish the active scientific trial before staging another ratio')
    baseline=Path(experiment['baseline']);name=f'{inference:02d}-infer-{science:02d}-science'
    if label:
        assert re.fullmatch(r'[a-z0-9][a-z0-9-]{0,23}',label)
        name+='-'+label
    trial=root/'trials'/name
    if trial.exists():raise RuntimeError('Trial already exists; inspect it instead of overwriting')
    pool=json.loads((root/'pool.json').read_text());active=[b for b in pool['backends'] if b.get('enabled',True)]
    assert len(active)==inference
    assert all(b['job_id'] in experiment['owned_job_ids'] for b in active)
    service=json.loads((root/'pool-service.json').read_text())
    cfg=yaml.safe_load((baseline/'campaign.yaml').read_text())
    trial.mkdir(parents=True);shutil.copytree(baseline/'inputs',trial/'inputs',ignore=shutil.ignore_patterns('__pycache__'))
    # Identifier-only compatibility fix; the three scientific recipes stay frozen.
    metadata_helper=Path(__file__).resolve().parents[1]/'skills/diffusion-models/rfd4-proteina/scripts/make_intact_followup_manifest.py'
    shutil.copyfile(metadata_helper,trial/'inputs'/metadata_helper.name)
    prepare(trial)
    (trial/'work').mkdir();(trial/'state').mkdir()
    cid='balance-'+root.name.split('-')[-1]+'-'+name
    # Rebase every baseline path before overriding the resource split.
    cfg=yaml.safe_load(yaml.safe_dump(cfg).replace(str(baseline),str(trial)).replace('jupiter-scale20-20261113',cid))
    cfg['rounds']=3
    for model in [*cfg['models'].values(),cfg['tournament']['generation'],cfg['tournament']['ranking']]:
        assert model['model']=='nemotron-3.5-super-vl'
        model['base_url']=service['base_url']
        assert model['api_key_env']=='LOAM_NEMOTRON_TOKEN'
    for role in cfg['models'].values():
        assert role['settings']['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    for model in (cfg['tournament']['generation'],cfg['tournament']['ranking']):
        assert not model.get('settings')
    hpc=cfg['execution']['hpc'];hpc['allocation']['nodes']=science;hpc['allocation']['walltime']='06:00:00'
    hpc['inspect_roots']=[str(trial),'/e/project1/deepwing/meigooni1/agent/software']
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));hpc['port']=sock.getsockname()[1]
    cfg['execution']['setup']=cfg['execution']['setup'].replace('one separate owned Nemotron node plus nineteen scientific\nnodes, 76 scientific GPU slots',f'{inference} separate owned Nemotron nodes plus {science} scientific nodes, {4*science} scientific GPU slots')
    cfg['execution']['setup']=cfg['execution']['setup'].replace('one separate owned Nemotron node plus nineteen scientific nodes, 76 scientific GPU slots',f'{inference} separate owned Nemotron nodes plus {science} scientific nodes, {4*science} scientific GPU slots')
    cfg['execution']['setup']=cfg['execution']['setup'].replace('stages_directory=data/intact-egfr-energy,','stages_directory=data/intact-egfr-energy-pathsafe,')
    oldgoal=(baseline/'goal.md').read_text();core=oldgoal.split('## Runtime execution observations')[0].strip()
    # Retain generic successful execution guidance, excluding old batch IDs/repairs.
    generic=oldgoal[oldgoal.index('INTACT MANIFEST FAST PATH'):]
    pin_start=generic.index('OPERATOR TOOL VERSION PIN:');pin_end=generic.index('EXACT ARTIFACT NAVIGATION:')
    generic=generic[:pin_start]+('OPERATOR TOOL VERSION PIN: Use tool_version=1 explicitly for the three operator-frozen recipes. Never register new versions or alter their scripts. Use run_code for metadata preparation.\n\n')+generic[pin_end:]
    goal=core+'\n\n## Runtime execution observations (scientific goal unchanged)\nUse the three operator-registered tools directly. Reuse actual owned receipts and completed stages. Preparation is metadata only. Keep all failed candidates and incomplete stages visible; never infer execution from a child narrative.\n\n'+generic
    (trial/'goal.md').write_text(goal)
    (trial/'navigation-note.txt').write_text('Use the supplied make_intact_followup_manifest.py in a fresh assigned run directory and submit the actual printed manifest path. It emits short metadata IDs and selects data/intact-egfr-energy-pathsafe, whose only change is cwd-relative native Amber filenames. The full UUID/run scientific seed namespace and all production scientific parameters remain unchanged. Never transcribe checksums, hard-code the legacy stage directory, overwrite operator recipes, or silently convert failed batches to success.\n')
    scientific_files=['run_binder_pipeline.py','run_binder_generation.py','run_intact_egfr_followup.py','summarize_binder_pipeline.py','intact-egfr-energy/cofold.py']
    proof=[]
    for filename in scientific_files:
        actual=hashlib.sha256((trial/'inputs'/filename).read_bytes()).hexdigest()
        assert actual==hashlib.sha256((baseline/'inputs'/filename).read_bytes()).hexdigest()
        proof.append({'file':filename,'sha256':actual,'matches_successful_baseline':True})
    (trial/'scientific-protocol.json').write_text(json.dumps({'generated_at':time.time(),'files':proof,'energy_filename_wrapper':'pathsafe-energy-stage.json','metadata_helper_sha256':hashlib.sha256(metadata_helper.read_bytes()).hexdigest()},indent=2)+'\n')
    (trial/'inputs/current-artifacts.json').write_text(json.dumps({'metadata_only':True,'scientific_evidence':False,'batches':[],'instructions':'No screen submitted yet. Execute this hypothesis own registered matched screen; this file is navigation metadata only.'})+'\n')
    for target in ('campaign.yaml','science-server.yaml'):(trial/target).write_text(yaml.safe_dump(cfg,sort_keys=False))
    load(trial/'campaign.yaml')
    # Matched proposal text, fresh identities and no inherited evidence/Elo.
    oldstate=json.loads((baseline/'state/jnana.json').read_text());seed=[];mapping=[]
    for i,old in enumerate(sorted(oldstate['hypotheses'],key=lambda h:h['created_at'])[:24]):
        h=copy.deepcopy(old);h['hypothesis_id']=str(uuid.uuid5(uuid.NAMESPACE_URL,str(trial)+':'+old['hypothesis_id']))
        h.update(elo_rating=1200,reviews=[],tournament_matches=[],tournament_wins=0,tournament_losses=0,last_tournament_time=None,created_at=time.time()+i*.001)
        seed.append(h);mapping.append({'baseline_id':old['hypothesis_id'],'trial_id':h['hypothesis_id'],'content_sha256':hashlib.sha256(h['content'].encode()).hexdigest()})
    state={k:{} for k in ('experiments','analyses','papers','agent_states','datasets','training_plans','evaluations')}
    state.update(hypotheses=seed,metadata={'research_goal':goal},tournament_state={'matches':[]})
    (trial/'state/jnana.json').write_text(json.dumps(state,indent=2)+'\n')
    (trial/'seeded-proposals.json').write_text(json.dumps({'source':str(baseline/'state/jnana.json'),'selection':'earliest 24 proposals; content only; reset evidence/Elo; fresh UUID and scientific seeds','mapping':mapping},indent=2)+'\n')
    progress={'campaign_id':cid,'phase':'staged; science allocation not submitted','inference_pool_experiment':str(root),'scientific_allocation_nodes':science,'target_total_nodes':20,'active_inference_job_ids':[b['job_id'] for b in active],'owned_job_ids':[b['job_id'] for b in active],'inference_submissions':[{'job_id':b['job_id']} for b in active],'baseline':str(baseline),'created_at':time.time()}
    (trial/'scale-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    pool['phase']='campaign-ready';(root/'pool.json').write_text(json.dumps(pool,indent=2)+'\n')
    print(json.dumps({'trial':str(trial),'campaign_id':cid,'science_nodes':science,'inference_nodes':inference,'mcp_port':hpc['port']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--inference',type=int,required=True);p.add_argument('--science',type=int,required=True);p.add_argument('--label');a=p.parse_args();stage(a.root.resolve(),a.inference,a.science,a.label)
