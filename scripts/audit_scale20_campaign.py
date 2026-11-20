#!/usr/bin/env python3
"""Independent evidence inventory for the twenty-node continuation.

This does not mark a goal complete. It reports individual requirements,
including failures and missing evidence. Operator pilots are excluded.
Use the RFD4 Python with --verify-models for coordinate/score verification.
"""
import argparse,ast,bisect,collections,hashlib,json,re,statistics,subprocess,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SOURCE1={346,348,353,355,356,357,358,382,384,408,409,411,412,415,417,418,438,440}
SOURCE2={467,468}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def input_protocol_issues(root,folder,tool):
    """Check delegated inputs as well as the immutable top-level executable.

    Mislabelled controls and changed delegated recipes are excluded from positive
    output counts; their original receipts and resource costs remain untouched.
    """
    row=read(folder/'input.json');issues=[]
    def check(label,actual,allowed):
        if actual not in allowed:issues.append({'field':label,'actual':actual,'allowed':allowed})
    if tool=='scale20_complete_binder_pipeline':
        condition=row.get('condition');p=root/'inputs/verified-egfr-inputs'/f'{condition}.cif'
        allowed=[sha(p)] if condition in ('dual','one_region','ridge') and p.is_file() else []
        check('condition_specific_input_sha256',row.get('input_sha256'),allowed)
        check('generation_script_sha256',row.get('generation_script_sha256'),[sha(root/'inputs/run_binder_generation.py')])
    elif tool=='scale20_intact_egfr_followup':
        for key,relative in [('cofold_script','intact-egfr-energy/cofold.py'),('target_annotation','intact-egfr/P00533.json')]:
            check(key+'_sha256',row.get(key+'_sha256'),[sha(root/'inputs'/relative)])
        allowed=[sha(p) for p in [root/'inputs/intact-egfr-energy/mmgbsa.py',root/'inputs/intact-egfr-energy-pathsafe/mmgbsa.py'] if p.is_file()]
        check('mmgbsa_script_sha256',row.get('mmgbsa_script_sha256'),allowed)
        s=read(folder/'followup-summary.json');source=Path(s['source_screen_summary'])
        issues.extend({'upstream_screen':str(source),**issue} for issue in input_protocol_issues(root,source.parent,'scale20_complete_binder_pipeline'))
    return issues
def reviewed_source_ast(path):
    """Ignore a module docstring and a harmless repeated models:models entry.

    This is deliberately narrow. Other script differences require independent
    review and stay excluded; numerical/scientific statements are untouched.
    """
    tree=ast.parse(path.read_text())
    if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):tree.body.pop(0)
    for node in ast.walk(tree):
        if not isinstance(node,ast.Dict):continue
        indices=[i for i,key in enumerate(node.keys) if isinstance(key,ast.Constant) and key.value=='models']
        if len(indices)!=2:continue
        first,last=indices
        if not all(isinstance(node.values[i],ast.Name) and node.values[i].id=='models' for i in indices):continue
        if not all(isinstance(node.values[i],(ast.Name,ast.Constant)) for i in range(first,last+1)):continue
        node.keys.pop(last);node.values.pop(last)
    return ast.dump(tree,include_attributes=False)
def claim_key(n,h,claim):
    return hashlib.sha256(json.dumps({'round':n,'hypothesis':h,'claim':claim['claim'],'source':claim['source']},sort_keys=True).encode()).hexdigest()
def timestamp(s):
    try:return datetime.fromisoformat(s).replace(tzinfo=ZoneInfo('Europe/Berlin')).timestamp()
    except (ValueError,TypeError):return None
def accounting(ids):
    fields=['JobIDRaw','JobName','State','NNodes','NodeList','Submit','Start','End','ElapsedRaw','AllocTRES']
    p=subprocess.run(['sacct','-n','-X','-j',','.join(ids),'--parsable2','--format='+','.join(fields)],capture_output=True,text=True,check=True)
    jobs={}
    for line in p.stdout.splitlines():
        values=line.strip().split('|')
        if not values or values[0] not in ids:continue
        d=dict(zip(fields,values));d['start_epoch']=timestamp(d['Start']);d['end_epoch']=timestamp(d['End']);d['submit_epoch']=timestamp(d['Submit'])
        d['queue_seconds']=max(0,d['start_epoch']-d['submit_epoch']) if d['start_epoch'] and d['submit_epoch'] else None
        if d['start_epoch'] and not d['end_epoch'] and d['State'].startswith(('RUNNING','COMPLETING')):d['end_epoch']=d['start_epoch']+int(d['ElapsedRaw'])
        n=d['NodeList'];d['hostnames']=[]
        if n and n not in ('None assigned','Unknown','(null)'):
            q=subprocess.run(['scontrol','show','hostnames',n],capture_output=True,text=True)
            if q.returncode==0:d['hostnames']=q.stdout.split()
        jobs[d['JobIDRaw']]=d
    return jobs
def pooled_overlap(science,inference_jobs,replicas):
    """Intersect simultaneous replica allocations; never add their durations."""
    if len(inference_jobs)!=replicas or not all(j.get('start_epoch') and j.get('end_epoch') for j in [science,*inference_jobs]):return []
    lo=max(j['start_epoch'] for j in [science,*inference_jobs]);hi=min(j['end_epoch'] for j in [science,*inference_jobs])
    return [{'start':lo,'end':hi,'seconds':hi-lo,'inference_jobs':[j['JobIDRaw'] for j in inference_jobs]}] if hi>lo else []
def amber_partition(topology,target_residues):
    """Read Amber's declared bond graph independently of OpenMM chain labels."""
    wanted={'ATOM_NAME','RESIDUE_POINTER','BONDS_INC_HYDROGEN','BONDS_WITHOUT_HYDROGEN'};sections={};key=None
    for line in topology.read_text().splitlines():
        if line.startswith('%FLAG '):key=line.split()[1];continue
        if line.startswith('%') or key not in wanted:continue
        sections.setdefault(key,[]).append(line)
    pointers=[int(x) for line in sections['RESIDUE_POINTER'] for x in line.split()]
    names=[line[i:i+4].strip() for line in sections['ATOM_NAME'] for i in range(0,len(line),4) if line[i:i+4].strip()]
    cross=[];ss=[]
    for key in ['BONDS_INC_HYDROGEN','BONDS_WITHOUT_HYDROGEN']:
        bonds=[int(x) for line in sections[key] for x in line.split()];assert len(bonds)%3==0
        for i in range(0,len(bonds),3):
            first,second=bonds[i]//3,bonds[i+1]//3;a,b=bisect.bisect_right(pointers,first+1),bisect.bisect_right(pointers,second+1)
            if (a<=target_residues)!=(b<=target_residues):cross.append([a,b])
            if names[first]==names[second]=='SG':ss.append(sorted([a,b]))
    return {'residues':len(pointers),'target_binder_covalent_bonds':cross,'disulfide_residue_pairs':sorted(ss)}
def verify_model(path,target_sequence,binder_sequence,expected_sha=None):
    import gemmi
    import numpy as np
    from scipy.spatial.distance import cdist
    path=path.resolve(strict=True)
    if expected_sha:assert sha(path)==expected_sha,'Coordinate checksum differs'
    model=gemmi.read_structure(str(path))[0];chains={c.name:c for c in model};assert set(chains)=={'A','B'}
    residues=lambda c:[r for r in c if gemmi.find_tabulated_residue(r.name).is_amino_acid()]
    seq=lambda rr:''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in rr)
    target,binder=residues(chains['A']),residues(chains['B']);assert seq(target)==target_sequence and seq(binder)==binder_sequence
    offset=285 if len(target)==205 else 0;assert len(target) in (205,622)
    one,two={i-offset for i in SOURCE1},{i-offset for i in SOURCE2};ids={r.seqid.num for r in target};assert one|two<=ids
    xyz=[];rid=[];other=[]
    for r in target:
        for atom in r:
            assert all(np.isfinite(atom.pos.tolist()))
            if atom.element.name not in ('H','D'):xyz.append(atom.pos.tolist());rid.append(r.seqid.num)
    for r in binder:
        for atom in r:
            assert all(np.isfinite(atom.pos.tolist()))
            if atom.element.name not in ('H','D'):other.append(atom.pos.tolist())
    distance=cdist(xyz,other);contacts={i for i,d in zip(rid,distance.min(1)) if d<=4}
    sp=path.with_name(path.name.replace('pred.','scores.')).with_suffix('.npz')
    with np.load(sp) as z:scores={k:float(z[k].reshape(-1)[0]) for k in ('iptm','ptm','aggregate_score')}
    assert all(np.isfinite(v) for v in scores.values())
    return {'path':str(path),'sha256':sha(path),'scores':scores,'region1_contact_residues':len(contacts&one),'region2_contact_residues':len(contacts&two),'dual_region_contact':bool(contacts&one and contacts&two),'minimum_interchain_heavy_A':float(distance.min()),'close_atom_pairs_lt2A':int((distance<2).sum())}
def verify_screen_model(item,folder,target_sequence,binder_sequence,resolutions):
    """Verify a declared prediction, including an observed compact copied manifest.

    A path/hash-only entry under models/ is accepted only when its bytes match
    the exact same-index original under this row's fold/, with its original
    score array present. Reconstruct missing per-model fields in the derived
    audit inventory only; never edit the source summary or count copies twice.
    Ordinary reported metrics retain their existing strict comparisons.
    """
    folder=folder.resolve(strict=True)
    declared=Path(item['path']).resolve(strict=True)
    assert declared.is_relative_to(folder),'Prediction escapes its owning row'
    compact=set(item)=={'path','sha256'}
    coordinate=declared
    if compact:
        assert declared.parent==(folder/'models').resolve(),'Unsupported compact prediction location'
        assert re.fullmatch(r'pred\.model_idx_[0-4]\.cif',declared.name),'Unsupported prediction filename'
        coordinate=(folder/'fold'/declared.name).resolve(strict=True)
        assert coordinate.is_relative_to(folder) and coordinate.parent==(folder/'fold').resolve() and coordinate.name==declared.name,'Original prediction is not the same row/index'
        assert sha(declared)==item['sha256']==sha(coordinate),'Copied prediction differs from the original'
        expected_score=coordinate.with_name(coordinate.name.replace('pred.','scores.')).with_suffix('.npz')
        score=expected_score.resolve(strict=True)
        assert score.parent==coordinate.parent and score.name==expected_score.name,'Original score array is not the same prediction index/directory'
    result=verify_model(coordinate,target_sequence,binder_sequence,item['sha256'])
    if compact:
        result['path']=str(declared)  # verified through exact byte identity above
        item.update({key:value for key,value in result.items() if key not in ('path','sha256')})
        resolutions.append({'summary':str(folder/'summary.json'),'summary_sha256':sha(folder/'summary.json'),
            'declared_coordinate':str(declared),'original_prediction':str(coordinate),
            'coordinate_sha256':item['sha256'],'original_score_file':str(score),'original_score_sha256':sha(score),
            'method':'Exact same-row, same-index byte-identical copied CIF; independent original CIF/NPZ verification and derived per-model metadata reconstruction. Original artifacts unchanged; copy is not an additional prediction.'})
    assert abs(result['scores']['iptm']-item['scores']['iptm'])<1e-6,'Reported per-model ipTM differs'
    assert result['region1_contact_residues']==item['region1_contact_residues'] and result['region2_contact_residues']==item['region2_contact_residues'],'Reported per-model contacts differ'
    return result
def check_pose_dispersion(summary):
    import gemmi
    import numpy as np
    poses=[]
    for item in summary['models']:
        structure=gemmi.read_structure(item['path'])[0]
        poses.append(tuple(np.array([r['CA'][0].pos.tolist() for r in structure[chain] if gemmi.find_tabulated_residue(r.name).is_amino_acid()]) for chain in ['A','B']))
    def rmsd(first,second):
        target,binder=first;reference,ref_binder=second
        left,_,right=np.linalg.svd((target-target.mean(0)).T@(reference-reference.mean(0)))
        rotation=left@np.diag([1,1,np.linalg.det(left@right)])@right
        return float(np.sqrt(np.mean(np.sum(((binder-target.mean(0))@rotation+reference.mean(0)-ref_binder)**2,axis=1))))
    pairs=[rmsd(poses[i],poses[j]) for i in range(5) for j in range(i)]
    zero=[rmsd(poses[i],poses[0]) for i in range(1,5)]
    assert abs(float(np.mean(pairs))-summary['mean_pairwise_target_aligned_binder_CA_RMSD_A'])<1e-5,'Pairwise pose dispersion differs'
    assert abs(float(np.mean(zero))-summary['mean_target_aligned_binder_CA_RMSD_to_model0_excluding_reference_A'])<1e-5,'Model-zero pose dispersion differs'
def check_minimized_geometry(summary,canonical):
    import gemmi
    import numpy as np
    from scipy.spatial.distance import cdist
    reported=summary['minimized_pose_geometry'];path=Path(reported['path']);assert sha(path)==reported['sha256']
    structure=gemmi.read_structure(str(path))[0]
    residues=[res for chain in structure for res in chain if gemmi.find_tabulated_residue(res.name).is_amino_acid()]
    assert ''.join(gemmi.find_tabulated_residue(res.name).one_letter_code for res in residues)==canonical+summary['binder_sequence']
    target,binder=residues[:622],residues[622:];assert len(binder)==60
    atoms=[(i+1,a.pos.tolist()) for i,res in enumerate(target) for a in res if a.element.name not in ('H','D')]
    other=[a.pos.tolist() for res in binder for a in res if a.element.name not in ('H','D')]
    distance=cdist([xyz for _,xyz in atoms],other);assert np.isfinite(distance).all()
    contacts={i for (i,_),d in zip(atoms,distance.min(1)) if d<=4}
    assert len(contacts&SOURCE1)==reported['region1_contact_residues'] and len(contacts&SOURCE2)==reported['region2_contact_residues']
    assert int((distance<2).sum())==reported['close_atom_pairs_lt2A']
    assert abs(float(distance.min())-reported['minimum_interchain_heavy_A'])<1e-5
    source=Path(summary['source_screen_summary']);assert sha(source)==summary['source_screen_sha256'];screen=read(source)
    assert screen['binder_sequence']==summary['binder_sequence']
    reference=max(screen['models'],key=lambda m:m['scores']['iptm']);assert sha(Path(reference['path']))==reference['sha256']
    ref=gemmi.read_structure(reference['path'])[0];ca=lambda rr:np.array([res['CA'][0].pos.tolist() for res in rr])
    crop=ca(target[285:490]);ref_target=ca(list(ref['A']));ref_binder=ca(list(ref['B']))
    left,_,right=np.linalg.svd((crop-crop.mean(0)).T@(ref_target-ref_target.mean(0)))
    rotation=left@np.diag([1,1,np.linalg.det(left@right)])@right
    rmsd=float(np.sqrt(np.mean(np.sum(((ca(binder)-crop.mean(0))@rotation+ref_target.mean(0)-ref_binder)**2,axis=1))))
    assert abs(rmsd-reported['target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A'])<1e-5,'Minimized binder pose differs'
def main():
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--verify-models',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args();root=a.campaign.resolve();progress=read(root/'scale-progress.json');errors=[]
    rounds={int(f.stem.split('-')[1]):read(f) for f in sorted((root/'state/rounds').glob('round-*.json'))}
    resolution_file=root/'claim-source-resolutions.json';resolutions=read(resolution_file) if resolution_file.exists() else {};source_resolutions=[]
    ownership={};unowned=[];accepted=collections.Counter();reviewed=collections.Counter();thread_errors=[];selected=collections.Counter()
    for n,record in rounds.items():
        for thread in record['threads']:
            h=thread['hypothesis']['id'];selected[n]+=1
            if thread.get('report') is not None and thread.get('review') is not None and not thread.get('error'):reviewed[n]+=1
            else:thread_errors.append({'round':n,'hypothesis':h,'error':thread.get('error'),'report_present':thread.get('report') is not None,'review_present':thread.get('review') is not None})
            if h not in ownership:
                roots=[];batches=set()
                for rp in (root/'state/hypotheses'/h/'runs').glob('r*.json'):
                    r=read(rp);roots.append(root/'work'/h/'outputs'/r['run_id']);batches.update(r.get('batches',[]))
                roots.extend(root/'batches'/b for b in batches);ownership[h]={'roots':roots,'batches':batches}
            for claim in thread.get('accepted',[]):
                source=claim.get('source','')
                if source.startswith(('http://','https://')):continue
                cleaned=re.sub(r'(?::\d+(?:-\d+)?|#.*)$','',source.strip(' `"\''));target=Path(cleaned)
                owned=lambda p:any(p.resolve().is_relative_to(q.resolve()) for q in ownership[h]['roots'])
                # A batch directory is a legitimate batch-level source. It must
                # exist, be owned, and contain an immutable batch receipt.
                if owned(target) and (target.is_file() or target.is_dir() and (target/'batch.json').is_file()):accepted[n]+=1
                else:
                    resolution=resolutions.get(claim_key(n,h,claim));paths=[Path(x['path']) for x in resolution.get('files',[])] if resolution else []
                    if paths and all(p.is_file() and owned(p) and sha(p)==x['sha256'] for p,x in zip(paths,resolution['files'])):
                        accepted[n]+=1;source_resolutions.append({'round':n,'hypothesis':h,'original_source':source,**resolution})
                    else:unowned.append({'round':n,'hypothesis':h,'source':source})
    candidates=[];followups=[];rows=[];validated=[];batch_states=[];per_round=collections.Counter();scientific_hosts=set();pending=0;source_classifications=[];input_classifications=[];model_location_resolutions=[]
    registrations=root/'campaign-tool-registrations.json';approved={}
    if registrations.exists():
        for tool in read(registrations):approved.setdefault(tool['tool'],set()).add(tool['sha256'])
    known_hashes={digest:name for name,digests in approved.items() for digest in digests}
    aliases=root/'operator-reviewed-source-aliases.json'
    if aliases.exists():
        for entry in read(aliases):
            candidate=Path(entry['source']);canonical=Path(entry['canonical_source'])
            assert candidate.resolve().is_relative_to((root/'batches').resolve()) and canonical.resolve().is_relative_to((root/'inputs').resolve())
            assert sha(candidate)==entry['sha256'] and sha(canonical)==entry['canonical_sha256'] in approved[entry['tool']]
            assert reviewed_source_ast(candidate)==reviewed_source_ast(canonical),'Reviewed source is not equivalent under the narrow documented normalization'
            known_hashes[entry['sha256']]=entry['tool']
    canonical=read(root/'inputs/intact-egfr/P00533.json')['sequence']['value'][24:646]
    screen_target=''.join(x for x in (root/'inputs/verified-target.fasta').read_text().splitlines() if not x.startswith('>'));assert canonical[285:490]==screen_target
    for bp in sorted(root.glob('batches/*/batch.json')):
        b=read(bp);labels=b.get('labels',{});hyp=labels.get('hypothesis_id');run=labels.get('run_id','');m=re.fullmatch(r'r(\d+)-\d+',run);n=int(m[1]) if m else None;tool=b.get('tool');done=failed=0
        recorded=(root/'state/hypotheses'/str(hyp)/'runs'/f'{run}.json')
        receipt=recorded.exists() and b['batch_id'] in read(recorded).get('batches',[])
        if not receipt:errors.append({'batch':b['batch_id'],'error':'No owning hypothesis run receipt'})
        script_hash=sha(bp.parent/'script.py')
        integrity=script_hash==b.get('script_sha256')
        if not integrity:
            errors.append({'batch':b['batch_id'],'error':'Executable source differs from its immutable receipt','script_sha256':script_hash,'tool':tool})
        # A direct submission can be byte-identical to an operator-reviewed
        # recipe. Recognize the source, retain its missing MCP contract, and
        # require the same independent coordinate/output checks. An altered
        # script is excluded from positive scientific counts, even on exit0.
        source_tool=known_hashes.get(script_hash) if integrity else None
        if source_tool!=tool or source_tool is None:
            source_classifications.append({'batch_id':b['batch_id'],'recorded_tool':tool,'reviewed_source_tool':source_tool,'script_sha256':script_hash,'countable_source':source_tool is not None,'MCP_output_contract':b.get('expects'),'note':'Exact reviewed source recognized independently; original receipt unchanged' if source_tool else 'Unapproved source excluded from positive scientific inventory; attempts remain in resource/failure accounting'})
        tool=source_tool
        for folder in sorted(bp.parent.glob('run_*')):
            status_path=folder/'status.json'
            if not (folder/'done').exists() or not status_path.exists():continue
            status=read(status_path);done+=1;bad=status.get('returncode')!=0 or bool(status.get('missing_outputs'));failed+=bool(bad)
            lease=read(folder/'lease.json') if (folder/'lease.json').exists() else {};host=status.get('host',lease.get('host','')).split('.')[0]
            rows.append({'batch_id':b['batch_id'],'row':folder.name,'round':n,'hypothesis':hyp,'tool':tool,'host':host,'gpus':b.get('gpus',0),'started_at':lease.get('started_at'),'seconds':status.get('seconds'),'returncode':status.get('returncode'),'recorded_owner':receipt})
            if b.get('gpus',0)>0:scientific_hosts.add(host)
            if bad or tool is None:continue
            try:
                if tool in ('scale20_complete_binder_pipeline','scale20_intact_egfr_followup'):
                    issues=input_protocol_issues(root,folder,tool)
                    if issues:
                        input_classifications.append({'batch_id':b['batch_id'],'row':folder.name,'tool':tool,'countable_input':False,'issues':issues,'note':'Excluded from positive scientific inventory; original artifacts and all attempts/costs retained'})
                        continue
                if tool=='scale20_complete_binder_pipeline':
                    s=read(folder/'summary.json');assert s['status']=='passed' and s['models_total']==len(s['models'])==5;assert s['target_sequence']==screen_target and len(s['binder_sequence'])==60
                    for item in s['models']:
                        path=Path(item['path']).resolve(strict=True);assert path.is_relative_to(folder.resolve())
                        if a.verify_models:
                            v=verify_screen_model(item,folder,screen_target,s['binder_sequence'],model_location_resolutions)
                            validated.append({**v,'round':n,'hypothesis':hyp,'condition':s['condition'],'binder_sequence':s['binder_sequence'],'kind':'crop'})
                    if a.verify_models:
                        check_pose_dispersion(s)
                        scores=[item['scores']['iptm'] for item in s['models']]
                        assert abs(s['best_iptm']-max(scores))<1e-6 and abs(s['median_iptm']-statistics.median(scores))<1e-6
                        assert s['dual_contact_models']==sum(item['dual_region_contact'] for item in s['models'])
                    candidates.append({'round':n,'hypothesis':hyp,'batch_id':b['batch_id'],'summary_path':str(folder/'summary.json'),'row':folder.name,**s});per_round[n]+=1
                elif tool=='scale20_intact_egfr_followup':
                    s=read(folder/'followup-summary.json');e=read(folder/'mmgbsa/report.json');assert s['status']=='passed' and e['mmgbsa_completed'] and e['minimization_completed']
                    assert len(e['disulfides_from_uniprot'])==25 and e['receptor_residues']==622 and e['binder_residues']==60
                    text=(folder/'mmgbsa/FINAL_RESULTS_MMPBSA.dat').read_text();value=float(re.search(r'^DELTA TOTAL\s+([-+\d.eE]+)',text,re.M)[1]);assert abs(value-s['static_MMGBSA_kcal_mol'])<1e-5
                    topology=amber_partition(folder/'mmgbsa/complex.prmtop',622);assert topology['residues']==682 and not topology['target_binder_covalent_bonds'];assert topology['disulfide_residue_pairs']==sorted(sorted(x) for x in e['disulfides_from_uniprot'])
                    if a.verify_models:
                        for item in read(folder/'validation-report.json')['predictions']:
                            v=verify_model(Path(item['cif']),canonical,s['binder_sequence']);validated.append({**v,'round':n,'hypothesis':hyp,'condition':s['condition'],'binder_sequence':s['binder_sequence'],'kind':'intact'})
                        check_minimized_geometry(s,canonical)
                    followups.append({'round':n,'hypothesis':hyp,'batch_id':b['batch_id'],'summary_path':str(folder/'followup-summary.json'),**s})
            except Exception as exc:errors.append({'batch':b['batch_id'],'row':folder.name,'error':str(exc),'type':type(exc).__name__})
        unfinished=b['total']-done;pending+=unfinished;batch_states.append({'batch_id':b['batch_id'],'tool':tool,'round':n,'hypothesis':hyp,'total':b['total'],'terminal':done,'failed':failed,'unfinished':unfinished})
    batch_lookup={entry['batch_id']:entry for entry in batch_states};fresh_reviewed_claims=collections.Counter()
    for n,record in rounds.items():
        for thread in record['threads']:
            for claim in thread.get('accepted',[]):
                match=re.search(r'batches/(b-[0-9a-f]+)/',claim.get('source',''))
                if not match:continue
                batch=batch_lookup.get(match[1],{});upstream=batch
                if batch.get('tool')=='scale20_summarize_binder_pipeline':
                    inputs=root/'batches'/match[1]/'inputs.jsonl'
                    source_rows=[json.loads(line) for line in inputs.read_text().splitlines()]
                    if len(source_rows)==1:upstream=batch_lookup.get(Path(source_rows[0]['batch_dir']).name,{})
                if (upstream.get('hypothesis')==thread['hypothesis']['id'] and upstream.get('round')==n
                        and upstream.get('tool') in ('scale20_complete_binder_pipeline','scale20_intact_egfr_followup')
                        and upstream.get('terminal',0)>upstream.get('failed',0)):
                    fresh_reviewed_claims[n]+=1
    jobs=accounting(progress['owned_job_ids']);science=jobs.get(progress.get('scientific_job'),{});science_nodes=set(science.get('hostnames',[]));overlap=None
    unified=progress.get('mode')=='single-allocation'
    pooled=bool(progress.get('inference_pool_experiment'));expected_science=int(progress.get('scientific_allocation_nodes',19))
    if unified:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from loam.allocation import read_allocation
        allocation=read_allocation(Path(progress['allocation_manifest']))
        science_nodes=set(allocation['science_nodes']);inference_nodes=set(allocation['inference_nodes'])
        expected_inference=int(progress['inference_allocation_nodes']);inference_jobs=[science];inference_ids=[allocation['job_id']]
        expected_total=int(progress['target_total_nodes']);intervals=[]
        allocation_valid=(list(jobs)==[allocation['job_id']] and int(science.get('NNodes',0))==expected_total
            and set(science.get('hostnames',[]))==set(allocation['all_nodes'])
            and len(science_nodes)==expected_science and len(inference_nodes)==expected_inference)
        if science.get('start_epoch') and science.get('end_epoch'):
            intervals=[{'start':science['start_epoch'],'end':science['end_epoch'],'seconds':science['end_epoch']-science['start_epoch'],'parent_job':allocation['job_id']}]
    else:
        expected_total=20
        inference_ids=progress['active_inference_job_ids'] if pooled else list(dict.fromkeys([x['job_id'] for x in progress.get('retired_inference_submissions',[])]+[progress['inference_submission']['job_id']]+([progress['standby_inference_submission']['job_id']] if progress.get('standby_inference_submission') else [])))
        expected_inference=len(inference_ids) if pooled else 1
        inference_jobs=[jobs[job] for job in inference_ids if job in jobs];intervals=[]
        inference_nodes=set().union(*(set(j.get('hostnames',[])) for j in inference_jobs))
        if pooled:intervals=pooled_overlap(science,inference_jobs,expected_inference)
        else:
            for inference in inference_jobs:
                if all(j.get('start_epoch') and j.get('end_epoch') for j in (science,inference)):
                    lo=max(j['start_epoch'] for j in (science,inference));hi=min(j['end_epoch'] for j in (science,inference))
                    if hi>lo:intervals.append({'start':lo,'end':hi,'seconds':hi-lo,'inference_job':inference['JobIDRaw']})
    if intervals:overlap={'start':min(x['start'] for x in intervals),'end':max(x['end'] for x in intervals),'seconds':sum(x['seconds'] for x in intervals),'intervals':intervals,'note':'Single parent allocation; roles are disjoint host subsets; GPU-hours count the parent only once' if unified else 'Intersection of simultaneous inference replicas with science allocation' if pooled else 'Separate actual allocation intervals; gaps between inference jobs are not treated as twenty-node time'}
    allocation_events=[]
    for job in jobs.values():
        if job.get('start_epoch') and job.get('end_epoch'):
            count=int(job['NNodes']);allocation_events.extend([(job['start_epoch'],count),(job['end_epoch'],-count)])
    active_nodes=peak_nodes=0
    for _,change in sorted(allocation_events,key=lambda x:(x[0],x[1])):
        active_nodes+=change;peak_nodes=max(peak_nodes,active_nodes)
    agent_times=[]
    if (root/'state/agent-timings.jsonl').exists():agent_times=[json.loads(s) for s in (root/'state/agent-timings.jsonl').read_text().splitlines()]
    covered={n:bool(intervals and (rr:=[x for x in agent_times if x.get('round')==n]) and all(any(x['started_at']>=window['start'] and x['finished_at']<=window['end']+1 for window in intervals) for x in rr)) for n in rounds}
    byseq=collections.defaultdict(list)
    for c in candidates:byseq[c['binder_sequence']].append(c['best_iptm'])
    best=[max(v) for v in byseq.values()];unique_models={v['sha256'] for v in validated}
    checks={'at_least_three_recorded_rounds':len(rounds)>=3,'every_recorded_thread_reviewed_without_orchestration_error':bool(rounds) and not thread_errors and all(reviewed[n]==selected[n] for n in rounds),'each_round_has_complete_scientific_pipeline_candidates':bool(rounds) and all(per_round[n]>0 for n in rounds),'each_round_has_accepted_owned_artifact_claims':bool(rounds) and all(accepted[n]>0 for n in rounds),'all_accepted_artifact_sources_owned_and_present':not unowned,'all_submitted_main_batches_terminal':bool(batch_states) and pending==0,'independent_model_checks_requested_and_passed':a.verify_models and bool(validated) and not errors,'twenty_distinct_allocated_nodes_overlap':bool(overlap) and len(science_nodes)==expected_science and expected_science+expected_inference==20 and (not pooled or len(inference_nodes)==expected_inference) and all(len(j.get('hostnames',[]))==1 and not science_nodes&set(j.get('hostnames',[])) for j in inference_jobs if j.get('start_epoch')),'all_science_nodes_executed_GPU_rows':len(science_nodes)==expected_science and science_nodes<=scientific_hosts,'recorded_scientific_rounds_inside_twenty_node_overlap':bool(rounds) and all(covered.values()),'no_inventory_or_provenance_errors':not errors}
    if unified:
        checks['single_allocation_contains_all_disjoint_roles']=bool(overlap) and allocation_valid
        checks['recorded_scientific_rounds_inside_parent_allocation']=checks.pop('recorded_scientific_rounds_inside_twenty_node_overlap')
        checks.pop('twenty_distinct_allocated_nodes_overlap')
    if not pooled:checks['all_nineteen_science_nodes_executed_GPU_rows']=checks['all_science_nodes_executed_GPU_rows']
    result={'generated_at':time.time(),'campaign':str(root),'scope':'Main continuation only; science-pilot/operator fixtures excluded. Checks are evidence, not automatic goal completion.','model_coordinates_independently_checked':a.verify_models,'checks':checks,'all_machine_checks_passed':all(checks.values()),'rounds':{str(n):{'selected':selected[n],'reviewed':reviewed[n],'accepted_owned_artifact_claims':accepted[n],'complete_screen_candidates':per_round[n],'inside_twenty_node_overlap':covered[n]} for n in rounds},'unique_tested_hypotheses':len({c['hypothesis'] for c in candidates if c['round'] in rounds}),'successful_screen_candidate_rows':len(candidates),'distinct_screen_sequences':len(byseq),'median_best_iptm_per_screen_sequence':statistics.median(best) if best else None,'unique_verified_coordinate_files':len(unique_models),'intact_energy_followups':len(followups),'unfinished_main_rows':pending,'scientific_execution_hosts':sorted(scientific_hosts),'jobs':jobs,'twenty_node_overlap':overlap,'thread_errors':thread_errors,'unowned_or_missing_claim_sources':unowned,'errors':errors,'batch_states':batch_states,'candidate_rows':candidates,'intact_followups':followups,'verified_models':validated,'row_timings':rows,'agent_timings':agent_times}
    result['configured_split']={'science':expected_science,'inference':expected_inference}
    result['operator_reviewed_source_aliases']=read(aliases) if aliases.exists() else []
    result['nonstandard_source_classifications']=source_classifications
    result['nonstandard_input_classifications']=input_classifications
    result['model_location_resolutions']=model_location_resolutions
    result['independent_verifier_sha256']=sha(Path(__file__))
    result['explicit_claim_source_resolutions']=source_resolutions
    claim_review=root/'independent-claim-review.json'
    result['independent_claim_review']=read(claim_review) if claim_review.exists() else None
    result['inference_jobs']=inference_jobs
    result['twenty_node_overlap_intervals']=intervals
    result['peak_concurrently_allocated_owned_nodes']=peak_nodes
    result['accepted_claims_on_current_round_scientific_batches']=dict(fresh_reviewed_claims)
    allocation_check='single_allocation_contains_all_disjoint_roles' if unified else 'twenty_distinct_allocated_nodes_overlap'
    coverage_check='recorded_scientific_rounds_inside_parent_allocation' if unified else 'recorded_scientific_rounds_inside_twenty_node_overlap'
    required={key:checks[key] for key in ('at_least_three_recorded_rounds','each_round_has_accepted_owned_artifact_claims','all_accepted_artifact_sources_owned_and_present','all_submitted_main_batches_terminal','independent_model_checks_requested_and_passed',allocation_check,'all_science_nodes_executed_GPU_rows',coverage_check,'no_inventory_or_provenance_errors')}
    if pooled:required.pop('all_science_nodes_executed_GPU_rows')  # Idle nodes remain a utilization finding, not invented work.
    required['each_recorded_round_has_reviewed_new_scientific_evidence']=bool(rounds) and all(fresh_reviewed_claims[n]>0 for n in rounds)
    required['owned_allocation_never_exceeded_target_nodes' if unified else 'owned_allocation_never_exceeded_twenty_nodes']=peak_nodes==expected_total
    watch=read(root/'controller-watch.json')
    required['campaign_driver_finished_successfully']=watch.get('state')=='controller exited' and watch.get('returncode')==0
    result['required_campaign_completion_checks']=required
    result['required_campaign_completion_checks_passed']=all(required.values())
    result['completion_scope_note']='User requirement: a complete tournament/science/review loop for at least three rounds at twenty total owned nodes, with reviewed fresh scientific evidence in each. Repository failure policy explicitly retains individual untested hypotheses without stopping the campaign. Every-thread success remains a separate diagnostic; no failed thread is silently counted as tested or repaired.'
    if unified:
        result['allocation_mode']='single-allocation';result['allocation_manifest']=allocation
        result['parent_allocation_interval']=result.pop('twenty_node_overlap')
        result['parent_allocation_intervals']=result.pop('twenty_node_overlap_intervals')
        for rr in result['rounds'].values():rr['inside_parent_allocation']=rr.pop('inside_twenty_node_overlap')
        result['completion_scope_note']=f'At least three tournament/science/review rounds inside a single {expected_total}-node allocation with {expected_inference} inference and {expected_science} science nodes; fresh reviewed scientific evidence in each round. Individual failed threads are retained and excluded from tested counts.'
    dest=a.output or root/'scaling-audit.json';dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['checks','successful_screen_candidate_rows','distinct_screen_sequences','intact_energy_followups','unfinished_main_rows','errors','nonstandard_source_classifications']},indent=2))
if __name__=='__main__':main()
