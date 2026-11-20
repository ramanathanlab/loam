"""Re-fold a real screened binder with intact EGFR, then minimized MM/GBSA.

Never apply MM/GBSA to the 205-residue crop: two native disulfides cross its
boundaries. This later funnel stage preserves the same designed sequence,
uses canonical EGFR25–646, and records its chemistry and source artifacts.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def measure_intact_geometry(root,screen,canonical_target):
    """Verify the intact model still engages the predeclared EGFR epitope."""
    import gemmi
    import numpy as np
    from scipy.spatial.distance import cdist
    region1={346,348,353,355,356,357,358,382,384,408,409,411,412,415,417,418,438,440}
    region2={467,468}
    assert canonical_target[285:490]==screen['target_sequence'],'Crop-to-intact canonical mapping differs'
    residues=lambda chain:[r for r in chain if gemmi.find_tabulated_residue(r.name).is_amino_acid()]
    sequence=lambda rr:''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in rr)
    ca=lambda rr:np.array([r['CA'][0].pos.tolist() for r in rr])
    reference=max(screen['models'],key=lambda m:m['scores']['iptm']);ref_path=Path(reference['path']).resolve(strict=True);assert sha(ref_path)==reference['sha256']
    ref=gemmi.read_structure(str(ref_path))[0];ref_target=ca(residues(ref['A']));ref_binder=ca(residues(ref['B']))
    assert sequence(residues(ref['A']))==screen['target_sequence'] and sequence(residues(ref['B']))==screen['binder_sequence']
    def metrics(target,binder,path,iptm=None):
        ids={r.seqid.num for r in target};assert region1|region2<=ids
        crop=[r for r in target if 286<=r.seqid.num<=490];assert sequence(crop)==screen['target_sequence']
        xyz=[];rid=[];bxyz=[]
        for r in target:
            for atom in r:
                if atom.element.name not in ('H','D'):xyz.append(atom.pos.tolist());rid.append(r.seqid.num)
        for r in binder:
            for atom in r:
                if atom.element.name not in ('H','D'):bxyz.append(atom.pos.tolist())
        assert np.isfinite(xyz).all() and np.isfinite(bxyz).all()
        distances=cdist(xyz,bxyz);contacts={i for i,d in zip(rid,distances.min(1)) if d<=4}
        tc,bc=ca(crop),ca(binder);x,y=tc-tc.mean(0),ref_target-ref_target.mean(0);u,_,vt=np.linalg.svd(x.T@y);rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
        aligned=(bc-tc.mean(0))@rotation+ref_target.mean(0);rmsd=float(np.sqrt(np.mean(np.sum((aligned-ref_binder)**2,axis=1))))
        one,two=len(contacts&region1),len(contacts&region2)
        return {'path':str(path),'sha256':sha(path),'iptm':iptm,'region1_contact_residues':one,'region2_contact_residues':two,'dual_region_contact':bool(one and two),'target_contact_residues_outside_original_crop':sum(not 286<=i<=490 for i in contacts),'minimum_interchain_heavy_A':float(distances.min()),'close_atom_pairs_lt2A':int((distances<2).sum()),'target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A':rmsd}
    models=[]
    for path in sorted((root/'cofold').glob('pred.model_idx_*.cif')):
        model=gemmi.read_structure(str(path))[0];target,binder=residues(model['A']),residues(model['B'])
        assert sequence(target)==canonical_target and sequence(binder)==screen['binder_sequence']
        sp=path.with_name(path.name.replace('pred.','scores.')).with_suffix('.npz');iptm=float(np.load(sp)['iptm'].reshape(-1)[0]);assert np.isfinite(iptm)
        models.append(metrics(target,binder,path,iptm))
    assert len(models)==5
    minimized=root/'mmgbsa/minimized.pdb';structure=gemmi.read_structure(str(minimized))[0]
    all_residues=[r for chain in structure for r in residues(chain)]
    assert sequence(all_residues)==canonical_target+screen['binder_sequence'],'Minimized residue partition lost sequence identity'
    target,binder=all_residues[:622],all_residues[622:];assert len(binder)==60
    assert [r.seqid.num for r in target]==list(range(1,623)),'Minimized target author numbering changed'
    post=metrics(target,binder,minimized)
    post['partition']='Sequence-verified residue ranges1-622 receptor,623-682 binder; Amber may collapse PDB chain labels. Covalent separation is independently checked in the Amber topology.'
    post['confidence_note']='No new ipTM is computed after minimization.'
    return {'models':models,'post_minimization':post,'source_screen_reference':reference,'mapping':'3NJP source A286-490 / screen A1-205 -> intact canonical EGFR25-646 chain A286-490. Source hotspot IDs equal intact native IDs.','contact_cutoff_A':4.0,'interpretation':'Intact-context contact persistence and target-aligned binder RMSD, including possible binder deformation. Five Chai models are before minimization; a separate record measures the single minimized energy frame. No measured affinity or dynamics.'}

def main():
    row=json.loads(Path(sys.argv[1]).read_text());out=Path.cwd();started=time.monotonic();source=Path(row['screen_summary']).resolve(strict=True)
    assert sha(source)==row['screen_summary_sha256'];screen=json.loads(source.read_text());assert screen['status']=='passed'
    binder=screen['binder_sequence'];assert len(binder)==60 and 'C' not in binder,'This reference requires a cysteine-free binder to avoid unverified binder disulfides'
    annotation=Path(row['target_annotation']).resolve(strict=True);assert sha(annotation)==row['target_annotation_sha256'];target=json.loads(annotation.read_text());assert target['primaryAccession']=='P00533'
    sequence=target['sequence']['value'][24:646];assert len(sequence)==622
    for f in target.get('features',[]):
        if f['type']=='Disulfide bond':
            i,j=f['location']['start']['value'],f['location']['end']['value'];assert (25<=i<=646)==(25<=j<=646),'Intact target boundary cuts an annotated disulfide'
    (out/'cofold.fasta').write_text(f'>protein|name=human_EGFR\n{sequence}\n>protein|name=designed_binder\n{binder}\n')
    (out/'sequence-report.json').write_text(json.dumps({'target_residues':622,'binder_residues':60,'canonical_interval':[25,646],'source_screen_summary':str(source),'source_screen_sha256':sha(source),'binder_sequence':binder,'target_annotation_sha256':sha(annotation)},indent=2)+'\n')
    software=Path(os.environ['LOAM_SOFTWARE_ROOT']);repo=Path(os.environ['HPC_LOAM_HOME']);stages=Path(row['stages_directory']).resolve(strict=True)
    effective={**row,'root':str(out),'repo':str(repo),'software':str(software),'purpose':'Hypothesis-owned intact EGFR follow-up of an actual RFD4/ProteinMPNN binder; confidence and single-frame MMGBSA are not measured affinity.'}
    inp=out/'effective-row.json';inp.write_text(json.dumps(effective,indent=2)+'\n');timings={}
    for name,env in [('cofold','RFD4-Proteina'),('mmgbsa','molecular-simulations')]:
        t=time.monotonic();script=stages/(name+'.py');assert sha(script)==row[name+'_script_sha256']
        with (out/(name+'-stage.log')).open('w') as log:
            subprocess.run(['bash','-lc','source "$1"; source "$2"; export OMP_NUM_THREADS=4; shift 2; exec "$@"','intact',str(repo/'scripts/jupiter_env.sh'),str(software/env/'.venv/bin/activate'),str(software/env/'.venv/bin/python'),str(script),str(inp)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        timings[name]=time.monotonic()-t
    energy=json.loads((out/'mmgbsa/report.json').read_text());assert energy['mmgbsa_completed'] and energy['minimization_completed']
    geometry=measure_intact_geometry(out,screen,sequence);(out/'intact-geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    energy_pose=next(m for m in geometry['models'] if Path(m['path']).resolve()==Path(energy['source_complex']).resolve())
    result={'status':'passed','source_screen_summary':str(source),'source_screen_sha256':sha(source),'screen_candidate_id':screen['candidate_id'],'condition':screen['condition'],'binder_sequence':binder,'target_residues':622,'canonical_interval':[25,646],'energy_report':str(out/'mmgbsa/report.json'),'static_MMGBSA_kcal_mol':energy['static_delta_total_kcal_mol'],'target_disulfides':len(energy['disulfides_from_uniprot']),'stage_seconds':timings,'elapsed_seconds':time.monotonic()-started,'interpretation':energy['method']}
    result.update(intact_iptm_best=max(m['iptm'] for m in geometry['models']),intact_iptm_median=float(__import__('statistics').median(m['iptm'] for m in geometry['models'])),intact_models_with_dual_region_contact=sum(m['dual_region_contact'] for m in geometry['models']),energy_input_pose_geometry=energy_pose,minimized_pose_geometry=geometry['post_minimization'],geometry_report=str(out/'intact-geometry.json'))
    (out/'followup-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
