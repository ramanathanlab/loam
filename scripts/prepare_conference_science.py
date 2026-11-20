#!/usr/bin/env python3
"""Prepare a concise conference slide from audited 256-node scientific evidence."""
import argparse,collections,hashlib,json
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
COLORS={'dual':'#078a92','one_region':'#e67d4d','ridge':'#8265ab'}
SOURCE1={346,348,353,355,356,357,358,382,384,408,409,411,412,415,417,418,438,440}
SOURCE2={467,468}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def residues(chain):return [r for r in chain if gemmi.find_tabulated_residue(r.name).is_amino_acid()]
def seq(chain):return ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues(chain))
def ca(chain):return np.array([r['CA'][0].pos.tolist() for r in residues(chain)])
def rigid(first,reference):
    a,b=first.mean(0),reference.mean(0);u,_,v=np.linalg.svd((first-a).T@(reference-b));rotation=u@np.diag([1,1,np.linalg.det(u@v)])@v
    return rotation,a,b
def apply(structure,rotation,origin,destination):
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                p=(np.array(atom.pos.tolist())-origin)@rotation+destination;atom.pos=gemmi.Position(*p)

def prepare(out):
    source=REPO/'reports/single-allocation-256-20261115/data.json';data=read(source);run=next(r for r in data['runs'] if r['nodes']==256)
    root=Path(data['experiment'])/'unified-256/campaign';audit=read(root/'scaling-audit.json');assert audit['required_campaign_completion_checks_passed']
    screens=[x for x in data['screen_trials'] if x['run']==run['label']];full=[x for x in data['expanded_trials'] if x['run']==run['label']]
    unique={}
    for row in full:unique.setdefault(row['binder_sequence'],row)
    top=sorted(unique.values(),key=lambda x:(x['minimized_dual_contact'],x['intact_models_with_dual_region_contact'],x['intact_iptm_best']),reverse=True)[:3]
    out.mkdir(exist_ok=True,parents=True);(out/'structures').mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    canonical=read(root/'inputs/intact-egfr/P00533.json')['sequence']['value'][24:646]
    reference=None;orientation=None;selected=[];files={str(source):sha(source),str(root/'scaling-audit.json'):sha(root/'scaling-audit.json')}
    render_colors=['#078a92','#286b78','#e67d4d']
    for i,row in enumerate(top):
        summary=read(row['summary_path']);pose=summary['minimized_pose_geometry'];path=Path(pose['path']);assert sha(path)==pose['sha256'];assert abs(summary['energy_input_pose_geometry']['iptm']-row['intact_iptm_best'])<1e-6
        original=gemmi.read_structure(str(path));rr=[r for ch in original[0] for r in residues(ch)];assert len(rr)==682
        assert ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in rr)==canonical+row['binder_sequence']
        # Amber can merge chain labels. Restore the sequence-verified partition,
        # preserving every atom and coordinate before rigid presentation transforms.
        structure=gemmi.Structure();model=gemmi.Model(1)
        for name,group in [('A',rr[:622]),('B',rr[622:])]:
            chain=gemmi.Chain(name)
            for number,residue in enumerate(group,1):
                copy=residue.clone();copy.seqid=gemmi.SeqId(number,' ');copy.subchain=name;chain.add_residue(copy)
            model.add_chain(chain)
        structure.add_model(model)
        assert len(structure)==1 and seq(structure[0]['A'])==canonical and seq(structure[0]['B'])==row['binder_sequence']
        target=structure[0]['A'];binder=structure[0]['B'];target_xyz=[];target_ids=[];binder_xyz=[];binder_ids=[]
        for chain,xyz,ids in [(target,target_xyz,target_ids),(binder,binder_xyz,binder_ids)]:
            for residue in residues(chain):
                for atom in residue:
                    if atom.element.name not in ('H','D'):xyz.append(atom.pos.tolist());ids.append(residue.seqid.num)
        distance=cdist(target_xyz,binder_xyz);ti=sorted({n for n,d in zip(target_ids,distance.min(1)) if d<=4});bi=sorted({n for n,d in zip(binder_ids,distance.min(0)) if d<=4})
        assert len(set(ti)&SOURCE1)==pose['region1_contact_residues'] and len(set(ti)&SOURCE2)==pose['region2_contact_residues']
        assert int((distance<2).sum())==pose['close_atom_pairs_lt2A']==0
        target_ca=ca(target)
        if reference is None:
            reference=target_ca.copy();contact_ca=np.array([r['CA'][0].pos.tolist() for r in residues(target) if r.seqid.num in ti]);bc=ca(binder);origin=np.r_[contact_ca,bc].mean(0)
            _,_,v=np.linalg.svd(bc-bc.mean(0));vertical=v[0];normal=bc.mean(0)-contact_ca.mean(0);depth=np.cross(normal,vertical);depth/=np.linalg.norm(depth);horizontal=np.cross(vertical,depth);horizontal/=np.linalg.norm(horizontal)
            orientation=(np.column_stack([horizontal,vertical,depth]),origin)
        else:
            rotation,a,b=rigid(target_ca[285:490],reference[285:490]);apply(structure,rotation,a,b)
        apply(structure,orientation[0],orientation[1],np.zeros(3))
        tag='binder-'+str(i+1);pdb=out/'structures'/(tag+'.pdb');structure.write_pdb(str(pdb))
        # Reparse exported geometry: presentation changes must be rigid only.
        exported=gemmi.read_structure(str(pdb));assert seq(exported[0]['A'])==canonical and seq(exported[0]['B'])==row['binder_sequence']
        before=np.linalg.norm(ca(target)[0]-ca(binder)[0]);after=np.linalg.norm(ca(exported[0]['A'])[0]-ca(exported[0]['B'])[0]);assert abs(before-after)<.003
        item={**row,'id':tag,'label':'Candidate '+str(i+1),'color':render_colors[i],'pdb_file':'structures/'+pdb.name,'pdb':pdb.read_text(),'pdb_sha256':sha(pdb),'source_coordinate':str(path),'source_coordinate_sha256':sha(path),'interface_target_residues':ti,'interface_binder_residues':bi,'interface_cutoff_A':4,'target_residues':622,'binder_residues':60,'rendered_pose':'Verified minimized full-context pose; sequence-verified chain partition and rigid presentation transform only. ipTM is from the pre-minimization Chai ensemble, not a new score of this minimized frame.','rendered_region1_contacts':len(set(ti)&SOURCE1),'rendered_region2_contacts':len(set(ti)&SOURCE2)}
        selected.append(item);files[str(path)]=sha(path);files[row['summary_path']]=sha(row['summary_path'])
    both5=sum(x['intact_models_with_dual_region_contact']==5 for x in full)
    statistics={'screened_unique_sequences':len({x['binder_sequence'] for x in screens}),'screen_trials':len(screens),'followup_trials':len(full),'followup_unique_sequences':len(unique),'both_region_contacts_in_all_five_poses':both5,'both_region_contacts_in_all_five_percent':100*both5/len(full),'minimized_dual_contact_trials':sum(x['minimized_dual_contact'] for x in full),'screen_trials_per_hour':run['screen_trials_per_hour'],'followup_trials_per_hour':run['expanded_trials_per_hour'],'job_hours':run['allocation_seconds']/3600,'nodes':256,'hypotheses_with_eligible_science':run['hypotheses_tested'],'rounds':3,'best_full_iptm':max(x['intact_iptm_best'] for x in full),'median_full_best_iptm':float(np.median([x['intact_iptm_best'] for x in full]))}
    manifest={'title':'From competing hypotheses to EGFR binder candidates','campaign_job':'2908365','statistics':statistics,'top_binders':selected,'source_files_sha256':files,'selection':'Three distinct sequences: earliest retained follow-up per sequence, sorted by minimized both-region contact, number of both-region Chai poses, then expanded-context best ipTM. All three have5/5 contact poses and no <2A interchain heavy-atom pairs. No experimental affinity measured.','distributions_scope':'All937 retained follow-up trials (767 distinct sequences); earliest successful identical sequence/seed/target-length trials retained. Repeated sequence evaluations are not independent biological replicates.','target':'Human EGFR, canonical residues25--646,622 amino acids.60-aa designed binders. Binding-site figures show crop-equivalent residues286--490 of the full predicted receptor, not a separately rescored crop.'}
    (out/'science-data.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plot(out,full,selected,statistics)
    print(json.dumps({'output':str(out),'statistics':statistics,'top_ids':[x['screen_candidate_id'] for x in selected]},indent=2))

def plot(out,full,selected,stats):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#213b4b','text.color':'#213b4b','xtick.color':'#465e6b','ytick.color':'#465e6b','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(6.0,6.5),layout='constrained')
    arms=['dual','one_region','ridge'];names=['Dual-region','One-region','Ridge control'];rng=np.random.default_rng(571)
    for y,(arm,name) in enumerate(zip(arms,names)):
        values=np.array([r['intact_iptm_best'] for r in full if r['condition']==arm]);vp=axes[0].violinplot([values],positions=[y],vert=False,widths=.68,showextrema=False)
        for body in vp['bodies']:body.set_facecolor(COLORS[arm]);body.set_alpha(.24);body.set_edgecolor(COLORS[arm])
        axes[0].scatter(values,y+rng.uniform(-.19,.19,len(values)),s=8,color=COLORS[arm],alpha=.25,linewidths=0,rasterized=True);q=np.quantile(values,[.25,.5,.75]);axes[0].plot(q[[0,2]],[y,y],color=COLORS[arm],linewidth=4,solid_capstyle='round');axes[0].scatter(q[1],y,color='white',edgecolor=COLORS[arm],s=25,zorder=5)
        axes[0].text(.99,y+.32,'n='+str(len(values)),ha='right',va='center',fontsize=10,color='#60727c')
    axes[0].set(yticks=range(3),yticklabels=names,xlim=(0,1),xlabel='Full-context ipTM (model confidence)',title='Confidence across all follow-up trials');axes[0].invert_yaxis();axes[0].grid(axis='x',alpha=.14);axes[0].set_axisbelow(True)
    for arm in arms:
        rows=[r for r in full if r['condition']==arm];axes[1].scatter([r['minimized_binder_RMSD_A'] for r in rows],[r['intact_iptm_best'] for r in rows],s=14,color=COLORS[arm],alpha=.40,linewidths=0,rasterized=True)
    for i,row in enumerate(selected):
        axes[1].scatter(row['minimized_binder_RMSD_A'],row['intact_iptm_best'],marker='*',s=145,color=row['color'],edgecolor='white',linewidth=.6,zorder=7)
    axes[1].set(xscale='log',xlim=(.3,130),ylim=(.1,.85),xlabel='Binder pose shift after context expansion (Å)',ylabel='Full-context ipTM',title='Strong candidates retain the designed pose');axes[1].grid(alpha=.14);axes[1].set_axisbelow(True)
    for ext in ('png','svg'):fig.savefig(out/'figures'/('binder-distributions.'+ext),dpi=240,bbox_inches='tight',transparent=False)
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();prepare(args.output.resolve())
