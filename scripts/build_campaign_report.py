#!/usr/bin/env python3
"""Audit frozen loam artifacts and build a standalone scientific/HPC report.

Run with the RFD4 Python environment (numpy/scipy/gemmi/matplotlib).
GPU occupancy is reconstructed from leases, never described as SM utilization.
"""
import argparse, csv, hashlib, html, io, json, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import gemmi
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOCAL = ZoneInfo('Europe/Berlin')
LOOP1 = {346,348,353,355,356,357,358,382,384,408,409,411,412,415,417,418,438,440}
LOOP2 = {467,468}
def read(p): return json.loads(p.read_text())
def iso(s): return datetime.fromisoformat(s).timestamp()
def local(s): return datetime.fromisoformat(s).replace(tzinfo=LOCAL).timestamp()
def safe(x): return html.escape(str(x))
def table(head, rows, ident=''):
    return '<div class="scroll"><table id="'+ident+'"><thead><tr>'+''.join('<th>'+safe(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+safe(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--scale-run',type=Path);a=ap.parse_args()
    run=a.run.resolve(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    verified=read(run/'campaign-verification.json'); inventory=read(run/'diagnostics/verified-model-inventory.json')
    original=gemmi.read_structure(str(run/'inputs/verified-egfr-inputs/3NJP.cif'))
    cut_disulfides=[]
    for bond in original.connections:
        if bond.type==gemmi.ConnectionType.Disulf and bond.partner1.chain_name==bond.partner2.chain_name=='A':
            i,j=bond.partner1.res_id.seqid.num,bond.partner2.res_id.seqid.num
            if (286<=i<=490)!=(286<=j<=490):cut_disulfides.append({'source_author_residues':[i,j],'record':bond.name,'reported_SG_distance_A':bond.reported_distance})
    candidates=list(csv.DictReader((run/'candidate-catalog.csv').open()))
    source={str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'campaign-verification.json',run/'candidate-catalog.csv',run/'diagnostics/verified-model-inventory.json',run/'state/jnana.json']}
    target_seq=''.join(x.strip() for x in (run/'verified-target.fasta').read_text().splitlines() if not x.startswith('>'))
    geometry=[]
    for item in inventory:
        p=Path(item['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
        model=gemmi.read_structure(str(p))[0];chains={c.name:c for c in model}
        target,binder=chains['A'],chains['B']
        residues=lambda c:[r for r in c if gemmi.find_tabulated_residue(r.name).is_amino_acid()]
        seq=lambda rr:''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in rr)
        tr,br=residues(target),residues(binder)
        assert seq(tr)==target_seq and seq(br)==item['binder_sequence'],p
        ids={r.seqid.num for r in tr};assert {x-285 for x in LOOP1|LOOP2}<=ids,p
        tc=[];ti=[];bc=[]
        for r in tr:
            for at in r:
                if at.element.name not in ('H','D'):
                    tc.append([at.pos.x,at.pos.y,at.pos.z]);ti.append(r.seqid.num)
        for r in br:
            for at in r:
                if at.element.name not in ('H','D'):bc.append([at.pos.x,at.pos.y,at.pos.z])
        dist=cdist(tc,bc);minimum=dist.min(axis=1);contact_ids={i for i,d in zip(ti,minimum) if d<=4}
        one=len(contact_ids & {x-285 for x in LOOP1});two=len(contact_ids & {x-285 for x in LOOP2})
        geometry.append({**item,'contact_atom_pairs_4A':int((dist<=4).sum()),'target_contact_residues_4A':len(contact_ids),'region1_contact_residues':one,'region2_contact_residues':two,'dual_region_contact':bool(one and two),'min_interchain_distance_A':float(dist.min()),'close_atom_pairs_2A':int((dist<2).sum())})
    (out/'model-metrics.json').write_text(json.dumps(geometry,indent=2)+'\n')
    byseq=defaultdict(list)
    for g in geometry:byseq[g['binder_sequence']].append(g)
    for c in candidates:
        gs=byseq[c['sequence']];c['dual_contact_models']=sum(x['dual_region_contact'] for x in gs);c['models']=len(gs)
        c['dual_contact_fraction']=c['dual_contact_models']/len(gs);c['min_distance_A']=min(x['min_interchain_distance_A'] for x in gs)
        c['models_with_close_pairs_2A']=sum(x['close_atom_pairs_2A']>0 for x in gs)
    with (out/'candidate-metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(candidates[0]));w.writeheader();w.writerows(candidates)
    # Leases are in UTC epoch seconds. Slurm and text logs are local CET.
    allocations=[{'job':'2870467','nodes':1,'start':local('2026-11-12T23:15:48'),'end':local('2026-11-13T01:09:13'),'note':'one-node development; scientific rounds 4–5'},
                 {'job':'2872507','nodes':5,'start':local('2026-11-13T01:09:41'),'end':local('2026-11-13T01:13:07'),'note':'failed bootstrap'},
                 {'job':'2872547','nodes':5,'start':local('2026-11-13T01:13:42'),'end':local('2026-11-13T01:16:13'),'note':'obsolete automatic retry'},
                 {'job':'2872548','nodes':5,'start':local('2026-11-13T01:13:52'),'end':local('2026-11-13T02:15:52'),'note':'completed scientific rounds 6–8'}]
    start,end=allocations[-1]['start'],allocations[-1]['end']
    rows=[]
    for bp in sorted(run.glob('batches/*/batch.json')):
        b=read(bp);label=b.get('labels',{});rid=label.get('run_id',label.get('run',''));match=re.match(r'r(\d+)',rid)
        rnd=int(match[1]) if match else None
        text=(b.get('tool','')+' '+label.get('stage','')+' '+b.get('source_script','')).lower()
        stage='CPU / preparation'
        if any(x in text for x in ['chai','cofold','design_fold','downstream']):stage='Sequence / complex folding'
        elif any(x in text for x in ['generation','generate','rfd4','backbone']):stage='Backbone generation'
        elif 'mpnn' in text or 'sequence' in text:stage='Sequence design'
        for sp in sorted(bp.parent.glob('run_*/status.json')):
            lp=sp.parent/'lease.json'
            if not lp.exists():continue
            st,lease=read(sp),read(lp);t=float(lease['started_at']);seconds=float(st.get('seconds',0));n=int(b.get('gpus',0))
            rows.append({'batch':b['batch_id'],'row':sp.parent.name,'hypothesis':label.get('hypothesis_id'),'round':rnd,'start':t,'end':t+seconds,'seconds':seconds,'host':st.get('host',lease.get('host','')).split('.')[0],'gpu':str(st.get('gpu','')),'gpus':n,'stage':stage,'returncode':st.get('returncode'),'tool':b.get('tool')})
    active=[r for r in rows if r['end']>start and r['start']<end]
    hosts=sorted({r['host'] for r in active});assert len(hosts)==5,hosts
    grid=np.arange(start,end,1.0)+.5;slot=np.zeros((5,len(grid)));science=np.zeros_like(slot)
    for r in active:
        ix=hosts.index(r['host']);lo=int(np.searchsorted(grid,r['start']));hi=int(np.searchsorted(grid,r['end']))
        slot[ix,lo:hi]+=r['gpus']
        if r['stage']!='CPU / preparation':science[ix,lo:hi]+=r['gpus']
    overlap=int((slot>4).sum());slot_seconds=sum(max(0,min(r['end'],end)-max(r['start'],start))*r['gpus'] for r in active)
    exact_peak=max(sum(r['gpus'] for r in active if r['start']<=t<r['end']) for t in sorted({r['start'] for r in active}|{r['end'] for r in active}))
    status=[]
    for line in (run/'logs/main.log').read_text().splitlines():
        if 'Status(' not in line:continue
        try:t=local(line[:19])
        except ValueError:continue
        m=re.search(r'nrunning_tasks=(\d+).*nfree_gpus=(\d+)',line)
        if start<=t<=end and m:
            running,free=map(int,m.groups())
            if 0<=free<=20 and (free or running):status.append([t,20-free,running])
    stage_rows=[]
    for name in sorted({r['stage'] for r in active}):
        rr=[r for r in active if r['stage']==name];dur=[r['seconds'] for r in rr]
        stage_rows.append({'stage':name,'rows':len(rr),'failed':sum(r['returncode']!=0 for r in rr),'median_seconds':float(np.median(dur)),'p90_seconds':float(np.percentile(dur,90)),'gpu_slot_hours':sum(r['seconds']*r['gpus'] for r in rr)/3600})
    hypotheses=read(run/'state/jnana.json')['hypotheses'];hlookup={h['hypothesis_id']:h for h in hypotheses};episodes=Counter(r['hypothesis'] for r in [])
    round_hyps={}
    for n in range(4,9):
        d=read(run/f'state/rounds/round-{n:03}.json');ids=[]
        for thread in d['threads']:
            hid=thread.get('hypothesis_id',thread.get('id',thread.get('hypothesis',{}).get('id')))
            if hid:ids.append(hid)
        if not ids:
            for hp in (run/'state/hypotheses').iterdir():
                if list((hp/'runs').glob(f'r{n:03}-*.json')):ids.append(hp.name)
        round_hyps[n]=sorted(set(ids));episodes.update(set(ids))
    assert sum(episodes.values())==17 and len(episodes)==7,(episodes,round_hyps)
    best=np.array([float(c['max_chai_iptm']) for c in candidates]);all_scores=np.array([g['scores']['iptm'] for g in geometry])
    phase_stats={};phase_sequences={}
    for phase,rounds in [('one_node',{4,5}),('five_node',{6,7,8})]:
        scores=defaultdict(list)
        for g in geometry:
            if g['round'] in rounds:scores[g['binder_sequence']].append(g['scores']['iptm'])
        values=[max(v) for v in scores.values()];phase_sequences[phase]=set(scores)
        phase_stats[phase]={'verified_models':sum(len(v) for v in scores.values()),'distinct_sequences':len(scores),'median_best_iptm':float(np.median(values)),'maximum_best_iptm':max(values),'sequences_best_iptm_ge_08':sum(v>=.8 for v in values),'sequences_best_iptm_ge_07':sum(v>=.7 for v in values)}
    stats={'run':str(run),'generated_utc':datetime.now(timezone.utc).isoformat(),'unique_hypotheses_tested':len(episodes),'hypothesis_round_evaluations':sum(episodes.values()),'proposed_hypotheses':len(hypotheses),'unique_sequences':len(candidates),'verified_models':len(geometry),'best_iptm_median':float(np.median(best)),'best_iptm_max':float(best.max()),'best_iptm_ge_08':int((best>=.8).sum()),'best_iptm_ge_07':int((best>=.7).sum()),'dual_contact_models':sum(g['dual_region_contact'] for g in geometry),'any_dual_contact_sequences':sum(c['dual_contact_models']>0 for c in candidates),'five_node':{'seconds':end-start,'allocated_gpu_hours':20*(end-start)/3600,'leased_gpu_hours':slot_seconds/3600,'mean_slot_occupancy_fraction':slot_seconds/(20*(end-start)),'peak_leased_slots':exact_peak,'fraction_seconds_zero_leased':float(np.mean(slot.sum(0)==0)),'fraction_seconds_all_twenty_leased':float(np.mean(slot.sum(0)>=20)),'rows':len(active),'failed_rows':sum(r['returncode']!=0 for r in active),'slot_overlap_node_seconds':overlap,'verified_models':215,'unique_models_per_hour':215/((end-start)/3600)},'stage_timings':stage_rows,'allocations':allocations,'source_sha256':source}
    (out/'analysis.json').write_text(json.dumps(stats,indent=2)+'\n');(out/'row-timings.json').write_text(json.dumps(rows,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
        svg=(out/(name+'.svg')).read_text();return svg[svg.index('<svg'):]
    charts={}
    fig,ax=plt.subplots(1,2,figsize=(11,3.5));ax[0].hist(all_scores,bins=np.linspace(0,1,21),color='#5b7caa');ax[0].set(xlabel='Chai ipTM (all 240 verified models)',ylabel='Models');ax[1].hist(best,bins=np.linspace(0,1,21),color='#187f70');ax[1].set(xlabel='Best Chai ipTM per distinct sequence (n=44)',ylabel='Sequences');charts['confidence']=save(fig,'confidence')
    ordered=sorted(candidates,key=lambda c:float(c['max_chai_iptm']),reverse=True)
    fig,ax=plt.subplots(figsize=(11,3.8));x=np.arange(44);ax.bar(x,[float(c['max_chai_iptm']) for c in ordered],color='#187f70',label='Best model');ax.scatter(x,[float(c['median_chai_iptm']) for c in ordered],s=12,c='#172f48',label='Median model');ax.set(xlabel='Distinct binder sequences, ranked by best ipTM',ylabel='Chai ipTM',ylim=(0,1));ax.legend();charts['ranked']=save(fig,'ranked-candidates')
    width=30;n=len(grid)//width;heat=slot[:,:n*width].reshape(5,n,width).mean(2)
    fig,ax=plt.subplots(figsize=(11,3.1));im=ax.imshow(heat,aspect='auto',origin='lower',extent=[0,n*width/60,-.5,4.5],vmin=0,vmax=4,cmap='YlGnBu');ax.set_yticks(range(5),hosts);ax.set(xlabel='Minutes since 01:13:52 CET, 13 November 2026',title='Leased GPU slots per node (30-second means; four slots per node)');fig.colorbar(im,ax=ax,label='Leased slots / node');charts['heat']=save(fig,'node-slot-saturation')
    fig,ax=plt.subplots(figsize=(11,3.1));ax.plot((grid-start)/60,slot.sum(0),lw=1,label='Row leases');ax.plot((grid-start)/60,science.sum(0),lw=1,label='Science-tagged row leases',alpha=.8)
    if status:
        st=np.array(status);ax.scatter((st[:,0]-start)/60,st[:,1],s=5,c='#a73b43',label='Scheduler 20 − free slots',alpha=.5)
    ax.axhline(20,c='gray',ls='--');ax.set(xlabel='Minutes since allocation start',ylabel='GPU slots',ylim=(0,21));ax.legend(ncol=3,fontsize=8);charts['occupancy']=save(fig,'aggregate-slot-occupancy')
    fig,ax=plt.subplots(figsize=(11,3.4));names=[r['stage'] for r in stage_rows];ax.barh(names,[r['median_seconds'] for r in stage_rows],color='#187f70',label='Median');ax.scatter([r['p90_seconds'] for r in stage_rows],names,c='#b76c2b',label='90th percentile',zorder=3);ax.set(xlabel='Row wall time (seconds), successful and failed attempts');ax.legend();charts['timing']=save(fig,'stage-timings')
    # Descriptive capacity sensitivity; no claim of measured 20-node speedup.
    f=stats['five_node']['mean_slot_occupancy_fraction'];ns=np.arange(5,20);fig,ax=plt.subplots(figsize=(11,3.1));ax.plot(ns,215*ns/5,label='Ideal proportional science capacity',c='#187f70');ax.axhline(215,c='#a73b43',label='Unchanged dispatch / orchestration ceiling');ax.fill_between(ns,215,215*ns/5,alpha=.12,color='#187f70');ax.set(xlabel='Scientific tool nodes (inference nodes are separate)',ylabel='Models per original 62-minute window',title='Scaling bounds: sensitivity scenario, not a measured forecast');ax.legend();charts['scaling']=save(fig,'scaling-sensitivity')
    corr=read(next(run.glob('batches/b-f9605d1be8e3/run_*/compact_summary.json')))
    hrows=[(hid,hlookup[hid].get('summary',''),num) for hid,num in episodes.items()]
    rr=[(r['round'],r['hypotheses'],r['effective_accepted_claims'],', '.join(f'{v} {k}' for k,v in r['outcomes'].items())) for r in verified['completed_scientific_rounds']]
    tr=[(r['stage'],r['rows'],r['failed'],f"{r['median_seconds']:.1f}",f"{r['p90_seconds']:.1f}",f"{r['gpu_slot_hours']:.3f}") for r in stage_rows]
    ar=[(r['job'],r['nodes'],datetime.fromtimestamp(r['start'],LOCAL).isoformat(),f"{(r['end']-r['start'])/60:.2f}",r['note']) for r in allocations]
    cr=[(c['candidate_id'],c['sequence'],f"{float(c['max_chai_iptm']):.4f}",f"{float(c['median_chai_iptm']):.4f}",c['models'],f"{c['dual_contact_models']}/{c['models']}",c['models_with_close_pairs_2A']) for c in ordered]
    occ=stats['five_node'];body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jupiter loam: five-node scientific and HPC audit</title>
<style>:root{{--ink:#152e43;--muted:#52687a;--line:#d6e2e9;--accent:#187f70}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f8;color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:32px 24px}}header{{background:#102d43;color:#fff;padding:36px;border-radius:14px}}h1{{line-height:1.15;font-size:2.2rem}}h2{{margin-top:0}}section{{background:white;border:1px solid var(--line);padding:28px;margin:24px 0;border-radius:12px}}nav a{{color:var(--accent);margin-right:18px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:20px 0}}.card{{border:1px solid var(--line);padding:15px;border-radius:8px}}.card b{{display:block;font-size:1.8rem}}.note{{padding:14px;background:#edf5f3;border-left:4px solid var(--accent)}}.warn{{background:#fff5e7;border-left-color:#be7d27}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:.85rem;width:100%}}th,td{{border-bottom:1px solid var(--line);text-align:left;padding:10px;vertical-align:top}}th{{background:#edf3f6;cursor:pointer}}td:nth-child(2){{overflow-wrap:anywhere;min-width:200px}}svg{{width:100%;height:auto}}code{{overflow-wrap:anywhere;font-size:.85em}}input{{padding:10px;width:min(100%,600px);margin-bottom:14px}}a{{color:#126e87}}.small{{font-size:.86rem;color:var(--muted)}}@media print{{body{{background:white}}section{{break-inside:avoid}}nav,input{{display:none}}}}</style>
<main><header><p>LOAM · JUPITER / JÜLICH · AUDITED 13 NOVEMBER 2026</p><h1>Five nodes: scientific output and scaling constraints</h1><p>A tournament-driven EGFR binder design campaign, with artifact-verified predictions and reconstructed execution occupancy. Headline counts describe the five-node phase; earlier one-node results are included below as campaign context.</p></header>
<nav><a href="#science">Scientific results</a><a href="#hpc">Throughput & saturation</a><a href="#scale">Twenty-node plan</a><a href="#candidates">Candidate catalog</a><a href="#methods">Methods & provenance</a></nav>
<div class="cards"><div class="card"><b>7</b>distinct hypotheses tested</div><div class="card"><b>{sum(len(round_hyps[n]) for n in (6,7,8))}</b>five-node hypothesis-round evaluations</div><div class="card"><b>{phase_stats["five_node"]["distinct_sequences"]}</b>five-node binder sequences</div><div class="card"><b>{phase_stats["five_node"]["verified_models"]}</b>five-node complex models</div><div class="card"><b>{phase_stats["five_node"]["median_best_iptm"]:.3f}</b>five-node median best ipTM / sequence</div><div class="card"><b>{occ['mean_slot_occupancy_fraction']:.1%}</b>mean leased GPU-slot occupancy</div></div>
<section id="science"><h2>Scientific findings</h2><p>The campaign generated 26 hypothesis proposals. Seven distinct hypotheses reached scientific execution and independent review across completed rounds 4–8; repeated evaluations of the same hypothesis count toward 17 evaluations, not toward seven independent hypotheses. Rounds 1–3 were preparation/debugging. Rounds 4–5 ran on one node; rounds 6–8 used five nodes.</p><p>The target was human EGFR (P00533), the 205-residue 3NJP chain-A crop, with 60-residue designed binders. The pipeline used RFD4-Proteina backbones, sequence design and Chai complex prediction with 3 trunk recycles, 200 diffusion steps and five models per folding invocation. The catalog contains 44 distinct sequences and 240 distinct verified prediction files; these are correlated model samples, not 240 independent binders.</p><p class="note warn"><b>Binding performance is predicted structural confidence.</b> No experimental Kd, IC50, validated binding free energy or biological activity was measured. Chai ipTM does not establish affinity. Best-model selection and repeated sampling can inflate apparent scores.</p><p>Best ipTM per sequence ranges from {best.min():.3f} to {best.max():.3f}; {int((best>=.8).sum())}/44 sequences reach 0.80 and {int((best>=.7).sum())}/44 reach 0.70. The median is {np.median(best):.3f}. These thresholds describe confidence only, with no calibrated success probability.</p>{charts['confidence']}{charts['ranked']}
<h3>Geometry and a controlled hypothesis test</h3><p>Independent report-time remeasurement verified all 240 coordinate hashes, exact target/binder sequences and the native hotspot residue IDs. Heavy-atom contacts within 4 Å reach both mapped EGF-contact regions in {stats['dual_contact_models']}/240 models and at least one model for {stats['any_dual_contact_sequences']}/44 sequences. This pooled contact count spans different design conditions and is descriptive, not a controlled estimate of a dual-conditioning benefit.</p><p>In the corrected matched cohort (four dual-region candidates and four controls, five models each), all four dual candidates engaged both regions in at least one model. However, target-aligned binder-CA RMSD to each candidate’s model 0 averaged <b>1.848 Å for dual candidates versus 1.291 Å for controls</b> (control sample SD 0.550 Å). The prespecified requirement, dual mean ≤ control mean − control SD, failed. Thus contact engagement was achieved but improved pose reproducibility was not demonstrated. This model-ensemble variation is not experimental dynamics.</p><p class="note warn">An earlier contact-analysis script used source residue numbers on renumbered models and returned invalid zero contacts. Two associated claims were retracted with preserved audit records. Those contact conclusions are excluded here; corrected round-8 results and independently verified coordinates are used.</p>{table(['Round','Hypotheses evaluated','Accepted claims after retraction','Validator outcomes'],rr)}<details><summary>Seven tested hypotheses and repeat evaluations</summary>{table(['Hypothesis ID','Scientific proposition (stored summary)','Evaluations'],hrows)}</details></section>
<section id="hpc"><h2>Throughput and node saturation</h2><p>The successful five-node allocation (job 2872548) lasted <b>62 minutes</b>, with 20 GH200 GPU slots and 20.667 allocated GPU-hours. It produced 215 of the 240 verified models: rounds 6/7/8 contributed 80/40/95. Gross throughput was <b>{occ['unique_models_per_hour']:.1f} verified models/hour</b> and {215/occ['allocated_gpu_hours']:.2f} models per allocated GPU-hour. The preceding one-node phase produced 25 models, but had different hypothesis workloads and debugging; it is not a matched strong-scaling baseline.</p><p>Reconstructed leases occupy <b>{occ['leased_gpu_hours']:.3f} GPU-slot-hours ({occ['mean_slot_occupancy_fraction']:.1%} of allocated capacity)</b>, with a peak of {exact_peak}/20 slots. Approximately {occ['fraction_seconds_zero_leased']:.1%} of one-second samples have no leased GPU rows. There were {len(active)} completed row attempts overlapping the allocation; {occ['failed_rows']} returned a nonzero status. Failed postprocessing may follow real GPU computation and is retained in resource accounting.</p><p class="note"><b>Occupancy is not hardware utilization.</b> These plots use row leases and wall time, cross-checked against scheduler free-slot logs. A leased GPU can be waiting, importing libraries, running CPU work or synchronizing. No prior nvidia-smi/DCGM samples were retained, so SM utilization, VRAM pressure and actual CPU saturation cannot be recovered. Parent Slurm CPUTime is allocated capacity, not measured computation.</p>{charts['heat']}{charts['occupancy']}<p class="small">Times shown in CET (UTC+1). Heatmap bins average 30 seconds. Lease start + recorded row seconds defines each interval; file modification times are not used. Overlapping-slot node-seconds above four: {overlap}. Initial uninitialized scheduler records with zero free slots and zero running tasks are excluded.</p>
<h3>Where time went</h3>{charts['timing']}{table(['Stage (tool-name classification)','Row attempts','Nonzero status','Median seconds','P90 seconds','Leased GPU-hours'],tr)}<p>Stage labels are inferred from registered tool names and source paths; combined sequence/folding rows include startup and postprocessing. Small CPU analysis rows sometimes reserved a GPU. Idle gaps include LLM orchestration, code generation/repair, dependencies, reporting and operator audit pauses. Prior HTTP logs lack request-start times and token usage, so these gaps cannot be assigned entirely to inference.</p>
<h3>Allocation history and cost boundary</h3>{table(['Job','Nodes','Start (CET)','Minutes','Purpose'],ar)}<p>Two short five-node bootstrap/retry allocations consumed an additional {(206+151)*20/3600:.3f} allocated GPU-hours, beyond the 20.667 successful-allocation denominator. Their runtimes overlap partially and are separately charged. Completed jobs were deliberately released after evidence was saved; scheduler CANCELLED status does not mean the scientific rounds failed. Other users’ and campaigns’ jobs are excluded and were not changed.</p></section>
<section id="scale"><h2>Scaling toward twenty total nodes</h2><p class="note">The twenty-node campaign is a new run, not a completed benchmark. One campaign-owned Nemotron node has been submitted for warmup and inference measurements. Three reservation nodes were occupied by other campaigns at planning time. The default plan stays inside bne_deepwing and grows as those nodes become available; no foreign job is cancelled.</p><p>The five-node trace shows that more scientific nodes alone will not yield a proportional end-to-end speedup. Legacy fan-out (five concurrent hypotheses × four rows) can expose only 20 GPU tasks at once. The next run needs enough independent hypothesis/candidate work to fill 72–76 scientific GPUs, and fewer serial LLM/repair transitions.</p><ol><li>Benchmark the owned local Nemotron endpoint with representative structured outputs and tool calls, at several concurrent request levels. Record service/queue latency, tokens per second and failures.</li><li>Start with one inference node and select a second only if measured inference queueing would starve scientific workers. Candidate configurations are 1 inference + 19 science or 2 inference + 18 science; initially fewer reservation nodes may be available.</li><li>Bundle validated backbone generation, fixed-target sequence design, complex folding and compact geometric analysis per GPU row. Preserve full scientific settings and matched control budgets; use independent recorded seeds and avoid padded duplicate work.</li><li>Increase tournament hypothesis and candidate fan-out to maintain a ready backlog. Keep CPU-only analysis at zero requested GPUs. Stagger 429/503/529 retries through the shared rate-limit gate; all Argo requests use gpt-5.6-luna.</li><li>Record per-node nvidia-smi utilization/memory/power and inference queue/token counters in the new run. Distinguish cold-start/model loading, queue wait, steady state, retries and shutdown.</li></ol>{charts['scaling']}<p>At the previous dispatch ceiling, extra nodes can remain idle. The upper line assumes unchanged per-GPU efficiency, unlimited independent work and no inference or filesystem bottleneck; it is a capacity bound, not a forecast. Real improvement must be measured in unique valid candidates/models per elapsed hour and per allocated GPU-hour, with hypothesis review completion and confidence/contact quality retained.</p></section>
<section id="candidates"><h2>Binder catalog</h2><p>Rows are ranked by best ipTM. Dual contacts mean a model has ≥1 heavy-atom contact within 4 Å to each mapped EGF-contact region. A model with interchain atom pairs &lt;2 Å is flagged for inspection; this simple distance screen is not a validated steric energy calculation.</p><input id="search" aria-label="Filter candidates" placeholder="Filter candidate ID or amino-acid sequence"><button id="fasta">Download sequences</button>{table(['Candidate ID','Sequence','Best ipTM','Median ipTM','Verified models','Dual-contact models','Models with <2 Å pairs'],cr,'catalog')}</section>
<section id="methods"><h2>Reproducibility and boundaries</h2><p>Source campaign: <code>{safe(run)}</code>. Target source SHA-256: <code>{verified['source_target_sha256']}</code>. Source author A286–490 maps to native A1–205 by subtracting 285 (UniProt310–514). Region 1 is a discontinuous set of 18 mapped EGF-contact residues; region 2 is source A467–468/native A182–183. These are two mapped contact regions, not necessarily two contiguous loops. Target self-contacts and unrelated receptor copies are excluded.</p><p>The report builder checks coordinate SHA-256, exact sequences, finite previously verified coordinates and hotspot presence before computing heavy-atom distances with Gemmi/SciPy. Duplicate coordinate hashes were removed in the source inventory. Distinct sequences can share a design history, so sequence count does not imply statistical independence. Scientific outcomes retain blocked/contradicted/inconclusive findings rather than discarding unsuccessful hypotheses.</p><p>Companion artifacts: <a href="analysis.json">analysis.json</a>, <a href="model-metrics.json">per-model geometry and provenance</a>, <a href="candidate-metrics.csv">candidate metrics CSV</a>, <a href="row-timings.json">row timings</a>. This HTML embeds every figure and needs no network connection. SVG and PNG plot files are also saved beside it. The source-code report builder is <code>scripts/build_campaign_report.py</code>.</p><details><summary>Source hashes</summary><pre>{safe(json.dumps(source,indent=2))}</pre></details><p class="small">Generated {safe(stats['generated_utc'])}. Observational HPC accounting and computational structural predictions; no experimental validation.</p></section></main>
<script>document.getElementById('search').addEventListener('input',e=>{{for(const r of document.querySelectorAll('#catalog tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase())}});document.getElementById('fasta').onclick=()=>{{const s={json.dumps((run/'candidate-binders.fasta').read_text())};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{{type:'text/plain'}}));a.download='candidate-binders.fasta';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};document.querySelectorAll('#catalog th').forEach((th,i)=>th.onclick=()=>{{const b=document.querySelector('#catalog tbody');const rows=[...b.rows];const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';rows.sort((a,b)=>{{let x=a.cells[i].textContent,y=b.cells[i].textContent;return (asc?1:-1)*(i>=2&&i!==5?Number(x)-Number(y):x.localeCompare(y))}});rows.forEach(r=>b.appendChild(r))}});</script></html>'''
    stats['crop_cut_disulfides']=cut_disulfides
    stats['scientific_phase_metrics']=phase_stats
    overlap_sequences=len(phase_sequences['one_node']&phase_sequences['five_node']);new_sequences=len(phase_sequences['five_node']-phase_sequences['one_node']);stats['five_node']['first_seen_sequences']=new_sequences
    phase_table=table(['Phase','Hypothesis-round evaluations','Distinct sequences represented','Verified models','Median best ipTM / sequence','Sequences with best ipTM ≥0.80'],[("One node (rounds 4–5)",2,phase_stats['one_node']['distinct_sequences'],25,f"{phase_stats['one_node']['median_best_iptm']:.3f}",phase_stats['one_node']['sequences_best_iptm_ge_08']),("Five nodes (rounds 6–8)",15,phase_stats['five_node']['distinct_sequences'],215,f"{phase_stats['five_node']['median_best_iptm']:.3f}",phase_stats['five_node']['sequences_best_iptm_ge_08'])])
    phase_note=f'<h3>Five-node-only scientific output</h3>{phase_table}<p>The five-node phase evaluated seven distinct hypotheses in fifteen hypothesis-round evaluations. It represents 43 distinct binder sequences; {overlap_sequences} were also represented in the one-node phase, leaving {new_sequences} sequences first seen during the five-node phase. Its median best ipTM is 0.779, with 17/43 sequences reaching 0.80 and 26/43 reaching 0.70. New-sequence throughput is {new_sequences/(62/60):.1f}/hour over the gross 62-minute allocation. The 44-sequence/240-model totals and confidence plots above describe the entire completed campaign. Shared sequences and different work budgets prevent a clean strong-scaling comparison.</p>'
    body=body.replace('<section id="hpc"><h2>Throughput and node saturation</h2>','<section id="hpc"><h2>Throughput and node saturation</h2>'+phase_note)
    chemistry='<p class="note warn"><b>Receptor-context limitation:</b> the source mmCIF explicitly records two disulfides crossing this crop boundary, source A482–491 and A486–499. The 205-residue crop therefore omits native disulfide partners and receptor context. Confidence/contact scores on this crop are screening observations. Energetic ranking requires intact-context re-folding and verified disulfide topology; no crop MM/GBSA result is claimed.</p>'
    body=body.replace('<h3>Geometry and a controlled hypothesis test</h3>',chemistry+'<h3>Geometry and a controlled hypothesis test</h3>')
    if a.scale_run and (a.scale_run/'inference-benchmark.json').exists():
        benchmarks=read(a.scale_run/'inference-benchmark.json');progress=read(a.scale_run/'scale-progress.json')
        b=benchmarks[-1];passed=sum(x['ok'] for group in benchmarks for x in group['results']);total=sum(len(group['results']) for group in benchmarks)
        addition=f'<h3>Measured local inference capacity</h3><p>The owned one-node Nemotron pilot passed {passed}/{total} structured tool-calling protocol tasks. At concurrency {b["concurrency"]}, it completed {b["successful_tasks_per_second"]:.2f} tasks/second in {b["wall_seconds"]:.2f} seconds; each task made two model calls, using about 16,000 total input tokens across both calls. This synthetic fixture uses repeated method documentation and benefits from prefix caching. It verifies the real loam adapter and tool/output protocol, not scientific reasoning quality or sustained campaign throughput.</p><p><b>Selected starting split: one inference node + nineteen scientific nodes.</b> Monitor queue depth and request latency under real validator/code-generation load before adding a second inference node. Hardware and vLLM telemetry are being recorded. Current stage: {safe(progress["phase"])}.</p>'
        body=body.replace('<section id="scale"><h2>Scaling toward twenty total nodes</h2>','<section id="scale"><h2>Scaling toward twenty total nodes</h2>'+addition)
        pilot_path=a.scale_run/'pilot-validation.json'
        if pilot_path.exists():
            pilot=read(pilot_path);follow=pilot['intact_followup']
            body=body.replace('<h3>Measured local inference capacity</h3>',f'<p>The new one-node bundled science pilot passed four candidates / twenty crop models. Median cold candidate wall time was {pilot["cold_median_candidate_seconds"]:.1f} seconds (generation {pilot["cold_median_stage_seconds"]["RFD4-Proteina"]:.1f}, sequence {pilot["cold_median_stage_seconds"]["proteinmpnn"]:.1f}, folding {pilot["cold_median_stage_seconds"]["chai-lab"]:.1f} seconds). An intact 622-residue EGFR follow-up also passed five predictions, 25 verified target disulfides, CUDA minimization and static MM/GBSA in {follow["elapsed_seconds"]:.1f} seconds. These are new-run operator validation results, excluded from the completed five-node campaign counts and from twenty-node tournament completion.</p><h3>Measured local inference capacity</h3>')
            stats['new_run_pilot']=pilot
        actual_path=a.scale_run/'initial-tournament-inference-timing.json'
        if actual_path.exists():
            actual=read(actual_path)
            actual_note=f'<h3>Actual initial tournament and launch status</h3><p>The owned local Nemotron service generated 24 initial hypotheses and completed their first Swiss tournament in {actual["wall_seconds"]:.1f} seconds while the science allocation waited. All {actual["requests"]} model requests succeeded; median request latency was {actual["median_request_seconds"]:.1f} seconds. This longer real workload differs from the short protocol fixture above. This initial pool and ranking are preparation, not scientific support; they are excluded from the completed five-node campaign counts. A hash-bound ranking cache avoids repeating this initial tournament after allocation.</p><p>Current inference route: {safe(progress.get("model_routes",{}).get("primary","see campaign configuration"))}. Argo remains an unused gated service; any Argo requests use gpt-5.6-luna. Science job {safe(progress.get("scientific_job"))} uses 19 nodes / 76 GPUs in bne_deepwing, with one additional local inference node. The durable controller verifies the twenty-node total. Current campaign stage: {safe(progress["phase"])}. See the <a href="../scale20-20261113/index.html">twenty-node continuation report</a> for measured outcomes, telemetry, failures and costs. CPU manifest preparation preserves 12 candidates per hypothesis (4 per condition), with 12 hypotheses concurrently and 144 independent rows available to 76 science GPUs.</p>'
            body=body.replace('<h3>Measured local inference capacity</h3>',actual_note+'<h3>Measured local inference capacity</h3>')
            stats['actual_initial_tournament']=actual
        stats['scale_run']=str(a.scale_run.resolve());stats['inference_benchmark']=benchmarks
    (out/'analysis.json').write_text(json.dumps(stats,indent=2)+'\n')
    (out/'index.html').write_text(body)
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
