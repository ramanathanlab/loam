#!/usr/bin/env python3
"""One-slide scientific summary, exported in clean/interface NGL variants."""
import argparse,csv,hashlib,html,json,shutil,zipfile
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import scaleup_slides as deck

TITLE='From competing hypotheses to EGFR binder candidates'
INK='#193647';TEAL='#078a92';MUTED='#5b707c';PALE='#eff6f7'

def slide(out,data,style):
    st=data['statistics'];s=deck.Slide.__new__(deck.Slide);s.title=TITLE;s.commands=[];s.dark=False
    s.footer=lambda page,total:None
    s.notes=json.dumps({'scope':data['target'],'selection':data['selection'],'distributions':data['distributions_scope'],'statistics':st,'candidate_sources':[{k:c[k] for k in ['screen_candidate_id','binder_sequence','source_coordinate','source_coordinate_sha256','intact_iptm_best','intact_models_with_dual_region_contact','minimized_binder_RMSD_A','static_MMGBSA_kcal_mol','rendered_pose']} for c in data['top_binders']],'interface_cutoff':'4Å interchain heavy-atom distance; contacting complete residues colored in highlighted version','rates':'Retained distinct-input trials divided by full parent Slurm elapsed11696seconds, including startup and all workflow idle time.102 inference+154 science nodes in one256-node allocation.','limitations':'Computational predictions; no experimental affinity. Chai ipTM is pre-minimization. Both-region contacts are a structural proxy. Follow-up trials share767unique sequences; not independent biological replicates. Three tournament/science/review rounds completed; optional reasoner synthesis failed context in all3rounds. Individual failed threads/noncanonical inputs remain in the full campaign report.'},indent=2)
    s.rect(0,0,960,540,'#ffffff');s.rect(32,22,35,3,TEAL);s.text(78,18,850,'LOAM  /  AGENTIC PROTEIN DESIGN  /  JUPITER',10,TEAL,'bold',max_height=18)
    s.text(32,45,896,TITLE,27,INK,'bold',max_height=39)
    s.text(32,88,896,'234 hypotheses · 3 tournament rounds · 60-aa binders evaluated with a 622-aa EGFR ectodomain',12.5,MUTED,max_height=20)
    metrics=[('4,309','unique designed sequences'),('937','full-context follow-up trials'),('390 / 937','both regions contacted in all 5 poses')]
    positions=[(32,174),(216,174),(400,188)]
    for (value,label),(x,w) in zip(metrics,positions):
        s.rect(x,121,w,61,PALE,radius=6);s.text(x+12,127,w-24,value,25,TEAL,'bold',max_height=32);s.text(x+12,158,w-24,label,10.4,INK,max_height=24)
    s.rect(609,121,319,61,'#f5f2ec',radius=6);s.text(622,128,293,'1,326 screens / hour',22,INK,'bold',max_height=29);s.text(622,157,293,'288 full-context trials / hour · 256 nodes',10.5,MUTED,max_height=19)
    for i,c in enumerate(data['top_binders']):
        x=32+i*187;s.text(x+4,195,177,c['label'],12,c['color'],'bold',max_height=18)
        image=out/'ngl-images'/f"{c['id']}-{style}-site.png";assert image.is_file();s.image(image,x,217,179,198)
        s.text(x+4,419,177,f"ipTM {c['intact_iptm_best']:.3f}",18,c['color'],'bold',max_height=25)
        s.text(x+4,446,177,'Both regions: 5 / 5 poses',10.8,INK,max_height=18)
        s.text(x+4,465,177,c['condition'].replace('_',' ')+' conditioning',10,MUTED,max_height=16)
    s.image(out/'figures/binder-distributions.png',598,191,338,304)
    s.text(608,497,320,'937 trials / 767 sequences · stars: featured candidates',8.7,MUTED,max_height=15)
    legend='EGFR: gray · binders: teal / orange'
    if style=='interface':legend+=' · interface residues: gold'
    s.text(36,490,549,legend,10.1,MUTED,max_height=23)
    s.rect(32,518,896,.6,'#d8e2e7')
    s.text(32,523,896,'Computational predictions; no measured affinity. Minimized poses shown; ipTM from Chai. Throughput: whole-job mean (3 h 15 min).',8.4,MUTED,max_height=14)
    return s

def main(out):
    data=json.loads((out/'science-data.json').read_text())
    source=json.loads((Path(__file__).resolve().parents[1]/'reports/single-allocation-256-20261115/data.json').read_text());rows=[r for r in source['expanded_trials'] if r['run'].startswith('256')]
    columns=['round','condition','seed','binder_sequence','intact_iptm_best','intact_iptm_median','intact_models_with_dual_region_contact','minimized_dual_contact','minimized_binder_RMSD_A','static_MMGBSA_kcal_mol']
    with (out/'cohort-metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (out/'featured-binders.fasta').write_text(''.join('>'+c['id']+' ipTM='+format(c['intact_iptm_best'],'.6f')+'\n'+c['binder_sequence']+'\n' for c in data['top_binders']))
    original=deck.pdfcanvas.Canvas
    class ConferenceCanvas(original):
        def setTitle(self,value):super().setTitle(TITLE)
    deck.pdfcanvas.Canvas=ConferenceCanvas
    for style in ['clean','interface']:
        folder=out/style;folder.mkdir(exist_ok=True);count=deck.export([slide(out,data,style)],folder);assert count==1
        prs=Presentation(folder/'scaleup.pptx');prs.core_properties.title=TITLE;prs.core_properties.subject='Verified in silico EGFR binder-design results from the 256-node Loam campaign';prs.core_properties.author='Loam campaign'
        # PDF/preview use the white composites. PowerPoint embeds actual alpha
        # PNGs, preserving transparency if the presenter changes its background.
        page=prs.slides[0];pictures=[x for x in page.shapes if x.shape_type==MSO_SHAPE_TYPE.PICTURE]
        assert len(pictures)==4
        for i,old in enumerate(pictures[:3]):
            image=out/'ngl-images'/f'binder-{i+1}-{style}-site-transparent.png';assert image.is_file()
            parent=old._element.getparent();position=parent.index(old._element);new=page.shapes.add_picture(str(image),old.left,old.top,width=old.width,height=old.height);parent.remove(old._element);parent.remove(new._element);parent.insert(position,new._element)
        prs.save(folder/'scaleup.pptx')
        for source,target in [(folder/'scaleup.pptx',out/f'loam-science-{style}.pptx'),(folder/'scaleup.pdf',out/f'loam-science-{style}.pdf'),(folder/'slides/slide-01.png',out/f'loam-science-{style}.png')]:shutil.copy2(source,target)
        (folder/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Loam science · {style}</title><style>body{{font:18px system-ui;background:#edf2f5;margin:20px}}img{{width:100%}}a{{margin:15px}}</style><a href="../index.html">Both versions</a><a href="scaleup.pptx">PowerPoint</a><a href="scaleup.pdf">PDF</a><a href="../ngl-gallery.html">Interactive NGL</a><img src="slides/slide-01.png" alt="One-slide EGFR scientific summary">')
    cards=[]
    for style,label in [('clean','Clean structures'),('interface','Interface residues highlighted')]:
        cards.append(f'<section><h2>{label}</h2><p><a href="loam-science-{style}.pptx">PowerPoint (one slide)</a> · <a href="loam-science-{style}.pdf">PDF</a> · <a href="loam-science-{style}.png">PNG</a></p><a href="loam-science-{style}.png"><img src="loam-science-{style}.png" alt="{label}"></a></section>')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Loam · conference science slide</title><style>body{margin:0;background:#edf2f5;color:#193647;font:17px system-ui}main{max-width:1450px;margin:auto;padding:26px}section{background:white;padding:20px;margin:22px 0;border-radius:12px}img{width:100%;height:auto}a{color:#087f87}p{line-height:1.5}</style><main><h1>One slide · EGFR binder science</h1><p>Choose a clean or interface-highlighted structure version. Both contain the same verified scientific data.</p><p><a href="ngl-gallery.html">Interactive NGL structures and PNG export</a> · <a href="transparent-renderings.html">Transparent molecular PNGs</a> · <a href="cohort-metrics.csv">Cohort metrics</a> · <a href="featured-binders.fasta">Featured sequences</a> · <a href="science-data.json">Data and provenance</a> · <a href="methods.txt">Methods and interpretation</a></p>'+''.join(cards)+'</main></html>')
    thumbnails=[]
    for p in sorted((out/'ngl-images').glob('*-transparent.png')):
        label=p.stem.replace('-transparent','').replace('-',' ');thumbnails.append('<article><h2>'+html.escape(label)+'</h2><a href="ngl-images/'+p.name+'" download><img src="ngl-images/'+p.name+'" alt="'+html.escape(label)+'"></a><p><a href="ngl-images/'+p.name+'" download>Download transparent PNG</a></p></article>')
    (out/'transparent-renderings.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Loam · transparent molecular renderings</title><style>body{font:16px system-ui;margin:24px;background:#f4f7f9;color:#193647}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{background:white;padding:15px;border:1px solid #d5e0e5;border-radius:10px}h2{font-size:17px}img{width:100%;height:320px;object-fit:contain;background:repeating-conic-gradient(#fff 0% 25%,#e1e5e8 0% 50%) 50% /24px 24px}a{color:#087f87}</style><h1>Transparent NGL renderings</h1><p><a href="index.html">One-slide versions</a> · <a href="ngl-gallery.html">Rotate and export in NGL</a></p><p>The checkerboard is a preview background; it is not part of the PNG images. Site views show the binding domain; context views show the full ectodomain.</p><main>'+''.join(thumbnails)+'</main></html>')
    methods='''Loam scientific conference slide — last 256-node campaign (Slurm 2908365)

Scope: 4,309 retained screen trials / unique 60-aa binder sequences; 937 deduplicated
full-context follow-up trials covering 767 distinct sequences. 234 hypotheses
produced canonical independently verified screen results across 3 tournament rounds.
The display is an in silico design campaign, not an experimental binding assay.

The three illustrated sequences are selected using the documented retained-trial
rule in science-data.json. Each has contacts to both designated epitope regions
in all 5 Chai cofold predictions and after minimization. Minimized structures are
shown, with chain A1--622 matched to EGFR UniProt P00533 residues 25--646 and B1--60
matched to the designed sequence. Only sequence-verified chain partition and
rigid presentation transforms were applied. The binding-site domain is shown
in the slide; the NGL gallery also provides the full ectodomain context.

ipTM is the best of 5 Chai poses before minimization; it is not a score recomputed
after minimization. RMSD is minimized binder C-alpha RMSD to the best screen pose
after aligning the target crop. Contacts: interchain heavy-atom distance <=4 A.
Highlighted residues are actual contacts in the shown minimized frame; carbon
atoms/backbone are gold on target, binder colors retained, N/O colors conventional.
937 trials include repeated evaluations of some sequences; neither trials nor
5 cofold poses are independent biological replicates. Statistical significance
and superior experimental affinity are not claimed for conditioning arms.

RFD4 backbone generation -> target-fixed ProteinMPNN -> Chai (screen and 622-residue
context) -> OpenMM/Amber static MMGBSA with 25 annotated target disulfides. Whole-parent
throughput uses all 11,696 seconds, including startup and workflow idle time:
1,326.3 screens/hour and 288.4 full-context trials/hour. 256 nodes = 102 Nemotron + 154 science.
Full audit/report:../single-allocation-256-20261115/index.html.

Rendering: NGL 2.4.0, software SwiftShader WebGL in temporary Chromium 130.0.6723.31.
NGL and molecular data are bundled for offline use. Open ngl-gallery.html locally;
no server, internet or API token is required. Use its buttons to change style,
choose whole-EGFR/site view, rotate and download high-resolution PNGs.
Transparent PNG backgrounds are the default. Twelve separate alpha PNGs are in
ngl-images/*-transparent.png (3 candidates x 2 styles x 2 views). The PowerPoint
slides also embed alpha PNGs; PDF and preview PNG backgrounds remain white.
NGL: https://nglviewer.org/ngl/ ; MIT license in vendor/NGL-LICENSE.txt.

Reliability context for presenter notes: the original campaign retained 208 failed
rows, 22 noncanonical inputs, 40 thread errors and 6 additional incomplete reviews.
Optional reasoner synthesis exceeded context in all 3 rounds; the three completed
tournament/science/review rounds did not receive a fresh reasoner steering brief.
The slide makes no claim that every mechanism proposed by a hypothesis was tested.
'''
    (out/'methods.txt').write_text(methods)
    manifest={'sources_sha256':data['source_files_sha256'],'builders_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('prepare_conference_science.py'),Path(__file__).with_name('build_ngl_conference.py'),Path(__file__).with_name('render_conference_ngl.py')]},'exports_sha256':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','verification.json']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,Path(out.name)/p.relative_to(out))
    print(json.dumps({'one_slide_variants':2,'output':str(out),'data':data['statistics']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output.resolve())
