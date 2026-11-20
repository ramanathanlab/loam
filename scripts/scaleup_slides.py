"""A shared slide scene rendered to editable PowerPoint, PDF, PNG and HTML.

The same prewrapped text and placement is used in every export. No office
application, browser, cloud upload, or cluster job is required.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import matplotlib
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

W, H = 960, 540
INK, TEAL, GOLD, GRAY, PALE, LINE = '#142f43', '#137f78', '#c78032', '#63758a', '#eef5f5', '#dbe5e9'
FONT_DIR = Path(matplotlib.get_data_path()) / 'fonts/ttf'
FONTS = {'regular': FONT_DIR/'DejaVuSans.ttf', 'bold': FONT_DIR/'DejaVuSans-Bold.ttf', 'mono': FONT_DIR/'DejaVuSansMono.ttf'}


def rgb(color):
    return tuple(int(color[i:i+2],16) for i in (1,3,5))


def font(size, kind='regular', scale=4):
    return ImageFont.truetype(str(FONTS[kind]), round(size*scale))


def wrap(text, width, size, kind='regular'):
    face = font(size, kind)
    fits = lambda s: face.getlength(s)/4 <= width
    result=[]
    for paragraph in str(text).split('\n'):
        words=paragraph.split()
        if not words:
            result.append('');continue
        line=''
        for word in words:
            while not fits(word):
                split=len(word)-1
                while split>1 and not fits(word[:split]):split-=1
                if line:result.append(line);line=''
                result.append(word[:split]);word=word[split:]
            trial=(line+' '+word).strip()
            if line and not fits(trial):result.append(line);line=word
            else:line=trial
        result.append(line)
    return result


class Slide:
    def __init__(self,title,eyebrow,subtitle='',notes='',dark=False):
        self.title,self.notes,self.commands,self.dark=title,notes,[],dark
        self.rect(0,0,W,H,INK if dark else '#ffffff')
        self.text(40,24,880,eyebrow.upper(),10.5,'#9bd6cc' if dark else TEAL,'bold',max_height=18)
        title_size=29
        while len(wrap(title,880,title_size,'bold'))>1 and title_size>22:title_size-=.4
        title_height=self.text(40,47,880,title,title_size,'#ffffff' if dark else INK,'bold',max_height=62)
        if subtitle:self.text(40,max(92,47+title_height+9),880,subtitle,13,'#c4d9e3' if dark else GRAY,max_height=34)

    def rect(self,x,y,w,h,fill,line=None,radius=0):
        self.commands.append({'type':'rect','x':x,'y':y,'w':w,'h':h,'fill':fill,'line':line,'radius':radius})

    def text(self,x,y,w,text,size=16,color=INK,kind='regular',max_height=None):
        lines=wrap(text,w,size,kind)
        while max_height is not None and len(lines)*size*1.23>max_height and size>9:
            size-=.3;lines=wrap(text,w,size,kind)
        height=len(lines)*size*1.23
        if max_height is not None and height>max_height+.1:
            raise ValueError(f'Text cannot fit on {self.title}: {str(text)[:85]} ({height:.1f}>{max_height})')
        self.commands.append({'type':'text','x':x,'y':y,'w':w,'h':height+4,'lines':lines,'size':size,'color':color,'kind':kind,'leading':size*1.23})
        return height

    def image(self,path,x,y,w,h):
        path=Path(path)
        with Image.open(path) as image:iw,ih=image.size
        ratio=min(w/iw,h/ih);ww,hh=iw*ratio,ih*ratio
        self.commands.append({'type':'image','path':str(path),'x':x+(w-ww)/2,'y':y+(h-hh)/2,'w':ww,'h':hh})

    def card(self,x,y,w,value,label,accent=TEAL):
        self.rect(x,y,w,66,PALE,radius=6)
        self.text(x+13,y+7,w-26,str(value),25,accent,'bold',max_height=34)
        self.text(x+13,y+43,w-26,label,10.5,GRAY,max_height=20)

    def table(self,x,y,widths,headers,rows,size=12):
        allrows=[headers]+rows
        yy=y
        for index,row in enumerate(allrows):
            kind='bold' if index==0 else 'regular'
            wrapped=[wrap(cell,width-16,size,kind) for cell,width in zip(row,widths)]
            height=max(len(lines) for lines in wrapped)*size*1.23+17
            self.rect(x,yy,sum(widths),height,PALE if index==0 else '#ffffff',line=LINE)
            xx=x
            for cell,width in zip(row,widths):
                self.text(xx+8,yy+8,width-16,cell,size,INK,kind)
                xx+=width
            yy+=height
        return yy

    def footer(self,page,total):
        self.rect(40,514,880,.7,'#385263' if self.dark else LINE)
        self.text(40,520,815,'Loamlite / Jupiter · Computational results; no measured affinity',8.4,'#c4d9e3' if self.dark else GRAY,max_height=13)
        self.text(886,520,38,f'{page}/{total}',8.4,'#c4d9e3' if self.dark else GRAY,max_height=13)

    def validate(self):
        for c in self.commands:
            if c['type']=='rect' and c['x']==c['y']==0:continue
            if c['x']<-.1 or c['y']<-.1 or c['x']+c['w']>W+.1 or c['y']+c['h']>H+.1:
                raise ValueError(f'Out-of-bounds {c["type"]} on {self.title}: {c}')


def round_slide(r, out):
    label=f'{r["nodes"]} {"node" if r["nodes"]==1 else "nodes"} / round {r["round"]}'
    s=Slide(f'{label}: hypotheses and binders',f'{r["campaign"]} campaign · actual selection snapshot',
            'Tournament rank is advisory. Binder rank is best verified crop ipTM per distinct sequence.',
            notes=r['note']+'\n\nSource: '+r['source_round_record']+'\n'+json.dumps({'top_hypotheses':r['top_hypotheses'],'top_binders':r['top_binders']},indent=2))
    for x,value,caption in [(40,r['verified_crop_files'],'verified crop model files'),(266,f'{r["distinct_sequences"]} / {r["new_sequences"]}','sequences / newly seen'),(492,f'{r["round_wall_seconds"]/60:.1f} min','whole-round time incl. restarts'),(718,f'{r["reviewed_threads"]}/{r["selected_hypotheses"]}','selected threads reviewed')]:
        s.card(x,135,202,value,caption)
    s.text(40,217,400,'Leading tournament selections',15,INK,'bold')
    s.text(474,217,446,'Highest-confidence binders',15,INK,'bold')
    for index,h in enumerate(r['top_hypotheses']):
        y=246+index*55
        tie=f' = {h["same_elo_selected_threads"]}-way tie' if h['same_elo_selected_threads']>1 else ''
        heading=f'#{h["selection_rank"]} {h["hypothesis_id"][:8]} · Elo {h["elo_at_selection"]:.1f}{tie}'
        s.text(40,y,410,heading,11.3,TEAL,'bold',max_height=15)
        s.text(40,y+16,410,h['short_proposition'],11.4,INK,max_height=30)
        s.text(40,y+44,410,'Stored outcome: '+h['stored_outcome']+(' · review overreach flagged' if h['reviewer_overreach_flag'] else ''),9.2,GRAY,max_height=12)
    table_rows=[[c['binder_id'][7:],f'{c["best_iptm"]:.3f}',f'{c["models_with_dual_contacts"]}/{c["verified_models"]}',str(c['cysteines']),str(c['best_model_pairs_below_2A'])] for c in r['top_binders']]
    bottom=s.table(474,244,[140,67,79,48,72],['Binder suffix','ipTM','Both regions','Cys','<2Å pairs'],table_rows,10.6)
    if r['lead_followup']:
        c=r['lead_followup'];pose=c['minimized_pose_geometry']
        suffix=hashlib.sha256(c['binder_sequence'].encode()).hexdigest()[:12]
        message=f'Expanded {suffix} ({c["condition"]}): ipTM {c["intact_iptm_best"]:.3f}; retained contacts; RMSD {pose["target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A"]:.2f} Å; MM/GBSA {c["static_MMGBSA_kcal_mol"]:.1f} kcal/mol.'
    else:
        message='Full IDs, sequences and coordinate sources are in the report. Close-pair counts refer to the highest-ipTM model; a high score alone does not validate a binder.'
    s.text(482,bottom+8,422,message,11.2,GRAY,max_height=32)
    lead=r['top_binders'][0]
    s.text(40,425,880,'Highest-confidence sequence · '+lead['binder_id'],11.5,TEAL,'bold',max_height=18)
    s.text(40,445,880,lead['sequence'],12.3,INK,'mono',max_height=24)
    s.text(40,476,880,r['note'],11,GRAY,max_height=32)
    return s


def make_slides(data,out):
    fig=out/'figures';repo=Path(__file__).resolve().parents[1]
    details=repo/'reports/scale20-20261113'
    slides=[]
    s=Slide('From 1 to 20 nodes','Agentic binder design on Jupiter','A measured scale-up story: throughput, hypothesis tournaments and structural candidates.',dark=True,
            notes='Scope: completed original scientific rounds4–8 and continuation rounds1–3. The independent crop inventory totals2390 files and402 distinct sequences. No experimental binding measurement is available.')
    s.text(40,164,880,'One local Nemotron node + nineteen science nodes',23,'#ffffff','bold',max_height=62)
    for x,value,label in [(40,'3.25×','gross throughput vs five nodes'),(340,'358','new continuation sequences'),(640,'3 rounds','completed at twenty-node scale')]:
        s.rect(x,263,280,102,'#244658',radius=8)
        s.text(x+18,278,244,value,33,'#a2e2d7','bold',max_height=47)
        s.text(x+18,336,244,label,11.4,'#d0e1e9',max_height=27)
    s.text(40,415,880,'Structural confidence, contact geometry and static energy estimates.\nNo measured affinity or biological activity.',17,'#d0e1e9',max_height=65)
    slides.append(s)
    s=Slide('The loop that was scaled','Method','Hypotheses competed using evidence gathered by code-running agents.',notes='Pipeline settings retained: RFD4 medium/front_loaded, fixed-target ProteinMPNN, Chai3 recycles /200 steps /5 predictions. Selected cysteine-free sequences were re-folded against EGFR25–646 before native-disulfide minimization and static MM/GBSA.')
    stages=[('1','Tournament','Generate hypotheses; Swiss pairings update Elo.'),('2','Scientific tools','RFD4 → ProteinMPNN → Chai; controls and independent seeds.'),('3','Evidence review','Check artifacts and ownership; reconcile accepted claims.'),('4','Next round','Synthesize evidence, retain failures and update priorities.')]
    for i,(number,title,body) in enumerate(stages):
        y=151+i*78;s.rect(40,y,880,65,PALE,radius=5);s.text(54,y+10,34,number,24,TEAL,'bold');s.text(106,y+8,230,title,17,INK,'bold');s.text(345,y+11,552,body,15,GRAY,max_height=44)
    s.text(40,477,880,'Target: human EGFR / 3NJP domain-III crop. Twenty-node follow-ups add expanded receptor context.',12,GRAY,max_height=28)
    slides.append(s)
    s=Slide('Measured throughput at 1, 5 and 20 nodes','HPC result','Primary denominator: complete science-allocation time, including idle, repair and audit time.',notes=data['throughput_method']+'\n'+json.dumps(data['phases'],indent=2))
    s.image(fig/'throughput-by-nodes.png',34,139,890,316)
    s.text(40,466,880,'1-node development included debugging. Its scientific-round window achieved 85.6 files/hour. The five-to-twenty-node increase was 3.25×; workloads and inference boundaries changed.',12,GRAY,max_height=36)
    slides.append(s)
    s=Slide('The allocation and comparison boundaries','Resource accounting','Four GH200 GPUs per Jupiter node; external inference was not metered in earlier jobs.',notes='Source: phase allocation records and scaling-audit.json. Twenty-node local inference cost includes both owned inference allocations, model warmup and waiting while science queued. Queue time itself has no allocated GPU-hours. Separate operator pilot and integration preflight are excluded from campaign output totals.')
    rows=[]
    for p in data['phases']:
        rows.append([str(p['nodes']),f'{p["science_nodes"]} / {p["local_inference_nodes"]}',f'{p["allocation_seconds"]/60:.1f}',str(p['verified_crop_files']),str(p['distinct_sequences']),f'{p["science_gpu_hours"]:.2f}',f'{p["local_inference_gpu_hours"]:.2f}' if p['local_inference_gpu_hours'] is not None else 'unmetered'])
    s.table(40,155,[65,155,113,104,117,120,206],['Total nodes','Science / inference','Allocation min','Crop files','Sequences','Science GPU-h','Local inference GPU-h'],rows,12)
    s.text(40,373,880,'Auxiliary configuration: two owned nodes (1 science + 1 shared Nemotron) in a recipe-validation pilot. It verified 20 crop files and 5 expanded-context files; it was not a tournament campaign.',16,INK,max_height=73)
    s.text(40,467,880,'These data show operational scaling. They do not isolate strong scaling at fixed scientific work.',13,TEAL,'bold',max_height=32)
    slides.append(s)
    s=Slide('Which node counts were actually allocated','Owned allocation timeline','Slurm accounting windows; pilots and bootstrap overlaps are not extra scaling benchmarks.',notes='Source: four verified original allocations and five owned continuation allocations. Old bootstrap retries briefly overlapped at10owned nodes. The continuation had2node preflights and a19node accounting gap during inference renewal. Main measured configurations are1,5and20nodes. Slurm epilog can retain nodes after accounting End; only owned jobs are shown, never foreign allocations.')
    s.image(fig/'allocation-timeline.png',35,133,890,342)
    s.text(40,478,880,'Transient 10-node bootstrap overlap and 19-node renewal intervals are shown, without invented throughput measurements.',12,GRAY,max_height=27)
    slides.append(s)
    for r in data['rounds'][:5]:slides.append(round_slide(r,out))
    s=Slide('Changes that enabled the twenty-node continuation','Port and execution','The scientific sampling settings were retained; execution and evidence plumbing changed.',notes='Timing evidence: five-node median generation38.9s, combined sequence/fold81.5s, mean leased-slot occupancy18.32%. Short local structured-tool protocol probe passed13/13 tasks. Those probe rates did not predict long scientific-agent latency.')
    changes=[('Jupiter software and offline execution','Use local software paths and cached scientific assets; compute nodes require no public internet.'),('Bundle work and preserve concurrency','One complete generation/sequence/fold GPU row; zero-GPU metadata tasks; hypothesis fan-out retained.'),('Make evidence recoverable','Owned artifact indexes, immutable tool-version pins and deterministic input manifests reduce repeated work.'),('Stabilize model interaction','Bound model-visible history, sequence dependent side effects and stagger rate-limit retries.')]
    for i,(title,body) in enumerate(changes):
        y=150+i*77;s.text(40,y,880,title,17,TEAL,'bold',max_height=25);s.text(40,y+28,880,body,14,GRAY,max_height=37)
    s.text(40,477,880,'Measured starting split: 1 local Nemotron node / 19 scientific nodes. It was not an optimized final split.',12,GRAY,max_height=28)
    slides.append(s)
    for r in data['rounds'][5:]:slides.append(round_slide(r,out))
    s=Slide('What changed scientifically across iterations','Scientific results','402 distinct sequences across both campaigns; repeated files are not independent binders.',notes='Sources: eight stored round records and the two independent inventories. The lower overall confidence median in the larger campaign includes many ridge controls and changed cohorts; it is not a matched quality comparison.')
    s.image(fig/'confidence-by-iteration.png',35,140,890,310)
    s.text(40,467,880,'Best examples remained high-confidence: original peak ipTM0.848; continuation peak0.860. The larger campaign contained 67/358 sequences reaching best crop ipTM≥0.80.',13,GRAY,max_height=37)
    slides.append(s)
    s=Slide('Leading binder poses from every iteration','Coordinate-backed examples','Receptor gray; binder teal. Each panel shows its round’s highest-ipTM verified crop model.',notes='Target-aligned two-dimensional CA-trace projections, using a shared target-derived camera. Exact paths, checksums, full sequences, contacts and close-pair diagnostics are in data.json and the report. These schematics are not all-atom clash validation.')
    s.image(fig/'leading-binder-poses.png',37,135,888,365)
    slides.append(s)
    s=Slide('Dual-region contacts did not establish a general advantage','Controlled-screen results','One-region controls matched or exceeded dual-region confidence in the larger campaign.',notes='Distinct sequences per arm: dual120, one_region119, ridge119. Median sequence-best ipTM:0.750151,0.767118,0.221671. Only2/33 complete matched screen batches passed the prespecified dispersion gate. Correlated adaptive samples prevent a causal interpretation of aggregate arms.')
    s.image(fig/'condition-comparison.png',35,140,890,293)
    s.rect(40,448,880,54,PALE,radius=5)
    s.text(54,457,852,'Only 2/33 complete matched screen batches passed the prespecified pose-dispersion gate. Contact presence alone was common even in ridge controls.',14,INK,max_height=39)
    slides.append(s)
    s=Slide('A favorable static energy can accompany the wrong pose','Expanded receptor follow-up','74 follow-up rows / 66 distinct binders; 36 minimized rows retained both epitope regions.',notes='Expanded target is canonical EGFR25–646,622residues with25 verified native disulfides. It is not the full membrane receptor. Membrane, intracellular domains, glycans and entropy are omitted. Lowest-energy example binder_a91ef18fe15e:−133.7411kcal/mol, expanded ipTM0.195716, minimized dual contactsfalse, binder displacement59.4416Å.')
    s.image(fig/'energy-versus-confidence.png',35,140,890,283)
    s.text(40,441,880,'Lowest energy: −133.74 kcal/mol, but ipTM0.196, lost epitope contacts and 59.44Å pose displacement. Interpret energy together with confidence and mapped geometry.',15,INK,max_height=62)
    slides.append(s)
    s=Slide('Peak concurrency was high; sustained science use was low','Measured node saturation','Gray intervals are missing observations. GPU busy time is not SM occupancy.',notes='Source: telemetry-summary.json and node-gpu-utilization.png. Mean observed node-minute GPU busy values: science'+str(data['telemetry_summary']['role_observed_node_minutes']['science']['mean_GPU_busy_percent'])+'%; inference'+str(data['telemetry_summary']['role_observed_node_minutes']['inference']['mean_GPU_busy_percent'])+'%. Earlier five-node hardware telemetry is unavailable; its18.32% is a different lease metric.')
    s.image(details/'node-gpu-utilization.png',30,139,640,356)
    t=data['telemetry_summary'];a=t['role_observed_node_minutes']
    for y,value,label in [(160,'74 / 76','peak leased science GPU slots'),(272,f'{a["science"]["mean_GPU_busy_percent"]:.1f}%','mean observed science GPU busy'),(384,f'{a["inference"]["mean_GPU_busy_percent"]:.1f}%','mean observed inference GPU busy')]:
        s.card(687,y,233,value,label)
    slides.append(s)
    s=Slide('What the measured imbalance suggests next','Interpretation and next experiment','The 1-inference / 19-science split demonstrated execution, not an optimal balance.',notes='Five-node mean leased science slots18.32%; twenty-node mean leased slots'+str(100*t['science_slot_leases']['mean_leased_slot_fraction'])+'%. Waiting-queue p95=0, max=2; low waiting depth does not measure running-request decode time. Proposed2inference+18science was not executed. No gain is forecast as a measured result.')
    s.image(details/'aggregate-gpu-activity.png',35,139,890,245)
    s.text(40,402,880,'Next controlled trial: two independent Nemotron replicas + eighteen science nodes, fixed candidate budgets and request routing across replicas. Compare elapsed time, science idle gaps and GPU-hours per reviewed result.',17,INK,max_height=80)
    slides.append(s)
    s=Slide('Limits, corrected claims and final state','Evidence quality','Three completed continuation rounds; 33/36 selected hypothesis-round threads reviewed.',notes='Two original round7 contact claims were retracted after a residue-index mapping error. Three continuation claims were independently flagged: two numerical errors and one mechanistic overreach. Raw reports, reviews, original tournament history and failed batches remain preserved. No foreign jobs were changed; owned jobs were released after audit.')
    points=[('Preserve failures','Two schema failures and one thread without a claim review remain incomplete.'),('Correct the scientific record','Two old contact claims were retracted; three new generated claims carry independent findings.'),('Separate rank from evidence','Tournament Elo and stored supported labels are not independent proof of mechanism.'),('Keep the performance boundary explicit','Report debugging, queueing, warmup, maintenance, repair work and external inference separately.')]
    for i,(title,body) in enumerate(points):
        y=151+i*77;s.text(40,y,880,title,17,TEAL,'bold');s.text(40,y+28,880,body,14,GRAY,max_height=36)
    s.text(40,477,880,'All owned allocations were released. Other campaigns and user-started parent endpoints were preserved.',12,GRAY,max_height=28)
    slides.append(s)
    s=Slide('Reproduce the story from the exported evidence','Downloads and definitions','The accompanying HTML report contains full per-round tables, sequences, sources and checksums.',notes='Source artifacts are local and were read without starting new science jobs or inference. Exports: scaleup.pptx, scaleup.pdf, slides.html, index.html, data.json, iteration-metrics.csv, hypotheses.csv, binder-metrics-by-iteration.csv, leading-binders.fasta, expanded-followups.csv and build-manifest.json.')
    definitions=[('Throughput','Verified crop coordinate files / whole science-allocation hour; round-window rate shown separately.'),('Hypothesis leader','Actual stored tournament selection order, with selection-time Elo and ties.'),('Binder leader','Highest verified crop ipTM per distinct sequence, with contacts, Cys and close-pair diagnostics.'),('Scientific validity','Confidence and geometry are computational readouts. No experimental affinity was measured.')]
    for i,(title,body) in enumerate(definitions):
        y=151+i*77;s.text(40,y,880,title,17,TEAL,'bold');s.text(40,y+28,880,body,14,GRAY,max_height=36)
    slides.append(s)
    return slides


def export(slides,out):
    for name,path in FONTS.items():pdfmetrics.registerFont(TTFont('Story-'+name,str(path)))
    presentation=Presentation();presentation.slide_width=Pt(W);presentation.slide_height=Pt(H)
    presentation.core_properties.title='Loamlite on Jupiter: binder design and GPU saturation'
    presentation.core_properties.subject='Verified throughput, hypothesis tournaments, binders and node utilization'
    presentation.core_properties.author='Loamlite campaign analysis'
    pdf=pdfcanvas.Canvas(str(out/'scaleup.pdf'),pagesize=(W,H));pdf.setTitle(presentation.core_properties.title)
    pngdir=out/'slides';pngdir.mkdir(exist_ok=True)
    transcript=[];pngfiles=[]
    for page,scene in enumerate(slides,1):
        scene.footer(page,len(slides));scene.validate()
        slide=presentation.slides.add_slide(presentation.slide_layouts[6])
        image=Image.new('RGB',(W*2,H*2),'white');draw=ImageDraw.Draw(image)
        words=[]
        for c in scene.commands:
            x,y,w,h=c['x'],c['y'],c['w'],c['h']
            if c['type']=='rect':
                box=(round(x*2),round(y*2),round((x+w)*2),round((y+h)*2))
                if c['radius']:draw.rounded_rectangle(box,round(c['radius']*2),fill=c['fill'],outline=c['line'])
                else:draw.rectangle(box,fill=c['fill'],outline=c['line'])
                pdf.setFillColorRGB(*(v/255 for v in rgb(c['fill'])))
                if c['line']:pdf.setStrokeColorRGB(*(v/255 for v in rgb(c['line'])))
                pdf.roundRect(x,H-y-h,w,h,c['radius'],fill=1,stroke=int(bool(c['line'])))
                shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if c['radius'] else MSO_SHAPE.RECTANGLE,Pt(x),Pt(y),Pt(w),Pt(h))
                shape.fill.solid();shape.fill.fore_color.rgb=RGBColor(*rgb(c['fill']))
                if c['line']:shape.line.color.rgb=RGBColor(*rgb(c['line']))
                else:shape.line.fill.background()
            elif c['type']=='image':
                with Image.open(c['path']) as picture:
                    picture=picture.convert('RGB').resize((max(1,round(w*2)),max(1,round(h*2))),Image.Resampling.LANCZOS)
                    image.paste(picture,(round(x*2),round(y*2)))
                pdf.drawImage(c['path'],x,H-y-h,width=w,height=h,mask='auto')
                slide.shapes.add_picture(c['path'],Pt(x),Pt(y),width=Pt(w),height=Pt(h))
            else:
                words.extend(c['lines'])
                face=font(c['size'],c['kind'],2)
                pdf.setFillColorRGB(*(v/255 for v in rgb(c['color'])));pdf.setFont('Story-'+c['kind'],c['size'])
                ascent=pdfmetrics.getAscent('Story-'+c['kind'])*c['size']/1000
                for i,line in enumerate(c['lines']):
                    yy=y+i*c['leading'];draw.text((round(x*2),round(yy*2)),line,font=face,fill=c['color'],anchor='lt')
                    pdf.drawString(x,H-yy-ascent,line)
                textbox=slide.shapes.add_textbox(Pt(x),Pt(y),Pt(w+2),Pt(h+2));tf=textbox.text_frame;tf.clear()
                tf.word_wrap=False;tf.vertical_anchor=MSO_ANCHOR.TOP
                tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                for i,line in enumerate(c['lines']):
                    paragraph=tf.paragraphs[0] if i==0 else tf.add_paragraph();paragraph.text=line
                    paragraph.space_before=paragraph.space_after=Pt(0);paragraph.line_spacing=Pt(c['leading'])
                    paragraph.font.name='DejaVu Sans Mono' if c['kind']=='mono' else 'DejaVu Sans'
                    paragraph.font.size=Pt(c['size']);paragraph.font.bold=c['kind']=='bold';paragraph.font.color.rgb=RGBColor(*rgb(c['color']))
        slide.notes_slide.notes_text_frame.text=scene.notes
        path=pngdir/f'slide-{page:02}.png';image.save(path,optimize=True);pngfiles.append(path)
        transcript.append(f'SLIDE {page}: {scene.title}\n'+'\n'.join(words)+'\n\nPRESENTER NOTES / SOURCES\n'+scene.notes+'\n')
        pdf.showPage()
    pdf.save();presentation.save(out/'scaleup.pptx')
    (out/'slide-transcript.txt').write_text('\n\n'.join(transcript))
    # Contact sheet is generated from the very same scenes for visual review.
    thumb_w,thumb_h=384,216;cols=3;rows=math.ceil(len(slides)/cols)
    sheet=Image.new('RGB',(cols*thumb_w,rows*(thumb_h+30)), '#e8eef2');d=ImageDraw.Draw(sheet)
    for i,path in enumerate(pngfiles):
        x=(i%cols)*thumb_w;y=(i//cols)*(thumb_h+30)
        with Image.open(path) as image:sheet.paste(image.resize((thumb_w,thumb_h),Image.Resampling.LANCZOS),(x,y))
        d.text((x+8,y+thumb_h+6),f'{i+1:02}  {slides[i].title[:51]}',fill=INK,font=font(10,'regular',1))
    sheet.save(out/'slide-contact-sheet.png')
    figures=[]
    for i,(scene,path) in enumerate(zip(slides,pngfiles)):
        encoded=base64.b64encode(path.read_bytes()).decode()
        figures.append(f'<figure class="slide{" active" if i==0 else ""}" aria-label="Slide {i+1}: {html.escape(scene.title)}"><img src="data:image/png;base64,{encoded}" alt="{html.escape(scene.title)}"><figcaption>{html.escape(scene.title)}</figcaption></figure>')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Loamlite / Jupiter scale-up slides</title><style>*{box-sizing:border-box}body{margin:0;background:#0c1f2e;color:white;font:14px system-ui}main{height:calc(100vh - 66px);display:grid;place-items:center}.slide{display:none;margin:0;width:100%;height:100%;align-items:center;justify-content:center}.slide.active{display:flex}.slide img{max-width:100%;max-height:100%;object-fit:contain}figcaption{position:absolute;width:1px;height:1px;overflow:hidden}nav{height:66px;display:flex;align-items:center;justify-content:center;gap:18px}button,a{padding:8px 13px;border-radius:5px;border:1px solid #6a8a9b;background:#1b3d50;color:white;text-decoration:none;font:inherit}button{cursor:pointer}@media print{@page{size:landscape;margin:0}body{background:white}main{display:block;height:auto}.slide,.slide.active{display:flex;height:100vh;break-after:page}nav{display:none}}</style><main>'''+''.join(figures)+'''</main><nav><button id="prev" aria-label="Previous slide">← Previous</button><span id="counter"></span><button id="next" aria-label="Next slide">Next →</button><a href="index.html">Report</a><a href="scaleup.pptx">PPTX</a><a href="scaleup.pdf">PDF</a><button id="full">Fullscreen</button></nav><script>const slides=[...document.querySelectorAll('.slide')];let current=0;function show(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===current));document.getElementById('counter').textContent=`${current+1} / ${slides.length}`;location.hash=String(current+1)}document.getElementById('prev').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);document.getElementById('full').onclick=()=>document.documentElement.requestFullscreen?.();document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(current+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(current-1)}if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1)});show((parseInt(location.hash.slice(1))||1)-1);</script></html>'''
    (out/'slides.html').write_text(doc)
    (out/'slide-scenes.json').write_text(json.dumps([{'title':s.title,'notes':s.notes,'commands':s.commands} for s in slides],indent=2)+'\n')
    return len(slides)


def build_deck(data,out):
    return export(make_slides(data,out),out)
