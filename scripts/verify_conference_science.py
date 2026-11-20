#!/usr/bin/env python3
"""Verify scientific summaries, rigid structure exports and one-slide files."""
import argparse,collections,csv,hashlib,io,json,re,time,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import numpy as np
import gemmi
from scipy.spatial.distance import pdist,cdist
from PIL import Image,ImageDraw
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def residues(s):return [r for c in s[0] for r in c if gemmi.find_tabulated_residue(r.name).is_amino_acid()]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src'):self.links.append(v)

def main(out):
    d=json.loads((out/'science-data.json').read_text());stats=d['statistics'];repo=Path(__file__).resolve().parents[1];source=json.loads((repo/'reports/single-allocation-256-20261115/data.json').read_text());run=next(r for r in source['runs'] if r['nodes']==256);full=[r for r in source['expanded_trials'] if r['run']==run['label']];screens=[r for r in source['screen_trials'] if r['run']==run['label']]
    assert len(full)==stats['followup_trials']==937 and len(screens)==stats['screen_trials']==4309
    assert len({r['binder_sequence'] for r in screens})==4309
    assert len({r['binder_sequence'] for r in full})==767
    assert sum(r['intact_models_with_dual_region_contact']==5 for r in full)==stats['both_region_contacts_in_all_five_poses']==390
    assert abs(stats['screen_trials_per_hour']-4309*3600/11696)<1e-9 and abs(stats['followup_trials_per_hour']-937*3600/11696)<1e-9
    structures=[]
    for c in d['top_binders']:
        original=Path(c['source_coordinate']);assert sha(original)==c['source_coordinate_sha256'];pdb=out/c['pdb_file'];assert sha(pdb)==c['pdb_sha256']
        a=residues(gemmi.read_structure(str(original)));b=gemmi.read_structure(str(pdb));bb=residues(b);assert len(a)==len(bb)==682
        assert len(b[0]['A'])==622 and len(b[0]['B'])==60
        assert ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in b[0]['B'])==c['binder_sequence']
        before=np.array([r['CA'][0].pos.tolist() for r in a]);after=np.array([r['CA'][0].pos.tolist() for r in bb]);error=float(np.max(np.abs(pdist(before)-pdist(after))));assert error<.003
        xyz=[];ids=[]
        for name in ('A','B'):
            values=[];numbers=[]
            for r in b[0][name]:
                for atom in r:
                    if atom.element.name not in ('H','D'):values.append(atom.pos.tolist());numbers.append(r.seqid.num)
            xyz.append(values);ids.append(numbers)
        distance=cdist(*xyz);target=sorted({n for n,v in zip(ids[0],distance.min(1)) if v<=4});binder=sorted({n for n,v in zip(ids[1],distance.min(0)) if v<=4});assert target==c['interface_target_residues'] and binder==c['interface_binder_residues'];assert int((distance<2).sum())==0
        structures.append({'id':c['id'],'max_CA_distance_change_A':error,'target_interface_residues':len(target),'binder_interface_residues':len(binder),'correct_sequence_partition':True})
    alpha=[]
    for p in sorted((out/'ngl-images').glob('*-transparent.png')):
        with Image.open(p) as im:
            assert im.mode=='RGBA';a=im.getchannel('A');assert a.getextrema()==(0,255);fraction=a.histogram()[0]/(im.width*im.height);assert fraction>.02;alpha.append({'file':str(p.relative_to(out)),'size':im.size,'fully_transparent_fraction':fraction})
    assert len(alpha)==12
    for style in ('clean','interface'):
        prs=Presentation(out/f'loam-science-{style}.pptx');assert len(prs.slides)==1
        pictures=[s for s in prs.slides[0].shapes if s.shape_type==MSO_SHAPE_TYPE.PICTURE];assert len(pictures)==4
        n=0
        for s in pictures:
            with Image.open(io.BytesIO(s.image.blob)) as im:
                if 'A' in im.getbands() and im.getchannel('A').getextrema()[0]==0:n+=1
        assert n==3,'PowerPoint must embed3actual alpha-channel molecular images'
        assert len(re.findall(rb'/Type\s*/Page\b',(out/f'loam-science-{style}.pdf').read_bytes()))==1
        with zipfile.ZipFile(out/f'loam-science-{style}.pptx') as z:assert z.testzip() is None
    for p in out.rglob('*.html'):
        parser=Links();parser.feed(p.read_text())
        for url in parser.links:
            u=urlsplit(url)
            if not u.path or u.scheme=='data':continue
            assert not u.scheme and not u.netloc,url
            dest=(p.parent/unquote(u.path)).resolve();assert dest.is_relative_to(out) and dest.is_file(),str(p)+' -> '+url
    # A checkerboard preview demonstrates real alpha without baking it into PNGs.
    p=out/'ngl-images/binder-1-interface-site-transparent.png'
    with Image.open(p) as im:
        im=im.copy();im.thumbnail((800,800));bg=Image.new('RGBA',im.size);draw=ImageDraw.Draw(bg)
        for y in range(0,im.height,28):
            for x in range(0,im.width,28):draw.rectangle((x,y,x+27,y+27),fill='#e3e7ea' if (x//28+y//28)%2 else '#ffffff')
        bg.alpha_composite(im);bg.convert('RGB').save(out/'transparent-background-preview.png')
    manifest=json.loads((out/'manifest.json').read_text())
    for path,digest in manifest['sources_sha256'].items():assert sha(Path(path))==digest
    for path,digest in manifest['builders_sha256'].items():assert sha(Path(path))==digest
    for rel,digest in manifest['exports_sha256'].items():assert sha(out/rel)==digest,rel
    browser=json.loads((out/'ngl-transparent-render-verification.json').read_text());assert browser['images']==12 and not browser['page_errors'] and browser['transparent_background']
    result={'checked_at':time.time(),'passed':True,'one_slide_PowerPoint_variants':2,'one_page_PDF_variants':2,'transparent_molecular_PNGs':12,'transparent_molecular_images_embedded_per_PowerPoint':3,'structures_verified':structures,'alpha_checks':alpha,'statistics_match_audited_campaign':True,'offline_HTML_links_present':True,'browser_render':browser,'scope':'Cohort arithmetic, coordinate source SHA, full CA-distance invariance, exact contact-residue selections, real alpha channels, single-page/slide exports, and browser-based NGL execution.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest['exports_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'};manifest['verifier_sha256']=sha(Path(__file__));(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,Path(out.name)/p.relative_to(out))
    print(json.dumps({k:v for k,v in result.items() if k not in ('alpha_checks','browser_render')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output.resolve())
