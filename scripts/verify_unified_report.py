#!/usr/bin/env python3
"""Verify local report exports without inference, science calculations or jobs."""
import argparse, csv, hashlib, json, re, time, zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Document(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=[];self.scripts=[];self.in_script=False;self.figures=0;self.options=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for k in ['src','href']:
            if k in a:self.links.append(a[k])
        if 'id' in a:self.ids.append(a['id'])
        if tag=='figure':self.figures+=1
        if tag=='option':self.options+=1
        if tag=='script':self.in_script=True;self.scripts.append('')
    def handle_endtag(self,tag):
        if tag=='script':self.in_script=False
    def handle_data(self,data):
        if self.in_script:self.scripts[-1]+=data


def main(out):
    out=out.resolve();m=json.loads((out/'build-manifest.json').read_text());data=json.loads((out/'data.json').read_text())
    for rel,digest in m['exports'].items():
        p=(out/rel).resolve();assert p.is_relative_to(out) and sha(p)==digest,rel
    for name,digest in m['sources'].items():assert sha(Path(name))==digest,name
    assert sha(Path(__file__).with_name('build_unified_report.py'))==m['builder_sha256']
    documents={}
    for name in ['index.html','slides.html']:
        d=Document();d.feed((out/name).read_text());assert len(d.ids)==len(set(d.ids)),name
        for link in d.links:
            u=urlsplit(link)
            if u.scheme=='data' or not u.path:continue
            assert not u.scheme and not u.netloc,'Unexpected external resource'
            p=(out/unquote(u.path)).resolve();assert p.is_relative_to(out) and p.is_file(),link
        documents[name]=d
    nodes=sum(r['nodes'] for r in data['runs']);assert documents['index.html'].options==nodes
    assert documents['index.html'].figures==m['figure_count']
    with zipfile.ZipFile(out/'scaleup.pptx') as z:
        assert z.testzip() is None
        pptx_count=sum(bool(re.fullmatch(r'ppt/slides/slide\d+\.xml',n)) for n in z.namelist())
        assert pptx_count==m['slide_count']
    pdf_count=len(re.findall(rb'/Type\s*/Page\b',(out/'scaleup.pdf').read_bytes()));assert pdf_count==pptx_count
    for filename,key in [('screen-trials.csv','screen_trials'),('expanded-trials.csv','expanded_trials'),('hypotheses.csv','hypotheses'),('node-utilization.csv','nodes')]:
        with (out/filename).open() as f:assert sum(1 for _ in csv.DictReader(f))==len(data[key]),filename
    assert len(data['screen_trials'])==sum(r['screen_trials'] for r in data['runs'])
    assert len(data['expanded_trials'])==sum(r['expanded_trials'] for r in data['runs'])
    assert all(r['completion_checks_passed'] and r['parent_state']=='COMPLETED' for r in data['runs'] if r['mode']=='campaign')
    # Only compare against this experiment's own secrets. Never print values.
    secrets=[];experiment=Path(data['experiment'])
    for planfile in experiment.glob('*/plan.json'):
        infra=Path(json.loads(planfile.read_text())['infra'])
        paths=list(infra.glob('replicas/*/state/nemotron/*/token'))+[infra/'redis/password']
        service=infra/'pool-service.json'
        if service.exists():
            token=Path(json.loads(service.read_text())['token_file']);assert token.resolve().is_relative_to(infra.resolve());paths.append(token)
        for p in paths:
            if p.exists():
                v=p.read_text().strip()
                if len(v)>=16:secrets.append(v.encode())
    textfiles=[p for p in out.rglob('*') if p.is_file() and p.suffix in {'.html','.json','.csv','.svg','.txt','.fasta'}]
    for p in textfiles:
        content=p.read_bytes();assert b'-----BEGIN PRIVATE KEY-----' not in content,p
        assert not any(v in content for v in secrets),'Credential found in '+str(p)
    record={'checked_at':time.time(),'passed':True,'slides':pptx_count,'figures':m['figure_count'],'node_plots':nodes,'export_hashes_checked':len(m['exports']),'source_hashes_checked':len(m['sources']),'owned_secret_values_checked':len(secrets),'HTML_relative_links_present':True,'CSV_counts_match':True,'PDF_PPTX_page_counts_match':True,'verifier_sha256':sha(Path(__file__))}
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n');m['exports']['verification.json']=sha(out/'verification.json');(out/'build-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,Path(out.name)/p.relative_to(out))
    with zipfile.ZipFile(out.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(record,indent=2))
    (out.parent/(out.name+'-offline-scripts.json')).write_text(json.dumps({k:v.scripts for k,v in documents.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
