#!/usr/bin/env python3
"""Record explicit, hashed resolutions for observed citation formatting errors.

Never change the claim or its original source. Only existing files inside a
batch/run recorded to that hypothesis can resolve a reference. Ambiguous or
unrecognized citations stay unresolved for independent review.
"""
import argparse, hashlib, json, re, time
from pathlib import Path
from audit_scale20_campaign import claim_key


def read(p):return json.loads(p.read_text())
def main(root,audit_file):
    audit=read(audit_file);destination=root/'claim-source-resolutions.json'
    resolutions=read(destination) if destination.exists() else {};remaining=[];added=[]
    rounds={int(p.stem.split('-')[1]):read(p) for p in (root/'state/rounds').glob('round-*.json')}
    for finding in audit['unowned_or_missing_claim_sources']:
        n=finding['round'];hyp=finding['hypothesis'];source=finding['source'];thread=next(t for t in rounds[n]['threads'] if t['hypothesis']['id']==hyp)
        batches=set();allowed=[]
        for p in (root/'state/hypotheses'/hyp/'runs').glob('r*.json'):
            run=read(p);batches.update(run.get('batches',[]));allowed.append(root/'work'/hyp/'outputs'/run['run_id'])
        allowed.extend(root/'batches'/batch for batch in batches)
        owned=lambda p:p.is_file() and any(p.resolve().is_relative_to(folder.resolve()) for folder in allowed)
        files=[];methods=[]
        if source.startswith('outputs/'):
            p=root/'work'/hyp/source
            if owned(p):files.append(p);methods.append('Hypothesis-relative output resolved inside its recorded owning run')
        for relative in re.finditer(r'\b(outputs/r[0-9]+-[0-9]+/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:jsonl|json|txt|dat|py))(?=$|[;,\s])',source):
            p=root/'work'/hyp/relative[1]
            if owned(p):files.append(p);methods.append('Exact hypothesis-relative filename before citation prose resolved inside its recorded owning run; no fuzzy matching')
        if source.startswith(str(root)+'/outputs/'):
            p=root/'work'/hyp/source[len(str(root))+1:]
            if owned(p):files.append(p);methods.append('Campaign-root output shorthand resolved inside this exact hypothesis recorded owning run')
        work=re.search(r'work/'+re.escape(hyp)+r'/(outputs/r\d+-\d+/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:json|txt|dat))',source)
        if work:
            p=root/'work'/hyp/work[1]
            if owned(p):files.append(p);methods.append('Incorrect campaign-root spelling resolved through exact hypothesis/run/file identity')
        directory_ref=re.fullmatch(r'.*?batches/(b-[0-9a-f]+)',source)
        if directory_ref and directory_ref[1] in batches:
            directory=root/'batches'/directory_ref[1]
            if owned(directory/'batch.json'):
                files.append(directory/'batch.json')
                for pattern in ('run_*/summary.json','run_*/followup-summary.json','run_*/intact-geometry.json','run_*/analysis.json'):
                    files.extend(p for p in directory.glob(pattern) if owned(p))
                methods.append('Exact owning batch-directory reference resolved to its immutable receipt and actual row reports')
        for match in re.finditer(r'\bbatches/(b-[0-9a-f]+)/?(?=$|[\s,;()])',source):
            batch=match[1];directory=root/'batches'/batch
            if batch not in batches or not owned(directory/'batch.json'):continue
            files.append(directory/'batch.json')
            for pattern in ('run_*/summary.json','run_*/followup-summary.json','run_*/intact-geometry.json','run_*/analysis.json'):
                files.extend(p for p in directory.glob(pattern) if owned(p))
            methods.append('Explicit owned batch directory among multiple citations/prose resolved to its receipt and actual row reports; no missing filename substituted')
        for match in re.finditer(r'\bbatches/(b-[0-9a-f]+)/(run_[A-Za-z0-9_.-]+)(?=$|[\s,;])',source):
            batch,row=match.groups();directory=root/'batches'/batch/row
            if batch not in batches or not owned(directory/'status.json') or not owned(directory/'input.json'):continue
            files.extend(directory/name for name in ('status.json','input.json','summary.json','followup-summary.json','intact-geometry.json') if owned(directory/name))
            methods.append('Explicit existing row-directory citation resolved inside the exact recorded owned batch, including input/status and actual reports')
        brace=re.search(r'\{([0-9]+(?:,[0-9]+){1,15})\}',source)
        if brace and source.count('{')==source.count('}')==1:
            expanded=[Path(source[:brace.start()]+item+source[brace.end():]) for item in brace[1].split(',')]
            expanded=[p if p.is_absolute() else root/p for p in expanded]
            if all(owned(p) for p in expanded):
                files.extend(expanded);methods.append('Explicit numeric brace citation expanded without a shell; every named file exists and belongs to this hypothesis')
        pattern=r'batches/(b-[0-9a-f]+)/((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:json|txt|dat))'
        for match in re.finditer(pattern,source):
            batch,suffix=match.groups();p=root/'batches'/batch/suffix
            if batch in batches and owned(p):files.append(p);methods.append('Exact named path resolved against this campaign root')
        for match in re.finditer(r'batches/(b-[0-9a-f]+)/run_\*/([A-Za-z0-9_.-]+\.(?:json|dat|txt))',source):
            batch,filename=match.groups();directory=root/'batches'/batch
            if batch not in batches:continue
            expanded=sorted(directory.glob('run_*/'+filename))
            if len(expanded)==read(directory/'batch.json')['total'] and all(owned(p) for p in expanded):
                files.extend(expanded);methods.append('Explicit run_* citation expanded to every actual recorded row report; no shell or cross-batch search')
        for match in re.finditer(r'batches/(b-[0-9a-f]+)/(run_[A-Za-z0-9_.-]*\*[A-Za-z0-9_.-]*)/([A-Za-z0-9_.-]+\.(?:json|dat|txt))',source):
            batch,row_pattern,filename=match.groups();directory=root/'batches'/batch
            if batch not in batches:continue
            rows=sorted(p for p in directory.glob(row_pattern) if p.is_dir());expanded=[p/filename for p in rows]
            if expanded and all(owned(p) for p in expanded):
                files.extend(expanded);methods.append('Explicit single-wildcard row pattern expanded within the exact owning batch; every matching row has the named file, with no shell or cross-batch search')
            elif rows:
                reports=[p for p in expanded if owned(p)];missing=[p for p in rows if not owned(p/filename)]
                failed_receipts=[]
                for row in missing:
                    status=row/'status.json';inputs=row/'input.json'
                    if not owned(status) or not owned(inputs) or not (row/'done').is_file():break
                    recorded=read(status)
                    if recorded.get('returncode') in (None,0) and not recorded.get('missing_outputs'):break
                    failed_receipts.extend([status,inputs])
                else:
                    if reports and missing:
                        files.extend([directory/'batch.json',*reports,*failed_receipts]);methods.append('Explicit owned wildcard resolves all existing named reports plus terminal failed-row input/status receipts for every missing report. Failed rows remain failed and have no fabricated summary; no successful row may lack the named report')
        batch_ids={match[1] for p in files if (match:=re.search(r'batches/(b-[0-9a-f]+)/',str(p)))}
        if len(batch_ids)==1:
            directory=root/'batches'/next(iter(batch_ids))
            for match in re.finditer(r'\b(run_[A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+\.(?:json|dat|txt))',source):
                p=directory/match[1]/match[2]
                if owned(p):files.append(p);methods.append('Explicit additional row/file resolved inside the unique cited owned batch')
            filenames={p.name for p in files}
            for match in re.finditer(r'\b(run_[A-Za-z0-9_.-]+)(?=$|[;,\s])',source):
                for filename in filenames:
                    p=directory/match[1]/filename
                    if owned(p):files.append(p);methods.append('Explicit bare row directory uses the same named report as the primary cited row')
        if 'energy_report/report.json' in source:
            for p in list(files):
                if p.name=='followup-summary.json' and (value:=read(p).get('energy_report')):
                    energy=Path(value)
                    if owned(energy):files.append(energy);methods.append('Energy-report shorthand resolved through the actual followup summary energy_report field')
        # A named sibling is allowed only alongside a verified exact file.
        for name in ('intact-geometry.json','full_analysis.json'):
            if 'and '+name in source:
                candidates=[p.parent/name for p in files if owned(p.parent/name)]
                if candidates:files.extend(candidates);methods.append('Explicit sibling filename resolved in each cited owned row directory')
        if 'sibling summary.json files' in source:
            batch_ids={p.parent.parent.name for p in files if p.name=='summary.json'}
            if len(batch_ids)==1:
                siblings=[p for p in (root/'batches'/next(iter(batch_ids))).glob('run_*/summary.json') if read(p).get('condition')=='dual' and owned(p)]
                if len(siblings)==4:files.extend(siblings);methods.append('Four actual dual candidate summaries resolve explicit sibling-summary citation')
        if '.../' in source and 'followup-summary.json' in source:
            batch_ids={p.parent.parent.name for p in files if p.name=='followup-summary.json'}
            if len(batch_ids)==1:
                siblings=[p for p in (root/'batches'/next(iter(batch_ids))).glob('run_*/followup-summary.json') if owned(p)]
                if len(siblings)==3 and {read(p)['condition'] for p in siblings}=={'dual','one_region','ridge'}:
                    files.extend(siblings);methods.append('Abbreviated same-batch references resolve to exactly one completed report per named conditioning arm')
        files=sorted(set(files))
        if not files:
            remaining.append(finding);continue
        for claim in thread['accepted']:
            if claim['source']!=source:continue
            key=claim_key(n,hyp,claim)
            if key in resolutions:
                previous=resolutions[key]
                for entry in previous['files']:assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],'Previously resolved evidence changed'
                files=sorted(set(files)|{Path(entry['path']) for entry in previous['files']})
                if {str(p) for p in files}=={entry['path'] for entry in previous['files']}:continue
            resolutions[key]={'resolved_at':resolutions.get(key,{}).get('resolved_at',time.time()),'updated_at':time.time(),'method':'; '.join(dict.fromkeys(methods)),'ownership':'Every file exists inside a batch/run recorded to this hypothesis; original citation and claim unchanged. This resolves file identity only, not numerical or mechanistic correctness.','files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
            added.append({'round':n,'hypothesis':hyp,'files':len(files),'claim_key':key})
    destination.write_text(json.dumps(resolutions,indent=2)+'\n')
    print(json.dumps({'added':added,'unresolved':remaining},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('audit',type=Path);a=p.parse_args();main(a.campaign.resolve(),a.audit.resolve())
