#!/usr/bin/env python3
"""Witness narrow AST equivalence to registered, frozen scientific recipes.

No code is executed. Other source changes stay excluded, with their attempts
and costs retained. The auditor independently rechecks every witness and model.
"""
import argparse,json,time
from pathlib import Path
from audit_scale20_campaign import sha,reviewed_source_ast


def main(root):
    definitions={'scale20_complete_binder_pipeline':'run_binder_pipeline.py','scale20_summarize_binder_pipeline':'summarize_binder_pipeline.py','scale20_intact_egfr_followup':'run_intact_egfr_followup.py'}
    registered={e['tool']:e['sha256'] for e in json.loads((root/'campaign-tool-registrations.json').read_text())}
    templates={}
    for name,filename in definitions.items():
        path=root/'inputs'/filename;assert sha(path)==registered[name]
        templates[name]=(path,reviewed_source_ast(path))
    destination=root/'operator-reviewed-source-aliases.json';entries=json.loads(destination.read_text()) if destination.exists() else [];known=set(registered.values())|{e['sha256'] for e in entries};added=[]
    for metadata in sorted(root.glob('batches/*/batch.json')):
        batch=json.loads(metadata.read_text());source=metadata.parent/'script.py';digest=sha(source)
        if digest in known or digest!=batch.get('script_sha256'):continue
        try:signature=reviewed_source_ast(source)
        except SyntaxError:continue
        for name,(canonical,expected) in templates.items():
            if signature!=expected:continue
            entry={'source':str(source),'canonical_source':str(canonical),'sha256':digest,'canonical_sha256':sha(canonical),'tool':name,'reviewed_at':time.time(),'review':'Whole AST equals the registered frozen recipe under the auditor narrow normalization: module docstring and a repeated models:models dictionary entry separated only by literal/name fields. No scientific statement or parameter ignored. Original direct receipt and missing output contract retained; independent coordinate/score checks remain required.'}
            entries.append(entry);known.add(digest);added.append({'batch':batch['batch_id'],'tool':name,'sha256':digest});break
    destination.write_text(json.dumps(entries,indent=2)+'\n');print(json.dumps({'added':added,'total_source_witnesses':len(entries)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args();main(a.campaign.resolve())
