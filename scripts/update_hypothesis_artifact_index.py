#!/usr/bin/env python3
"""Navigation metadata for hypothesis-owned batches; never scientific evidence."""
import argparse
import hashlib
import json
import fcntl
import tempfile
import time
from collections import defaultdict
from pathlib import Path


def update(campaign):
    receipts=json.loads((campaign/'campaign-tool-registrations.json').read_text())
    known={entry['sha256']:entry['tool'] for entry in receipts}
    by_hypothesis=defaultdict(list)
    for metadata in sorted(campaign.glob('batches/*/batch.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        batch=json.loads(metadata.read_text());labels=batch.get('labels',{});hyp=labels.get('hypothesis_id');run=labels.get('run_id')
        record=campaign/'state/hypotheses'/str(hyp)/'runs'/f'{run}.json'
        if not record.exists() or batch['batch_id'] not in json.loads(record.read_text()).get('batches',[]):continue
        root=metadata.parent;tool=known.get(batch.get('script_sha256'));entry={'batch_id':batch['batch_id'],'owning_run':run,'directory':str(root),'reviewed_source_tool':tool,'recorded_tool':batch.get('tool'),'tool_version':batch.get('tool_version'),'total':batch['total'],'terminal':0,'successful':0,'reports':[]}
        for folder in sorted(root.glob('run_*')):
            if not(folder/'done').exists() or not(folder/'status.json').exists():continue
            status=json.loads((folder/'status.json').read_text());entry['terminal']+=1
            if status.get('returncode')!=0 or status.get('missing_outputs'):continue
            entry['successful']+=1
            if tool:
                for name in ('analysis.json','full_analysis.json','summary.json','followup-summary.json','intact-geometry.json'):
                    path=folder/name
                    if path.is_file():
                        report={'file':name,'path':str(path)}
                        if name=='summary.json' and tool=='scale20_complete_binder_pipeline':
                            summary=json.loads(path.read_text())
                            report.update(candidate_id=summary['candidate_id'],condition=summary['condition'],best_iptm=summary['best_iptm'],cysteine_free='C' not in summary['binder_sequence'])
                        entry['reports'].append(report)
        entry['unfinished']=entry['total']-entry['terminal'];by_hypothesis[hyp].append(entry)
    for workspace in (campaign/'work').iterdir():
        if not(workspace/'data').is_dir():continue
        data={'metadata_only':True,'scientific_evidence':False,'hypothesis_id':workspace.name,'updated_at':time.time(),'batches':by_hypothesis[workspace.name],
              'instructions':'Read the exact report paths, then cite their owning scientific artifacts. Do not append outputs/work or guess a row directory. If no completed screen is listed, generate this hypothesis own screen before intact follow-up. Reuse completed stages. write_files/run_code preparation must finish before a dependent run_batch call in a later model turn.'}
        note=campaign/'navigation-note.txt'
        if note.exists():data['operator_execution_note']=note.read_text().strip()
        destination=workspace/'data/current-artifacts.json'
        with tempfile.NamedTemporaryFile(mode='w',dir=destination.parent,prefix='.artifacts-',delete=False) as stream:
            stream.write(json.dumps(data,separators=(',',':'))+'\n');temporary=Path(stream.name)
        try:temporary.replace(destination)
        finally:temporary.unlink(missing_ok=True)
    return {'hypotheses':len(by_hypothesis),'recorded_batches':sum(map(len,by_hypothesis.values()))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--watch',action='store_true');a=p.parse_args();r=a.campaign.resolve()
    if a.watch:
        lock=(r/'.artifact-index-watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        print(json.dumps({'timestamp':time.time(),**update(r)}),flush=True)
        if not a.watch:break
        watch=json.loads((r/'controller-watch.json').read_text());pid=watch['controller_pid']
        try:alive=Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
        except FileNotFoundError:alive=False
        if not alive:break
        time.sleep(30)
