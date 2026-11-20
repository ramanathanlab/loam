#!/usr/bin/env python3
"""Inventory actual owned scientific row deadlines without changing evidence."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time


def collect(experiment,output):
    experiment=Path(experiment).resolve(strict=True);output=Path(output).resolve()
    if output.exists():raise RuntimeError('Refuse to overwrite an existing budget inventory')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    read=lambda p:json.loads(p.read_text())
    ledger=experiment/'experiment.json';owned=set(read(ledger)['owned_job_ids'])
    sources={str(ledger):sha(ledger),str(Path(__file__).resolve()):sha(Path(__file__))}
    rows=[];missing=[];seen=set()
    for path in sorted(experiment.glob('*/submission.json')):
        receipt=read(path);jid=receipt['job_id']
        if jid not in owned:continue
        assert jid not in seen;seen.add(jid)
        job=path.parent.resolve();planpath=job/'plan.json';plan=read(planpath)
        assert plan['job_name']==receipt['job_name'] and Path(plan['job_root']).resolve()==job
        assert (job.parent/'experiment.json').resolve()==ledger.resolve()
        campaign=Path(plan['campaign']).resolve(strict=True);assert campaign==job/'campaign'
        sources[str(path)]=sha(path);sources[str(planpath)]=sha(planpath)
        groups=collections.Counter()
        for batch in sorted((campaign/'batches').glob('*/batch.json')):
            metadata=read(batch);sources[str(batch)]=sha(batch)
            timeout=metadata.get('timeout_each');tool=metadata.get('tool','unnamed')
            if timeout is None:missing.append({'job_id':jid,'path':str(batch),'tool':tool})
            groups[(tool,timeout,metadata.get('gpus'))]+=1
        for (tool,timeout,gpus),count in groups.items():
            rows.append({'trial':job.name,'job_id':jid,'tool':tool,'timeout_each_seconds':timeout,
                         'gpus_per_row':gpus,'submitted_batches':count})
    assert seen==owned,'Missing an owned submission receipt'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'generated_at':time.time(),'rows':rows,
        'missing_timeout_metadata':missing,'source_sha256':sources,
        'scope':'Observed batch metadata, including failed/repeated submissions. Batch counts are not scientific outputs. Row deadlines differ from model, agent, MCP and Slurm timers.'},indent=2)+'\n')
    return len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args();print(collect(a.experiment,a.output or a.experiment/'diagnostics/report-batch-budgets.json'))
