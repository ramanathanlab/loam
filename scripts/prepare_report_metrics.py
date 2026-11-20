"""Cache report-scoped metrics for terminal, numerically audited diagnostics."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import full_goal_cases
from full_goal_cases import load_run,measure


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(job,output):
    assert not output.exists(), 'Inspect/reuse the existing metric cache rather than overwriting it'
    source=Path(full_goal_cases.__file__)
    body=source.read_bytes()
    digest=hashlib.sha256(body).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    snapshot=output.with_name('metric-adapter-source-'+digest+'.py')
    if not snapshot.exists():snapshot.write_bytes(body)
    assert sha(snapshot)==digest
    # Import the pinned adapter before integrating; source edits during a long
    # telemetry scan cannot change which implementation this cache attests.
    spec=importlib.util.spec_from_file_location('pinned_metric_adapter_'+digest,snapshot)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    run=adapter.load_run(job)
    metric=adapter.measure(job.resolve())
    sources={str(snapshot):sha(snapshot),str(run['audit_path']):sha(run['audit_path']),
             str(run['job']/'allocation.json'):sha(run['job']/'allocation.json'),
             str(run['job']/'submission.json'):sha(run['job']/'submission.json'),
             str(run['job']/'plan.json'):sha(run['job']/'plan.json'),
             str(full_goal_cases.FROZEN/'audit_saturation_campaign.py'):sha(full_goal_cases.FROZEN/'audit_saturation_campaign.py')}
    value={'current':metric,'source_sha256':sources,
           'scope':'Report-only native integration, occupancy and distinct numerical production. Original campaign/claim/science gates remain unchanged. Diagnostic output counts do not certify a campaign as successful.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'cache':str(output),'job_id':run['receipt']['job_id'],
                      'summary':{k:v for k,v in metric['summary'].items() if not isinstance(v,(dict,list))}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.job,args.output)
