#!/usr/bin/env python3
"""Compare serial and concurrent canonical rows inside one owned GPU lease.

Replays are diagnostics, excluded from novel campaign output. No scheduler
submission/cancellation and no modification to any scientific recipe.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import ClusterClient
from loam.allocation import validate_role


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def driver(folder):
    """This runs on a science node with one physical GPU and two CPU cores."""
    request=json.loads((folder/'request.json').read_text())
    assert os.environ['SLURM_JOB_ID']==request['job_id']
    cpus=sorted(os.sched_getaffinity(0));assert len(cpus)==2
    visible=os.environ['CUDA_VISIBLE_DEVICES'];assert len(visible.split(','))==1
    result={'job_id':request['job_id'],'host':socket.gethostname(),'gpu':visible,
            'cpus':cpus,'phases':[],'scope':'Paired replay diagnostic; no new campaign science.'}
    for phase,parallel in [('serial',False),('parallel',True)]:
        section={'phase':phase,'started_at':time.time(),'rows':[]};processes=[]
        for index,selection in enumerate(request['selected']):
            target=folder/phase/str(index);target.mkdir(parents=True)
            (target/'input.json').write_text(json.dumps(selection['row'],indent=2)+'\n')
            bootstrap='import os; os.sched_setaffinity(0,{'+str(cpus[index])+'}); os.execv('+repr(request['python'])+','+repr([request['python'],request['script'],'input.json'])+')'
            stdout=(target/'stdout.txt').open('w');stderr=(target/'stderr.txt').open('w')
            p=subprocess.Popen([request['python'],'-c',bootstrap],cwd=target,stdout=stdout,stderr=stderr)
            item={'output':str(target),'pid':p.pid,'cpu':cpus[index],'started_at':time.time()}
            section['rows'].append(item);processes.append((p,item,stdout,stderr))
            if not parallel:
                item['returncode']=p.wait();item['finished_at']=time.time()
        for p,item,stdout,stderr in processes:
            if 'returncode' not in item:
                item['returncode']=p.wait();item['finished_at']=time.time()
            stdout.close();stderr.close()
        section['finished_at']=time.time();section['elapsed_seconds']=section['finished_at']-section['started_at']
        result['phases'].append(section);write(folder/'progress.json',result)
    result['all_completed']=all(r['returncode']==0 and (Path(r['output'])/request['summary_file']).is_file() for p in result['phases'] for r in p['rows'])
    result['speedup']=result['phases'][0]['elapsed_seconds']/result['phases'][1]['elapsed_seconds']
    write(folder/'result.json',result)
    assert result['all_completed'],result


def main(job):
    allocation=validate_role(job/'allocation.json','science')
    receipt=json.loads((job/'submission.json').read_text())
    assert receipt['job_id']==allocation['job_id']
    assert receipt['job_id'] in json.loads((job.parent/'experiment.json').read_text())['owned_job_ids']
    assert socket.gethostname().split('.')[0]==allocation['coordinator_host']
    campaign=job/'campaign';diagnostic=job/'diagnostics/gpu-sharing';diagnostic.mkdir()
    tools={'screen':'scale20_complete_binder_pipeline','full_context':'scale20_intact_egfr_followup'}
    requests=[]
    for kind,tool in tools.items():
        summary_file='summary.json' if kind=='screen' else 'followup-summary.json'
        selected=[];seen=set();script=campaign/'tools'/tool/'v1/script.py'
        for registry in sorted((campaign/'batches/registry').glob('*.json')):
            entry=json.loads(registry.read_text())
            if entry.get('tool')!=tool:continue
            assert entry['script_sha256']==hashlib.sha256(script.read_bytes()).hexdigest()
            for folder in sorted((campaign/'batches'/entry['batch_id']).glob('run_*')):
                if not (folder/summary_file).is_file() or not (folder/'status.json').is_file():continue
                status=json.loads((folder/'status.json').read_text());row=json.loads((folder/'input.json').read_text())
                if status['returncode']!=0:continue
                identity=json.dumps({k:v for k,v in row.items() if k not in ['id','out_dir']},sort_keys=True)
                if identity in seen:continue
                seen.add(identity);selected.append({'row':row,'source':str(folder)})
                if len(selected)==2:break
            if len(selected)==2:break
        assert len(selected)==2
        output=diagnostic/kind;output.mkdir()
        request={'kind':kind,'summary_file':summary_file,'job_id':receipt['job_id'],'script':str(script),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'selected':selected,
                 'python':'/e/project1/deepwing/meigooni1/agent/software/RFD4-Proteina/.venv/bin/python'}
        write(output/'request.json',request);requests.append((output,request))
    checkpoint=list(campaign.glob('ckpt_*/main/main_comm.json'));assert len(checkpoint)==1
    os.environ['EL_LOGDIR']=str(diagnostic/'logs')
    client=ClusterClient(checkpoint_dir=str(checkpoint[0].parents[1]),node_id='main',checkpoint_timeout=30)
    intent={'job_id':receipt['job_id'],'at':time.time(),'tasks':[],'scope':'Real canonical scientific replay diagnostics; exclude from novel output and hypotheses.'}
    write(diagnostic/'intent.json',intent)
    futures=[]
    try:
        client.start()
        for output,request in requests:
            task_id='diagnostic-sharing-'+uuid.uuid4().hex
            task=Task(task_id=task_id,nnodes=1,ppn=2,ngpus_per_process=.5,
                executable=subprocess.check_call,args=([str(Path(__file__).resolve().parents[1]/'.venv/bin/python'),str(Path(__file__).resolve()),'--driver',str(output)],))
            resource=task.get_resource_requirements().resources[0]
            assert resource.cpu_count==2 and resource.gpu_count==1
            item={'task_id':task_id,'kind':request['kind'],'output':str(output),'requested_CPUs':2,'physical_GPUs':1,'at':time.time()}
            intent['tasks'].append(item);write(diagnostic/'intent.json',intent)
            futures.append((client.submit(task),item))
        for future,item in futures:
            try:
                future.result(timeout=2400)
                item['result']=json.loads((Path(item['output'])/'result.json').read_text())
            except Exception as exc:item['error']=str(exc)
            write(diagnostic/'progress.json',intent)
        intent['finished_at']=time.time();write(diagnostic/'result.json',intent)
        print(json.dumps({'path':str(diagnostic),'finished':True,'errors':[x.get('error') for x in intent['tasks']]},indent=2),flush=True)
    finally:client.teardown()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',nargs='?',type=lambda p:Path(p).resolve());parser.add_argument('--driver',type=Path)
    args=parser.parse_args()
    if args.driver:driver(args.driver)
    else:main(args.job)
