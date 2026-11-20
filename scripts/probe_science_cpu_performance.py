#!/usr/bin/env python3
"""Paired CPU-resource diagnostic using an existing receipt-owned allocation.

Replays one completed canonical screen per condition at one and eight CPUs.
Duplicate diagnostic seeds are excluded from scientific campaign throughput.
Never submits or cancels a Slurm job and never changes the live controller.
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
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def main(job):
    allocation = validate_role(job/'allocation.json', 'science')
    receipt = json.loads((job/'submission.json').read_text())
    assert receipt['job_id'] == allocation['job_id']
    assert receipt['job_id'] in json.loads((job.parent/'experiment.json').read_text())['owned_job_ids']
    assert socket.gethostname().split('.')[0] == allocation['coordinator_host']
    campaign = job/'campaign'
    script = campaign/'tools/scale20_complete_binder_pipeline/v1/script.py'
    protocol = json.loads((campaign/'scientific-protocol.json').read_text())
    original = next(x['sha256'] for x in protocol['files'] if x['file']=='run_binder_pipeline.py')
    assert hashlib.sha256(script.read_bytes()).hexdigest() == original
    selected = {}
    for registry in sorted((campaign/'batches/registry').glob('*.json')):
        entry = json.loads(registry.read_text())
        if entry.get('tool') != 'scale20_complete_binder_pipeline': continue
        for folder in sorted((campaign/'batches'/entry['batch_id']).glob('run_*')):
            if not (folder/'summary.json').is_file() or not (folder/'status.json').is_file(): continue
            summary = json.loads((folder/'summary.json').read_text())
            status = json.loads((folder/'status.json').read_text())
            if status['returncode'] != 0 or summary.get('status') != 'passed': continue
            row = json.loads((folder/'input.json').read_text())
            selected.setdefault(row['condition'], {'row':row, 'source':str(folder)})
        if len(selected)==3: break
    assert set(selected)=={'dual','one_region','ridge'}
    diagnostic = job/'diagnostics/cpu-performance'
    diagnostic.mkdir()  # refuse duplicate diagnostic submissions
    experiment = {'job_id':receipt['job_id'], 'started_at':time.time(),
        'script':str(script), 'script_sha256':original, 'selected_inputs':selected,
        'scope':'Paired diagnostic replays only; no new hypotheses or scientific throughput.', 'tasks':[]}
    write(diagnostic/'intent.json', experiment)
    checkpoint = list(campaign.glob('ckpt_*/main/main_comm.json'))
    assert len(checkpoint)==1
    os.environ['EL_LOGDIR'] = str(diagnostic/'logs')
    client = ClusterClient(checkpoint_dir=str(checkpoint[0].parents[1]), node_id='main', checkpoint_timeout=30)
    science_python = '/e/project1/deepwing/meigooni1/agent/software/RFD4-Proteina/.venv/bin/python'
    repo = Path(__file__).resolve().parents[1]
    futures = []
    try:
        client.start()
        for condition, selection in selected.items():
            for cpus in (1, 8):
                output = diagnostic/(condition+'-cpu'+str(cpus)); output.mkdir()
                (output/'input.json').write_text(json.dumps(selection['row'],indent=2)+'\n')
                # A stdlib callable can cross the installed process-pool's final
                # pickle boundary; the child inherits the real EL CPU/GPU binding.
                bootstrap = 'import os,json,subprocess,time; from pathlib import Path; '
                bootstrap += 'p=Path('+repr(str(output))+'); os.chdir(p); '
                bootstrap += 'Path("binding.json").write_text(json.dumps({"host":os.uname().nodename,"at":time.time(),"cpus_allowed":sorted(os.sched_getaffinity(0)),"gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),"job_id":os.environ.get("SLURM_JOB_ID")})); '
                bootstrap += 'subprocess.run(['+repr(science_python)+','+repr(str(script))+',"input.json"],check=True,stdout=open("stdout.txt","w"),stderr=open("stderr.txt","w"))'
                task_id = 'diagnostic-science-cpu-'+str(cpus)+'-'+uuid.uuid4().hex
                task = Task(task_id=task_id, nnodes=1, ppn=cpus, ngpus_per_process=1/cpus,
                    executable=subprocess.check_call, args=([science_python,'-c',bootstrap],),
                    env={'LOAM_SOFTWARE_ROOT':'/e/project1/deepwing/meigooni1/agent/software', 'HPC_LOAM_HOME':str(repo)})
                request = task.get_resource_requirements().resources[0]
                assert request.cpu_count==cpus and request.gpu_count==1
                item = {'task_id':task_id,'condition':condition,'cpus':cpus,'output':str(output),'submitted_at':time.time()}
                experiment['tasks'].append(item); write(diagnostic/'intent.json',experiment)
                futures.append((client.submit(task),item))
        for future,item in futures:
            try:
                item['return_value'] = future.result(timeout=1800)
                item['summary_exists'] = (Path(item['output'])/'summary.json').is_file()
            except Exception as exc:
                item['error'] = str(exc)
            item['observed_finished_at'] = time.time()
            write(diagnostic/'progress.json',experiment)
        experiment['finished_at'] = time.time()
        experiment['all_completed'] = all(x.get('summary_exists') and not x.get('error') for x in experiment['tasks'])
        write(diagnostic/'result.json',experiment)
        print(json.dumps({'all_completed':experiment['all_completed'],'tasks':len(futures),'path':str(diagnostic)},indent=2),flush=True)
    finally:
        client.teardown()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=lambda p:Path(p).resolve())
    main(parser.parse_args().job)
