#!/usr/bin/env python3
"""Read-only pinning proof for actual canonical rows in an owned allocation.

No GPU work, signals, environment dumps or new allocation. Only three explicitly
selected, nonsecret environment fields leave each science node.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def worker(job):
    allocation=read(job/'allocation.json');host=socket.gethostname().split('.')[0]
    assert os.environ['SLURM_JOB_ID']==allocation['job_id'] and host in allocation['science_nodes']
    batches=(job/'campaign/batches').resolve();rows=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            if process.stat().st_uid!=os.getuid():continue
            cwd=(process/'cwd').resolve(strict=True)
            if not cwd.is_relative_to(batches):continue
            args=(process/'cmdline').read_bytes().split(b'\0')
            if len(args)<2 or Path(os.fsdecode(args[1])).name!='script.py':continue
            selected={}
            for entry in (process/'environ').read_bytes().split(b'\0'):
                for field in ['SLURM_JOB_ID','CUDA_VISIBLE_DEVICES','LOAM_EL_GPU_SLOTS']:
                    prefix=field.encode()+b'='
                    if entry.startswith(prefix):selected[field]=entry[len(prefix):].decode()
            if selected.get('SLURM_JOB_ID')!=allocation['job_id']:continue
            logical=int(selected['LOAM_EL_GPU_SLOTS']);gpu=int(selected['CUDA_VISIBLE_DEVICES'])
            rows.append({'pid':int(process.name),'row_directory':str(cwd.relative_to(batches)),
                         'logical_slot':logical,'physical_GPU':gpu,'mapping_valid':0<=logical<8 and 0<=gpu<4 and logical%4==gpu})
        except (OSError,ValueError,KeyError):continue
    proof={'at':time.time(),'job_id':allocation['job_id'],'host':host,'canonical_row_processes':rows,
           'scope':'Top-level canonical row Python processes and their pins at one instant. Nested scientific subprocesses are excluded; this is not a kernel-concurrency or completion measurement.'}
    (job/'diagnostics'/('active-science-'+host+'.json')).write_text(json.dumps(proof,indent=2)+'\n')


def main(job):
    receipt=read(job/'submission.json');allocation=read(job/'allocation.json')
    assert receipt['job_id'] in read(job.parent/'experiment.json')['owned_job_ids']
    raw=subprocess.run(['scontrol','show','job',receipt['job_id'],'-o'],capture_output=True,text=True,check=True,timeout=30).stdout
    fields=dict(token.split('=',1) for token in raw.split() if '=' in token)
    assert fields['JobName']==receipt['job_name'] and fields['WorkDir']==str(job) and fields['JobState']=='RUNNING'
    nodes=allocation['science_nodes'];script=Path(__file__).resolve()
    with (job/'diagnostics/active-science-probe.log').open('a') as log:
        result=subprocess.run(['srun','--jobid='+receipt['job_id'],'--overlap','--exact','--mpi=none',
            '--nodes='+str(len(nodes)),'--ntasks='+str(len(nodes)),'--ntasks-per-node=1','--cpus-per-task=1',
            '--gpus-per-node=0','--nodelist='+','.join(nodes),sys.executable,str(script),str(job),'--worker'],stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0
    results=[read(job/'diagnostics'/('active-science-'+host+'.json')) for host in nodes]
    rows=[(result['host'],row) for result in results for row in result['canonical_row_processes']]
    counts=collections.Counter((host,row['physical_GPU']) for host,row in rows)
    proof={'checked_at':time.time(),'job_id':receipt['job_id'],'nodes_probed':len(nodes),'canonical_row_processes':len(rows),
           'physical_GPUs_with_rows':len(counts),'physical_GPUs_with_two_rows':sum(n==2 for n in counts.values()),
           'maximum_rows_per_GPU':max(counts.values(),default=0),'all_recorded_mappings_valid':all(row['mapping_valid'] for _,row in rows),
           'probe_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
           'scope':'Sequential per-node instantaneous observations from one zero-GPU job step within the existing parent. No scientific output or sustained utilization is asserted.'}
    (job/'diagnostics/active-science-pinning-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=lambda p:Path(p).resolve());parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    worker(args.job) if args.worker else main(args.job)
