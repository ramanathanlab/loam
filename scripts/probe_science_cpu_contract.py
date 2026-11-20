#!/usr/bin/env python3
"""Probe CPU/GPU binding through an already owned EL cluster; no GPU kernels.

Run on the receipt-owned coordinator. This submits one tiny diagnostic task,
never a Slurm job, and tears down only its own ClusterClient connection.
"""
import argparse
import json
import os
import socket
import inspect
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import ClusterClient
from loam.allocation import validate_role


def inspect_binding(job_id, output, expected_cpus):
    import json
    import os
    import socket
    import time
    from pathlib import Path
    assert os.environ['SLURM_JOB_ID']==job_id
    result={'job_id':job_id,'host':socket.gethostname().split('.')[0],
            'pid':os.getpid(),'at':time.time(),
            'CPUs_allowed':sorted(os.sched_getaffinity(0)),
            'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES',''),
            'requested_CPUs':expected_cpus,'requested_physical_GPUs':1,
            'scope':'Resource binding only; no scientific/GPU computation.'}
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main(job, cpus):
    allocation=validate_role(job/'allocation.json','science')
    receipt=json.loads((job/'submission.json').read_text())
    assert receipt['job_id']==allocation['job_id']
    assert receipt['job_id'] in json.loads((job.parent/'experiment.json').read_text())['owned_job_ids']
    assert socket.gethostname().split('.')[0]==allocation['coordinator_host']
    manifests=list((job/'campaign').glob('ckpt_*/main/main_comm.json'))
    assert len(manifests)==1,'Need one unambiguous owned live cluster'
    checkpoint=manifests[0].parents[1]
    folder=job/'diagnostics/cpu-contract';folder.mkdir(parents=True,exist_ok=True)
    task_id='diagnostic-cpu-'+str(cpus)+'-'+uuid.uuid4().hex
    output=folder/(task_id+'-result.json')
    # The installed process-pool executor uses stdlib pickle for the final
    # dispatch. A function defined in __main__ cannot cross that boundary.
    code=inspect.getsource(inspect_binding)+'\ninspect_binding('+repr(receipt['job_id'])+','+repr(str(output))+','+repr(cpus)+')\n'
    task=Task(task_id=task_id,nnodes=1,ppn=cpus,ngpus_per_process=1/cpus,
              executable=subprocess.check_call,args=([sys.executable,'-c',code],))
    req=task.get_resource_requirements().resources[0]
    assert req.cpu_count==cpus and req.gpu_count==1
    intent={'job_id':receipt['job_id'],'task_id':task_id,'at':time.time(),
            'checkpoint':str(checkpoint),'output':str(output),
            'requested_CPUs':cpus,'requested_physical_GPUs':1,
            'scope':'Tiny binding diagnostic in an existing owned allocation, not new scientific output or a new Slurm job.'}
    (folder/(task_id+'-intent.json')).write_text(json.dumps(intent,indent=2)+'\n')
    os.environ['EL_LOGDIR']=str(folder/'logs')
    client=None
    try:
        client=ClusterClient(checkpoint_dir=str(checkpoint),checkpoint_timeout=30)
        client.start()
        client.submit(task).result(timeout=600)
        result=json.loads(output.read_text())
        assert result['host'] in allocation['science_nodes']
        assert len(result['CPUs_allowed'])==cpus,result
        assert len(result['CUDA_VISIBLE_DEVICES'].split(','))==1 and result['CUDA_VISIBLE_DEVICES'],result
        proof={**intent,'result':result,'passed':True,'finished_at':time.time()}
        (job/'diagnostics/cpu-contract-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
        print(json.dumps(proof,indent=2),flush=True)
    except Exception:
        (folder/(task_id+'-error.txt')).write_text(traceback.format_exc())
        raise
    finally:
        if client is not None:
            client.teardown()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=lambda p:Path(p).resolve())
    parser.add_argument('--cpus',type=int,default=8);args=parser.parse_args()
    assert args.cpus>0
    main(args.job,args.cpus)
