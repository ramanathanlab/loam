#!/usr/bin/env python3
"""Serve one assigned Nemotron node as a step of the parent allocation."""
import asyncio,json,os,socket,sys,time,traceback
from pathlib import Path
from loam.allocation import validate_role

infra=Path(sys.argv[1]);replica=sys.argv[2];allocation=validate_role(infra.parent/'allocation.json','inference')
entry=next(e for e in json.loads((infra/'pool.json').read_text())['backends'] if e['id']==replica)
assert entry['hostname']==socket.gethostname().split('.')[0]
attempt=Path(entry['campaign'])/'state/nemotron'/entry['attempt']
(attempt/'step.json').write_text(json.dumps({'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ.get('SLURM_STEP_ID'),'host':socket.gethostname(),'pid':os.getpid(),'started_at':time.time()})+'\n')
local=Path(os.environ.get('SLURM_TMPDIR','/tmp'))/('loam-'+allocation['job_id']+'-'+replica);local.mkdir(mode=0o700,parents=True,exist_ok=True)
os.environ.update(VLLM_CACHE_ROOT=str(local/'vllm'),TRITON_CACHE_DIR=str(local/'triton'),TORCHINDUCTOR_CACHE_DIR=str(local/'inductor'))
sys.path.insert(0,str(infra/'inference-runtime/scripts'));import jupiter_nemotron
for retry in range(2):
    try:
        asyncio.run(jupiter_nemotron.serve(attempt));break
    except Exception as exc:
        (attempt/f'failure-{retry}.json').write_text(json.dumps({'error_type':type(exc).__name__,'at':time.time(),'retry':retry})+'\n');traceback.print_exc()
        if retry:raise
        time.sleep(20)
