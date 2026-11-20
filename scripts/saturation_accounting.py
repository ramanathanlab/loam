"""Count every allocated attempt of an owned parent job, including Slurm requeues."""
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

FIELDS=['JobIDRaw','JobName','State','NNodes','AllocTRES','Submit','Start','End','ElapsedRaw','NodeList','ExitCode','Reason','FailedNode']


def timestamp(value):
    try:
        moment=datetime.fromisoformat(value)
        if moment.tzinfo is None:moment=moment.replace(tzinfo=ZoneInfo('Europe/Berlin'))
        return moment.timestamp()
    except (TypeError,ValueError):return None


def parse_attempts(content,job_id):
    allocations={};unallocated=[]
    for line in content.splitlines():
        row=dict(zip(FIELDS,line.strip().split('|')))
        if row.get('JobIDRaw')!=job_id:continue
        start=timestamp(row.get('Start'));end=timestamp(row.get('End'))
        if start is None:
            unallocated.append(row);continue
        elapsed=float(row.get('ElapsedRaw') or 0)
        if end is None:
            if row['State'].startswith(('RUNNING','COMPLETING')):end=start+elapsed
            else:raise ValueError('Allocated attempt has no authoritative end: '+str(row['State']))
        elapsed=max(elapsed,end-start)
        if elapsed<=0:continue
        nodes=int(row['NNodes']);tres=dict(part.split('=',1) for part in row['AllocTRES'].split(',') if '=' in part)
        if 'gres/gpu' in tres:gpus=int(tres['gres/gpu'])
        else:gpus=sum(int(value) for key,value in tres.items() if key.startswith('gres/gpu:'))
        if not nodes or gpus!=4*nodes:raise ValueError('Unexpected Jupiter GPU allocation count')
        record={**row,'start_epoch':start,'end_epoch':end,'allocated_seconds':elapsed,
                'allocated_GPUs':gpus,'allocated_GPU_seconds':gpus*elapsed}
        # sacct may expose repeated records for one controller transition. They
        # describe one physical allocation, not additional billed node-time.
        key=(start,end,nodes,row['NodeList'],gpus)
        allocations[key]=record
    values=sorted(allocations.values(),key=lambda r:r['start_epoch'])
    return {'job_id':job_id,'allocations':values,'unallocated_records':unallocated,
            'allocation_attempts':len(values),
            'total_allocated_seconds':sum(r['allocated_seconds'] for r in values),
            'total_allocated_GPU_seconds':sum(r['allocated_GPU_seconds'] for r in values),
            'scope':'Each physical parent allocation counted once. Queue-only records and job steps contribute no GPU time.'}


def allocation_attempts(job_id):
    if not job_id.isdigit():raise ValueError('Expected exact numeric parent job ID')
    result=subprocess.run(['sacct','--duplicates','-X','-n','-P','-j',job_id,
                           '--format='+','.join(FIELDS)],capture_output=True,text=True,check=True,timeout=60)
    return parse_attempts(result.stdout,job_id)
