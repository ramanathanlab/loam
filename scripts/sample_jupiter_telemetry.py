#!/usr/bin/env python3
"""Sample hardware and optional vLLM counters inside an owned Slurm allocation."""
import argparse, fcntl, json, os, socket, subprocess, time, urllib.request
from pathlib import Path

def node_counters():
    cpu=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    mem={line.split(':',1)[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    return cpu,mem

def main():
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--process',type=Path);p.add_argument('--allocation',type=Path);p.add_argument('--interval',type=float,default=10);a=p.parse_args()
    receipt=json.loads((a.campaign/'scale-progress.json').read_text());job=os.environ.get('SLURM_JOB_ID')
    if job not in receipt['owned_job_ids']:raise RuntimeError('Refusing telemetry outside a recorded campaign-owned job')
    d=a.campaign/'telemetry';d.mkdir(exist_ok=True);host=socket.gethostname();dest=d/f'{job}-{host}.jsonl'
    role=None
    if a.allocation:
        from loam.allocation import read_allocation,validate_role
        allocated=read_allocation(a.allocation);role='inference' if host.split('.')[0] in allocated['inference_nodes'] else 'science';validate_role(a.allocation,role)
        if role=='inference':
            infra=a.allocation.parent/'infra';entries=json.loads((infra/'pool.json').read_text())['backends'];entry=next(e for e in entries if e['hostname']==host.split('.')[0]);a.process=Path(entry['campaign'])/'state/nemotron'/entry['attempt']/'process.json'
    lock=(d/f'{job}-{host}.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A sampler already holds this owned job/node lock')
    previous=None
    while True:
        row={'timestamp':time.time(),'host':host,'job_id':job}
        if role:row.update(role=role,step_id=os.environ.get('SLURM_STEP_ID'))
        try:
            cpu,mem=node_counters();row['host_memory_used_percent']=100*(1-mem['MemAvailable']/mem['MemTotal']);row['logical_cpus']=os.cpu_count()
            if previous:
                total=sum(cpu)-sum(previous)
                if total>0:
                    row['cpu_busy_percent']=100*(1-((cpu[3]-previous[3])+(cpu[4]-previous[4]))/total)
                    row['cpu_iowait_percent']=100*(cpu[4]-previous[4])/total
            previous=cpu
        except (OSError,ValueError,KeyError) as error:row['cpu_memory_error']=type(error).__name__
        try:
            q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
            row['gpu_csv']=q.stdout.strip();row['nvidia_returncode']=q.returncode
        except (OSError,subprocess.TimeoutExpired) as error:
            row['nvidia_error']=type(error).__name__
        if a.process and a.process.exists():
            try:
                proc=json.loads(a.process.read_text());base=proc['raw_base_url'].removesuffix('/v1')
                # Bypass all proxies for the job-local metrics socket.
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                metrics=opener.open(base+'/metrics',timeout=3).read().decode()
                row['vllm_metrics']=[s for s in metrics.splitlines() if s.startswith('vllm:') and any(k in s for k in ('num_requests_running','num_requests_waiting','kv_cache_usage_perc','prompt_tokens_total','generation_tokens_total','request_success_total','e2e_request_latency_seconds_sum','e2e_request_latency_seconds_count','time_to_first_token_seconds_sum','time_to_first_token_seconds_count'))]
            except Exception as e:row['metrics_error']=type(e).__name__
        with dest.open('a') as f:f.write(json.dumps(row)+'\n')
        time.sleep(a.interval)
if __name__=='__main__':main()
