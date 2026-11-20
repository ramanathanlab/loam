#!/usr/bin/env python3
"""Batch-job owner: local control services plus disjoint inference/science steps.

This program never submits or cancels a Slurm job. It starts services only inside
the parent allocation and cleans up its own child processes/steps on exit.
"""
import asyncio,hashlib,json,os,secrets,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
import httpx,yaml

REPO=Path(__file__).resolve().parents[1];APP=Path(os.environ.get('LOAM_PYTHON',REPO/'.venv/bin/python'));SOFTWARE=Path(os.environ.get('LOAM_SOFTWARE_ROOT',REPO.parent/'software'))

def read(p):return json.loads(p.read_text())
def write(p,d):
    t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def free_port():
    with socket.socket() as s:s.bind(('0.0.0.0',0));return s.getsockname()[1]

async def run(job):
    plan=read(job/'plan.json');parent=os.environ.get('SLURM_JOB_ID','');assert parent.isdigit() and os.environ.get('SLURM_JOB_NAME')==plan['job_name']
    from loam.config import load
    from loam.allocation import validate_campaign_paths
    validate_campaign_paths(load(Path(plan['campaign'])/'campaign.yaml'),Path(plan['campaign']))
    integrity_path=job/'inputs-integrity-preflight.json'
    if integrity_path.exists():
        for name,digest in read(integrity_path)['input_sha256'].items():
            assert hashlib.sha256((Path(plan['campaign'])/'inputs'/name).read_bytes()).hexdigest()==digest,'Scientific input changed while queued: '+name
    seal_path=job/'prepared-control-source.json'
    if seal_path.exists():
        seal=read(seal_path)
        assert Path(seal['runtime_repo']).resolve()==REPO
        for name,digest in seal['source_sha256'].items():
            assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest,'Frozen control source changed while queued: '+name
        for name,digest in seal.get('native_source_sha256',{}).items():
            assert hashlib.sha256((Path(plan['infra'])/name).read_bytes()).hexdigest()==digest,'Native source changed while queued: '+name
        assert hashlib.sha256(Path(plan['inference_request_template']).read_bytes()).hexdigest()==seal['inference_request_template_sha256'],'Inference profile changed while queued'
    physical_gpus=int(plan.get('gpus_per_node',4));software=Path(plan.get('software_root',SOFTWARE))
    hosts=subprocess.run(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],capture_output=True,text=True,check=True,timeout=30).stdout.split();assert len(hosts)==plan['nodes'] and len(set(hosts))==len(hosts)
    sci=hosts[:plan['science_count']];inf=hosts[plan['science_count']:];assert socket.gethostname().split('.')[0] in sci
    control=socket.gethostbyname(socket.gethostname());infra=Path(plan['infra']);campaign=Path(plan['campaign']);manifest=job/'allocation.json'
    allocation={'mode':'single-allocation','job_id':parent,'job_name':plan['job_name'],'all_nodes':hosts,'science_nodes':sci,'inference_nodes':inf,'coordinator_host':socket.gethostname().split('.')[0],'coordinator_address':control,'started_at':time.time(),'job_root':str(job)};write(manifest,allocation)
    ownership=subprocess.run(['scontrol','show','job',parent,'-o'],capture_output=True,text=True,check=True,timeout=30).stdout;assert 'JobName='+plan['job_name']+' ' in ownership and 'WorkDir='+str(job)+' ' in ownership
    receipt=read(job/'submission.json');assert receipt['job_id']==parent and receipt['job_name']==plan['job_name']
    assert hashlib.sha256((job/'job.sh').read_bytes()).hexdigest()==receipt['script_sha256']
    protocol=read(campaign/'scientific-protocol.json')
    for item in protocol['files']:
        assert hashlib.sha256((campaign/'inputs'/item['file']).read_bytes()).hexdigest()==item['sha256'],'Staged science recipe changed before launch'
    sources=['scripts/run_single_allocation.py', 'scripts/run_single_replica.py', 'scripts/run_single_allocation_campaign.py', 'loam/allocation.py', 'loam/inference_pool.py', 'loam/config.py', 'loam/execution.py', 'loam/orchestrator.py', 'loam/synthesis.py', 'loam/agents.py', 'loam/history.py', 'loam/tournament.py', 'scripts/sample_jupiter_telemetry.py', 'skills/hpc-deployment/scripts/hpc_agent.py', 'skills/hpc-deployment/scripts/mcp_server.py', 'loam/allocation_startup.py', 'loam/jnana_checkpoint.py', 'loam/skills.py']
    write(campaign/'control-plane-at-start.json',{'job_id':parent,'captured_at':time.time(),'source_sha256':{name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in sources}})
    progress={'campaign_id':'unified-'+plan['job_name'],'mode':'single-allocation','scientific_job':parent,'parent_job_id':parent,'allocation_manifest':str(manifest),'scientific_allocation_nodes':len(sci),'inference_allocation_nodes':len(inf),'target_total_nodes':len(hosts),'active_inference_job_ids':[parent],'owned_job_ids':[parent],'inference_pool_experiment':str(infra),'baseline':str(plan.get('reference_case',REPO/'examples/scaling/egfr')),'phase':'bootstrapping single allocation','created_at':time.time()};write(campaign/'scale-progress.json',progress);write(infra/'scale-progress.json',{'owned_job_ids':[parent],'allocation_manifest':str(manifest)})
    children=[];open_logs=[];state={**allocation,'phase':'bootstrapping'}
    def status(phase,**kw):
        state.update(phase=phase,updated_at=time.time(),**kw);write(job/'status.json',state);print(json.dumps({'phase':phase,**kw}),flush=True)
    def start(args,name,env=None):
        log=(job/(name+'.log')).open('ab',buffering=0);open_logs.append(log);p=subprocess.Popen(list(map(str,args)),cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append((name,p));write(job/(name+'-process.json'),{'pid':p.pid,'args_sha256':hashlib.sha256(json.dumps(list(map(str,args))).encode()).hexdigest(),'parent_job':parent,'started_at':time.time()});return p
    async def healthy(url,headers=None,timeout=7200):
        deadline=time.monotonic()+timeout
        async with httpx.AsyncClient(trust_env=False,timeout=10) as client:
            while time.monotonic()<deadline:
                try:
                    q=await client.get(url,headers=headers)
                    if q.status_code==200:return q.json()
                except (httpx.HTTPError,ValueError):pass
                await asyncio.sleep(2)
        raise TimeoutError('Local service readiness deadline exceeded')
    try:
        # Password/TLS credentials are minted after allocation begins, so queue
        # waiting cannot consume their lifetime. Redis is private to this job.
        redis=infra/'redis';redis.mkdir(mode=0o700);password=secrets.token_urlsafe(40);password_file=redis/'password';password_file.write_text(password+'\n');password_file.chmod(0o600);redis_port=free_port();conf=redis/'redis.conf';conf.write_text(f'bind 127.0.0.1 {control}\nport {redis_port}\nprotected-mode yes\nrequirepass {password}\nsave ""\nappendonly no\ndir {redis}\n');conf.chmod(0o600)
        # Redis was built with GCC 14; scope that runtime to this process so
        # scientific tool library selection remains unchanged.
        redis_env=dict(os.environ)
        redis_lib=os.environ.get('LOAM_REDIS_LIBRARY_PATH','/e/software/fs/jupiter/stages/2026/software/GCCcore/14.3.0/lib64')
        if Path(redis_lib).is_dir():redis_env['LD_LIBRARY_PATH']=redis_lib+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
        redis_process=start([Path(plan.get('redis_executable',software/'redis/bin/redis-server')),conf],'redis',redis_env)
        import redis.asyncio as redis_client
        async with redis_client.Redis(host=control,port=redis_port,password=password) as connection:
            for retry in range(60):
                if redis_process.poll() is not None:raise RuntimeError('Private Redis exited during startup; see redis.log')
                try:
                    if await connection.ping():break
                except (ConnectionError,redis_client.ConnectionError):pass
                await asyncio.sleep(1)
            else:raise TimeoutError('Private Redis readiness timeout')
        base={**os.environ,'PYTHONPATH':str(REPO)+':'+str(REPO/'skills/hpc-deployment/scripts')+':'+os.environ.get('PYTHONPATH',''),'ACADEMY_EXCHANGE_TYPE':'redis','ACADEMY_REDIS_HOST':control,'ACADEMY_REDIS_PORT':str(redis_port),'ACADEMY_REDIS_PASSWORD_FILE':str(password_file),'ACADEMY_REDIS_DB':'0','HPC_LAUNCH_MODE':'existing','HPC_SCIENCE_NODELIST':','.join(sci),'HPC_MIN_BATCH_ROWS':'1','HPC_MIN_BATCH_DISTINCT':'1','HPC_GPUS_PER_NODE':str(physical_gpus),'HPC_CPUS_PER_NODE':str(plan.get('cpus_per_node',288)),'HPC_SCIENCE_WORKERS_PER_GPU':str(plan.get('science_workers_per_gpu',1))}
        request_template=read(Path(plan['inference_request_template']));entries=[]
        for i,host in enumerate(inf):
            replica=f'nemotron-{i+1:03d}';rep=infra/'replicas'/replica;attempt_name='loam-nemotron-'+secrets.token_hex(8);attempt=rep/'state/nemotron'/attempt_name;attempt.mkdir(mode=0o700,parents=True)
            (attempt/'token').write_text(secrets.token_urlsafe(32)+'\n');(attempt/'token').chmod(0o600)
            subprocess.run(['openssl','req','-x509','-newkey','ec','-pkeyopt','ec_paramgen_curve:P-256','-noenc','-days','7','-subj','/CN='+attempt_name,'-keyout',str(attempt/'tls.key'),'-out',str(attempt/'tls.pem')],check=True,capture_output=True,timeout=30);(attempt/'tls.key').chmod(0o600)
            request={**request_template,'campaign':str(rep),'login_address':control,'warmup_seconds':plan['model_startup_timeout'],'reservation':plan['reservation'],'walltime':plan['walltime'],'allocation_manifest':str(manifest),'assigned_host':host};write(attempt/'request.json',request);write(attempt/'submission.json',{'job_id':parent,'mode':'job-step','hostname':host});write(rep/'state/nemotron/current.json',{'attempt':attempt_name})
            entries.append({'id':replica,'job_id':parent,'campaign':str(rep),'attempt':attempt_name,'hostname':host,'allocation_manifest':str(manifest),'enabled':True})
        write(infra/'pool.json',{'model':'nemotron-3.5-super-vl','policy':'least_loaded_affinity','phase':'initializing','max_inflight_per_backend':16,'backends':entries})
        from loam.allocation_startup import wait_for_allocation_telemetry
        telemetry=start(['srun','--overlap','--exact','--mpi=none','--nodes='+str(len(hosts)),'--ntasks='+str(len(hosts)),'--ntasks-per-node=1','--cpus-per-task=1','--gpus-per-node='+str(physical_gpus),'--gpu-bind=none',APP,REPO/'scripts/sample_jupiter_telemetry.py',campaign,'--allocation',manifest],'telemetry',base)
        status('telemetry_starting',inference_ready=0,science_nodes=len(sci))
        await wait_for_allocation_telemetry(telemetry,campaign,allocation,timeout=900)
        for entry in entries:
            start(['srun','--overlap','--exact','--mpi=none','--nodes=1','--ntasks=1','--cpus-per-task=32','--gpus-per-node='+str(physical_gpus),'--gpu-bind=none','--nodelist='+entry['hostname'],APP,REPO/'scripts/run_single_replica.py',infra,entry['id']],entry['id'],base)
            await asyncio.sleep(0.25)
        gateway=start([APP,REPO/'scripts/serve_nemotron_pool.py',infra,'--port','0'],'gateway',base)
        while not (infra/'pool-service.json').exists():
            if gateway.poll() is not None:raise RuntimeError('Gateway exited during startup')
            await asyncio.sleep(1)
        service=read(infra/'pool-service.json');headers={'Authorization':'Bearer '+Path(service['token_file']).read_text().strip()}
        cfg=yaml.safe_load((campaign/'campaign.yaml').read_text());hpc=cfg['execution']['hpc'];hpc['port']=free_port()
        for model in [*cfg['models'].values(),cfg['tournament']['generation'],cfg['tournament']['ranking']]:model['base_url']=service['base_url']
        (campaign/'campaign.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        from loam.config import load
        typed=load(campaign/'campaign.yaml');env={**base,**typed.execution.hpc.environment()};env.update(base);env['HPC_CLUSTER_STARTUP_TIMEOUT']='7200';env['HPC_AGENT_STARTUP_TIMEOUT']='900'
        mcp=start([APP,typed.execution.hpc.server_script],'science-mcp',env)
        status('services_starting',inference_ready=0,science_nodes=len(sci))
        from loam.execution import check_mcp_tools
        typed.execution.mcp_url=f'http://127.0.0.1:{hpc["port"]}/mcp'
        deadline=time.monotonic()+plan['model_startup_timeout'];last=0;science_ready=False
        async with httpx.AsyncClient(trust_env=False,timeout=15) as client:
            while time.monotonic()<deadline:
                if mcp.poll() is not None or gateway.poll() is not None:raise RuntimeError('Infrastructure service exited during startup')
                q=await client.get(service['status_url'],headers=headers);q.raise_for_status();pool=q.json();ready=pool['ready_replicas']
                if not science_ready:
                    try:await asyncio.wait_for(check_mcp_tools(typed),timeout=2);science_ready=True
                    except Exception:pass
                if time.monotonic()-last>30:status('services_starting',inference_ready=ready,inference_total=len(inf),science_ready=science_ready);last=time.monotonic()
                if ready==len(inf) and science_ready:break
                await asyncio.sleep(5)
            else:raise TimeoutError('Inference/science startup timeout')
        write(campaign/'server-ready.json',{'mcp_url':typed.execution.mcp_url,'parent_job_id':parent,'ready_at':time.time()});pool_cfg=read(infra/'pool.json');pool_cfg['phase']=plan['job_name'];write(infra/'pool.json',pool_cfg)
        status('campaign_running',inference_ready=len(inf),science_ready=True)
        controller_env={**base,'LOAM_NEMOTRON_TOKEN':Path(service['token_file']).read_text().strip(),'CUDA_VISIBLE_DEVICES':''}
        controller=start([APP,REPO/'scripts/run_single_allocation_campaign.py',job],'controller',controller_env);write(campaign/'controller-watch.json',{'state':'controller running','controller_pid':controller.pid,'controller_started_at':time.time(),'scientific_job':parent})
        indexer=None
        while controller.poll() is None:
            if indexer is None and (campaign/'campaign-tool-registrations.json').exists():indexer=start([APP,REPO/'scripts/update_hypothesis_artifact_index.py',campaign,'--watch'],'artifact-index',base)
            await asyncio.sleep(5)
        watch=read(campaign/'controller-watch.json');watch.update(state='controller exited',returncode=controller.returncode,controller_finished_at=time.time());write(campaign/'controller-watch.json',watch)
        if controller.returncode:raise RuntimeError('Campaign controller failed; see controller.log')
        status('draining_submitted_science')
        deadline=time.monotonic()+7200
        while True:
            pending=[]
            for p in campaign.glob('batches/*/batch.json'):
                b=read(p);terminal=sum((r/'done').exists() and (r/'status.json').exists() for r in p.parent.glob('run_*'))
                if terminal<b['total']:pending.append(b['batch_id'])
            if not pending:break
            if time.monotonic()>deadline:raise TimeoutError('Submitted scientific rows did not drain')
            status('draining_submitted_science',pending_batches=pending);await asyncio.sleep(15)
        status('completed',controller_returncode=0,submitted_batches_terminal=True)
    except Exception as exc:
        status('failed',error_type=type(exc).__name__);traceback.print_exc();raise
    finally:
        # Terminate only subprocess groups we created. Never scancel the parent
        # or any other allocation; normal batch exit releases this one job.
        for name,p in reversed(children):
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        end=time.monotonic()+45
        for name,p in reversed(children):
            if p.poll() is None:
                try:await asyncio.to_thread(p.wait, max(1,end-time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
        for log in open_logs:log.close()
        write(job/'child-cleanup.json',{'at':time.time(),'parent_job':parent,'children':[{'name':n,'pid':p.pid,'returncode':p.poll()} for n,p in children]})

if __name__=='__main__':
    asyncio.run(run(Path(sys.argv[1]).resolve()))
