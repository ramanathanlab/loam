"""The single-allocation coordinator: one batch job that runs a whole campaign on its own nodes.

    python -m loam.coordinator JOB

is what a staged job's `job.sh` runs (loam.staging), on the first science host. It never
submits or cancels a job. In order, it:

1. checks the job before touching the scheduler: every campaign path is the job's own, the
   inputs and frozen runtime are unchanged since staging, and it runs in the submitted job;
2. splits the allocation's hosts into science and inference roles (`allocation.json`);
3. starts per-host telemetry and waits until every GPU on every host reports (loam.telemetry,
   loam.allocation_startup);
4. starts one inference replica per inference host, from the job's inference profile, behind
   one `loam.inference_pool`, and points the campaign's served models at the pool;
5. runs the campaign controller (`python -m loam.coordinator --controller JOB`), which attaches
   the MCP server to the science hosts and runs every round;
6. waits for submitted science to finish, then stops only the processes it started.

Everything scheduler-specific (host discovery, per-host steps, the job id variable) comes from
the campaign's System. Progress is written to JOB/status.json. Ported from the moeen branch's
run_single_allocation.py and run_single_allocation_campaign.py; the Nemotron-specific replica
setup and operator smoke mode stay there.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import signal
import socket
import subprocess
import time
import traceback
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import httpx
import yaml

from .allocation import validate_campaign_paths, validate_role
from .allocation_startup import wait_for_allocation_telemetry
from .config import load
from .staging import InferenceProfile, check, check_gc_threshold, read, write
from .systems import System

RUNTIME = Path(__file__).resolve().parents[1]
POOL_TOKEN = 'LOAM_POOL_TOKEN'
TELEMETRY_TIMEOUT = 900
DRAIN_TIMEOUT = 7200
STOP_GRACE = 45


def plan_allocation(plan: dict, hosts: list[str], *, job_id: str, hostname: str) -> dict:
    """The role manifest for this allocation: the first `science_count` hosts do science, the rest inference."""
    if len(hosts) != plan['nodes'] or len(set(hosts)) != len(hosts):
        raise RuntimeError(f'The allocation has {len(hosts)} hosts; the plan needs {plan["nodes"]} distinct hosts')
    science, inference = hosts[:plan['science_count']], hosts[plan['science_count']:]
    host = hostname.split('.')[0]
    if host not in science:
        raise RuntimeError(f'The coordinator runs on {host}, which is not a science host')
    return {'mode': 'single-allocation', 'job_id': job_id, 'job_name': plan['job_name'], 'all_nodes': hosts,
            'science_nodes': science, 'inference_nodes': inference, 'coordinator_host': host, 'started_at': time.time(),
            'job_root': plan['job_root']}


def replica_command(profile: InferenceProfile, host: str) -> list[str]:
    return [part.replace('{host}', host).replace('{port}', str(profile.port)) for part in profile.command]


def replica_steps(system: System, profile: InferenceProfile, hosts: list[str]) -> list[list[str]]:
    """One step per inference host, each serving the model on that host alone."""
    return [system.host_step([host], replica_command(profile, host), cpus=profile.cpus_per_task) for host in hosts]


def pool_config(profile: InferenceProfile, hosts: list[str], *, job_id: str) -> dict:
    """pool.json for `loam.inference_pool`: every replica, by the URL its host serves on."""
    config = {'model': profile.model, 'policy': 'least_loaded_affinity', 'phase': 'single-allocation',
              'backends': [{'id': f'replica-{i + 1:03d}', 'job_id': job_id, 'hostname': host,
                            'base_url': f'http://{host}:{profile.port}/v1',
                            **({'checkpoint': profile.checkpoint} if profile.checkpoint else {})}
                           for i, host in enumerate(hosts)]}
    if profile.checkpoint:
        config['checkpoint'] = profile.checkpoint
    return config


def point_models_at_pool(config: Path, roles: list[str], base_url: str, *, mcp_port: int) -> None:
    """Send the served roles' requests (and Jnana's, if it uses the same model) to the pool."""
    raw = yaml.safe_load(config.read_text())
    served = {raw['models'][role]['model'] for role in roles}
    specs = [*(raw['models'][role] for role in roles),
             *(spec for spec in (raw['tournament']['generation'], raw['tournament']['ranking'])
               if spec['model'] in served)]
    for spec in specs:
        spec.update(base_url=base_url, api_key_env=POOL_TOKEN)
        spec.pop('token_refresh', None)  # the local pool owns its token; external refresh no longer applies
    raw['execution']['hpc']['port'] = mcp_port
    config.write_text(yaml.safe_dump(raw, sort_keys=False))


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def configure_controller_gc(plan: dict) -> dict:
    """Set the controller's young-generation GC threshold if the plan asks; never disable collection.

    Only the controller process calls this; replicas and science workers keep their own policy.
    """
    before = gc.get_threshold()
    requested = plan.get('controller_gc_threshold0')
    check_gc_threshold(requested)
    if requested is not None:
        if not gc.isenabled():
            raise ValueError('Controller GC tuning requires automatic collection to be enabled')
        gc.set_threshold(requested, *before[1:])
    return {'requested_threshold0': requested, 'thresholds_before': list(before),
            'thresholds_applied': list(gc.get_threshold()), 'automatic_gc_enabled': gc.isenabled()}


class Children:
    """Processes the coordinator started, each in its own session with its own log, stopped together."""

    def __init__(self, job: Path, job_id: str):
        self.job, self.job_id = job, job_id
        self.started: list[tuple[str, subprocess.Popen]] = []
        self.logs = []

    def start(self, name: str, args: list, env: Mapping[str, str]) -> subprocess.Popen:
        log = (self.job / f'{name}.log').open('ab', buffering=0)
        self.logs.append(log)
        process = subprocess.Popen([str(a) for a in args], cwd=RUNTIME, env=dict(env), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        self.started.append((name, process))
        write(self.job / f'{name}-process.json', {'pid': process.pid, 'parent_job': self.job_id,
                                                  'args': [str(a) for a in args], 'started_at': time.time()})
        return process

    async def stop(self) -> None:
        """Terminate only the process groups started here; the job itself ends when the script exits."""
        for _, process in reversed(self.started):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + STOP_GRACE
        for _, process in reversed(self.started):
            if process.poll() is None:
                try:
                    await asyncio.to_thread(process.wait, max(1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        for log in self.logs:
            log.close()
        write(self.job / 'child-cleanup.json', {'at': time.time(), 'parent_job': self.job_id, 'children': [
            {'name': name, 'pid': process.pid, 'returncode': process.poll()} for name, process in self.started]})


def check_job(job: Path, plan: dict, environ: Mapping[str, str], runtime: Path) -> tuple:
    """Everything checked before the scheduler is asked anything. Returns (config, System, job id)."""
    check(job, runtime=runtime)
    campaign = Path(plan['campaign'])
    cfg = load(campaign / 'campaign.yaml')
    system = cfg.execution.hpc.system_definition()
    receipt = read(job / 'submission.json')
    job_id = environ.get(system.job_id_variable)
    if job_id != receipt['job_id']:
        raise RuntimeError(f'Running in job {job_id}, not the submitted job {receipt["job_id"]}')
    system.check_gpu_telemetry()
    return cfg, system, job_id


async def wait_for_file(path: Path, process: subprocess.Popen, what: str, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f'{what} exited during startup')
        if time.monotonic() > deadline:
            raise TimeoutError(f'{what} did not start')
        await asyncio.sleep(1)


async def wait_for_replicas(service: dict, token: str, count: int, processes: list[subprocess.Popen],
                            timeout: float, status) -> None:
    """Wait until the pool reports every replica ready; fail if any replica or the pool exits first."""
    deadline = time.monotonic() + timeout
    last = 0.0
    async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('An inference replica or the pool exited during startup')
            reply = await client.get(service['status_url'], headers={'Authorization': f'Bearer {token}'})
            reply.raise_for_status()
            ready = reply.json()['ready_replicas']
            if ready == count:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f'{ready} of {count} inference replicas ready at the startup deadline')
            if time.monotonic() - last > 30:
                status('inference_starting', inference_ready=ready, inference_total=count)
                last = time.monotonic()
            await asyncio.sleep(5)


def pending_batches(campaign: Path) -> list[str]:
    """Submitted science batches with rows that have not yet finished."""
    pending = []
    for path in campaign.glob('batches/*/batch.json'):
        batch = read(path)
        finished = sum((row / 'done').exists() and (row / 'status.json').exists() for row in path.parent.glob('run_*'))
        if finished < batch['total']:
            pending.append(batch['batch_id'])
    return pending


def report_status(job: Path, phase: str, **details) -> None:
    """Update progress from the coordinator or its controller while the other waits."""
    path = job / 'status.json'
    state = read(path) if path.exists() else {}
    state.update(phase=phase, updated_at=time.time(), **details)
    write(path, state)
    print(json.dumps({'phase': phase, **details}), flush=True)


async def run(job: Path, *, environ: Mapping[str, str] | None = None, runtime: Path = RUNTIME) -> None:
    """Run the staged job's campaign inside the allocation it was submitted as."""
    environ = os.environ if environ is None else environ
    job = Path(job).resolve()
    plan = read(job / 'plan.json')
    campaign, infra = Path(plan['campaign']), Path(plan['infra'])
    cfg, system, job_id = check_job(job, plan, environ, Path(runtime))
    allocation = plan_allocation(plan, system.allocation_hosts(environ), job_id=job_id, hostname=socket.gethostname())
    manifest = job / 'allocation.json'
    write(manifest, allocation)
    science, inference = allocation['science_nodes'], allocation['inference_nodes']
    python = plan['python']
    env = {**environ, **system.step_environment(),
           'PYTHONPATH': os.pathsep.join(filter(None, [str(runtime), environ.get('PYTHONPATH')]))}
    write(job / 'status.json', allocation)
    status = partial(report_status, job)

    children = Children(job, job_id)
    try:
        status('telemetry_starting', science_nodes=len(science))
        hosts = allocation['all_nodes']
        telemetry = children.start('telemetry', system.host_step(
            hosts, [python, '-m', 'loam.telemetry', campaign, '--allocation', manifest]), env)
        await wait_for_allocation_telemetry(telemetry, campaign, allocation, system=system, timeout=TELEMETRY_TIMEOUT)

        profile = InferenceProfile.model_validate(read(plan['inference_profile']))
        replicas = [children.start(f'replica-{i + 1:03d}', step, env)
                    for i, step in enumerate(replica_steps(system, profile, inference))]
        pool_root = infra / 'pool'
        pool_root.mkdir(exist_ok=True)
        write(pool_root / 'pool.json', pool_config(profile, inference, job_id=job_id))
        pool = children.start('pool', [python, '-m', 'loam.inference_pool', pool_root, '--host', '127.0.0.1',
                                       '--port', '0'], env)
        await wait_for_file(pool_root / 'pool-service.json', pool, 'The inference pool')
        service = read(pool_root / 'pool-service.json')
        token = Path(service['token_file']).read_text().strip()
        status('inference_starting', inference_ready=0, inference_total=len(inference))
        await wait_for_replicas(service, token, len(inference), [*replicas, pool], profile.startup_timeout, status)
        point_models_at_pool(campaign / 'campaign.yaml', plan['served_roles'], service['base_url'],
                             mcp_port=free_port())

        status('campaign_running', inference_ready=len(inference))
        controller = children.start('controller', [python, '-m', 'loam.coordinator', '--controller', job],
                                    # The controller uses no GPU: they all belong to science.
                                    {**env, POOL_TOKEN: token, 'HPC_SCIENCE_NODELIST': ','.join(science),
                                     system.gpu_pin_var: ''})
        while controller.poll() is None:
            await asyncio.sleep(5)
        if controller.returncode:
            raise RuntimeError(f'The campaign controller failed with status {controller.returncode}; see controller.log')

        status('completed', controller_returncode=0)
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error)[:2000])
        traceback.print_exc()
        raise
    finally:
        await children.stop()


async def control(job: Path, *, environ: Mapping[str, str] | None = None) -> None:
    """Run every round and drain submitted science before releasing the MCP server and its workers."""
    from .launcher import managed_hpc_server
    from .orchestrator import Campaign
    from .token_refresh import refresh_tokens

    environ = os.environ if environ is None else environ
    job = Path(job).resolve()
    plan = read(job / 'plan.json')
    campaign = Path(plan['campaign'])
    cfg = load(campaign / 'campaign.yaml')
    validate_campaign_paths(cfg, campaign)
    system = cfg.execution.hpc.system_definition()
    allocation = validate_role(job / 'allocation.json', 'science', job_id=environ.get(system.job_id_variable))
    if cfg.execution.hpc.allocation.nodes != len(allocation['science_nodes']):
        raise RuntimeError('The campaign is not sized to the allocation\'s science hosts')
    policy = configure_controller_gc(plan)
    diagnostics = job / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    write(diagnostics / 'controller-gc-policy.json', {'at': time.time(), 'controller_pid': os.getpid(), **policy})
    async with managed_hpc_server(cfg):
        await refresh_tokens(cfg)  # the allocation wait can outlive an endpoint's token
        await Campaign(cfg).run(cfg.rounds)
        deadline = time.monotonic() + DRAIN_TIMEOUT
        while pending := pending_batches(campaign):
            if time.monotonic() > deadline:
                raise TimeoutError('Submitted science did not finish: ' + ', '.join(pending))
            report_status(job, 'draining_submitted_science', pending_batches=pending)
            await asyncio.sleep(15)
        report_status(job, 'science_drained', pending_batches=[])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='python -m loam.coordinator', description=__doc__.split('\n\n')[0])
    parser.add_argument('job', type=Path, help='the staged job directory')
    parser.add_argument('--controller', action='store_true', help='run the campaign controller (started by the coordinator)')
    args = parser.parse_args(argv)
    if args.controller:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
        asyncio.run(control(args.job))
    else:
        asyncio.run(run(args.job))


if __name__ == '__main__':
    main()
