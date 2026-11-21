"""Self-hosted Nemotron replicas as an optional backend of the inference pool (`pip install loam[nemotron]`).

Ported from the moeen branch. Each replica runs vLLM on one node behind a small relay that
accepts only the pool's host, over TLS with a certificate the pool pins, and a bearer token
minted when the replica was added. Nothing here submits or cancels jobs, nor names a scheduler:
whoever holds the allocation starts `loam-nemotron serve` on each replica's node.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web
from pydantic import Field, StrictFloat, StrictInt

from . import inference_pool
from .allocation import read_allocation, validate_role
from .config import ConfigModel, NonEmptyStr
from .inference_pool import PoolConfig, ReplicaEndpoint, safe_error, write_json_atomic
from .systems import SYSTEMS

ROUTES = {('GET', '/v1/models'), ('POST', '/v1/chat/completions')}
ATTEMPT = re.compile(r'loam-nemotron-[0-9a-f]{16}')
PROBE = 'LOAM_NEMOTRON_READY'

Positive = Annotated[StrictInt, Field(ge=1)]


class Toolchain(ConfigModel):
    """The C toolchain vLLM compiles kernels with at startup; each defaults to the one on PATH."""

    cc: NonEmptyStr | None = None
    cxx: NonEmptyStr | None = None
    ld: NonEmptyStr | None = None


class ServingProfile(ConfigModel):
    """The measured serving profile: model, weights and vLLM budgets (see docs/scaling-recipe-book.md)."""

    model: NonEmptyStr                      # served model name, which pool.json's `model` must match
    checkpoint: NonEmptyStr                 # e.g. nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16
    checkpoint_revision: Annotated[str, Field(pattern=r'^[0-9a-f]{40}$')]
    model_dir: NonEmptyStr                  # local snapshot directory, named with that revision
    command: Annotated[list[NonEmptyStr], Field(min_length=1)]   # the vLLM executable
    max_model_len: Positive
    max_num_seqs: Positive
    tensor_parallel_size: Positive = 4
    gpu_memory_utilization: Annotated[StrictFloat, Field(gt=0, lt=1)] = 0.85
    warmup_seconds: Positive = 5400
    compiler: Toolchain = Toolchain()


def server_arguments(profile: Mapping, port: int) -> list[str]:
    """The vLLM command line for `profile`, listening on 127.0.0.1:`port` (only the relay reaches it)."""
    p = ServingProfile.model_validate(profile)
    return p.command + [
        'serve', p.model_dir, '--host', '127.0.0.1', '--port', str(port),
        '--served-model-name', p.model, '--tensor-parallel-size', str(p.tensor_parallel_size),
        '--gpu-memory-utilization', str(p.gpu_memory_utilization), '--max-model-len', str(p.max_model_len),
        '--max-num-seqs', str(p.max_num_seqs), '--language-model-only', '--trust-remote-code',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'nemotron_v3',
    ]


def private_address(address: str) -> str:
    """`address`, if it is a private IPv4 address: replicas are reached only inside the System's network."""
    ip = ipaddress.IPv4Address(address)
    if not any(ip in ipaddress.IPv4Network(net) for net in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')):
        raise ValueError(f'{address} is not a private IPv4 address')
    return str(ip)


def model_snapshot(profile: ServingProfile) -> None:
    """Refuse a model directory that is not a complete local Nemotron snapshot of the pinned revision."""
    path = Path(profile.model_dir).resolve(strict=True)
    if path.name != profile.checkpoint_revision:
        raise ValueError('model_dir must be the local snapshot directory named with checkpoint_revision')
    if json.loads((path / 'config.json').read_text()).get('model_type') != 'nemotron_h_omni':
        raise ValueError('The local snapshot is not the expected Nemotron Super VL architecture')
    shards = set(json.loads((path / 'model.safetensors.index.json').read_text())['weight_map'].values())
    if not shards or any(Path(name).name != name or not (path / name).is_file() or not (path / name).stat().st_size
                         for name in shards):
        raise ValueError('The local Nemotron snapshot is missing model shards')


def _pool_config(root: Path, profile: ServingProfile) -> dict:
    path = root / 'pool.json'
    if not path.exists():
        return {'model': profile.model, 'checkpoint': profile.checkpoint, 'backends': []}
    config = PoolConfig.model_validate_json(path.read_text())
    if (config.model, config.checkpoint) != (profile.model, profile.checkpoint):
        raise ValueError(f'pool.json serves model {config.model!r} ({config.checkpoint}), '
                         f'not {profile.model!r} ({profile.checkpoint})')
    return json.loads(path.read_text())


def _new_attempt(directory: Path, request: dict) -> str:
    """A replica's private directory: its request, bearer token and self-signed TLS key pair."""
    name = 'loam-nemotron-' + secrets.token_hex(8)
    attempt = directory / name
    attempt.mkdir(mode=0o700, parents=True)
    token = attempt / 'token'
    token.touch(mode=0o600)
    token.write_text(secrets.token_urlsafe(32) + '\n')
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:P-256', '-nodes',
                    '-days', '7', '-subj', '/CN=' + name, '-keyout', str(attempt / 'tls.key'),
                    '-out', str(attempt / 'tls.pem')], check=True, capture_output=True, timeout=30)
    (attempt / 'tls.key').chmod(0o600)
    write_json_atomic(attempt / 'request.json', request)
    return name


def add_replicas(root: Path, profile: Mapping, *, system: str, job_id: str, expires_at: float, pool_address: str,
                 hosts: list[str | None] = (None,), allocation_manifest: Path | None = None) -> list[dict]:
    """Prepare one replica per host in job `job_id` and list each in ROOT/pool.json; returns their entries.

    The pool routes to a replica once `loam-nemotron serve ROOT/replicas/<attempt>` reports it ready,
    on that host, in that job, before `expires_at` (epoch seconds: when the allocation ends). The pool
    connects from `pool_address`, the only address a replica accepts requests from.
    """
    root = Path(root).resolve()
    serving = ServingProfile.model_validate(profile)
    model_snapshot(serving)
    pool_address = private_address(pool_address)
    if system not in SYSTEMS:
        raise ValueError(f'Unknown System {system!r}')
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.replicas.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = _pool_config(root, serving)
        numbers = [int(m[1]) for b in config['backends'] if (m := re.fullmatch(r'nemotron-(\d+)', b['id']))]
        first = max(numbers, default=-1) + 1
        added = []
        for index, host in enumerate(hosts, start=first):
            request = {'pool': str(root), 'system': system, 'job_id': str(job_id), 'expires_at': expires_at,
                       'host': host, 'pool_address': pool_address, 'profile': serving.model_dump(),
                       'allocation_manifest': str(Path(allocation_manifest).resolve()) if allocation_manifest else None}
            added.append({'id': f'nemotron-{index:02d}', 'job_id': str(job_id), 'enabled': True,
                          'attempt': _new_attempt(root / 'replicas', request)})
        config['backends'] += added
        write_json_atomic(root / 'pool.json', config)
    return added


def _attempt(root: Path, entry: Mapping) -> Path:
    name = entry.get('attempt')
    if not isinstance(name, str) or not ATTEMPT.fullmatch(name):
        raise ValueError('Invalid replica attempt name')
    return root / 'replicas' / name


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def readiness(attempt: Path, request: Mapping) -> dict:
    """The replica's readiness record, if it is current and names this attempt, job, host, model and weights."""
    record = _read(attempt / 'ready.json')
    profile = request['profile']
    host = request.get('host')
    if (record['attempt'] != attempt.name or record['job_id'] != request['job_id']
            or record['model'] != profile['model'] or record['checkpoint'] != profile['checkpoint']
            or record['checkpoint_revision'] != profile['checkpoint_revision']
            or (host and record['hostname'].split('.')[0] != host)
            or record['expires_at'] <= time.time()):
        raise RuntimeError('Replica readiness record is stale or mismatched')
    url = urlsplit(record['base_url'])
    if (url.scheme != 'https' or url.username or url.password or url.path != '/v1'
            or not url.port or url.query or url.fragment):
        raise ValueError('Invalid replica relay URL')
    private_address(url.hostname or '')
    return record


def pinned_tls(certificate: Path) -> aiohttp.Fingerprint:
    """Accept exactly the replica's own self-signed certificate."""
    return aiohttp.Fingerprint(hashlib.sha256(ssl.PEM_cert_to_DER_cert(certificate.read_text())).digest())


def replica_resolver(root: Path) -> Callable[[Mapping], ReplicaEndpoint]:
    """The pool's resolver for replicas added to ROOT: each must prove it belongs to this pool, runs in
    its job (and, in a single allocation, on one of its inference hosts) and is serving the pinned
    weights, before the pool loads its credentials."""
    root = Path(root).resolve()

    def resolve(entry: Mapping) -> ReplicaEndpoint:
        attempt = _attempt(root, entry)
        request = _read(attempt / 'request.json')
        if request['pool'] != str(root) or request['job_id'] != str(entry['job_id']):
            raise RuntimeError('Replica belongs to another pool or job')
        if request.get('allocation_manifest'):
            allocated = read_allocation(Path(request['allocation_manifest']))
            if allocated['job_id'] != request['job_id'] or request.get('host') not in allocated['inference_nodes']:
                raise RuntimeError('Replica is outside the allocation inference role')
        record = readiness(attempt, request)
        return ReplicaEndpoint(record['base_url'], token=(attempt / 'token').read_text().strip(),
                               tls=pinned_tls(attempt / 'tls.pem'), checkpoint=record['checkpoint'])

    return resolve


def relay_app(session: aiohttp.ClientSession, token: str, upstream, *, model: str,
              allowed_address: str | None = None) -> web.Application:
    """Forward the two OpenAI routes the pool uses (SSE included) to `await upstream()`'s
    (base_url, token, tls), for callers with `token` (and, if given, from `allowed_address`) only.

    Backend error text never passes through, except a context overflow's numbers (see safe_error).
    """
    async def forward(request: web.Request) -> web.StreamResponse:
        if (request.method, request.path) not in ROUTES:
            raise web.HTTPNotFound()
        if allowed_address and request.remote != allowed_address:
            raise web.HTTPForbidden()
        if not hmac.compare_digest(request.headers.get('Authorization', '').encode(), f'Bearer {token}'.encode()):
            raise web.HTTPUnauthorized()
        data = await request.read() if request.method == 'POST' else None
        if data is not None:
            try:
                requested = json.loads(data)['model']
            except (ValueError, KeyError, TypeError):
                raise web.HTTPBadRequest(text='Expected an OpenAI chat request') from None
            if requested != model:
                raise web.HTTPBadRequest(text=f'This replica serves {model!r}')
        reply = None
        try:
            base, upstream_token, tls = await upstream()
            headers = {'Content-Type': 'application/json'}
            if upstream_token:
                headers['Authorization'] = f'Bearer {upstream_token}'
            async with session.request(request.method, base + request.path.removeprefix('/v1'), data=data,
                                       headers=headers, ssl=tls, allow_redirects=False) as response:
                if response.status >= 300:
                    error = safe_error(response.status, await response.content.read(16385))
                    return web.json_response(error, status=response.status if response.status >= 400 else 502)
                reply = web.StreamResponse(status=response.status, headers={
                    'Content-Type': response.headers.get('Content-Type', 'application/json'),
                    'Cache-Control': 'no-cache'})
                await reply.prepare(request)
                async for chunk in response.content.iter_any():
                    await reply.write(chunk)
                await reply.write_eof()
                return reply
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError, RuntimeError):
            if reply is not None and reply.prepared:
                # The caller must see a broken stream, never a fabricated complete one.
                if request.transport is not None:
                    request.transport.close()
                return reply
            return web.json_response({'error': {'message': 'Replica unavailable'}}, status=503)

    app = web.Application(client_max_size=16 * 1024 * 1024)
    for method, path in ROUTES:
        app.router.add_route(method, path, forward)
    return app


async def _drained(root: Path, ids: set[str], poll_seconds: float) -> None:
    """Wait until the running pool (if any) has stopped routing to `ids` and has nothing in flight there."""
    service = root / 'pool-service.json'
    if not service.exists():
        return
    record = _read(service)
    headers = {'Authorization': 'Bearer ' + Path(record['token_file']).read_text().strip()}
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=15)) as session:
        while True:
            async with session.get(record['status_url'], headers=headers) as response:
                response.raise_for_status()
                backends = {b['id']: b for b in (await response.json())['backends']}
            if all(not backends[i]['enabled'] and not backends[i]['inflight'] for i in ids if i in backends):
                return
            await asyncio.sleep(poll_seconds)


async def retire_replicas(root: Path, *, keep: int, poll_seconds: float = 2) -> list[dict]:
    """Keep the first `keep` enabled replicas in ROOT/pool.json and retire the rest; returns their entries.

    A retired replica is disabled in pool.json first. Once the running pool has finished every request
    it sent there, the replica's `stop` file tells its `loam-nemotron serve` to shut vLLM down; the
    caller then releases its node however its scheduler does.
    """
    root = Path(root).resolve()
    with (root / '.replicas.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = _read(root / 'pool.json')
        active = [b for b in config['backends'] if b.get('enabled', True)]
        if not 0 <= keep <= len(active):
            raise ValueError(f'Cannot keep {keep} of {len(active)} enabled replicas')
        retired = active[keep:]
        for entry in retired:
            entry['enabled'] = False
        write_json_atomic(root / 'pool.json', config)
    await _drained(root, {entry['id'] for entry in retired}, poll_seconds)
    for entry in retired:
        (_attempt(root, entry) / 'stop').touch()
    return retired


def native_runtime_env(profile: ServingProfile, inherited: Mapping[str, str]) -> dict[str, str]:
    """`inherited`, with vLLM's directory and the toolchain first on PATH, and CC/CXX set."""
    env = dict(inherited)
    tools = {}
    for name, executable in (('cc', 'gcc'), ('cxx', 'g++'), ('ld', 'ld')):
        found = getattr(profile.compiler, name) or shutil.which(executable, path=env.get('PATH'))
        if not found or not os.access(found, os.X_OK):
            raise RuntimeError(f'The native vLLM toolchain is missing {name} ({found or executable})')
        tools[name] = str(Path(found).resolve())
    directories = [str(Path(profile.command[0]).parent), *(str(Path(t).parent) for t in tools.values()),
                   *env.get('PATH', os.defpath).split(os.pathsep)]
    env['PATH'] = os.pathsep.join(dict.fromkeys(directories))
    env.update(CC=tools['cc'], CXX=tools['cxx'])
    library = Path(tools['cc']).parent.parent / 'lib64'
    if library.is_dir():
        env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, [env.get('LD_LIBRARY_PATH'), str(library)]))
    return env


def check_compiler(env: Mapping[str, str]) -> None:
    """Compile and run a tiny program before spending the warm-up on loading weights."""
    with tempfile.TemporaryDirectory(prefix='loam-nemotron-compiler-') as directory:
        source, binary = Path(directory) / 'probe.c', Path(directory) / 'probe'
        source.write_text('int main(void) { return 0; }\n')
        result = subprocess.run([env['CC'], str(source), '-o', str(binary)], env=env,
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f'Native vLLM compiler check failed: {result.stderr[-2000:]}')
        subprocess.run([str(binary)], env=env, check=True, capture_output=True, timeout=10)


async def completion_check(session: aiohttp.ClientSession, base: str, model: str, token: str = '', *,
                           tls=None, timeout: float = 120) -> None:
    """Readiness means a real completion from the expected model, not an open port."""
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    body = {'model': model, 'messages': [{'role': 'user', 'content': f'Reply exactly: {PROBE}'}],
            'max_tokens': 128, 'chat_template_kwargs': {'enable_thinking': False}}
    async with session.post(base + '/chat/completions', json=body, headers=headers, ssl=tls,
                            timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=False) as response:
        response.raise_for_status()
        result = await response.json()
    if result.get('model') != model or PROBE not in (result['choices'][0]['message'].get('content') or ''):
        raise RuntimeError('Replica did not return the expected model and warm-up response')


def _check_placement(request: Mapping) -> str:
    """This process's job id, if it runs in the replica's job (and, in a single allocation, on its host)."""
    job_id = os.environ.get(SYSTEMS[request['system']].job_id_variable)
    if job_id != request['job_id']:
        raise RuntimeError('The replica must run in the job it was added for')
    host = socket.gethostname().split('.')[0]
    if request.get('allocation_manifest'):
        validate_role(Path(request['allocation_manifest']), 'inference', job_id=job_id)
    if request.get('host') and host != request['host']:
        raise RuntimeError('The replica must run on the host it was added for')
    return job_id


def _reserve_port(address: str) -> socket.socket:
    listener = socket.socket()
    try:
        listener.bind((address, 0))
        return listener
    except BaseException:
        listener.close()
        raise


async def serve(attempt: Path) -> None:
    """Run the replica added as ATTEMPT: start vLLM, wait for a real completion, then serve it through the
    relay over TLS to the pool's address only, until the allocation ends, vLLM exits or the replica is retired."""
    attempt = Path(attempt).resolve()
    request = _read(attempt / 'request.json')
    profile = ServingProfile.model_validate(request['profile'])
    job_id = _check_placement(request)
    expires_at, pool_address = request['expires_at'], private_address(request['pool_address'])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((pool_address, 1))
        address = private_address(route.getsockname()[0])
    with _reserve_port('127.0.0.1') as listener:
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}/v1'
    env = native_runtime_env(profile, {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    check_compiler(env)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)

    def ending() -> bool:
        return stop.is_set() or (attempt / 'stop').exists() or time.time() >= expires_at

    with (attempt / 'vllm.log').open('ab') as log:
        process = subprocess.Popen(server_arguments(request['profile'], port), env=env, stdout=log, stderr=log,
                                   start_new_session=True)
    try:
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(
                total=None, connect=10, sock_read=600)) as session:
            deadline = min(time.time() + profile.warmup_seconds, expires_at)
            while True:
                if ending():
                    return
                if process.poll() is not None:
                    raise RuntimeError(f'vLLM exited with code {process.returncode}. See vllm.log.')
                if time.time() >= deadline:
                    raise TimeoutError('vLLM did not pass a real completion before the startup deadline')
                try:
                    await completion_check(session, base, profile.model, timeout=min(60, max(1, deadline - time.time())))
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(5)

            async def upstream():
                if process.poll() is not None or time.time() >= expires_at:
                    raise RuntimeError('Replica allocation ended')
                return base, '', None

            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(attempt / 'tls.pem', attempt / 'tls.key')
            token = (attempt / 'token').read_text().strip()
            runner = web.AppRunner(relay_app(session, token, upstream, model=profile.model,
                                             allowed_address=pool_address), access_log=None)
            await runner.setup()
            with _reserve_port(address) as listener:
                await web.SockSite(runner, listener, ssl_context=tls).start()
                write_json_atomic(attempt / 'ready.json', {
                    'attempt': attempt.name, 'job_id': job_id, 'hostname': socket.gethostname(),
                    'base_url': f'https://{address}:{listener.getsockname()[1]}/v1',
                    'model': profile.model, 'checkpoint': profile.checkpoint,
                    'checkpoint_revision': profile.checkpoint_revision, 'expires_at': expires_at,
                    'ready_at': time.time()})
                print(f'Nemotron replica {attempt.name} ready in job {job_id}', flush=True)
                try:
                    while not ending() and process.poll() is None:
                        await asyncio.sleep(2)
                finally:
                    (attempt / 'ready.json').unlink(missing_ok=True)
                    await runner.cleanup()
            if process.poll() is not None and not ending():
                raise RuntimeError(f'vLLM exited with code {process.returncode}. See vllm.log.')
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.to_thread(process.wait, 20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                await asyncio.to_thread(process.wait)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='loam-nemotron', description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='action', required=True)
    add = commands.add_parser('add', help='prepare replicas in a running job and list them in ROOT/pool.json')
    add.add_argument('root', type=Path)
    add.add_argument('--profile', required=True, type=Path, help='serving profile JSON (see ServingProfile)')
    add.add_argument('--system', required=True, choices=sorted(SYSTEMS))
    add.add_argument('--job-id', required=True)
    add.add_argument('--expires-at', required=True, type=float, help='epoch seconds when the job ends')
    add.add_argument('--pool-address', required=True, help='private IPv4 address the pool connects from')
    add.add_argument('--host', action='append', dest='hosts', help='one replica per host (repeat)')
    add.add_argument('--allocation-manifest', type=Path, help='single-allocation manifest naming inference hosts')
    run = commands.add_parser('serve', help="run one replica on its node (prints nothing until it is ready)")
    run.add_argument('attempt', type=Path, help='ROOT/replicas/<attempt>')
    retire = commands.add_parser('retire', help='drain and stop all but the first KEEP replicas')
    retire.add_argument('root', type=Path)
    retire.add_argument('--keep', required=True, type=int)
    pool = commands.add_parser('pool', help='serve ROOT/pool.json over these replicas (see loam-inference-pool)')
    pool.add_argument('root', type=Path)
    pool.add_argument('--host', default='127.0.0.1')
    pool.add_argument('--port', type=int, default=0, help='0 picks a free port')
    args = parser.parse_args(argv)
    try:
        if args.action == 'add':
            added = add_replicas(args.root, json.loads(args.profile.read_text()), system=args.system,
                                 job_id=args.job_id, expires_at=args.expires_at, pool_address=args.pool_address,
                                 hosts=args.hosts or [None], allocation_manifest=args.allocation_manifest)
            print(json.dumps(added))
        elif args.action == 'retire':
            print(json.dumps(asyncio.run(retire_replicas(args.root, keep=args.keep))))
        elif args.action == 'pool':
            asyncio.run(inference_pool.serve(args.root, args.host, args.port, resolver=replica_resolver(args.root)))
        else:
            try:
                asyncio.run(serve(args.attempt))
            except Exception as error:
                write_json_atomic(args.attempt / 'failure.json', {'error': type(error).__name__, 'at': time.time()})
                raise
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'loam-nemotron: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
