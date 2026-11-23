"""Route OpenAI-compatible chat requests across a pool of inference replicas.

The pool serves one model to campaign agents from one URL. It forwards each request to a replica
unchanged (thinking, tools and token budgets are never rewritten), with per-replica capacity,
queue backpressure, cooldown after a failure, and failover for requests that have not yet started
streaming. The backend serving each replica (vLLM, SGLang, a hosted endpoint) does its own
continuous batching; `POST /pool/batch` only fans independent requests out, it never merges them.

`pool.json` in the pool's root directory configures it, and is re-read every couple of seconds so
replicas can be added or retired while the pool runs:

    {"model": "...", "checkpoint": "...", "policy": "least_loaded_affinity",
     "max_inflight_per_backend": 16,
     "backends": [{"id": "r0", "base_url": "http://host:8000/v1", "token_file": "r0.token"}]}

`model` is the served model name every request must name; `checkpoint` (optional) names the
weights, and a replica whose endpoint reports other weights (or none) is never sent requests.
Neither can change while the pool runs. How a backend entry becomes a replica endpoint is the
resolver's job. The default takes `base_url`, credentials and `checkpoint` from the entry itself,
so for it the checkpoint is what the operator declares, not something the replica proves. An
optional backend (e.g. self-hosted replicas that must prove they belong to this campaign and which
weights they loaded) passes its own resolver, closing over the pool root if it needs files there.
Nothing here knows which System it runs on.

The pool appends one timing record per attempt to `pool-request-timings.jsonl`, without prompt,
completion or credential contents, and never passes a backend's own error text back to callers.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import signal
import ssl
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import aiohttp
from aiohttp import web
from pydantic import ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator

from .config import ConfigModel, NonEmptyStr

RETRYABLE = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 3
QUEUE_TIMEOUT = 590
REFRESH_SECONDS = 2
REPLICA_HEADER = 'X-Loam-Replica'
RESOLVE_ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError)


class PoolBackendEntry(ConfigModel):
    """One replica in pool.json. Keys beyond these belong to the resolver (the default reads
    `base_url`, `token_file`, `token_env` and `checkpoint`), so they are kept, not refused."""

    model_config = ConfigDict(extra='allow')

    id: NonEmptyStr
    # A scheduler job id is often written as a number; it only marks when an id is reused.
    job_id: StrictStr | StrictInt = ''
    enabled: StrictBool = True


class PoolConfig(ConfigModel):
    """pool.json, re-read while the pool runs."""

    model: NonEmptyStr
    checkpoint: NonEmptyStr | None = None
    policy: Literal['round_robin', 'least_loaded', 'least_loaded_affinity'] = 'least_loaded_affinity'
    max_inflight_per_backend: Annotated[StrictInt, Field(ge=1, le=64)] = 16
    # A free-form label (e.g. the campaign stage) copied into each timing record.
    phase: Annotated[str, Field(max_length=128)] = 'unassigned'
    backends: list[PoolBackendEntry]

    @field_validator('backends')
    @classmethod
    def ids_are_unique(cls, backends):
        ids = [b.id for b in backends]
        if len(set(ids)) != len(ids):
            raise ValueError('backend ids must be unique')
        return backends


@dataclass(frozen=True)
class ReplicaEndpoint:
    """Where one replica serves the OpenAI-compatible API, and how to reach it."""

    base_url: str
    token: str = field(default='', repr=False)
    tls: ssl.SSLContext | aiohttp.Fingerprint | bool | None = field(default=None, repr=False)
    checkpoint: str | None = None


Resolver = Callable[[Mapping], ReplicaEndpoint]


def listed_endpoint(entry: Mapping, root: Path | None = None) -> ReplicaEndpoint:
    """The default resolver: the entry names its own `base_url`, and optionally a bearer token
    through `token_file` (relative to the pool root) or `token_env`, and the `checkpoint` it serves.
    """
    base_url = entry['base_url']
    if not isinstance(base_url, str) or not re.match(r'https?://', base_url):
        raise ValueError('base_url must be an http(s) URL')
    token = ''
    if entry.get('token_file'):
        token = (Path(root or '.') / entry['token_file']).read_text().strip()
    elif entry.get('token_env'):
        token = os.environ[entry['token_env']]
    return ReplicaEndpoint(base_url.rstrip('/'), token=token, checkpoint=entry.get('checkpoint'))


@dataclass
class Backend:
    name: str
    job_id: str = ''
    endpoint: ReplicaEndpoint | None = field(default=None, repr=False)
    enabled: bool = True
    ready: bool = False
    capacity: int = 16
    inflight: int = 0
    estimated_tokens: int = 0
    completed: int = 0
    failures: int = 0
    cooldown_until: float = 0
    last_error: str | None = None

    def available(self, now):
        return self.enabled and self.ready and now >= self.cooldown_until


def request_key(body):
    """Hash a request's stable task identity for cache affinity; never keep the prompt text."""
    messages = body.get('messages', [])
    user = next((m.get('content', '') for m in messages if m.get('role') == 'user'), '')
    system = next((m.get('content', '') for m in messages if m.get('role') == 'system'), '')
    raw = json.dumps([system, user], sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def safe_error(status, payload):
    """The error a caller sees: a context-length error keeps its numbers, so the caller can shrink
    the prompt; any other backend text is dropped."""
    fallback = {'error': {'message': f'Inference backend returned HTTP {status}'}}
    if status != 400 or len(payload) > 16384:
        return fallback
    try:
        data = json.loads(payload)
    except ValueError:
        return fallback
    error = data.get('error', data) if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return fallback
    message = error.get('message', '')
    message = message if isinstance(message, str) else ''
    context = re.search(r'maximum context length is\s+(\d{1,10})\s+tokens', message, re.I)
    count = re.search(r'prompt contains\s+(at least\s+)?(\d{1,10})\s+input tokens', message, re.I)
    detail = 'Maximum context length exceeded. Please reduce the length of the input prompt.'
    # Native vLLM uses numeric code 400; only recognized overflow counts survive the filter.
    if context and count and int(count[2]) > int(context[1]):
        qualifier = 'at least ' if count[1] else ''
        detail = (f"This model's maximum context length is {int(context[1])} tokens. However, your prompt "
                  f'contains {qualifier}{int(count[2])} input tokens, which exceeds this limit even without any output '
                  'tokens. Please reduce the length of the input prompt.')
    elif error.get('code') != 'context_length_exceeded':
        return fallback
    return {'error': {'message': detail, 'type': 'invalid_request_error',
                      'code': 'context_length_exceeded', 'param': 'input_tokens'}}


def unavailable(message):
    return web.json_response({'error': {'message': message}}, status=503, headers={'Retry-After': '5'})


class StreamAccounting:
    """Usage, finish reason and time to first token, read from an SSE stream as it passes."""

    def __init__(self):
        self.buffer = b''
        self.usage = {}
        self.finish_reason = None
        self.first_token_at = None

    def feed(self, chunk):
        self.buffer += chunk
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            if not line.startswith(b'data:'):
                continue
            try:
                data = json.loads(line[5:].strip())
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            if data.get('usage'):
                self.usage = data['usage']
            for choice in data.get('choices', []):
                delta = choice.get('delta') or {}
                if self.first_token_at is None and any(
                        delta.get(k) for k in ('content', 'reasoning', 'reasoning_content', 'tool_calls')):
                    self.first_token_at = time.time()
                if choice.get('finish_reason'):
                    self.finish_reason = choice['finish_reason']
        if len(self.buffer) > 1024 * 1024:
            self.buffer = b''


class InferencePool:
    def __init__(self, root: Path, *, resolver: Resolver | None = None, session: aiohttp.ClientSession | None = None):
        self.root = Path(root).resolve()
        self.resolver = resolver or (lambda entry: listed_endpoint(entry, self.root))
        self.session = session
        self.owns_session = False
        self.model: str | None = None
        self.checkpoint: str | None = None
        self.policy = 'least_loaded_affinity'
        self.phase = 'unassigned'
        self.backends: dict[str, Backend] = {}
        self.condition = asyncio.Condition()
        self.affinity = OrderedDict()
        self.cursor = 0
        self.metrics = Counter()
        self.refresh_task = None
        self.token = ''
        self.last_config_error = None

    def validate(self, body):
        if not isinstance(body, dict) or body.get('model') != self.model or not isinstance(body.get('messages'), list):
            raise web.HTTPBadRequest(text=f'Expected a chat request for model {self.model!r} with a messages list')

    async def refresh(self):
        config = PoolConfig.model_validate_json((self.root / 'pool.json').read_text())
        if self.model is not None and (config.model, config.checkpoint) != (self.model, self.checkpoint):
            raise ValueError('pool.json cannot change the model or checkpoint of a running pool')
        self.model, self.checkpoint = config.model, config.checkpoint
        self.policy, self.phase = config.policy, config.phase
        seen = set()
        for item in config.backends:
            entry = item.model_dump()
            name, job = item.id, str(item.job_id)
            seen.add(name)
            if name in self.backends and self.backends[name].job_id != job:
                if self.backends[name].inflight:
                    raise RuntimeError('Cannot replace a busy backend identity')
                del self.backends[name]
            backend = self.backends.setdefault(name, Backend(name, job))
            backend.enabled = item.enabled
            backend.capacity = config.max_inflight_per_backend
            try:
                # A resolver may read files or load TLS material; with many replicas that must not
                # stall the event loop while it serves campaign requests.
                endpoint = await asyncio.to_thread(self.resolver, entry)
                if not isinstance(endpoint, ReplicaEndpoint):
                    raise TypeError('A resolver must return a ReplicaEndpoint')
                if self.checkpoint is not None and endpoint.checkpoint != self.checkpoint:
                    raise RuntimeError('Replica serves a different checkpoint')
            except RESOLVE_ERRORS as error:
                backend.ready = False
                backend.last_error = type(error).__name__
            else:
                backend.endpoint = endpoint
                backend.ready = True
                backend.last_error = None
        for name, backend in self.backends.items():
            if name not in seen:
                backend.enabled = False
        self.last_config_error = None
        async with self.condition:
            self.condition.notify_all()

    async def watch(self):
        while True:
            await asyncio.sleep(REFRESH_SECONDS)
            try:
                await self.refresh()
            except RESOLVE_ERRORS as error:
                self.last_config_error = type(error).__name__
                for backend in self.backends.values():
                    backend.ready = False

    async def acquire(self, body, *, excluded=(), timeout=QUEUE_TIMEOUT):
        key = request_key(body)
        estimate = max(1, len(json.dumps(body.get('messages', []))) // 4)
        estimate += min(int(body.get('max_tokens', body.get('max_completion_tokens', 8192)) or 8192), 4096)
        started = time.time()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            async with self.condition:
                now = time.time()
                possible = [b for b in self.backends.values() if b.name not in excluded and b.enabled and b.ready]
                if not possible:
                    raise RuntimeError('No ready inference replicas')
                choices = sorted((b for b in possible if b.available(now) and b.inflight < b.capacity),
                                 key=lambda b: b.name)
                if choices:
                    selected, hit = self.select(choices, key)
                    selected.inflight += 1
                    selected.estimated_tokens += estimate
                    self.affinity[key] = selected.name
                    self.affinity.move_to_end(key)
                    while len(self.affinity) > 10000:
                        self.affinity.popitem(last=False)
                    return selected, estimate, time.time() - started, hit
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError('Inference pool backpressure deadline exceeded')
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=min(remaining, 2))
                except TimeoutError:
                    pass

    def select(self, choices, key):
        if self.policy == 'round_robin':
            selected = choices[self.cursor % len(choices)]
            self.cursor += 1
            return selected, False

        def score(b):
            return (b.inflight + .2 * b.estimated_tokens / 32768, b.completed + b.failures, b.name)

        selected = min(choices, key=score)
        sticky = self.affinity.get(key)
        if self.policy == 'least_loaded_affinity' and sticky:
            preferred = next((b for b in choices if b.name == sticky), None)
            if preferred and score(preferred)[0] <= score(selected)[0] + 1:
                return preferred, True
        return selected, False

    async def release(self, backend, estimate):
        async with self.condition:
            backend.inflight -= 1
            backend.estimated_tokens -= estimate
            assert backend.inflight >= 0 and backend.estimated_tokens >= 0
            self.condition.notify_all()

    def record(self, item):
        self.metrics[str(item['status'])] += 1
        with (self.root / 'pool-request-timings.jsonl').open('a') as stream:
            stream.write(json.dumps(item, separators=(',', ':')) + '\n')

    def another_available(self, excluded=()):
        """Whether any replica outside `excluded` could take a request now."""
        now = time.time()
        return any(b.available(now) and b.name not in excluded for b in self.backends.values())

    async def dispatch(self, body, client_request=None):
        self.validate(body)
        stream = bool(body.get('stream', False))
        if stream and client_request is None:
            raise web.HTTPBadRequest(text='Batch fan-out requires non-streaming requests')
        excluded = set()
        request_id = uuid.uuid4().hex
        whole_start = time.time()
        for attempt_number in range(1, MAX_ATTEMPTS + 1):
            backend = reply = None
            estimate = 0
            started = time.time()
            accounting = StreamAccounting()
            record = {'request_id': request_id, 'attempt': attempt_number, 'started_at': started,
                      'policy': self.policy, 'phase': self.phase,
                      'thinking_enabled': (body.get('chat_template_kwargs') or {}).get('enable_thinking', True),
                      'stream': stream, 'status': 503,
                      'prompt_bytes': len(json.dumps(body.get('messages', [])).encode())}
            try:
                backend, estimate, queued, hit = await self.acquire(body, excluded=excluded)
                excluded.add(backend.name)
                record.update(backend=backend.name, job_id=backend.job_id, queue_seconds=queued, affinity_hit=hit)
                endpoint = backend.endpoint
                headers = {'Content-Type': 'application/json'}
                if endpoint.token:
                    headers['Authorization'] = 'Bearer ' + endpoint.token
                async with self.session.post(endpoint.base_url + '/chat/completions', json=body, headers=headers,
                                             ssl=endpoint.tls, allow_redirects=False) as response:
                    status = response.status
                    record['status'] = status
                    record['headers_seconds'] = time.time() - started
                    if status >= 300:
                        error = safe_error(status, await response.content.read(16385))
                        if status in RETRYABLE:
                            try:
                                delay = float(response.headers.get('Retry-After', '5'))
                            except ValueError:
                                delay = 5
                            backend.cooldown_until = time.time() + max(1, min(120, delay)) + secrets.randbelow(1000) / 1000
                            backend.failures += 1
                            if self.another_available(excluded):
                                continue
                        return web.json_response(error, status=status if status >= 400 else 502,
                                                 headers={'Retry-After': '5'} if status == 429 else None)
                    if stream:
                        reply = web.StreamResponse(status=200, headers={
                            'Content-Type': response.headers.get('Content-Type', 'text/event-stream'),
                            'Cache-Control': 'no-cache', REPLICA_HEADER: backend.name})
                        await reply.prepare(client_request)
                        async for chunk in response.content.iter_any():
                            accounting.feed(chunk)
                            await reply.write(chunk)
                        await reply.write_eof()
                    else:
                        payload = await response.read()
                        try:
                            result = json.loads(payload)
                            accounting.usage = result.get('usage') or {}
                            accounting.finish_reason = next(
                                (c.get('finish_reason') for c in result.get('choices', []) if c.get('finish_reason')), None)
                        except (ValueError, AttributeError):
                            raise RuntimeError('Backend did not return a JSON completion') from None
                        reply = web.Response(body=payload, status=200, content_type='application/json',
                                             headers={REPLICA_HEADER: backend.name})
                    backend.completed += 1
                    return reply
            except asyncio.CancelledError:
                record.update(status=499, error_type='CancelledError')
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError, ValueError) as error:
                record['error_type'] = type(error).__name__
                record['status'] = 503
                if backend:
                    backend.failures += 1
                    backend.cooldown_until = time.time() + 5
                # A stream that has started emitting is never replayed on another replica.
                if reply is not None and reply.prepared:
                    if client_request.transport is not None:
                        client_request.transport.close()
                    return reply
                if backend is None or not self.another_available(excluded):
                    return unavailable('Inference pool unavailable')
            finally:
                usage = accounting.usage
                record.update(finished_at=time.time(), seconds=time.time() - started,
                              total_request_seconds=time.time() - whole_start,
                              time_to_first_token_seconds=(accounting.first_token_at - started)
                              if accounting.first_token_at else None,
                              finish_reason=accounting.finish_reason,
                              prompt_tokens=usage.get('prompt_tokens'),
                              completion_tokens=usage.get('completion_tokens'),
                              reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
                self.record(record)
                if backend:
                    await self.release(backend, estimate)
        return unavailable('Inference replicas failed')

    def status(self):
        now = time.time()
        return {'model': self.model, 'checkpoint': self.checkpoint, 'policy': self.policy, 'phase': self.phase,
                'time': now, 'config_error': self.last_config_error,
                'ready_replicas': sum(b.available(now) for b in self.backends.values()),
                'request_attempt_statuses': dict(self.metrics),
                'backends': [{'id': b.name, 'job_id': b.job_id, 'enabled': b.enabled, 'ready': b.ready,
                              'available': b.available(now), 'capacity': b.capacity, 'inflight': b.inflight,
                              'estimated_tokens': b.estimated_tokens, 'completed': b.completed,
                              'failed_attempts': b.failures, 'cooldown_seconds': max(0, b.cooldown_until - now),
                              'readiness_error': b.last_error}
                             for b in self.backends.values()]}

    def app(self):
        @web.middleware
        async def authorize(request, handler):
            expected = ('Bearer ' + self.token).encode()
            if not hmac.compare_digest(request.headers.get('Authorization', '').encode(), expected):
                raise web.HTTPUnauthorized()
            return await handler(request)

        async def models(request):
            if not self.another_available():
                return unavailable('Inference replicas still warming')
            return web.json_response({'object': 'list', 'data': [
                {'id': self.model, 'object': 'model', 'owned_by': 'loam', 'root': self.checkpoint or self.model}]})

        async def complete(request):
            try:
                body = await request.json()
            except ValueError:
                raise web.HTTPBadRequest(text='Expected JSON') from None
            return await self.dispatch(body, request)

        async def batch(request):
            try:
                items = (await request.json())['items']
            except (ValueError, KeyError, TypeError):
                raise web.HTTPBadRequest(text='Expected items with custom_id and body') from None
            if not isinstance(items, list) or not 1 <= len(items) <= 64:
                raise web.HTTPBadRequest(text='Expected one to sixty-four independent requests')
            ids = [x.get('custom_id') for x in items if isinstance(x, dict)]
            if len(ids) != len(items) or any(not isinstance(k, str) or len(k) > 128 for k in ids) or len(set(ids)) != len(ids):
                raise web.HTTPBadRequest(text='custom_id values must be unique strings')
            for item in items:
                self.validate(item.get('body'))
                if item['body'].get('stream'):
                    raise web.HTTPBadRequest(text='Batch items must not stream')
            semaphore = asyncio.Semaphore(max(1, min(32, sum(b.capacity for b in self.backends.values()))))

            async def one(item):
                async with semaphore:
                    response = await self.dispatch(item['body'])
                    return {'custom_id': item['custom_id'], 'status': response.status,
                            'response': json.loads(response.body)}

            return web.json_response({'items': await asyncio.gather(*(one(item) for item in items))})

        async def status(request):
            return web.json_response(self.status())

        application = web.Application(client_max_size=16 * 1024 * 1024, middlewares=[authorize])
        application.router.add_get('/v1/models', models)
        application.router.add_post('/v1/chat/completions', complete)
        application.router.add_get('/pool/status', status)
        application.router.add_post('/pool/batch', batch)
        return application

    async def start(self, *, token=None):
        """Read pool.json, open the outbound session and start watching for config changes.

        Without an explicit token, callers authenticate with the one in `<root>/pool-token`,
        created (mode 0600) on first start.
        """
        if token is None:
            path = self.root / 'pool-token'
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'w') as stream:
                    stream.write(secrets.token_urlsafe(32) + '\n')
            except FileExistsError:
                pass
            token = path.read_text().strip()
        self.token = token
        await self.refresh()
        if self.session is None:
            # Keep the small-pool budget, but let a large pool reach its configured capacity
            # (102 replicas x 16 in flight = 1632) instead of capping it at 256 connections.
            capacity = sum(b.capacity for b in self.backends.values() if b.enabled)
            self.session = aiohttp.ClientSession(
                trust_env=False, timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=600),
                connector=aiohttp.TCPConnector(limit=max(256, min(2048, capacity)), limit_per_host=64))
            self.owns_session = True
        self.refresh_task = asyncio.create_task(self.watch())

    async def close(self):
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        if self.owns_session:
            await self.session.close()


def write_json_atomic(path: Path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


async def serve(root: Path, host: str, port: int, *, resolver: Resolver | None = None):
    """Run the pool until SIGINT/SIGTERM. A backend with its own resolver serves through this too."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    pool = InferencePool(root, resolver=resolver)
    await pool.start()
    runner = web.AppRunner(pool.app(), access_log=None)
    await runner.setup()
    try:
        site = web.TCPSite(runner, host, port)
        await site.start()
        actual = site._server.sockets[0].getsockname()[1]
        base = f'http://{host}:{actual}'
        write_json_atomic(pool.root / 'pool-service.json', {
            'pid': os.getpid(), 'model': pool.model, 'base_url': base + '/v1', 'status_url': base + '/pool/status',
            'batch_url': base + '/pool/batch', 'token_file': str(pool.root / 'pool-token')})
        print(json.dumps({'pool_url': base + '/v1', 'pid': os.getpid()}), flush=True)
        await stop.wait()
    finally:
        await runner.cleanup()
        await pool.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='loam-inference-pool',
        description='Serve the replicas listed in ROOT/pool.json as one OpenAI-compatible endpoint. '
                    'The URL and token file are written to ROOT/pool-service.json.')
    parser.add_argument('root', type=Path, help='directory holding pool.json')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0, help='0 picks a free port')
    args = parser.parse_args(argv)
    asyncio.run(serve(args.root, args.host, args.port))


if __name__ == '__main__':
    main()
