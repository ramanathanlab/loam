"""The Nemotron backend: adding and retiring replicas, the pool's view of them, and the relay."""
import asyncio
import hashlib
import json
import ssl
import stat
import time
from contextlib import asynccontextmanager

import pytest

pytest.importorskip('aiohttp', reason='needs the nemotron extra')

import aiohttp  # noqa: E402
from aiohttp import web  # noqa: E402

import loam.inference_pool  # noqa: E402
from loam import nemotron  # noqa: E402
from loam.inference_pool import InferencePool, ReplicaEndpoint  # noqa: E402

REVISION = 'f' * 40
MODEL = 'nemotron-3.5-super-vl'
CHECKPOINT = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16'


@pytest.fixture
def profile(tmp_path):
    """A serving profile whose model directory looks like a complete local snapshot."""
    snapshot = tmp_path / 'weights' / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text(json.dumps({'model_type': 'nemotron_h_omni'}))
    (snapshot / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'shard-1.safetensors'}}))
    (snapshot / 'shard-1.safetensors').write_bytes(b'weights')
    return {'model': MODEL, 'checkpoint': CHECKPOINT, 'checkpoint_revision': REVISION, 'model_dir': str(snapshot),
            'command': ['/installed/bin/vllm'], 'max_model_len': 131072, 'max_num_seqs': 16}


def add(root, profile, **kwargs):
    options = {'system': 'jupiter', 'job_id': '2979745', 'expires_at': 4e9, 'pool_address': '10.1.2.3', **kwargs}
    return nemotron.add_replicas(root, profile, **options)


def pool_json(root):
    return json.loads((root / 'pool.json').read_text())


def test_add_registers_one_replica_per_host_with_fresh_credentials(tmp_path, profile):
    root = tmp_path / 'pool'
    added = add(root, profile, hosts=['jpbo-001', 'jpbo-002'])
    config = pool_json(root)
    assert (config['model'], config['checkpoint']) == (MODEL, CHECKPOINT)
    assert [b['id'] for b in config['backends']] == ['nemotron-00', 'nemotron-01']
    assert [b['job_id'] for b in config['backends']] == ['2979745', '2979745']
    assert added == config['backends']
    tokens = set()
    for entry, host in zip(config['backends'], ['jpbo-001', 'jpbo-002']):
        attempt = root / 'replicas' / entry['attempt']
        request = json.loads((attempt / 'request.json').read_text())
        assert request['pool'] == str(root.resolve()) and request['host'] == host
        assert request['profile']['model'] == MODEL
        assert stat.S_IMODE((attempt / 'token').stat().st_mode) == 0o600
        assert stat.S_IMODE((attempt / 'tls.key').stat().st_mode) == 0o600
        assert (attempt / 'tls.pem').read_text().startswith('-----BEGIN CERTIFICATE-----')
        tokens.add((attempt / 'token').read_text())
    assert len(tokens) == 2


def test_adding_again_continues_the_numbering(tmp_path, profile):
    add(tmp_path, profile)
    add(tmp_path, profile, hosts=['a', 'b'])
    assert [b['id'] for b in pool_json(tmp_path)['backends']] == ['nemotron-00', 'nemotron-01', 'nemotron-02']


def test_a_pool_serving_another_model_refuses_the_replica(tmp_path, profile):
    (tmp_path / 'pool.json').write_text(json.dumps({'model': 'other', 'backends': []}))
    with pytest.raises(ValueError, match='model'):
        add(tmp_path, profile)
    assert not (tmp_path / 'replicas').exists()


@pytest.mark.parametrize('breakage', ['revision', 'architecture', 'shard'])
def test_an_incomplete_or_foreign_snapshot_is_refused(tmp_path, profile, breakage):
    snapshot = tmp_path / 'weights' / REVISION
    if breakage == 'revision':
        profile['checkpoint_revision'] = 'e' * 40
    elif breakage == 'architecture':
        (snapshot / 'config.json').write_text(json.dumps({'model_type': 'llama'}))
    else:
        (snapshot / 'shard-1.safetensors').write_bytes(b'')
    with pytest.raises(ValueError):
        add(tmp_path / 'pool', profile)


def test_the_pool_address_must_be_private(tmp_path, profile):
    with pytest.raises(ValueError, match='private'):
        add(tmp_path, profile, pool_address='8.8.8.8')


def ready(root, entry, **overrides):
    """Write the readiness record `loam-nemotron serve` writes once its replica answers a real completion."""
    attempt = root / 'replicas' / entry['attempt']
    request = json.loads((attempt / 'request.json').read_text())
    record = {'attempt': entry['attempt'], 'job_id': entry['job_id'], 'hostname': f"{request['host'] or 'node'}.local",
              'base_url': 'https://10.0.0.5:8443/v1', 'model': MODEL, 'checkpoint': CHECKPOINT,
              'checkpoint_revision': REVISION, 'expires_at': time.time() + 3600, **overrides}
    (attempt / 'ready.json').write_text(json.dumps(record))
    return attempt


async def refreshed(root):
    pool = InferencePool(root, resolver=nemotron.replica_resolver(root))
    await pool.refresh()
    return pool


def test_a_ready_replica_is_reached_over_pinned_tls_with_its_own_token(tmp_path, profile):
    [entry] = add(tmp_path, profile, hosts=['jpbo-001'])
    attempt = ready(tmp_path, entry)
    endpoint = nemotron.replica_resolver(tmp_path)(entry)
    assert endpoint.base_url == 'https://10.0.0.5:8443/v1'
    assert endpoint.token == (attempt / 'token').read_text().strip()
    assert endpoint.checkpoint == CHECKPOINT
    der = ssl.PEM_cert_to_DER_cert((attempt / 'tls.pem').read_text())
    assert isinstance(endpoint.tls, aiohttp.Fingerprint)
    assert endpoint.tls.fingerprint == hashlib.sha256(der).digest()
    pool = asyncio.run(refreshed(tmp_path))
    assert pool.backends['nemotron-00'].ready


@pytest.mark.parametrize('change', [
    {'expires_at': 1.0}, {'job_id': '1'}, {'attempt': 'loam-nemotron-' + 'a' * 16},
    {'model': 'other'}, {'checkpoint_revision': 'e' * 40}, {'hostname': 'jpbo-999'},
    {'base_url': 'http://10.0.0.5:8443/v1'}, {'base_url': 'https://8.8.8.8:8443/v1'},
    {'base_url': 'https://10.0.0.5/v1'}, {'base_url': 'https://u:p@10.0.0.5:8443/v1'},
])
def test_a_stale_or_mismatched_readiness_record_keeps_the_replica_out(tmp_path, profile, change):
    [entry] = add(tmp_path, profile, hosts=['jpbo-001'])
    ready(tmp_path, entry, **change)
    pool = asyncio.run(refreshed(tmp_path))
    assert not pool.backends['nemotron-00'].ready


def test_a_replica_that_has_not_reported_ready_is_not_routed(tmp_path, profile):
    add(tmp_path, profile)
    backend = asyncio.run(refreshed(tmp_path)).backends['nemotron-00']
    assert not backend.ready and backend.last_error == 'FileNotFoundError'


def test_rejects_foreign_pool_before_loading_credentials(tmp_path, profile):
    [entry] = add(tmp_path / 'foreign', profile)
    ready(tmp_path / 'foreign', entry)
    own = tmp_path / 'own'
    (own / 'replicas').mkdir(parents=True)
    (tmp_path / 'foreign' / 'replicas' / entry['attempt']).rename(own / 'replicas' / entry['attempt'])
    (own / 'replicas' / entry['attempt'] / 'token').unlink()
    with pytest.raises(RuntimeError, match='another pool'):
        nemotron.replica_resolver(own)(entry)


@pytest.mark.parametrize('attempt', ['../../elsewhere', 'loam-nemotron-zz', 7])
def test_only_attempt_names_the_backend_mints_are_read(tmp_path, attempt):
    with pytest.raises(ValueError):
        nemotron.replica_resolver(tmp_path)({'id': 'r', 'job_id': '1', 'attempt': attempt})


def allocation(tmp_path, **overrides):
    path = tmp_path / 'allocation.json'
    path.write_text(json.dumps({'mode': 'single-allocation', 'job_id': '2979745', 'all_nodes': ['s1', 'i1'],
                                'science_nodes': ['s1'], 'inference_nodes': ['i1'], **overrides}))
    return path


@pytest.mark.parametrize('host,manifest,routed', [
    ('i1', {}, True), ('s1', {}, False), ('i1', {'job_id': '1'}, False)])
def test_in_a_single_allocation_only_its_inference_hosts_serve(tmp_path, profile, host, manifest, routed):
    [entry] = add(tmp_path / 'pool', profile, hosts=[host], allocation_manifest=allocation(tmp_path, **manifest))
    ready(tmp_path / 'pool', entry)
    assert asyncio.run(refreshed(tmp_path / 'pool')).backends['nemotron-00'].ready is routed


@asynccontextmanager
async def serve(app):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    try:
        yield f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    finally:
        await runner.cleanup()


def fake_vllm(reply):
    async def models(request):
        return web.json_response({'data': [{'id': MODEL}]})

    app = web.Application()
    app.router.add_post('/v1/chat/completions', reply)
    app.router.add_get('/v1/models', models)
    return app


async def through_relay(reply, method, path, *, headers=None, allowed_address=None, **kwargs):
    """Send one request through the relay in front of a fake vLLM; returns (status, body bytes)."""
    async with serve(fake_vllm(reply)) as upstream, aiohttp.ClientSession() as session:
        async def target():
            return upstream + '/v1'

        app = nemotron.relay_app(session, 'relay-token', target, model=MODEL, allowed_address=allowed_address)
        async with serve(app) as relay:
            async with session.request(method, relay + path, headers=headers or RELAY_AUTH, **kwargs) as r:
                return r.status, await r.read()


RELAY_AUTH = {'Authorization': 'Bearer relay-token'}


def chat(**kwargs):
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': 'hi'}], **kwargs}


async def completion(request):
    return web.json_response({'model': MODEL, 'choices': [{'message': {'content': 'ok'}}]})


def test_the_relay_forwards_chat_and_streams_unchanged():
    async def stream(request):
        reply = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await reply.prepare(request)
        for part in (b'data: {"choices":[]}\n\n', b'data: [DONE]\n\n'):
            await reply.write(part)
        await reply.write_eof()
        return reply

    assert asyncio.run(through_relay(completion, 'POST', '/v1/chat/completions', json=chat()))[0] == 200
    status, data = asyncio.run(through_relay(stream, 'POST', '/v1/chat/completions', json=chat(stream=True)))
    assert status == 200 and data == b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
    assert asyncio.run(through_relay(completion, 'GET', '/v1/models'))[0] == 200


@pytest.mark.parametrize('method,path,kwargs,status', [
    ('POST', '/v1/chat/completions', {'json': chat(), 'headers': {'Authorization': 'Bearer wrong'}}, 401),
    ('POST', '/v1/chat/completions', {'json': chat(model='other')}, 400),
    ('POST', '/v1/chat/completions', {'data': b'not json'}, 400),
    ('POST', '/v1/completions', {'json': chat()}, 404),
    ('POST', '/v1/chat/completions', {'json': chat(), 'allowed_address': '10.9.9.9'}, 403),
])
def test_the_relay_refuses_what_the_pool_would_never_send(method, path, kwargs, status):
    assert asyncio.run(through_relay(completion, method, path, **kwargs))[0] == status


@pytest.mark.parametrize('format', ['openai', 'vllm', 'legacy-vllm'])
def test_the_relay_passes_on_context_overflow_numbers_but_no_other_backend_text(format):
    async def overflow(request):
        error = {'code': 'context_length_exceeded' if format == 'openai' else 400,
                 'type': 'BadRequestError', 'private': 'secret', 'message':
                 "This model's maximum context length is 100 tokens. However, your prompt "
                 'contains at least 101 input tokens. Private path: /secret/model'}
        payload = error if format == 'legacy-vllm' else {'error': error}
        return web.json_response(payload, status=400)

    async def crash(request):
        return web.json_response({'error': {'message': 'secret /path/on/node'}}, status=500)

    status, data = asyncio.run(through_relay(overflow, 'POST', '/v1/chat/completions', json=chat()))
    assert status == 400
    assert json.loads(data) == {'error': {
        'message': "This model's maximum context length is 100 tokens. However, your prompt contains "
                   'at least 101 input tokens, which exceeds this limit even without any output tokens. '
                   'Please reduce the length of the input prompt.',
        'type': 'invalid_request_error', 'code': 'context_length_exceeded', 'param': 'input_tokens'}}
    status, data = asyncio.run(through_relay(crash, 'POST', '/v1/chat/completions', json=chat()))
    assert status == 500 and b'secret' not in data


@pytest.mark.parametrize('payload', [
    b'not json: secret',
    b'["secret"]',
    b'{"error": "secret"}',
    b'{"error": {"code": 400, "message": ["secret"]}}',
    b'{"error": {"code": 400, "message": "Invalid request at /secret/model"}}',
    json.dumps({'error': {'code': 400, 'message': "This model's maximum context length is 100 tokens. "
                        'However, your prompt contains 99 input tokens. /secret/model'}}).encode(),
    json.dumps({'error': {'code': 400, 'message': "This model's maximum context length is 100 tokens. "
                        'However, your prompt contains 101 input tokens.', 'private': 'x' * 16384}}).encode(),
], ids=['invalid-json', 'non-object', 'non-object-error', 'non-string-message', 'unrelated-error',
        'within-context-limit', 'oversized-error'])
def test_the_relay_scrubs_unrecognized_or_oversized_errors(payload):
    async def error(request):
        return web.Response(body=payload, content_type='application/json', status=400)

    status, data = asyncio.run(through_relay(error, 'POST', '/v1/chat/completions', json=chat()))

    assert status == 400
    assert json.loads(data) == {'error': {'message': 'Inference backend returned HTTP 400'}}


def test_the_relay_reports_an_ended_replica_as_unavailable():
    async def run():
        async with aiohttp.ClientSession() as session:
            async def ended():
                raise RuntimeError('allocation ended')

            async with serve(nemotron.relay_app(session, 'relay-token', ended, model=MODEL)) as relay:
                async with session.post(relay + '/v1/chat/completions', json=chat(), headers=RELAY_AUTH) as r:
                    return r.status
    assert asyncio.run(run()) == 503


def test_retire_keeps_the_first_replicas_and_stops_the_rest(tmp_path, profile):
    entries = add(tmp_path, profile, hosts=['a', 'b', 'c'])
    retired = asyncio.run(nemotron.retire_replicas(tmp_path, keep=1))
    assert [r['id'] for r in retired] == ['nemotron-01', 'nemotron-02']
    assert [b['enabled'] for b in pool_json(tmp_path)['backends']] == [True, False, False]
    stops = [(tmp_path / 'replicas' / e['attempt'] / 'stop').exists() for e in entries]
    assert stops == [False, True, True]
    assert asyncio.run(nemotron.retire_replicas(tmp_path, keep=1)) == []


def test_retire_refuses_to_keep_more_replicas_than_are_running(tmp_path, profile):
    add(tmp_path, profile)
    with pytest.raises(ValueError):
        asyncio.run(nemotron.retire_replicas(tmp_path, keep=2))


def test_a_retired_replica_finishes_its_request_before_it_is_stopped(tmp_path, profile, monkeypatch):
    monkeypatch.setattr(loam.inference_pool, 'REFRESH_SECONDS', 0.05)
    entries = add(tmp_path, profile, hosts=['a', 'b'])
    stop = tmp_path / 'replicas' / entries[1]['attempt'] / 'stop'
    release = asyncio.Event()
    stopped_while_busy = []

    async def slow(request):
        await release.wait()
        stopped_while_busy.append(stop.exists())
        return web.json_response({'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]})

    async def scenario(pool, url, client):
        (tmp_path / 'pool-token').write_text('pool-token\n')
        (tmp_path / 'pool-service.json').write_text(json.dumps(
            {'status_url': url + '/pool/status', 'token_file': str(tmp_path / 'pool-token')}))
        pool.backends['nemotron-00'].cooldown_until = float('inf')  # route the request to nemotron-01

        async def ask():
            async with client.post(url + '/v1/chat/completions', json=chat(),
                                   headers={'Authorization': 'Bearer pool-token'}) as r:
                return r.status

        async def busy():
            while not pool.backends['nemotron-01'].inflight:
                await asyncio.sleep(0.01)

        request = asyncio.create_task(ask())
        await asyncio.wait_for(busy(), 5)
        retiring = asyncio.create_task(nemotron.retire_replicas(tmp_path, keep=1, poll_seconds=0.02))
        await asyncio.sleep(0.3)
        assert not retiring.done() and not stop.exists()
        release.set()
        assert await asyncio.wait_for(request, 5) == 200
        return [r['id'] for r in await asyncio.wait_for(retiring, 5)]

    async def run():
        async with serve(fake_vllm(slow)) as upstream, aiohttp.ClientSession() as client:
            pool = InferencePool(tmp_path, resolver=lambda entry: ReplicaEndpoint(upstream + '/v1', checkpoint=CHECKPOINT))
            await pool.start(token='pool-token')
            try:
                async with serve(pool.app()) as url:
                    try:
                        return await scenario(pool, url, client)
                    finally:
                        release.set()  # a failure must not leave the fake replica's handler waiting
            finally:
                await pool.close()

    assert asyncio.run(run()) == ['nemotron-01']
    assert stopped_while_busy == [False] and stop.exists()


def test_a_retire_that_failed_while_draining_is_finished_by_the_next_one(tmp_path, profile):
    entries = add(tmp_path, profile, hosts=['a', 'b'])
    (tmp_path / 'pool-token').write_text('t\n')
    (tmp_path / 'pool-service.json').write_text(json.dumps(  # a pool that has since died
        {'status_url': 'http://127.0.0.1:9/pool/status', 'token_file': str(tmp_path / 'pool-token')}))
    with pytest.raises(aiohttp.ClientError):
        asyncio.run(nemotron.retire_replicas(tmp_path, keep=1))
    (tmp_path / 'pool-service.json').unlink()
    assert [r['id'] for r in asyncio.run(nemotron.retire_replicas(tmp_path, keep=1))] == ['nemotron-01']
    assert (tmp_path / 'replicas' / entries[1]['attempt'] / 'stop').exists()


def test_retire_gives_up_on_a_pool_that_never_drains(tmp_path, profile, monkeypatch):
    add(tmp_path, profile, hosts=['a', 'b'])
    monkeypatch.setattr(nemotron, 'DRAIN_SECONDS', 0.2)

    async def run():
        async def status(request):
            return web.json_response({'backends': [{'id': 'nemotron-01', 'enabled': False, 'inflight': 1}]})

        app = web.Application()
        app.router.add_get('/pool/status', status)
        async with serve(app) as url:
            (tmp_path / 'pool-token').write_text('t\n')
            (tmp_path / 'pool-service.json').write_text(json.dumps(
                {'status_url': url + '/pool/status', 'token_file': str(tmp_path / 'pool-token')}))
            await nemotron.retire_replicas(tmp_path, keep=1, poll_seconds=0.02)

    with pytest.raises(TimeoutError):
        asyncio.run(run())
