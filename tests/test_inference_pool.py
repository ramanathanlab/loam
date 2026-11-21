"""The inference pool, driven over HTTP against fake OpenAI-compatible backends."""
import asyncio
import json
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

import loam.inference_pool
from loam.inference_pool import InferencePool, ReplicaEndpoint, safe_error

MODEL = 'served-model'
AUTH = {'Authorization': 'Bearer test-client'}


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


def fake_backend(handler, name):
    app = web.Application()

    async def invoke(request):
        return await handler(name, request)

    app.router.add_post('/v1/chat/completions', invoke)
    return app


def ok(content='ok', **extra):
    return web.json_response({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}], **extra})


@asynccontextmanager
async def pool_over(root, handler, *, capacity=1, policy='least_loaded', **config):
    """Two fake backends listed by base_url in pool.json, behind a started pool."""
    async with serve(fake_backend(handler, 'a')) as a, serve(fake_backend(handler, 'b')) as b:
        config = {'model': MODEL, 'policy': policy, 'max_inflight_per_backend': capacity,
                  'backends': [{'id': 'a', 'base_url': a + '/v1'}, {'id': 'b', 'base_url': b + '/v1'}], **config}
        (root / 'pool.json').write_text(json.dumps(config))
        pool = InferencePool(root)
        await pool.start(token='test-client')
        try:
            async with serve(pool.app()) as url, aiohttp.ClientSession() as client:
                yield pool, url, client
        finally:
            await pool.close()


def body(i=0, **kwargs):
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': f'task{i}'}], 'max_tokens': 8192,
            'chat_template_kwargs': {'enable_thinking': False}, **kwargs}


def timings(root):
    return [json.loads(line) for line in (root / 'pool-request-timings.jsonl').read_text().splitlines()]


def test_bounded_fanout_uses_both_replicas_and_preserves_bodies(tmp_path):
    async def run():
        active = {'a': 0, 'b': 0}
        peaks = {'a': 0, 'b': 0}
        seen = []

        async def handler(name, request):
            data = await request.json()
            seen.append(data)
            active[name] += 1
            peaks[name] = max(peaks[name], active[name])
            await asyncio.sleep(.02)
            active[name] -= 1
            return ok(usage={'prompt_tokens': 8, 'completion_tokens': 2})

        async with pool_over(tmp_path, handler) as (pool, url, client):
            requests = [body(i, tools=[{'type': 'function', 'function': {'name': 'submit'}}]) for i in range(8)]

            async def one(data):
                async with client.post(url + '/v1/chat/completions', json=data, headers=AUTH) as r:
                    assert r.status == 200
                    return await r.json()

            await asyncio.gather(*(one(data) for data in requests))
            assert peaks == {'a': 1, 'b': 1}
            assert pool.backends['a'].completed and pool.backends['b'].completed
            assert sorted(seen, key=lambda x: x['messages'][0]['content']) == requests
            assert not any(b.inflight or b.estimated_tokens for b in pool.backends.values())
            logs = timings(tmp_path)
            assert len(logs) == 8
            assert all(x['thinking_enabled'] is False and x['completion_tokens'] == 2 for x in logs)
            assert all('messages' not in x for x in logs)
    asyncio.run(run())


def test_pool_batch_preserves_independent_ids_and_bodies(tmp_path):
    async def run():
        seen = []

        async def handler(name, request):
            data = await request.json()
            seen.append(data)
            await asyncio.sleep(.01)
            return ok(data['messages'][0]['content'])

        async with pool_over(tmp_path, handler) as (pool, url, client):
            first = body(1)
            first.pop('chat_template_kwargs')
            items = [{'custom_id': 'first', 'body': first}, {'custom_id': 'second', 'body': body(2)}]
            async with client.post(url + '/pool/batch', json={'items': items}, headers=AUTH) as r:
                assert r.status == 200
                result = await r.json()
            assert [x['custom_id'] for x in result['items']] == ['first', 'second']
            assert [x['response']['choices'][0]['message']['content'] for x in result['items']] == ['task1', 'task2']
            assert 'chat_template_kwargs' not in next(x for x in seen if x['messages'][0]['content'] == 'task1')
            assert sum(b.completed for b in pool.backends.values()) == 2
    asyncio.run(run())


def test_initial_backend_failure_fails_over_without_leaking_error(tmp_path):
    async def run():
        async def handler(name, request):
            if name == 'a':
                return web.json_response({'error': {'message': 'private server credential'}}, status=503)
            return ok()

        async with pool_over(tmp_path, handler, policy='round_robin') as (pool, url, client):
            async with client.post(url + '/v1/chat/completions', json=body(), headers=AUTH) as r:
                assert r.status == 200 and r.headers['X-Loam-Replica'] == 'b'
            assert pool.backends['a'].failures == 1 and pool.backends['b'].completed == 1
            log = (tmp_path / 'pool-request-timings.jsonl').read_text()
            assert 'credential' not in log
            assert len(log.splitlines()) == 2
    asyncio.run(run())


def test_stream_passes_tool_calls_and_usage_and_releases_capacity(tmp_path):
    async def run():
        async def handler(name, request):
            reply = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await reply.prepare(request)
            call = {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'test', 'arguments': '{}'}}]},
                                 'finish_reason': None}]}
            data = b'data: ' + json.dumps(call).encode() + b'\n\n'
            await reply.write(data[:13])
            await reply.write(data[13:])
            await reply.write(b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],'
                              b'"usage":{"prompt_tokens":10,"completion_tokens":4}}\n\ndata: [DONE]\n\n')
            await reply.write_eof()
            return reply

        async with pool_over(tmp_path, handler) as (pool, url, client):
            async with client.post(url + '/v1/chat/completions', json=body(stream=True), headers=AUTH) as r:
                result = await r.text()
                assert r.status == 200 and 'tool_calls' in result and '[DONE]' in result
            log = timings(tmp_path)[0]
            assert log['completion_tokens'] == 4 and log['finish_reason'] == 'tool_calls'
            assert log['time_to_first_token_seconds'] is not None
            assert not any(b.inflight for b in pool.backends.values())
    asyncio.run(run())


def test_auth_and_batch_validation_do_not_call_backends(tmp_path):
    async def run():
        async def handler(name, request):
            raise AssertionError('Backend should not be called')

        async with pool_over(tmp_path, handler) as (pool, url, client):
            async with client.post(url + '/v1/chat/completions', json=body()) as r:
                assert r.status == 401
            duplicate = [{'custom_id': 'x', 'body': body()}, {'custom_id': 'x', 'body': body()}]
            async with client.post(url + '/pool/batch', json={'items': duplicate}, headers=AUTH) as r:
                assert r.status == 400
            streaming = [{'custom_id': 'x', 'body': body(stream=True)}]
            async with client.post(url + '/pool/batch', json={'items': streaming}, headers=AUTH) as r:
                assert r.status == 400
    asyncio.run(run())


def test_context_error_keeps_numeric_recovery_and_strips_other_fields():
    error = {'error': {'code': 'context_length_exceeded', 'private': 'secret',
                       'message': "This model's maximum context length is 100 tokens. "
                                  'However, your prompt contains 101 input tokens.'}}
    result = safe_error(400, json.dumps(error).encode())
    assert result['error']['code'] == 'context_length_exceeded'
    assert '101' in result['error']['message']
    assert 'secret' not in str(result)
    assert 'secret' not in str(safe_error(500, b'secret'))


def test_model_and_checkpoint_come_from_pool_config(tmp_path):
    async def run():
        async def handler(name, request):
            return ok(name)

        async with pool_over(tmp_path, handler, checkpoint='org/weights-v2') as (pool, url, client):
            # Replica 'a' reports the configured weights; 'b' lists none, so it cannot prove it.
            config = json.loads((tmp_path / 'pool.json').read_text())
            config['backends'][0]['checkpoint'] = 'org/weights-v2'
            (tmp_path / 'pool.json').write_text(json.dumps(config))
            await pool.refresh()

            async with client.get(url + '/v1/models', headers=AUTH) as r:
                assert (await r.json())['data'] == [
                    {'id': MODEL, 'object': 'model', 'owned_by': 'loam', 'root': 'org/weights-v2'}]
            async with client.get(url + '/pool/status', headers=AUTH) as r:
                status = await r.json()
            assert (status['model'], status['checkpoint']) == (MODEL, 'org/weights-v2')
            assert {b['id']: b['ready'] for b in status['backends']} == {'a': True, 'b': False}

            for i in range(3):
                async with client.post(url + '/v1/chat/completions', json=body(i), headers=AUTH) as r:
                    assert r.headers['X-Loam-Replica'] == 'a'
            async with client.post(url + '/v1/chat/completions', json={**body(), 'model': 'other'}, headers=AUTH) as r:
                assert r.status == 400
    asyncio.run(run())


def test_running_pool_refuses_a_changed_model(tmp_path):
    async def run():
        async def handler(name, request):
            return ok()

        async with pool_over(tmp_path, handler) as (pool, url, client):
            config = json.loads((tmp_path / 'pool.json').read_text())
            (tmp_path / 'pool.json').write_text(json.dumps({**config, 'model': 'swapped'}))
            with pytest.raises(ValueError, match='cannot change the model'):
                await pool.refresh()
            assert pool.model == MODEL
    asyncio.run(run())


def test_listed_replica_token_file_is_sent_as_bearer(tmp_path):
    async def run():
        tokens = {}

        async def handler(name, request):
            tokens[name] = request.headers.get('Authorization')
            return ok()

        (tmp_path / 'a.token').write_text('replica-secret\n')
        async with pool_over(tmp_path, handler, policy='round_robin') as (pool, url, client):
            config = json.loads((tmp_path / 'pool.json').read_text())
            config['backends'][0]['token_file'] = 'a.token'
            (tmp_path / 'pool.json').write_text(json.dumps(config))
            await pool.refresh()
            for i in range(2):
                async with client.post(url + '/v1/chat/completions', json=body(i), headers=AUTH) as r:
                    assert r.status == 200
            assert tokens == {'a': 'Bearer replica-secret', 'b': None}
            assert 'replica-secret' not in (tmp_path / 'pool-request-timings.jsonl').read_text()
    asyncio.run(run())


@pytest.mark.parametrize('requests', [616, 918])
def test_large_pool_can_admit_campaign_requests_across_102_backends(tmp_path, requests):
    """A resolver supplied by a backend (here, a lookup table) in place of listed base URLs."""
    async def run():
        count = 0
        barrier = asyncio.Event()

        async def handler(request):
            nonlocal count
            await request.json()
            count += 1
            if count == requests:
                barrier.set()
            await barrier.wait()
            return ok()

        app = web.Application()
        app.router.add_post('/v1/chat/completions', handler)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        urls = {}
        pool = None
        try:
            for i in range(102):
                site = web.TCPSite(runner, '127.0.0.1', 0)
                await site.start()
                urls[str(i)] = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1'
            config = {'model': MODEL, 'policy': 'least_loaded', 'max_inflight_per_backend': 16,
                      'backends': [{'id': key, 'job_id': str(i + 1)} for i, key in enumerate(urls)]}
            (tmp_path / 'pool.json').write_text(json.dumps(config))
            pool = InferencePool(tmp_path, resolver=lambda entry: ReplicaEndpoint(urls[entry['id']]))
            await pool.start(token='test-client')
            assert pool.session.connector.limit == 1632
            connector = aiohttp.TCPConnector(limit=1024)
            async with serve(pool.app()) as url, aiohttp.ClientSession(connector=connector) as client:
                async def one(i):
                    async with client.post(url + '/v1/chat/completions', json=body(i), headers=AUTH) as response:
                        assert response.status == 200
                        await response.json()

                await asyncio.wait_for(asyncio.gather(*(one(i) for i in range(requests))), timeout=60)
            assert count == requests
            assert sum(b.completed for b in pool.backends.values()) == requests
            assert all(b.completed for b in pool.backends.values())
        finally:
            if pool is not None:
                await pool.close()
            await runner.cleanup()
    asyncio.run(run())


def test_core_pool_does_not_reach_for_a_system_specific_backend():
    source = open(loam.inference_pool.__file__).read()
    assert 'sys.path' not in source
    assert 'nemotron' not in source.lower()


def test_resolver_returning_something_else_leaves_replica_not_ready(tmp_path):
    async def run():
        (tmp_path / 'pool.json').write_text(json.dumps({'model': MODEL, 'backends': [{'id': 'a'}]}))
        pool = InferencePool(tmp_path, resolver=lambda entry: {'base_url': 'http://127.0.0.1:1/v1'})
        await pool.start(token='test-client')
        try:
            assert pool.status()['backends'][0]['ready'] is False
            assert pool.status()['backends'][0]['readiness_error'] == 'TypeError'
        finally:
            await pool.close()
    asyncio.run(run())
