import asyncio,json,time
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from loam.inference_pool import InferencePool,StreamAccounting,safe_error


@asynccontextmanager
async def server(app):
    runner=web.AppRunner(app,access_log=None);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    try:yield f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    finally:await runner.cleanup()


@asynccontextmanager
async def fixture(root,handler,capacity=1,policy='least_loaded'):
    entries=[];runners=[]
    async with server(make_backend(handler,'a')) as a,server(make_backend(handler,'b')) as b:
        urls={'a':a+'/v1','b':b+'/v1'}
        config={'model':'nemotron-3.5-super-vl','policy':policy,'max_inflight_per_backend':capacity,'backends':[{'id':k,'job_id':str(i+1)} for i,k in enumerate(urls)]}
        (root/'pool.json').write_text(json.dumps(config));pool=InferencePool(root,resolver=lambda e:{'base_url':urls[e['id']]})
        await pool.start(token='test-client')
        async with server(pool.app()) as url,aiohttp.ClientSession() as client:
            try:yield pool,url,client
            finally:await pool.close()


def make_backend(handler,name):
    app=web.Application()
    async def invoke(request):return await handler(name,request)
    app.router.add_post('/v1/chat/completions',invoke);return app


def body(i=0,**kwargs):
    return {'model':'nemotron-3.5-super-vl','messages':[{'role':'user','content':f'task{i}'}],'max_tokens':8192,'chat_template_kwargs':{'enable_thinking':False},**kwargs}


@pytest.mark.parametrize('requests', [616, 918])
def test_large_pool_can_admit_campaign_requests_across_102_backends(tmp_path, requests):
    """Exercise active-thread and larger initial-generation HTTP bursts."""
    async def run():
        count=0;barrier=asyncio.Event();replicas=102
        async def handler(request):
            nonlocal count
            await request.json();count+=1
            if count==requests:barrier.set()
            await barrier.wait()
            return web.json_response({'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}]})
        app=web.Application();app.router.add_post('/v1/chat/completions',handler)
        runner=web.AppRunner(app,access_log=None);await runner.setup()
        urls={};pool=None
        try:
            for i in range(replicas):
                site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
                urls[str(i)]=f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1'
            config={'model':'nemotron-3.5-super-vl','policy':'least_loaded','max_inflight_per_backend':16,
                    'backends':[{'id':key,'job_id':str(i+1)} for i,key in enumerate(urls)]}
            (tmp_path/'pool.json').write_text(json.dumps(config))
            pool=InferencePool(tmp_path,resolver=lambda entry:{'base_url':urls[entry['id']]})
            await pool.start(token='test-client')
            assert pool.session.connector.limit==1632
            async with server(pool.app()) as url,aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1024)) as client:
                async def one(i):
                    async with client.post(url+'/v1/chat/completions',json=body(i),headers={'Authorization':'Bearer test-client'}) as response:
                        assert response.status==200;await response.json()
                await asyncio.wait_for(asyncio.gather(*(one(i) for i in range(requests))),timeout=60)
            assert count==requests
            assert sum(b.completed for b in pool.backends.values())==requests
            assert all(b.completed for b in pool.backends.values())
        finally:
            if pool is not None:await pool.close()
            await runner.cleanup()
    asyncio.run(run())


def test_bounded_fanout_uses_both_replicas_and_preserves_bodies(tmp_path):
    async def run():
        active={'a':0,'b':0};peaks={'a':0,'b':0};seen=[]
        async def handler(name,request):
            data=await request.json();seen.append(data);active[name]+=1;peaks[name]=max(peaks[name],active[name]);await asyncio.sleep(.02);active[name]-=1
            return web.json_response({'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}],'usage':{'prompt_tokens':8,'completion_tokens':2}})
        async with fixture(tmp_path,handler) as (pool,url,client):
            requests=[body(i,tools=[{'type':'function','function':{'name':'submit'}}]) for i in range(8)]
            async def one(data):
                async with client.post(url+'/v1/chat/completions',json=data,headers={'Authorization':'Bearer test-client'}) as r:assert r.status==200;return await r.json()
            await asyncio.gather(*(one(data) for data in requests))
            assert peaks=={'a':1,'b':1};assert pool.backends['a'].completed and pool.backends['b'].completed
            assert sorted(seen,key=lambda x:x['messages'][0]['content'])==requests
            assert not any(b.inflight or b.estimated_tokens for b in pool.backends.values())
            logs=[json.loads(s) for s in (tmp_path/'pool-request-timings.jsonl').read_text().splitlines()]
            assert len(logs)==8 and all(x['thinking_enabled'] is False and x['completion_tokens']==2 for x in logs)
            assert all('messages' not in x for x in logs)
    asyncio.run(run())


def test_pool_batch_preserves_independent_ids_and_tournament_default(tmp_path):
    async def run():
        seen=[]
        async def handler(name,request):
            data=await request.json();seen.append(data);await asyncio.sleep(.01);return web.json_response({'choices':[{'message':{'content':data['messages'][0]['content']},'finish_reason':'stop'}]})
        async with fixture(tmp_path,handler) as (pool,url,client):
            first=body(1);first.pop('chat_template_kwargs')
            async with client.post(url+'/pool/batch',json={'items':[{'custom_id':'first','body':first},{'custom_id':'second','body':body(2)}]},headers={'Authorization':'Bearer test-client'}) as r:
                result=await r.json();assert r.status==200
            assert [x['custom_id'] for x in result['items']]==['first','second'];assert 'chat_template_kwargs' not in seen[0]
            assert sum(b.completed for b in pool.backends.values())==2
    asyncio.run(run())


def test_initial_backend_failure_fails_over_without_leaking_error(tmp_path):
    async def run():
        async def handler(name,request):
            if name=='a':return web.json_response({'error':{'message':'private server credential'}},status=503)
            return web.json_response({'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}]})
        async with fixture(tmp_path,handler,policy='round_robin') as (pool,url,client):
            async with client.post(url+'/v1/chat/completions',json=body(),headers={'Authorization':'Bearer test-client'}) as r:
                result=await r.json();assert r.status==200 and r.headers['X-Loamlite-Replica']=='b'
            assert pool.backends['a'].failures==1 and pool.backends['b'].completed==1
            log=(tmp_path/'pool-request-timings.jsonl').read_text();assert 'credential' not in log;assert len(log.splitlines())==2
    asyncio.run(run())


def test_stream_passes_tool_calls_and_usage_and_releases_capacity(tmp_path):
    async def run():
        async def handler(name,request):
            reply=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await reply.prepare(request)
            data=b'data: '+json.dumps({'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'name':'test','arguments':'{}'}}]},'finish_reason':None}]}).encode()+b'\n\n'
            await reply.write(data[:13]);await reply.write(data[13:]);await reply.write(b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":4}}\n\ndata: [DONE]\n\n');await reply.write_eof();return reply
        async with fixture(tmp_path,handler) as (pool,url,client):
            async with client.post(url+'/v1/chat/completions',json=body(stream=True),headers={'Authorization':'Bearer test-client'}) as r:
                result=await r.text();assert r.status==200 and 'tool_calls' in result and '[DONE]' in result
            logs=[json.loads(s) for s in (tmp_path/'pool-request-timings.jsonl').read_text().splitlines()]
            assert logs[0]['completion_tokens']==4 and logs[0]['finish_reason']=='tool_calls' and logs[0]['time_to_first_token_seconds'] is not None
            assert not any(b.inflight for b in pool.backends.values())
    asyncio.run(run())


def test_auth_and_batch_validation_do_not_call_backends(tmp_path):
    async def run():
        async def handler(name,request):raise AssertionError('Backend should not be called')
        async with fixture(tmp_path,handler) as (pool,url,client):
            async with client.post(url+'/v1/chat/completions',json=body()) as r:assert r.status==401
            async with client.post(url+'/pool/batch',json={'items':[{'custom_id':'x','body':body()},{'custom_id':'x','body':body()}]},headers={'Authorization':'Bearer test-client'}) as r:assert r.status==400
            async with client.post(url+'/pool/batch',json={'items':[{'custom_id':'x','body':body(stream=True)}]},headers={'Authorization':'Bearer test-client'}) as r:assert r.status==400
    asyncio.run(run())


def test_rejects_foreign_campaign_before_loading_credentials(tmp_path):
    foreign=tmp_path/'foreign';foreign.mkdir();own=tmp_path/'own';own.mkdir();pool=InferencePool(own)
    import pytest
    with pytest.raises(RuntimeError,match='outside owned experiment'):
        pool.resolve_owned({'campaign':str(foreign),'attempt':'loam-nemotron-'+'a'*16,'job_id':'123'})


def test_context_error_keeps_numeric_recovery_and_strips_other_fields():
    error={'error':{'code':'context_length_exceeded','message':"This model's maximum context length is 100 tokens. However, your prompt contains 101 input tokens.",'private':'secret'}}
    result=safe_error(400,json.dumps(error).encode());assert result['error']['code']=='context_length_exceeded';assert '101' in result['error']['message'];assert 'secret' not in str(result)
    assert 'secret' not in str(safe_error(500,b'secret'))
