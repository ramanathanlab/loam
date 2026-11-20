"""Owned Nemotron replica routing, backpressure and transparent chat/SSE forwarding.

Replica URLs and credentials come only from validated campaign ownership and
readiness files. The gateway never changes thinking, tools or token budgets.
vLLM performs continuous batching inside each replica; /pool/batch provides
bounded fan-out for independent non-streaming requests without merging prompts.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import aiohttp
from aiohttp import web


@dataclass
class Backend:
    name: str
    job_id: str
    base_url: str = ''
    token: str = field(default='', repr=False)
    tls: object = field(default=None, repr=False)
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
    """Hash stable task identity for cache affinity; never persist prompt text."""
    messages=body.get('messages',[])
    user=next((m.get('content','') for m in messages if m.get('role')=='user'),'')
    system=next((m.get('content','') for m in messages if m.get('role')=='system'),'')
    raw=json.dumps([system,user],sort_keys=True,ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def safe_error(status,payload):
    fallback={'error':{'message':f'Nemotron API returned HTTP {status}'}}
    if status!=400 or len(payload)>16384:return fallback
    try:error=json.loads(payload).get('error',{})
    except (ValueError,AttributeError):return fallback
    if not isinstance(error,dict) or error.get('code')!='context_length_exceeded':return fallback
    message=error.get('message','');context=re.search(r'maximum context length is\s+(\d{1,10})\s+tokens',message,re.I) if isinstance(message,str) else None
    count=re.search(r'prompt contains\s+(?:at least\s+)?(\d{1,10})\s+input tokens',message,re.I) if isinstance(message,str) else None
    detail='Maximum context length exceeded. Please reduce the length of the input prompt.'
    if context and count:
        detail=f"This model's maximum context length is {int(context[1])} tokens. However, your prompt contains {int(count[1])} input tokens, which exceeds this limit even without any output tokens. Please reduce the length of the input prompt."
    return {'error':{'message':detail,'type':'invalid_request_error','code':'context_length_exceeded','param':'input_tokens'}}


class StreamAccounting:
    def __init__(self):
        self.buffer=b'';self.usage={};self.finish_reason=None;self.first_token_at=None

    def feed(self, chunk):
        self.buffer+=chunk
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            if not line.startswith(b'data:'):continue
            try:data=json.loads(line[5:].strip())
            except (ValueError,UnicodeDecodeError):continue
            if not isinstance(data,dict):continue
            if data.get('usage'):self.usage=data['usage']
            for choice in data.get('choices',[]):
                delta=choice.get('delta') or {}
                if self.first_token_at is None and any(delta.get(k) for k in ('content','reasoning','reasoning_content','tool_calls')):
                    self.first_token_at=time.time()
                if choice.get('finish_reason'):self.finish_reason=choice['finish_reason']
        if len(self.buffer)>1024*1024:self.buffer=b''


class InferencePool:
    def __init__(self, root: Path, *, resolver: Callable | None=None, session=None):
        self.root=Path(root).resolve();self.backends={};self.policy='least_loaded_affinity'
        self.phase='unassigned'
        self.model='nemotron-3.5-super-vl';self.condition=asyncio.Condition()
        self.affinity=OrderedDict();self.cursor=0;self.resolver=resolver
        self.session=session;self.metrics=Counter();self.refresh_task=None
        self.token='';self.owns_session=False;self.last_config_error=None

    def validate(self, body):
        if not isinstance(body,dict) or body.get('model')!=self.model or not isinstance(body.get('messages'),list):
            raise web.HTTPBadRequest(text='Expected a Nemotron chat request with a messages list')

    async def refresh(self):
        config=json.loads((self.root/'pool.json').read_text())
        if config.get('model')!=self.model or config.get('policy') not in {'round_robin','least_loaded','least_loaded_affinity'}:
            raise ValueError('Invalid pool configuration')
        self.policy=config['policy'];self.phase=str(config.get('phase','unassigned'))[:128];seen=set()
        for entry in config['backends']:
            name=entry['id'];job=str(entry['job_id']);seen.add(name)
            if name in self.backends and self.backends[name].job_id!=job:
                if self.backends[name].inflight:raise RuntimeError('Cannot replace a busy backend identity')
                del self.backends[name]
            backend=self.backends.setdefault(name,Backend(name,job))
            backend.enabled=bool(entry.get('enabled',True));backend.capacity=int(config.get('max_inflight_per_backend',16))
            if not 1<=backend.capacity<=64:raise ValueError('Invalid per-backend capacity')
            try:
                # At102 replicas, synchronous metadata/TLS reads must not stop
                # the HTTP event loop while it serves campaign requests.
                info=self.resolver(entry) if self.resolver else await asyncio.to_thread(self.resolve_owned,entry)
                backend.base_url=info['base_url'];backend.token=info.get('token','');backend.tls=info.get('tls')
                backend.ready=True;backend.last_error=None
            except (OSError,ValueError,KeyError,RuntimeError) as error:
                backend.ready=False;backend.last_error=type(error).__name__
        for name,backend in self.backends.items():
            if name not in seen:backend.enabled=False
        self.last_config_error=None
        async with self.condition:self.condition.notify_all()

    def resolve_owned(self, entry):
        campaign=Path(entry['campaign']).resolve(strict=True)
        if not campaign.is_relative_to(self.root/'replicas'):raise RuntimeError('Replica is outside owned experiment')
        if not re.fullmatch(r'loam-nemotron-[0-9a-f]{16}',entry['attempt']):raise ValueError('Invalid attempt name')
        import sys
        path=str(self.root/'inference-runtime/scripts')
        if path not in sys.path:sys.path.insert(0,path)
        from jupiter_nemotron_api import readiness,pinned_tls
        pointer=json.loads((campaign/'state/nemotron/current.json').read_text())
        if pointer['attempt']!=entry['attempt']:raise RuntimeError('Replica attempt pointer changed')
        attempt=campaign/'state/nemotron'/entry['attempt']
        owner=json.loads((attempt/'request.json').read_text());submission=json.loads((attempt/'submission.json').read_text())
        progress=json.loads((self.root/'scale-progress.json').read_text())
        if owner['campaign']!=str(campaign) or str(submission['job_id'])!=str(entry['job_id']) or str(entry['job_id']) not in progress['owned_job_ids']:
            raise RuntimeError('Replica ownership proof failed')
        if entry.get('allocation_manifest'):
            from .allocation import read_allocation
            manifest=Path(entry['allocation_manifest']).resolve(strict=True)
            if manifest!=self.root.parent/'allocation.json':raise RuntimeError('Unexpected parent allocation manifest')
            allocated=read_allocation(manifest)
            if allocated['job_id']!=str(entry['job_id']) or entry.get('hostname') not in allocated['inference_nodes'] or owner.get('allocation_manifest')!=str(manifest):
                raise RuntimeError('Replica is outside the parent allocation inference role')
        record=readiness(attempt,str(entry['job_id']))
        if entry.get('hostname') and record['hostname'].split('.')[0]!=entry['hostname']:
            raise RuntimeError('Replica readiness hostname differs from its assigned node')
        return {'base_url':record['base_url'],'token':(attempt/'token').read_text().strip(),'tls':pinned_tls(attempt/'tls.pem')}

    async def watch(self):
        while True:
            try:await self.refresh()
            except (OSError,ValueError,KeyError,RuntimeError) as error:
                self.last_config_error=type(error).__name__
                for backend in self.backends.values():backend.ready=False
            await asyncio.sleep(2)

    async def acquire(self, body, *, excluded=(), timeout=590):
        key=request_key(body);estimate=max(1,len(json.dumps(body.get('messages',[])) )//4)
        estimate+=min(int(body.get('max_tokens',body.get('max_completion_tokens',8192)) or 8192),4096)
        started=time.time();deadline=asyncio.get_running_loop().time()+timeout
        while True:
            async with self.condition:
                now=time.time();possible=[b for b in self.backends.values() if b.name not in excluded and b.enabled and b.ready]
                if not possible:raise RuntimeError('No owned ready inference replicas')
                choices=[b for b in possible if b.available(now) and b.inflight<b.capacity]
                if choices:
                    choices.sort(key=lambda b:b.name);hit=False
                    if self.policy=='round_robin':
                        selected=choices[self.cursor%len(choices)];self.cursor+=1
                    else:
                        score=lambda b:(b.inflight+.2*b.estimated_tokens/32768,b.completed+b.failures,b.name)
                        selected=min(choices,key=score)
                        sticky=self.affinity.get(key)
                        if self.policy=='least_loaded_affinity' and sticky:
                            preferred=next((b for b in choices if b.name==sticky),None)
                            if preferred and score(preferred)[0]<=score(selected)[0]+1:
                                selected=preferred;hit=True
                    selected.inflight+=1;selected.estimated_tokens+=estimate
                    self.affinity[key]=selected.name;self.affinity.move_to_end(key)
                    while len(self.affinity)>10000:self.affinity.popitem(last=False)
                    return selected,estimate,time.time()-started,hit
                remaining=deadline-asyncio.get_running_loop().time()
                if remaining<=0:raise TimeoutError('Inference pool backpressure deadline exceeded')
                try:await asyncio.wait_for(self.condition.wait(),timeout=min(remaining,2))
                except TimeoutError:pass

    async def release(self, backend, estimate):
        async with self.condition:
            backend.inflight-=1;backend.estimated_tokens-=estimate
            assert backend.inflight>=0 and backend.estimated_tokens>=0
            self.condition.notify_all()

    def record(self, item):
        self.metrics[str(item['status'])]+=1
        with (self.root/'pool-request-timings.jsonl').open('a') as stream:
            stream.write(json.dumps(item,separators=(',',':'))+'\n')

    async def dispatch(self, body, client_request=None):
        self.validate(body);stream=bool(body.get('stream',False))
        if stream and client_request is None:raise web.HTTPBadRequest(text='Batch fan-out requires non-streaming requests')
        excluded=set();request_id=uuid.uuid4().hex;whole_start=time.time()
        for attempt_number in range(1,4):
            backend=None;reply=None;status=503;estimate=0;started=time.time();accounting=StreamAccounting()
            record={'request_id':request_id,'attempt':attempt_number,'started_at':started,'policy':self.policy,'phase':self.phase,
                    'thinking_enabled':body.get('chat_template_kwargs',{}).get('enable_thinking',True),
                    'stream':stream,'status':503,'prompt_bytes':len(json.dumps(body.get('messages',[])).encode())}
            try:
                backend,estimate,queued,hit=await self.acquire(body,excluded=excluded)
                excluded.add(backend.name);record.update(backend=backend.name,job_id=backend.job_id,queue_seconds=queued,affinity_hit=hit)
                headers={'Content-Type':'application/json'}
                if backend.token:headers['Authorization']='Bearer '+backend.token
                async with self.session.post(backend.base_url+'/chat/completions',json=body,headers=headers,ssl=backend.tls,allow_redirects=False) as response:
                    status=response.status;record['status']=status;record['headers_seconds']=time.time()-started
                    if status>=300:
                        payload=await response.content.read(16385)
                        # Numeric context errors remain useful; arbitrary backend text is never exposed.
                        error=safe_error(status,payload)
                        if status in (429,500,502,503,504):
                            try:delay=float(response.headers.get('Retry-After','5'))
                            except ValueError:delay=5
                            backend.cooldown_until=time.time()+max(1,min(120,delay))+secrets.randbelow(1000)/1000
                            backend.failures+=1
                            if any(b.available(time.time()) and b.name not in excluded for b in self.backends.values()):continue
                        return web.json_response(error,status=status if status>=400 else 502,headers={'Retry-After':'5'} if status==429 else None)
                    if stream:
                        reply=web.StreamResponse(status=200,headers={'Content-Type':response.headers.get('Content-Type','text/event-stream'),'Cache-Control':'no-cache','X-Loamlite-Replica':backend.name})
                        await reply.prepare(client_request)
                        async for chunk in response.content.iter_any():
                            accounting.feed(chunk);await reply.write(chunk)
                        await reply.write_eof()
                    else:
                        payload=await response.read()
                        try:
                            result=json.loads(payload);accounting.usage=result.get('usage') or {}
                            accounting.finish_reason=next((c.get('finish_reason') for c in result.get('choices',[]) if c.get('finish_reason')),None)
                        except (ValueError,AttributeError):
                            raise RuntimeError('Backend did not return a JSON completion') from None
                        reply=web.Response(body=payload,status=200,content_type='application/json',headers={'X-Loamlite-Replica':backend.name})
                    backend.completed+=1
                    return reply
            except asyncio.CancelledError:
                record.update(status=499,error_type='CancelledError')
                raise
            except (aiohttp.ClientError,asyncio.TimeoutError,OSError,RuntimeError,ValueError) as error:
                record['error_type']=type(error).__name__;record['status']=503
                if backend:
                    backend.failures+=1;backend.cooldown_until=time.time()+5
                if reply is not None and reply.prepared:
                    if client_request.transport is not None:client_request.transport.close()
                    return reply
                if backend is None or not any(b.available(time.time()) and b.name not in excluded for b in self.backends.values()):
                    return web.json_response({'error':{'message':'Owned inference pool unavailable'}},status=503,headers={'Retry-After':'5'})
            finally:
                record.update(finished_at=time.time(),seconds=time.time()-started,total_request_seconds=time.time()-whole_start,
                              time_to_first_token_seconds=(accounting.first_token_at-started) if accounting.first_token_at else None,
                              finish_reason=accounting.finish_reason,
                              prompt_tokens=accounting.usage.get('prompt_tokens'),completion_tokens=accounting.usage.get('completion_tokens'),
                              reasoning_tokens=(accounting.usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
                self.record(record)
                if backend:await self.release(backend,estimate)
        return web.json_response({'error':{'message':'Owned inference replicas failed'}},status=503,headers={'Retry-After':'5'})

    def status(self):
        now=time.time()
        return {'model':self.model,'policy':self.policy,'phase':self.phase,'time':now,'config_error':self.last_config_error,
                'ready_replicas':sum(b.available(now) for b in self.backends.values()),'request_attempt_statuses':dict(self.metrics),
                'backends':[{'id':b.name,'job_id':b.job_id,'enabled':b.enabled,'ready':b.ready,'available':b.available(now),
                             'capacity':b.capacity,'inflight':b.inflight,'estimated_tokens':b.estimated_tokens,'completed':b.completed,
                             'failed_attempts':b.failures,'cooldown_seconds':max(0,b.cooldown_until-now),'readiness_error':b.last_error}
                            for b in self.backends.values()]}

    def app(self):
        @web.middleware
        async def authorize(request,handler):
            expected=('Bearer '+self.token).encode()
            if not hmac.compare_digest(request.headers.get('Authorization','').encode(),expected):raise web.HTTPUnauthorized()
            return await handler(request)
        async def models(request):
            if not any(b.available(time.time()) for b in self.backends.values()):
                return web.json_response({'error':{'message':'Inference replicas still warming'}},status=503,headers={'Retry-After':'5'})
            return web.json_response({'object':'list','data':[{'id':self.model,'object':'model','owned_by':'loam-experiment'}]})
        async def complete(request):
            try:body=await request.json()
            except ValueError:raise web.HTTPBadRequest(text='Expected JSON') from None
            return await self.dispatch(body,request)
        async def batch(request):
            try:body=await request.json();items=body['items']
            except (ValueError,KeyError,TypeError):raise web.HTTPBadRequest(text='Expected items with custom_id and body') from None
            if not isinstance(items,list) or not 1<=len(items)<=64:raise web.HTTPBadRequest(text='Expected one to sixty-four independent requests')
            ids=[x.get('custom_id') for x in items if isinstance(x,dict)]
            if len(ids)!=len(items) or any(not isinstance(k,str) or len(k)>128 for k in ids) or len(set(ids))!=len(ids):
                raise web.HTTPBadRequest(text='custom_id values must be unique strings')
            for item in items:
                self.validate(item.get('body'))
                if item['body'].get('stream'):raise web.HTTPBadRequest(text='Batch items must not stream')
            semaphore=asyncio.Semaphore(max(1,min(32,sum(b.capacity for b in self.backends.values()))))
            async def one(item):
                async with semaphore:
                    response=await self.dispatch(item['body'])
                    return {'custom_id':item['custom_id'],'status':response.status,'response':json.loads(response.body)}
            return web.json_response({'items':await asyncio.gather(*(one(item) for item in items))})
        async def status(request):return web.json_response(self.status())
        application=web.Application(client_max_size=16*1024*1024,middlewares=[authorize])
        application.router.add_get('/v1/models',models);application.router.add_post('/v1/chat/completions',complete)
        application.router.add_get('/pool/status',status);application.router.add_post('/pool/batch',batch)
        return application

    async def start(self, *, token=None):
        if token is None:
            path=self.root/'pool-token'
            try:
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'w') as stream:stream.write(secrets.token_urlsafe(32)+'\n')
            except FileExistsError:pass
            token=path.read_text().strip()
        self.token=token
        await self.refresh()
        if self.session is None:
            # Preserve the small-run budget, but let a large pool reach its
            # configured bounded capacity instead of capping all102 replicas
            # at256 outbound requests. At102x16 the limit is1632.
            capacity=sum(b.capacity for b in self.backends.values() if b.enabled)
            connection_limit=max(256,min(2048,capacity))
            self.session=aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=None,connect=10,sock_read=600),connector=aiohttp.TCPConnector(limit=connection_limit,limit_per_host=64))
            self.owns_session=True
        self.refresh_task=asyncio.create_task(self.watch())

    async def close(self):
        if self.refresh_task:
            self.refresh_task.cancel()
            try:await self.refresh_task
            except asyncio.CancelledError:pass
        if self.owns_session:await self.session.close()
