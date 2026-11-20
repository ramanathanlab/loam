#!/usr/bin/env python3
"""Login-local, bearer-authenticated relay to this campaign's owned Nemotron job."""
import argparse, asyncio, json, os, secrets, sys, time
from pathlib import Path

async def main(campaign):
    sys.path.insert(0,str(campaign/'inference-runtime/scripts'))
    from jupiter_nemotron_api import atomic_json, completion_check, pinned_tls, proxy_app, readiness
    import aiohttp
    from aiohttp import web
    state=campaign/'state/nemotron'
    token_file=state/'relay-token'
    if not token_file.exists():
        fd=os.open(token_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(32)+'\n')
    token=token_file.read_text().strip()
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=None,connect=10,sock_read=600)) as session:
        async def upstream():
            progress=json.loads((campaign/'scale-progress.json').read_text());attempt=state/progress['inference_submission']['attempt'];job=progress['inference_submission']['job_id']
            if job not in progress['owned_job_ids']:raise RuntimeError('Unowned inference job')
            r=readiness(attempt,job)
            return r['base_url'],(attempt/'token').read_text().strip(),pinned_tls(attempt/'tls.pem')
        while True:
            try:
                base,key,tls=await upstream();await completion_check(session,base,key,tls=tls);break
            except (FileNotFoundError,RuntimeError,aiohttp.ClientError,asyncio.TimeoutError):await asyncio.sleep(10)
        async def audit(request,handler):
            started=time.time();response=None
            try:response=await handler(request);return response
            finally:
                with (state/'request-timings.jsonl').open('a') as f:f.write(json.dumps({'start':started,'end':time.time(),'seconds':time.time()-started,'status':getattr(response,'status',None),'path':request.path})+'\n')
        app=proxy_app(session,token,upstream);app.middlewares.append(web.middleware(audit))
        runner=web.AppRunner(app,access_log=None);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1]
        atomic_json(state/'relay.json',{'base_url':f'http://127.0.0.1:{port}/v1','token_file':str(token_file),'pid':os.getpid(),'model':'nemotron-3.5-super-vl','started_at':time.time()})
        print('Owned Nemotron relay ready',port,flush=True)
        try:await asyncio.Event().wait()
        finally:await runner.cleanup()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args();asyncio.run(main(a.campaign.resolve()))
