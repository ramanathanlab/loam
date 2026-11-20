#!/usr/bin/env python3
"""Serve the experiment's owned inference replicas on one stable login URL."""
import argparse,asyncio,json,os,signal
from pathlib import Path
from aiohttp import web
from loam.inference_pool import InferencePool

async def main(root,port):
    import sys
    sys.path.insert(0,str(root/'inference-runtime/scripts'))
    from jupiter_nemotron_api import atomic_json
    stop=asyncio.Event();loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,stop.set)
    pool=InferencePool(root)
    await pool.start()
    runner=web.AppRunner(pool.app(),access_log=None);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',port);await site.start();actual=site._server.sockets[0].getsockname()[1]
    atomic_json(root/'pool-service.json',{'pid':os.getpid(),'base_url':f'http://127.0.0.1:{actual}/v1','status_url':f'http://127.0.0.1:{actual}/pool/status','batch_url':f'http://127.0.0.1:{actual}/pool/batch','token_file':str(root/'pool-token')})
    print(json.dumps({'pool_url':f'http://127.0.0.1:{actual}/v1','pid':os.getpid()}),flush=True)
    try:await stop.wait()
    finally:await runner.cleanup();await pool.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=lambda x:Path(x).resolve());parser.add_argument('--port',type=int,default=0);args=parser.parse_args();asyncio.run(main(args.experiment,args.port))
