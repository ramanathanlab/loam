#!/usr/bin/env python3
"""Hold a managed science server for bounded operator validation via MCP."""
import argparse, asyncio, json, logging, os, signal
from pathlib import Path
from loam.config import load
from loam.launcher import managed_hpc_server

async def main(config):
    cfg=load(config);base=Path(cfg.execution.hpc.base_dir);stop=asyncio.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,stop.set)
    (base/'server-pid.json').write_text(json.dumps({'pid':os.getpid(),'config':str(config)})+'\n')
    async with managed_hpc_server(cfg):
        (base/'server-ready.json').write_text(json.dumps({'mcp_url':cfg.execution.mcp_url,'pid':os.getpid()})+'\n')
        while not stop.is_set() and not (base/'stop-server').exists():
            try:await asyncio.wait_for(stop.wait(),timeout=5)
            except TimeoutError:pass
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);a=p.parse_args();logging.basicConfig(level=logging.INFO);asyncio.run(main(a.config.resolve()))
