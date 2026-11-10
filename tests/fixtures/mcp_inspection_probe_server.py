"""Loopback-only fixture: no arbitrary code execution or unrestricted file reads."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skills/hpc-deployment/scripts'))
from fastmcp import FastMCP
from mcp_dispatch import dispatch_action

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()
server = FastMCP('loam-inspection-probe')


async def answered_action(name, *args, **kwargs):
    if name == 'inspect_filesystem':
        raise FileNotFoundError('fixture path does not exist')
    return {'stdout': 'LOAM_DISPATCH_OK\n', 'stderr': '', 'returncode': '0'}


class Pool:
    async def next_agent(self, launch_one):
        return SimpleNamespace(handle=SimpleNamespace(action=answered_action))


class UnavailablePool:
    async def next_agent(self, launch_one):
        raise ConnectionError('fixture worker pool is unavailable')


@server.tool(annotations={"readOnlyHint": True})
async def inspect_filesystem(operation: str, path: str) -> dict:
    return await dispatch_action(Pool(), None, 'inspect_filesystem', operation, path)


@server.tool(annotations={"readOnlyHint": True})
async def run_code() -> dict:
    return await dispatch_action(Pool(), None, 'run_code')


@server.tool(annotations={"readOnlyHint": True})
async def pool_failure() -> dict:
    return await dispatch_action(UnavailablePool(), None, 'inspect_filesystem', 'ls', '/fixture')


server.run(transport='http', host='127.0.0.1', port=args.port, show_banner=False)
