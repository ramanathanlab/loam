"""Lifecycle for the YAML-configured local server that fronts managed HPC."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager

from .config import Config
from .execution import check_mcp_tools


@asynccontextmanager
async def managed_hpc_server(cfg: Config):
    """Start the configured server, wait for its tools, and shut it down cleanly."""
    deployment = cfg.execution.hpc
    if cfg.execution.mode != 'hpc' or deployment is None:
        yield
        return

    url_host = f'[{deployment.host}]' if ':' in deployment.host else deployment.host
    client_host = '127.0.0.1' if deployment.host in {'0.0.0.0', '::'} else url_host
    cfg.execution.mcp_url = f'http://{client_host}:{deployment.port}/mcp'
    env = os.environ.copy()
    env.update(deployment.environment())
    script_dir = str(deployment.server_script.parent)
    current_pythonpath = env.get('PYTHONPATH')
    env['PYTHONPATH'] = os.pathsep.join(
        part for part in (script_dir, current_pythonpath) if part
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(deployment.server_script), env=env,
    )
    deadline = asyncio.get_running_loop().time() + deployment.startup_timeout
    last_error: Exception | None = None
    try:
        while True:
            if process.returncode is not None:
                raise RuntimeError(
                    f'managed HPC MCP server exited with status {process.returncode} during startup'
                )
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RuntimeError(
                    f'managed HPC MCP server was not ready at {cfg.execution.mcp_url} '
                    f'within {deployment.startup_timeout:g} seconds: {last_error}'
                ) from last_error
            try:
                await asyncio.wait_for(check_mcp_tools(cfg), timeout=min(1.0, remaining))
                break
            except Exception as exc:
                last_error = exc
            try:
                await asyncio.wait_for(process.wait(), timeout=min(0.5, remaining))
            except TimeoutError:
                pass
        yield
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
        await process.wait()
