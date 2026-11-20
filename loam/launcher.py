"""Lifecycle for the YAML-configured local server that fronts managed HPC."""

from __future__ import annotations

import asyncio
import os
import logging
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
    # The cluster probe may wait through the entire queue allowance. Give the
    # outer server deadline additional time for registration and client startup.
    startup_budget = deployment.startup_timeout + deployment.cold_endpoint_timeout + 600
    deadline = asyncio.get_running_loop().time() + startup_budget
    logger = logging.getLogger('loam')
    logger.info('Waiting for HPC allocation and server: %.1fh queue allowance plus %.0fs startup grace',
                deployment.startup_timeout / 3600, startup_budget - deployment.startup_timeout)
    next_progress = asyncio.get_running_loop().time() + 60
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
                    f'within {startup_budget:g} seconds: {last_error}'
                ) from last_error
            try:
                await asyncio.wait_for(check_mcp_tools(cfg), timeout=min(1.0, remaining))
                break
            except Exception as exc:
                last_error = exc
            now = asyncio.get_running_loop().time()
            if now >= next_progress:
                logger.info('Still waiting for HPC allocation/server; %.1fh startup allowance remaining',
                            max(0, deadline - now) / 3600)
                next_progress = now + 60
            try:
                await asyncio.wait_for(process.wait(), timeout=min(0.5, remaining))
            except TimeoutError:
                pass
        await _refresh_inference_token(cfg)
        yield
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
        await process.wait()


async def _refresh_inference_token(cfg: Config) -> None:
    """Queue waiting can outlive the access token obtained at launch."""
    command = os.environ.get('ALCF_TOKENS_BIN')
    specs = [*cfg.models.values(), cfg.tournament.generation, cfg.tournament.ranking]
    if not command or not any(s.api_key_env == 'ALCF_INFERENCE_TOKEN' for s in specs):
        return
    process = await asyncio.create_subprocess_exec(
        command, 'get-token', 'inference',
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=120)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError('ALCF token refresh timed out after HPC startup') from None
    token = stdout.decode().strip()
    if process.returncode or not token:
        raise RuntimeError('ALCF token refresh failed after HPC startup; run alcf-tokens login inference')
    os.environ['ALCF_INFERENCE_TOKEN'] = token
    logging.getLogger('loam').info('Refreshed ALCF inference token after HPC startup')
