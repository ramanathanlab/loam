"""Fresh ALCF inference tokens for the endpoints configured with `token_refresh: alcf`.

An ALCF access token can expire while a campaign waits in a scheduler queue, so loam fetches a
fresh one just before round 1 with the `alcf-tokens` CLI (`pip install loam[alcf-auth]`, then
`alcf-tokens login inference` once). `ALCF_TOKENS_BIN` names another install, e.g. a separate venv.
Endpoints without `token_refresh` are never touched.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

from .config import Config

TIMEOUT = 120


def _alcf_tokens() -> str:
    command = os.environ.get('ALCF_TOKENS_BIN') or shutil.which(
        'alcf-tokens', path=os.pathsep.join([str(Path(sys.executable).parent), os.environ.get('PATH', '')]))
    if not command:
        raise RuntimeError('token_refresh: alcf needs the alcf-tokens CLI: pip install loam[alcf-auth], '
                           'or set ALCF_TOKENS_BIN')
    return command


async def _fetch(command: str) -> str:
    process = await asyncio.create_subprocess_exec(command, 'get-token', 'inference', stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.DEVNULL)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=TIMEOUT)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError('ALCF token refresh timed out') from None
    token = stdout.decode().strip()
    if process.returncode or not token:
        raise RuntimeError(f'ALCF token refresh failed; authenticate with: {command} login inference')
    return token


async def refresh_tokens(cfg: Config) -> None:
    """Set a fresh token in each api_key_env that an endpoint with `token_refresh: alcf` reads."""
    specs = [*cfg.models.values(), cfg.tournament.generation, cfg.tournament.ranking]
    if cfg.memory is not None:
        specs += [cfg.memory.llm, cfg.memory.embeddings]
    variables = sorted({spec.api_key_env for spec in specs if spec.token_refresh == 'alcf'})
    if not variables:
        return
    token = await _fetch(_alcf_tokens())
    for variable in variables:
        os.environ[variable] = token
    logging.getLogger('loam').info('Refreshed the ALCF inference token in %s', ', '.join(variables))
