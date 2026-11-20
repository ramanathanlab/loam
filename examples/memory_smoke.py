"""Opt-in smoke test for loam's Hindsight memory (issue #34). Not part of the test suite.

Starts `hindsight-api` as a child process on an embedded Postgres instance kept
apart from the real memory bank, then checks, against real model endpoints:
  1. the server starts and passes its health check (remote embeddings, rrf reranker, no torch);
  2. a lesson is retained synchronously with strict-schema extraction;
  3. a lesson is retained asynchronously and the queued operation completes;
  4. recall finds the lessons and returns loam's tags and metadata intact.
Then it stops the server.

    uv sync --extra memory
    .venv/bin/python examples/memory_smoke.py --base-url http://localhost:61612/v1 \\
        --llm-model argo:gpt-5.6-luna --embed-model argo:text-embedding-3-small

Exits 0 only if every check passes. The server log is written next to --log.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def server_env(args: argparse.Namespace, port: int) -> dict[str, str]:
    key = os.environ.get(args.api_key_env, '') if args.api_key_env else ''
    key = key or 'none'  # an OpenAI client refuses an empty key; argo-proxy ignores it
    return {
        **os.environ,
        'HINDSIGHT_API_DATABASE_URL': args.db,
        'HINDSIGHT_API_HOST': '127.0.0.1',
        'HINDSIGHT_API_PORT': str(port),
        'HINDSIGHT_API_WORKER_ID': 'loam-memory-smoke',
        'HINDSIGHT_API_LLM_PROVIDER': 'openai',
        'HINDSIGHT_API_LLM_BASE_URL': args.base_url,
        'HINDSIGHT_API_LLM_MODEL': args.llm_model,
        'HINDSIGHT_API_LLM_API_KEY': key,
        'HINDSIGHT_API_LLM_STRICT_SCHEMA_RETAIN': 'true',
        'HINDSIGHT_API_EMBEDDINGS_PROVIDER': 'openai',
        'HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL': args.base_url,
        'HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL': args.embed_model,
        'HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY': key,
        'HINDSIGHT_API_RERANKER_PROVIDER': 'rrf',
    }


def wait_healthy(proc: subprocess.Popen, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'hindsight-api exited with code {proc.returncode} before becoming healthy')
        try:
            if httpx.get(f'{url}/health', timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(f'hindsight-api not healthy after {timeout:.0f}s')


LESSONS = [
    ('Chai-1 folding of a 1,450-residue complex crashed with CUDA out-of-memory on a 40 GB A100; '
     'complexes above roughly 1,200 residues need an 80 GB GPU or must be cropped to the binding region.',
     {'kind': 'procedural', 'round': '2', 'hypothesis_id': 'smoke-h7', 'run_id': 'r2-1'}),
    ('The reviewer rejected a binding claim because the validator cited ipTM from a single Chai-1 seed; '
     'interface confidence claims need at least five seeds with the spread reported.',
     {'kind': 'review', 'round': '2', 'hypothesis_id': 'smoke-h3', 'run_id': ''}),
]


async def checks(url: str, bank: str) -> list[str]:
    from hindsight_client import Hindsight

    failures: list[str] = []
    client = Hindsight(base_url=url, timeout=600)
    try:
        await client.acreate_bank(
            bank, enable_observations=False,
            retain_mission='Extract reusable lessons about computational tools, skills, parameters, failure '
                           'modes and methodological pitfalls. Omit run ids, file paths and campaign-specific '
                           'identifiers.')
        print('ok   bank created/updated')

        (text, meta), (text2, meta2) = LESSONS
        tags = lambda m: ['campaign:smoke', f"kind:{m['kind']}"]  # noqa: E731

        t = time.monotonic()
        r = await client.aretain(bank, text, tags=tags(meta), metadata=meta,
                                 document_id='smoke:r2:smoke-h7', update_mode='replace')
        print(f'ok   sync retain ({time.monotonic() - t:.1f}s): items={r.items_count} usage={r.usage}')

        t = time.monotonic()
        r = await client.aretain(bank, text2, tags=tags(meta2), metadata=meta2,
                                 document_id='smoke:r2:smoke-h3', update_mode='replace', retain_async=True)
        op = r.operation_id or (r.operation_ids or [None])[0]
        if not op:
            failures.append(f'async retain returned no operation id: {r}')
        else:
            status = None
            while time.monotonic() - t < 600:
                status = await client.operations.get_operation_status(bank, op)
                if status.status in {'completed', 'failed', 'cancelled', 'not_found'}:
                    break
                await asyncio.sleep(2)
            if status is None or status.status != 'completed':
                failures.append(f'async retain did not complete: {status}')
            else:
                print(f'ok   async retain completed ({time.monotonic() - t:.1f}s)')

        for query, want_kind, filt in [
            ('fold a large protein complex with Chai-1 on GPU', 'procedural', ['kind:procedural']),
            ('how many seeds are needed for an ipTM interface claim', 'review', None),
        ]:
            res = await client.arecall(bank, query, max_tokens=1000, tags=filt,
                                       tags_match='any_strict' if filt else 'any')
            hits = [x for x in res.results if (x.metadata or {}).get('kind') == want_kind]
            print(f'..   recall {query!r}: {len(res.results)} result(s)')
            for x in res.results[:5]:
                print(f'       [{x.type}] tags={x.tags} metadata={x.metadata} :: {x.text[:110]}')
            if not hits:
                failures.append(f'recall {query!r} returned no {want_kind} lesson with metadata intact')
            elif 'campaign:smoke' not in (hits[0].tags or []):
                failures.append(f'recall {query!r}: campaign tag missing: {hits[0].tags}')
            else:
                print(f'ok   recall found the {want_kind} lesson with tags and metadata')
    finally:
        await client.aclose()
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-url', required=True, help='OpenAI-compatible endpoint for both LLM and embeddings')
    p.add_argument('--llm-model', required=True)
    p.add_argument('--embed-model', required=True)
    p.add_argument('--api-key-env', default=None, help='env var holding the endpoint key (omit for argo-proxy)')
    p.add_argument('--db', default='pg0://loam-memory-smoke', help='kept apart from the real memory bank')
    p.add_argument('--bank', default='loam-smoke')
    p.add_argument('--startup-timeout', type=float, default=300)
    p.add_argument('--log', type=Path, default=Path('memory_smoke.log'))
    args = p.parse_args()

    exe = shutil.which('hindsight-api', path=str(Path(sys.executable).parent)) or shutil.which('hindsight-api')
    if not exe:
        print('hindsight-api not found: uv sync --extra memory', file=sys.stderr)
        return 2
    port = free_port()
    url = f'http://127.0.0.1:{port}'
    log = args.log.open('w')
    print(f'..   starting {exe} on {url} (db {args.db}); log -> {args.log}')
    t = time.monotonic()
    proc = subprocess.Popen([exe, '--host', '127.0.0.1', '--port', str(port), '--no-access-log'],
                            env=server_env(args, port), stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_healthy(proc, url, args.startup_timeout)
        print(f'ok   healthy after {time.monotonic() - t:.1f}s')
        failures = asyncio.run(checks(url, args.bank))
    except Exception as exc:
        failures = [repr(exc)]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
    torch_loaded = 'torch' in args.log.read_text().lower() and 'import torch' in args.log.read_text()
    if torch_loaded:
        failures.append('server log mentions importing torch')
    for f in failures:
        print(f'FAIL {f}')
    print('PASS' if not failures else f'{len(failures)} check(s) failed')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
