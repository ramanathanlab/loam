"""Opt-in smoke test for loam's Hindsight memory (issues #34, #35). Not part of the test suite.

Drives `loam.hindsight.HindsightMemory` itself, on its own embedded Postgres instance kept apart from
the real memory bank, against real model endpoints:
  1. a different embedding model on an existing instance is refused before any server starts;
  2. start launches `hindsight-api`, waits for health and sets up the bank;
  3. retain queues one round's lessons (two for one hypothesis, one for another) asynchronously;
  4. close drains the queued retains and stops the server;
  5. start again on the same instance (the guard accepts the same model);
  6. recall finds the lessons, labelled with their campaign, and the kind filter keeps review
     lessons out of a procedural-only recall;
  7. close.

    uv sync --all-extras
    .venv/bin/python examples/memory_smoke.py --base-url http://localhost:61612/v1 \\
        --llm-model argo:gpt-5.6-luna --embed-model argo:text-embedding-3-small

Exits 0 only if every check passes. The server log and guard marker go under --home.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from loam.config import MemoryConfig
from loam.hindsight import DEFAULT_HOME, HindsightMemory
from loam.memory import Lesson, MemoryUnavailable

CAMPAIGN = 'memory-smoke'

LESSONS = [
    Lesson('procedural',
           'Chai-1 folding of a 1,450-residue complex crashed with CUDA out-of-memory on a 40 GB A100; '
           'complexes above roughly 1,200 residues need an 80 GB GPU or must be cropped to the binding region.',
           round=2, hypothesis_id='smoke-h7', run_id='r2-1'),
    Lesson('review',
           'The reviewer rejected a binding claim because it rested on contacts from a single folded model; '
           'an interface claim needs contacts that persist across several independent models.',
           round=2, hypothesis_id='smoke-h7'),
    Lesson('review',
           'The reviewer rejected a binding claim because the validator cited ipTM from a single Chai-1 seed; '
           'interface confidence claims need at least five seeds with the spread reported.',
           round=2, hypothesis_id='smoke-h3'),
]


def config(args: argparse.Namespace, embed_model: str | None = None) -> MemoryConfig:
    spec = lambda model: {'model': model, 'base_url': args.base_url, 'api_key_env': args.api_key_env}  # noqa: E731
    return MemoryConfig.model_validate({
        'campaign': CAMPAIGN, 'bank': args.bank, 'instance': args.instance,
        'llm': spec(args.llm_model), 'embeddings': spec(embed_model or args.embed_model),
        'timeout': 120, 'drain_timeout': 600})


async def run(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    cfg = config(args)

    # 1. guard: only meaningful once the instance has a marker, so record it first if this is a fresh home
    from loam.hindsight import check_embedding_model
    check_embedding_model(cfg, args.home)
    try:
        await HindsightMemory(config(args, embed_model='argo:some-other-embedding'), home=args.home).start()
        failures.append('a different embedding model was not refused')
    except MemoryUnavailable as exc:
        print(f'ok   guard refused a different embedding model: {str(exc)[:90]}...')

    # 2-4. start, retain, drain
    mem = HindsightMemory(cfg, home=args.home)
    t = time.monotonic()
    await mem.start()
    print(f'ok   started and bank ready ({time.monotonic() - t:.1f}s)')
    t = time.monotonic()
    await mem.retain(LESSONS)
    queued = len(mem._pending)
    print(f'ok   retain queued {queued} operation(s) ({time.monotonic() - t:.1f}s)')
    if queued < 2:
        failures.append(f'expected a queued retain operation per hypothesis (2), got {queued}')
    t = time.monotonic()
    await mem.close()
    if mem._pending:
        failures.append(f'{len(mem._pending)} retain(s) still pending after close')
    else:
        print(f'ok   close drained the retains and stopped the server ({time.monotonic() - t:.1f}s)')

    # 5-7. start again, recall
    mem = HindsightMemory(cfg, home=args.home)
    t = time.monotonic()
    await mem.start()
    print(f'ok   restarted on the same instance ({time.monotonic() - t:.1f}s)')
    try:
        for query, kinds, want, unwanted in [
            ('fold a large protein complex with Chai-1 on GPU', ['procedural'], 'out-of-memory', 'review'),
            ('how many seeds are needed for an ipTM interface claim', None, 'seed', None),
            ('are contacts from one folded model enough for an interface claim', ['review'], 'contact', 'procedural'),
        ]:
            text = await mem.recall(query, kinds=kinds)
            print(f'..   recall {query!r} kinds={kinds}:')
            for line in text.splitlines()[:6]:
                print(f'       {line[:150]}')
            ours = [line for line in text.splitlines() if f'[{CAMPAIGN}' in line]
            if not any(want.lower() in line.lower() for line in ours):
                failures.append(f'recall {query!r}: no {CAMPAIGN} lesson mentioning {want!r}')
            elif unwanted and any(f', {unwanted}]' in line for line in text.splitlines()):
                failures.append(f'recall {query!r}: kind filter let a {unwanted} lesson through')
            else:
                print(f'ok   recall found the lesson, labelled with campaign {CAMPAIGN!r}')
    finally:
        await mem.close()
    print('ok   closed')
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-url', required=True, help='OpenAI-compatible endpoint for both LLM and embeddings')
    p.add_argument('--llm-model', required=True)
    p.add_argument('--embed-model', required=True)
    p.add_argument('--api-key-env', default=None, help='env var holding the endpoint key (omit for argo-proxy)')
    p.add_argument('--instance', default='loam-memory-smoke', help='kept apart from the real memory database')
    p.add_argument('--bank', default='loam-smoke')
    p.add_argument('--home', type=Path, default=DEFAULT_HOME, help='guard markers and server log')
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    try:
        failures = asyncio.run(run(args))
    except Exception as exc:
        failures = [repr(exc)]
    log = args.home / f'{args.instance}.server.log'
    if log.exists() and 'import torch' in log.read_text():
        failures.append('server log mentions importing torch')
    for f in failures:
        print(f'FAIL {f}')
    print('PASS' if not failures else f'{len(failures)} check(s) failed')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
