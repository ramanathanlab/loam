"""Hindsight-backed cross-campaign memory (issue #35). See loam/memory.py for the interface and policy."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import httpx
from hindsight_client import Hindsight   # the `memory` extra; missing -> memory.from_config's ImportError

from .config import MemoryConfig, ModelSpec
from .memory import Kind, Lesson, MemoryUnavailable

log = logging.getLogger('loam')

DEFAULT_HOME = Path.home() / '.loam' / 'memory'
WORKER_ID = 'loam-memory'   # fixed: Hindsight recovers queued work on restart by worker id
STARTUP_TIMEOUT = 300.0     # first start initialises the embedded Postgres; ~13 s seen, allow plenty
DONE = {'completed', 'failed', 'cancelled', 'not_found'}
RETAIN_MISSION = (
    'Extract reusable lessons about computational tools, skills, parameters, failure modes and '
    'methodological pitfalls, stated so they apply beyond the campaign they came from. Omit run ids, '
    'file paths, hypothesis ids and other campaign-specific identifiers.')


def _key(spec: ModelSpec, name: str) -> str:
    """The endpoint key. A named but unset api_key_env is a misconfiguration, not 'no key'."""
    if spec.api_key_env and not os.environ.get(spec.api_key_env):
        raise MemoryUnavailable(f'memory.{name}.api_key_env names {spec.api_key_env}, which is not set')
    return spec.api_key(custom_endpoint=True)   # 'none' without a key: the OpenAI client refuses ''


def _marker(cfg: MemoryConfig, marker_dir: Path) -> Path:
    return marker_dir / f'{cfg.instance_name}.json'


def check_embedding_model(cfg: MemoryConfig, marker_dir: Path) -> None:
    """Refuse to start a loam-owned database with an embedding model other than the one it was created
    with. Must run BEFORE the server starts: Hindsight resizes the embedding column at startup."""
    marker = _marker(cfg, marker_dir)
    if not marker.exists():
        return
    have, want = json.loads(marker.read_text()).get('embeddings_model'), cfg.embeddings.model
    if have != want:
        raise MemoryUnavailable(
            f'memory instance {cfg.instance_name!r} was created with embedding model {have!r}, but the config '
            f'names {want!r}. The embedding model is fixed per database: use a new memory.instance '
            f'(or switch back to {have!r}). Marker: {marker}')


def record_embedding_model(cfg: MemoryConfig, marker_dir: Path) -> None:
    """Pin the instance to its embedding model. Only once the server is healthy, i.e. has sized the
    vector column for it: a typo'd model or a failed first start must not lock the instance."""
    marker = _marker(cfg, marker_dir)
    if not marker.exists():
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'embeddings_model': cfg.embeddings.model}) + '\n')


TagKey = Literal['campaign', 'kind']


def tag(key: TagKey, value: str) -> str:
    """Encode a lesson tag: `campaign:<name>` or `kind:<kind>`."""
    return f'{key}:{value}'


def tag_value(tags: Sequence[str] | None, key: TagKey) -> str | None:
    """Decode the first `key` tag in `tags`, or None."""
    prefix = tag(key, '')
    return next((t.removeprefix(prefix) for t in tags or () if t.startswith(prefix)), None)


def retain_batches(lessons: Sequence[Lesson], campaign: str) -> list[list[dict]]:
    """One batch of retain items per (round, hypothesis), one item per lesson kind.

    Each item is its own replace-mode document, `{campaign}:r{round}:{hypothesis}:{kind}`, holding every
    lesson of that kind, so a re-run round replaces them. Hindsight's async retain refuses a batch with
    duplicate document_ids, and separate replace-mode retains of one document would delete each other,
    hence the merge."""
    groups: dict[tuple[int, str], dict[str, list[Lesson]]] = {}
    for lesson in lessons:
        groups.setdefault((lesson.round, lesson.hypothesis_id), {}).setdefault(lesson.kind, []).append(lesson)
    batches = []
    for (rnd, hypothesis_id), by_kind in groups.items():
        batches.append([{
            'content': '\n\n'.join(lesson.text for lesson in kind_lessons),
            'document_id': f'{campaign}:r{rnd}:{hypothesis_id}:{kind}',
            'tags': [tag('campaign', campaign), tag('kind', kind)],
            'metadata': {'round': str(rnd), 'hypothesis_id': hypothesis_id, 'kind': kind,
                         'run_id': ','.join(lesson.run_id for lesson in kind_lessons if lesson.run_id)},
            'update_mode': 'replace',
        } for kind, kind_lessons in by_kind.items()])
    return batches


def recall_tags(kinds: Sequence[Kind] | None) -> list[str] | None:
    return [tag('kind', k) for k in kinds] if kinds else None


def format_recall(results: Sequence) -> str:
    """One line per recalled fact, labelled with its source campaign. The campaign comes from the TAG:
    Hindsight may split one lesson into several facts, and metadata can be None on those."""
    lines = []
    for r in results:
        campaign = tag_value(r.tags, 'campaign') or 'unknown campaign'
        kind = tag_value(r.tags, 'kind')
        label = f'{campaign}, {kind}' if kind else campaign
        lines.append(f'- [{label}] {" ".join(r.text.split())}')
    return '\n'.join(lines)


def server_env(cfg: MemoryConfig, port: int) -> dict[str, str]:
    """Environment for a loam-owned `hindsight-api`. Everything not in the YAML is fixed here."""
    return {
        **os.environ,
        'HINDSIGHT_API_DATABASE_URL': f'pg0://{cfg.instance_name}',
        'HINDSIGHT_API_HOST': '127.0.0.1',
        'HINDSIGHT_API_PORT': str(port),
        'HINDSIGHT_API_WORKER_ID': WORKER_ID,
        'HINDSIGHT_API_LLM_PROVIDER': 'openai',
        'HINDSIGHT_API_LLM_BASE_URL': cfg.llm.base_url,
        'HINDSIGHT_API_LLM_MODEL': cfg.llm.model,
        'HINDSIGHT_API_LLM_API_KEY': _key(cfg.llm, 'llm'),
        'HINDSIGHT_API_LLM_STRICT_SCHEMA_RETAIN': 'true',
        'HINDSIGHT_API_EMBEDDINGS_PROVIDER': 'openai',
        'HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL': cfg.embeddings.base_url,
        'HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL': cfg.embeddings.model,
        'HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY': _key(cfg.embeddings, 'embeddings'),
        'HINDSIGHT_API_RERANKER_PROVIDER': 'rrf',
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class HindsightMemory:
    """Memory on a Hindsight server: loam's own child `hindsight-api` (no `url`), or a shared one (`url`).

    `home` holds loam's per-instance embedding-model markers and the server log; it lives outside the
    repo and is overridable for tests.
    """

    def __init__(self, cfg: MemoryConfig, *, home: Path | None = None):
        self.cfg = cfg
        self.home = home or DEFAULT_HOME
        self._client = None
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._pending: list[str] = []   # async retain operation ids, drained by close()

    # ---- lifecycle --------------------------------------------------------------------------------

    async def start(self) -> None:
        cfg = self.cfg
        try:
            for spec, name in ((cfg.llm, 'llm'), (cfg.embeddings, 'embeddings')):
                _key(spec, name)
            url = cfg.url.rstrip('/') if cfg.url else await self._launch()
            self._client = Hindsight(base_url=url, timeout=cfg.timeout)
            try:
                await asyncio.wait_for(
                    self._client.acreate_bank(cfg.bank, enable_observations=False, retain_mission=RETAIN_MISSION),
                    cfg.timeout)
            except Exception as exc:
                raise MemoryUnavailable(f'memory bank {cfg.bank!r} setup failed on {url}: {exc!r}') from exc
            log.info('memory ready: bank %r on %s (campaign %r)', cfg.bank, url, cfg.campaign)
        except BaseException:
            await self._release()
            raise

    async def _launch(self) -> str:
        """Start loam's own hindsight-api on the embedded instance; return its URL once healthy."""
        cfg = self.cfg
        port = _free_port()
        env = server_env(cfg, port)              # raises on an unset api_key_env
        check_embedding_model(cfg, self.home)    # BEFORE the server: it resizes the vector column at startup
        exe = (shutil.which('hindsight-api', path=str(Path(sys.executable).parent))
               or shutil.which('hindsight-api'))
        if not exe:
            raise MemoryUnavailable('hindsight-api not found: install the memory extra (uv sync --all-extras)')
        url = f'http://127.0.0.1:{port}'
        self.home.mkdir(parents=True, exist_ok=True)
        log_path = self.home / f'{cfg.instance_name}.server.log'
        self._log = log_path.open('a')
        log.info('starting %s on %s (instance %r); log -> %s', exe, url, cfg.instance_name, log_path)
        self._proc = subprocess.Popen([exe, '--host', '127.0.0.1', '--port', str(port), '--no-access-log'],
                                      env=env, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + STARTUP_TIMEOUT
        async with httpx.AsyncClient() as http:
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    raise MemoryUnavailable(f'hindsight-api exited with code {self._proc.returncode} before '
                                            f'becoming healthy; see {log_path}')
                with contextlib.suppress(httpx.HTTPError):
                    if (await http.get(f'{url}/health', timeout=2)).status_code == 200:
                        record_embedding_model(cfg, self.home)
                        return url
                await asyncio.sleep(1)
        raise MemoryUnavailable(f'hindsight-api not healthy after {STARTUP_TIMEOUT:.0f}s; see {log_path}')

    async def close(self) -> None:
        try:
            await self._drain()
        except Exception:
            log.exception('memory: draining queued retains failed')
        await self._release()

    async def _drain(self) -> None:
        if not self._client or not self._pending:
            return
        deadline = time.monotonic() + self.cfg.drain_timeout
        log.info('memory: waiting up to %.0fs for %d queued retain(s)', self.cfg.drain_timeout, len(self._pending))
        while self._pending and time.monotonic() < deadline:
            for op in list(self._pending):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    status = (await asyncio.wait_for(
                        self._client.operations.get_operation_status(self.cfg.bank, op),
                        min(self.cfg.timeout, remaining))).status
                except Exception as exc:
                    log.warning('memory: status of retain %s unknown: %r', op, exc)
                    continue
                if status in DONE:
                    self._pending.remove(op)
                    if status != 'completed':
                        log.warning('memory: retain %s ended %s', op, status)
            if self._pending:
                await asyncio.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        if self._pending:
            log.warning('memory: %d retain(s) still queued after %.0fs; they resume on the next start',
                        len(self._pending), self.cfg.drain_timeout)

    async def _release(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                log.exception('memory: closing the client failed')
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()   # the server drains for up to 20 s on SIGTERM
            try:
                await asyncio.to_thread(proc.wait, 30)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait)
        if self._log is not None:
            self._log.close()
            self._log = None

    # ---- recall / retain --------------------------------------------------------------------------

    async def recall(self, query: str, *, kinds: Sequence[Kind] | None = None) -> str:
        if self._client is None or not query.strip():
            return ''
        tags = recall_tags(kinds)
        try:
            res = await asyncio.wait_for(
                self._client.arecall(self.cfg.bank, query, max_tokens=self.cfg.recall_tokens, tags=tags,
                                     tags_match='any_strict' if tags else 'any'),
                self.cfg.timeout)
            return format_recall(res.results or [])
        except Exception as exc:
            log.warning('memory: recall failed, continuing without lessons: %r', exc)
            return ''

    async def retain(self, lessons: Sequence[Lesson]) -> None:
        if self._client is None:
            return
        for items in retain_batches(lessons, self.cfg.campaign):
            try:
                res = await asyncio.wait_for(
                    self._client.aretain_batch(self.cfg.bank, items, retain_async=True), self.cfg.timeout)
                ops = [o for o in (getattr(res, 'operation_ids', None) or [res.operation_id]) if o]
                self._pending.extend(ops)
            except Exception as exc:
                log.warning('memory: dropping lessons for %s: %r', [i['document_id'] for i in items], exc)
