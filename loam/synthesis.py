"""Bound the *first* synthesis request, retaining a complete reduction lineage.

History trimming cannot fix a first request larger than the model context. Each
leaf consumes an explicit portion of the evidence, and each reduction consumes
all its children. Raw evidence and every request/response remain on disk; the
language-model summaries are navigation aids, never new accepted claims.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from .workspace import write_json


def encoded(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def size(value: str) -> int:
    return len(value.encode('utf-8'))


def prompt(items: list[dict], level: int) -> str:
    return (
        f'TOURNAMENT SYNTHESIS, reduction level {level}.\n'
        'Synthesize only the evidence supplied below. Intermediate summaries are '
        'derived navigation, not new observations or independently accepted claims. '
        'Preserve hypothesis IDs, source/dependency identifiers, uncertainty, negative '
        'results and concrete discriminating tests. Do not infer unseen results. '
        'This may be one portion of the round: scope conclusions to this portion. '
        'At higher levels combine ALL supplied portions without double-counting '
        'shared observations. Use the ordinary report/tournament_brief schema. '
        'Higher levels receive compact steering briefs from every child; the '
        'referenced full reports are archived lineage, not additional text you have read. '
        'Keep report under 5000 UTF-8 bytes and tournament_brief under 1800 UTF-8 '
        'bytes; do not enumerate every thread in the brief.\n\n'
        + encoded(items)
    )


def partition(items: list[dict], budget: int, level: int) -> list[list[dict]]:
    """Pack every input exactly once, fragmenting oversized single records losslessly."""
    fragments = []
    for index, item in enumerate(items):
        if size(prompt([item], level)) <= budget:
            fragments.append(item)
            continue
        text = encoded(item)
        # JSON escaping can enlarge a text fragment; check the actual request
        # below as well. Splitting Python strings preserves Unicode code points.
        width = max(1, (budget - size(prompt([], level)) - 1024) // 12)
        pieces = [text[i:i + width] for i in range(0, len(text), width)]
        fragments.extend({'kind': 'record_fragment', 'record': index, 'part': n,
                          'parts': len(pieces), 'json_fragment': part}
                         for n, part in enumerate(pieces))
    groups, current = [], []
    for item in fragments:
        if size(prompt([item], level)) > budget:
            raise ValueError('Cannot fit a synthesis record fragment in the input budget')
        if current and size(prompt(current + [item], level)) > budget:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


async def synthesize(agents, cfg, digest, table, previous_brief, round_number):
    instructions = (cfg.goal_text() + cfg.rules_text()
                    + (Path(__file__).parent / 'prompts/reasoner.md').read_text())
    # Deliberately conservative UTF-8 byte limit. The campaign model has a
    # 131072-token context, including instructions and configured output tokens.
    output_tokens = int(cfg.models['reasoner'].settings.get('max_tokens') or 8192)
    budget = min(64000, (cfg.max_history_bytes or 96000) // 2,
                 120000 - size(instructions) - output_tokens)
    if budget < 16000:
        raise ValueError('Campaign instructions leave too little safe synthesis context')
    directory = cfg.state_dir / 'synthesis' / f'round-{round_number:03d}' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    raw = {'threads': digest, 'steering': table, 'previous_brief': previous_brief}
    write_json(directory / 'input.json', raw)
    concurrency = min(cfg.synthesis_concurrency, cfg.concurrency)
    manifest = {'status': 'running', 'request_concurrency': concurrency, 'input_sha256': hashlib.sha256(
        (directory / 'input.json').read_bytes()).hexdigest(), 'prompt_budget_bytes': budget,
        'thread_count': len(digest), 'steering_count': len(table), 'calls': [],
        'scope': 'Derived synthesis only; full accepted claims and provenance retained in input.json'}
    write_json(directory / 'manifest.json', manifest)
    items = ([{'kind': 'previous_brief', 'text': previous_brief}]
             + [{'kind': 'thread', 'value': value} for value in digest]
             + [{'kind': 'steering', 'value': value} for value in table])
    gate = asyncio.Semaphore(concurrency)

    async def call(group, level, index):
        node = f'level-{level:02d}-part-{index:04d}'
        text = prompt(group, level)
        write_json(directory / (node + '-request.json'), {'prompt': text})
        info = {'node': node, 'prompt_bytes': size(text), 'inputs': len(group)}
        manifest['calls'].append(info)
        try:
            async with gate:
                result = (await agents.run('reasoner', text)).model_dump()
                write_json(directory / (node + '-response.json'), result)
            info.update(status='complete', output_bytes=size(encoded(result)),
                        brief_bytes=size(result['tournament_brief']),
                        response_sha256=hashlib.sha256((directory/(node+'-response.json')).read_bytes()).hexdigest())
            return {'kind': 'derived_summary', 'node': node, **result}
        except BaseException as exc:
            info.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            write_json(directory / 'manifest.json', manifest)

    try:
        for level in range(12):
            groups = partition(items, budget, level)
            results = await asyncio.gather(*(call(group, level, i) for i, group in enumerate(groups)),
                                           return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            if len(results) == 1:
                final = results[0]
                manifest.update(status='complete', levels=level + 1, final_node=final['node'])
                write_json(directory / 'manifest.json', manifest)
                return {'report': final['report'], 'tournament_brief': final['tournament_brief'],
                        'synthesis_receipt': str(directory / 'manifest.json')}
            # The compact brief is the agent's durable steering memory. Carry
            # that upward, retaining the complete detailed report by hash.
            # A valid long report must not invalidate an otherwise safe round;
            # every subsequent *request* is still partitioned to the byte cap.
            items = [{'kind':'derived_summary','node':r['node'],
                      'tournament_brief':r['tournament_brief'],
                      'full_response_file':r['node']+'-response.json',
                      'full_response_sha256':hashlib.sha256(
                          (directory/(r['node']+'-response.json')).read_bytes()).hexdigest()}
                     for r in results]
        raise ValueError('Synthesis reduction failed to converge')
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__)
        write_json(directory / 'manifest.json', manifest)
        raise
