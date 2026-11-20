"""Round synthesis: reduce a round's threads and steering table to the next brief, within the context.

History trimming cannot fix a first request larger than the model's context. So no request here
exceeds a byte budget: the evidence is packed into leaf requests, each covering an explicit portion,
and each level's compact briefs are reduced again until one brief remains. Every input record is
sent exactly once; a record too large for one request is split losslessly into fragments.

Raw evidence and every request and response stay on disk under
state_dir/synthesis/round-NNN/<id>/, with a manifest as the receipt. The reasoner's summaries are
navigation aids, never new accepted claims.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from .workspace import write_json

# Deliberately conservative: the campaign model has a 131072-token context, which must hold the
# instructions, the request and the reasoner's output. Bytes stand in for tokens (a token is >= 1 byte).
CONTEXT_BYTES = 120_000
MAX_PROMPT_BYTES = 64_000
MIN_PROMPT_BYTES = 16_000
OUTPUT_TOKENS = 8192            # reserved for the reasoner's reply
MAX_LEVELS = 12


def encoded(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def size(text: str) -> int:
    return len(text.encode('utf-8'))


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
    """Pack every item exactly once into groups whose prompt fits the budget, fragmenting
    an item too large for a prompt of its own. Concatenating a record's fragments restores it."""
    fragments = []
    for index, item in enumerate(items):
        if size(prompt([item], level)) <= budget:
            fragments.append(item)
            continue
        text = encoded(item)
        # JSON escaping can enlarge a fragment up to 12x (a 4-byte code point as two \u escapes);
        # every packed prompt is still measured below. Slicing a str keeps code points whole.
        width = max(1, (budget - size(prompt([], level)) - 1024) // 12)
        pieces = [text[i:i + width] for i in range(0, len(text), width)]
        fragments.extend({'kind': 'record_fragment', 'record': index, 'part': n,
                          'parts': len(pieces), 'json_fragment': piece}
                         for n, piece in enumerate(pieces))
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def synthesize(agents, cfg, digest: list[dict], table: list[dict], previous_brief: str,
                     round_number: int) -> dict:
    """The round's brief: {'report', 'tournament_brief', 'synthesis_receipt'}. Raises if the
    reasoner fails or the instructions leave too little room; the caller keeps the previous brief."""
    instructions = (cfg.goal_text() + cfg.rules_text()
                    + (Path(__file__).parent / 'prompts/reasoner.md').read_text())
    budget = min(MAX_PROMPT_BYTES, (cfg.max_history_bytes or 2 * MAX_PROMPT_BYTES) // 2,
                 CONTEXT_BYTES - size(instructions) - OUTPUT_TOKENS)
    if budget < MIN_PROMPT_BYTES:
        raise ValueError('Campaign instructions leave too little safe synthesis context')
    directory = cfg.state_dir / 'synthesis' / f'round-{round_number:03d}' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    write_json(directory / 'input.json', {'threads': digest, 'steering': table, 'previous_brief': previous_brief})
    concurrency = min(cfg.synthesis_concurrency, cfg.concurrency)
    manifest = {'status': 'running', 'request_concurrency': concurrency,
                'input_sha256': _sha256(directory / 'input.json'), 'prompt_budget_bytes': budget,
                'thread_count': len(digest), 'steering_count': len(table), 'calls': [],
                'scope': 'Derived synthesis only; full accepted claims and provenance retained in input.json'}
    write_json(directory / 'manifest.json', manifest)
    items = ([{'kind': 'previous_brief', 'text': previous_brief}]
             + [{'kind': 'thread', 'value': value} for value in digest]
             + [{'kind': 'steering', 'value': value} for value in table])
    gate = asyncio.Semaphore(concurrency)

    async def call(group: list[dict], level: int, index: int) -> dict:
        node = f'level-{level:02d}-part-{index:04d}'
        text = prompt(group, level)
        write_json(directory / f'{node}-request.json', {'prompt': text})
        info = {'node': node, 'prompt_bytes': size(text), 'inputs': len(group)}
        manifest['calls'].append(info)
        try:
            async with gate:
                result = (await agents.run('reasoner', text)).model_dump()
                write_json(directory / f'{node}-response.json', result)
            info.update(status='complete', output_bytes=size(encoded(result)),
                        brief_bytes=size(result['tournament_brief']),
                        response_sha256=_sha256(directory / f'{node}-response.json'))
            return {'node': node, **result}
        except BaseException as exc:
            info.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            write_json(directory / 'manifest.json', manifest)

    try:
        for level in range(MAX_LEVELS):
            groups = partition(items, budget, level)
            results = await asyncio.gather(*(call(group, level, i) for i, group in enumerate(groups)),
                                           return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            if len(results) == 1:
                [final] = results
                manifest.update(status='complete', levels=level + 1, final_node=final['node'])
                write_json(directory / 'manifest.json', manifest)
                return {'report': final['report'], 'tournament_brief': final['tournament_brief'],
                        'synthesis_receipt': str(directory / 'manifest.json')}
            # Carry each child's compact brief upward; its full report stays on disk, named by hash,
            # so a long but valid report cannot overflow the next level's request.
            items = [{'kind': 'derived_summary', 'node': r['node'], 'tournament_brief': r['tournament_brief'],
                      'full_response_file': f"{r['node']}-response.json",
                      'full_response_sha256': _sha256(directory / f"{r['node']}-response.json")}
                     for r in results]
        raise ValueError('Synthesis reduction failed to converge')
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__)
        write_json(directory / 'manifest.json', manifest)
        raise
