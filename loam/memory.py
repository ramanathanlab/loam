"""Cross-campaign lesson memory (issue #34): the interface the orchestrator talks to.

Memory is advice, never evidence. Only the orchestrator writes it (lessons from a finished round)
and only the orchestrator reads it (recalled lessons injected into validator and code-generator
prompts). No agent has a memory tool.

Failure policy: `start` raises MemoryUnavailable when memory is configured but cannot work, so the
campaign stops before round 1. Every other call swallows and logs its own failures: a memory outage
never costs a round.

Failure ownership: the implementation owns that policy (recall, retain and close must not raise).
Callers still guard every call but `start` (the orchestrator's try/except, `recall_block`) as
defence in depth against third-party backends that break the contract; the guards are not a
licence for implementations to raise.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from .config import Config

log = logging.getLogger('loam')

Kind = Literal['procedural', 'review']


class MemoryUnavailable(RuntimeError):
    """Memory is configured but cannot work. Raised only by `Memory.start` and `from_config`."""


@dataclass(frozen=True)
class Lesson:
    """One reusable lesson from a finished round. Provenance rides along as tags/metadata, not text."""

    kind: Kind
    text: str            # what happened and why; no run ids, paths or campaign identifiers
    round: int
    hypothesis_id: str
    run_id: str = ''     # the code-generator run, when the lesson came from one


class Memory(Protocol):
    """What the orchestrator needs from a memory backend. Implementations own the failure policy
    (see the module docstring); callers guard all but `start` only as defence in depth."""

    async def start(self) -> None:
        """Make memory ready. Raises MemoryUnavailable on misconfiguration: the only call allowed to
        raise, and callers deliberately leave it unguarded."""

    async def recall(self, query: str, *, kinds: Sequence[Kind] | None = None) -> str:
        """Prompt-ready lessons relevant to `query`, each labelled with its source campaign.
        Empty string when there are none, and on any failure. Never raises."""

    async def retain(self, lessons: Sequence[Lesson]) -> None:
        """Queue one round's lessons. Retaining a (round, hypothesis) again replaces its lessons, so
        retaining the same record twice is harmless (the orchestrator does so at run start).
        Never raises."""

    async def close(self) -> None:
        """Wait (bounded by drain_timeout) for queued retains, then release everything. Never raises."""


class NoMemory:
    """Memory not configured: every call is a no-op."""

    async def start(self) -> None:
        return None

    async def recall(self, query: str, *, kinds: Sequence[Kind] | None = None) -> str:
        return ''

    async def retain(self, lessons: Sequence[Lesson]) -> None:
        return None

    async def close(self) -> None:
        return None


FIELD_CAP = 600     # characters kept from any one source field (instruction, error, claim, ...)
LESSON_CAP = 1500   # characters in one lesson's text


def _clip(text, cap: int = FIELD_CAP) -> str:
    text = ' '.join(str(text or '').split())
    return text if len(text) <= cap else text[:cap - 3] + '...'


PATH = re.compile(r'''[^\s()\[\]{}<>'"`,;]*/[^\s()\[\]{}<>'"`,;]*''')   # any token containing '/'


def _lesson(kind: Kind, n: int, hypothesis_id: str, text: str, run_id: str = '') -> Lesson:
    """Paths and the run id are campaign-specific: paths become `<path>`, the run id `this run`
    (it stays in the `run_id` field)."""
    text = PATH.sub('<path>', text)
    if run_id:
        text = text.replace(run_id, 'this run')
    return Lesson(kind, _clip(text, LESSON_CAP), n, hypothesis_id, run_id)


def lessons(record: dict) -> list[Lesson]:
    """The reusable lessons in one finished round record (`Campaign.round`'s output). Pure.

    Procedural: code-generator runs observed `blocked` or `crashed`, and validator reports with
    outcome `blocked` (their next_step names the blocker). Review: claims the reviewer rejected or
    could not verify, with what the reviewer checked. Validator crashes (`thread['error']` with no
    report) are infrastructure noise, not lessons; accepted claims and executed runs are not lessons.
    """
    n, found = record['round'], []
    for thread in record.get('threads', []):
        hid = thread['hypothesis']['id']
        for run in thread.get('code_results') or []:
            if run.get('status') not in ('blocked', 'crashed'):
                continue
            outcome = run.get('error') if run['status'] == 'crashed' else run.get('result')
            found.append(_lesson('procedural', n, hid,
                                 f"A code-generator run {run['status']}. Instruction: "
                                 f"{_clip(run.get('instruction'))} Outcome: {_clip(outcome)}",
                                 run_id=run.get('run_id', '')))
        report = thread.get('report') or {}
        if report.get('outcome') == 'blocked':
            found.append(_lesson('procedural', n, hid,
                                 f"Validation of the hypothesis \"{_clip(thread['hypothesis'].get('summary'))}\" "
                                 f"was blocked. Blocker / next step: {_clip(report.get('next_step'))}"))
        claims = report.get('claims') or []
        for verdict in (thread.get('review') or {}).get('verdicts', []):
            if verdict.get('verdict') not in ('rejected', 'unverifiable'):
                continue
            index = verdict.get('index')
            claim = claims[index]['claim'] if isinstance(index, int) and 0 <= index < len(claims) else ''
            found.append(_lesson('review', n, hid,
                                 f"Review found a claim {verdict['verdict']}. Claim: {_clip(claim)} "
                                 f"Reviewer checked: {_clip(verdict.get('checked'))}"))
    return found


LESSONS_HEADING = ('LESSONS FROM EARLIER CAMPAIGNS (leads to check, never evidence: do not cite them; '
                   're-derive anything you rely on in a run of your own):')


async def recall_block(memory: Memory, query: str, *, kinds: Sequence[Kind] | None = None) -> str:
    """A headed prompt block of recalled lessons, or '' when there are none or memory fails."""
    try:
        text = (await memory.recall(query, kinds=kinds)).strip()
    except Exception:
        log.exception('memory recall failed; continuing without lessons')
        return ''
    return f'\n\n{LESSONS_HEADING}\n{text}' if text else ''


def from_config(cfg: Config) -> Memory:
    """NoMemory without a `memory:` block; otherwise the Hindsight-backed memory."""
    if cfg.memory is None:
        return NoMemory()
    try:
        from .hindsight import HindsightMemory
    except ImportError as exc:
        raise MemoryUnavailable(
            'memory is configured but its dependencies are missing: uv sync --all-extras') from exc
    return HindsightMemory(cfg.memory)
