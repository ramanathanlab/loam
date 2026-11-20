"""Cross-campaign lesson memory (issue #34): the interface the orchestrator talks to.

Memory is advice, never evidence. Only the orchestrator writes it (lessons from a finished round)
and only the orchestrator reads it (recalled lessons injected into validator and code-generator
prompts). No agent has a memory tool.

Failure policy: `start` raises MemoryUnavailable when memory is configured but cannot work, so the
campaign stops before round 1. Every other call swallows and logs its own failures: a memory outage
never costs a round.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from .config import Config

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
    async def start(self) -> None:
        """Make memory ready. Raises MemoryUnavailable on misconfiguration."""

    async def recall(self, query: str, *, kinds: Sequence[Kind] | None = None) -> str:
        """Prompt-ready lessons relevant to `query`, each labelled with its source campaign.
        Empty string when there are none, and on any failure. Never raises."""

    async def retain(self, lessons: Sequence[Lesson]) -> None:
        """Queue one round's lessons. Retaining a (round, hypothesis) again replaces its lessons.
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
