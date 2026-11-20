"""Jnana hypothesis tournament: generate (or top up) hypotheses, Elo-rank them, return the top k.

Rankings are model-mediated and advisory; they are never evidence.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from unittest.mock import patch

from .config import Config, RateLimits, TournamentConfig
from .rate_limits import jnana_coscientist
from .schemas import Hypothesis

log = logging.getLogger('loam')

class GenerationTimedOut(RuntimeError):
    """Generation exceeded its cap; ranking must not race possibly live workers."""



class Tournament:
    def __init__(self, cfg: TournamentConfig, state_dir: Path, *, model_retries: int = 8,
                 request_timeout: float = 600, rate_limits: RateLimits | None = None):
        self.cfg = cfg
        self.model_retries = model_retries
        self.request_timeout = request_timeout
        self.rate_limits = rate_limits
        self.state_file = state_dir / 'jnana.json'
        if str(cfg.jnana_root) not in sys.path:
            sys.path.insert(0, str(cfg.jnana_root))

    @classmethod
    def for_campaign(cls, cfg: Config) -> Tournament:
        """This campaign's tournament: Jnana's model requests get the campaign's SDK retries,
        request timeout and shared rate-limit gate, like the agents' own."""
        return cls(cfg.tournament, cfg.state_dir, model_retries=cfg.model_retries,
                   request_timeout=cfg.timeouts.model_request, rate_limits=cfg.rate_limits)

    def run(self, goal: str, *, initial: bool, brief: str = '') -> list[Hypothesis]:
        """Blocking (Jnana is threaded); call through asyncio.to_thread."""
        import importlib

        from jnana.protognosis.core.coscientist import CoScientist
        periodic = importlib.import_module('jnana.protognosis.periodic_tournament')
        # Generation and ranking both use this subclass, so their requests pass the shared gate.
        # Ranking builds its own CoScientist from periodic's module global, patched for the call only.
        campaign_scientist = jnana_coscientist(CoScientist, retries=self.model_retries,
                                               timeout=self.request_timeout, limits=self.rate_limits)

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        cs = campaign_scientist(llm_config=self.cfg.generation.jnana(), storage_path=str(self.state_file),
                                max_workers=max(self.cfg.initial_hypotheses, self.cfg.new_per_round))
        existing = len(cs.get_all_hypotheses())
        count = max(0, self.cfg.initial_hypotheses - existing) if initial else self.cfg.new_per_round
        if count:
            steering = f'\n\nSTEERING FROM THE LAST ROUND (advisory, from evidence review):\n{brief}' if brief else ''
            cs.set_research_goal(goal + steering)
            cs.start()
            completed = False
            try:
                cs.generate_hypotheses(count=count, batch_size=self.cfg.gen_batch_size)
                completed = cs.wait_for_completion(timeout=self.cfg.max_seconds)
                if completed:
                    cs.memory.save()
            finally:
                cs.stop()
            if not completed:
                raise GenerationTimedOut(
                    f'Jnana generation exceeded {self.cfg.max_seconds}s; ranking was not started')

        # scheme/rounds/workers were previously left at Jnana's defaults, which
        # silently gave every campaign a single serial batch of random pairings.
        # match_count is still passed because Jnana ignores it under swiss.
        with patch.object(periodic, 'CoScientist', campaign_scientist):
            result = periodic.run_incremental_tournament(
                state_file=self.state_file, llm_config=self.cfg.ranking.jnana(),
                match_count=self.cfg.matches, top_k=self.cfg.top_k, max_seconds=self.cfg.max_seconds,
                scheme=self.cfg.scheme, rounds=self.cfg.swiss_rounds, workers=self.cfg.workers)
        bodies = {h.hypothesis_id: h for h in cs.get_all_hypotheses()}
        return [
            Hypothesis(id=r['hypothesis_id'], summary=r['summary'], elo=float(r['elo_rating']),
                       matches=_match_count(r.get('tournament_matches')),
                       content=str(getattr(bodies.get(r['hypothesis_id']), 'content', '') or r['summary']))
            for r in result.get('top_hypotheses', [])
        ]

    def pool(self) -> dict[str, Hypothesis]:
        """Every valid hypothesis in Jnana state; malformed persisted rows are skipped."""
        try:
            state = json.loads(self.state_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        if not isinstance(state, dict) or not isinstance(state.get('hypotheses', []), list):
            return {}
        pool = {}
        for row in state.get('hypotheses', []):
            if not isinstance(row, dict):
                continue
            hypothesis_id = row.get('hypothesis_id')
            summary = ' '.join(str(row.get('summary') or '').split())[:240]
            content = str(row.get('content') or row.get('summary') or '').strip()
            if not isinstance(hypothesis_id, str) or not hypothesis_id.strip() or not summary or not content:
                continue
            try:
                elo = float(row.get('elo_rating') or 1200.0)
                if not math.isfinite(elo):
                    continue
                hypothesis = Hypothesis(id=hypothesis_id, summary=summary, content=content, elo=elo,
                                        matches=_match_count(row.get('tournament_matches')))
            except (TypeError, ValueError):
                continue
            pool[hypothesis.id] = hypothesis
        return pool

    def top(self) -> list[Hypothesis]:
        """The pool's top_k by Elo, without running a tournament: the fallback when Jnana fails."""
        return sorted(self.pool().values(), key=lambda h: -h.elo)[:self.cfg.top_k]


def _match_count(value) -> int:
    """Jnana has stored this as either a count or a list of match records."""
    if isinstance(value, list):
        return len(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
