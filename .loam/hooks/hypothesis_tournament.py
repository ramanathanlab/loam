"""Seed, refresh and rank a bounded Jnana hypothesis tournament before an LLM call.

Advisory only: the rankings it returns are model-mediated, not empirical validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import knowledge_brief
from campaign import campaign_context


class Tournament:
    # Keys a `generation:` / `ranking:` block in the profile may override.
    role_keys = (
        'provider',
        'model',
        'inference_endpoint',
        'endpoint_is_chat_completions',
        'api_key_env',
        'omit_temperature',
        'json_max_tokens',
    )
    brief_heading = (
        "Latest round synthesis from the campaign's reasoner (advisory, and "
        "unvalidated unless independently reconciled):"
    )

    def __init__(self, payload: Mapping[str, Any]):
        self.payload = payload
        self.config = payload.get('loam_hypothesis_tournament') or {}
        self.enabled = self.config.get('enabled') is True
        self.jnana_root = Path(self.config.get('jnana_root', '')).expanduser()
        self.state_file = Path(self.config.get('state_file', '')).expanduser()
        self.reports_dir = Path(self.config.get('reports_dir', '')).expanduser()

    def context(self) -> str:
        """Run one tournament iteration and return the rendered rankings."""
        if not self.enabled:
            return ''

        if str(self.jnana_root) not in sys.path:
            sys.path.insert(0, str(self.jnana_root))

        from jnana.protognosis.periodic_tournament import run_incremental_tournament

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        generation = self._generate()
        # The tournament must finish INSIDE the hook's own budget. _invoke kills
        # this subprocess at timeout_seconds and returns None after a bare
        # logger.warning, so an overrun silently costs the round its hypothesis
        # selection: no validators are dispatched and the compute allocation
        # goes unused. Hand jnana a deadline short of that kill so it returns a
        # ranking instead of being killed mid-tournament. Generation already
        # consumed part of the budget, so measure what is left.
        budget = float(self.config.get('timeout_seconds') or 900)
        spent = max(0.0, time.monotonic() - started)
        max_seconds = max(30.0, (budget - spent) * 0.8)
        result = dict(
            run_incremental_tournament(
                state_file=self.state_file,
                llm_config=self._llm('ranking', 1),
                match_count=self.config.get('matches_per_run', 1),
                top_k=self.config.get('top_k', 3),
                max_seconds=max_seconds,
            )
        )
        result['generation'] = generation
        self._report(result)
        return self._render(result)

    def latest_report(self) -> tuple[Path, Mapping[str, Any]] | None:
        """The newest completed tournament report for this workflow, and its path."""
        if not self.config.get('reports_dir'):
            return None
        workflow_id = self.payload.get('workflow_id', '')
        for path in sorted(self.reports_dir.glob('tournament-*.json'), reverse=True):
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                # Still being written. tool_policy reads this, and a raise there allows the call.
                continue
            if workflow_id and report.get('workflow_id') != workflow_id:
                continue
            if report['result'].get('status') == 'completed':
                return path, report
        return None

    def hypotheses(self) -> dict[str, Mapping[str, Any]]:
        """Every hypothesis in the Jnana state file, by id."""
        if not self.state_file.is_file():
            return {}
        try:
            state = json.loads(self.state_file.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return {}  # Jnana is mid-save
        return {item['hypothesis_id']: item for item in state.get('hypotheses', [])}

    def _wait_budget(self) -> float:
        """Seconds any single ensemble wait may take, from the hook's own budget.

        _invoke kills this subprocess at timeout_seconds and returns None after
        a bare logger.warning, which silently costs the round its hypothesis
        selection. Every wait in here must therefore end well short of that.
        Half the budget leaves room for the tournament that follows.
        """
        budget = float(self.config.get('timeout_seconds') or 900)
        return max(30.0, budget * 0.5)

    def _llm(self, role: str, min_json_tokens: int) -> Any:
        """The LLMConfig for one role: the shared config overlaid with that role's block."""
        from jnana.protognosis.core.multi_llm_config import LLMConfig

        override = {
            key: value for key, value in (self.config.get(role) or {}).items()
            if key in self.role_keys and str(value).strip()
        }
        resolved = {**self.config, **override}
        base_url = resolved['inference_endpoint'].strip().rstrip('/')
        if resolved.get('endpoint_is_chat_completions') is True:
            base_url = base_url.removesuffix('/chat/completions')
       
        api_key_env = resolved.get('api_key_env')
        
        return LLMConfig(
            provider=resolved['provider'],
            model=resolved['model'],
            api_key=os.environ.get(api_key_env, '') if api_key_env else 'local-no-auth',
            base_url=base_url,
            model_adapter={
                'omit_temperature': resolved.get('omit_temperature') is True,
                'json_max_tokens': max(min_json_tokens, resolved.get('json_max_tokens', 1024)),
            },
        )

    def _goal(self) -> tuple[str, str]:
        """The campaign goal when there is one, else the turn's message, and which it was."""
        campaign = campaign_context(self.payload)
        if campaign is not None and campaign.goal:
            return campaign.goal, f'campaign:{campaign.goal_name}'
        
        return ' '.join(self.payload.get('user_message', '').split()), 'user_message'

    def _compose(self, goal: str, source: str, knowledge: str) -> str:
        """The goal Jnana stores: objective, knowledge brief, then the reasoner brief.

        A turn-message goal already carries the reasoner brief (`prepare_payload`
        appends it to the message), so it is only added to a campaign goal.
        """
        blocks = [goal, knowledge]
        brief = self.payload.get('loam_reasoner_brief', '').strip()
        if brief and source.startswith('campaign:'):
            blocks.append(f'{self.brief_heading}\n{brief}')
        return '\n\n'.join(block for block in blocks if block)

    def _generate(self) -> dict[str, Any]:
        """Seed an empty pool, or refresh a seeded one when the campaign goal has moved."""
        from jnana.protognosis.core.coscientist import CoScientist

        initial = self.config.get('initial_hypotheses', 3)
        refresh = self.config.get('refresh_hypotheses', 0)
        llm = self._llm('generation', 4096)
        coscientist = CoScientist(
            llm_config=llm,
            storage_path=str(self.state_file),
            max_workers=max(initial, refresh),
        )
        # Start the agent ensemble ONLY when hypotheses must actually be
        # generated. CoScientist.start() launches every agent -- generation,
        # reflection, ranking, evolution, proximity, meta-review -- and they
        # enqueue their own work indefinitely, so the matching stop() has to
        # drain a queue that keeps refilling. Counting hypotheses, re-setting
        # the goal and stamping the digest are all plain memory operations that
        # need no workers at all.
        #
        # Measured 2026-10-31: starting it unconditionally with
        # max_workers=max(initial, refresh)=10 held this hook for 7+ minutes on
        # a 6-hypothesis pool with refresh_hypotheses: 0 -- adding 25 unasked
        # tournament matches on the way -- and the round never reached the
        # tournament, let alone a validator. periodic_tournament escapes this
        # only because it passes max_workers=1.
        started = False

        def ensure_started() -> None:
            nonlocal started
            if not started:
                coscientist.start()
                started = True

        try:
            existing = len(coscientist.get_all_hypotheses())
            goal, source = self._goal()
            if existing < 2:
                if not goal:
                    return {'status': 'skipped', 'reason': 'missing_research_goal', 'generated_hypotheses': 0}
                ensure_started()
                return self._seed(coscientist, llm, goal, source, existing, max(initial, 2 - existing))

            outcome: dict[str, Any] = {'status': 'not_needed', 'generated_hypotheses': 0}
            if not goal:
                return outcome
            outcome['goal_source'] = source
            if not source.startswith('campaign:'):
                # The turn's message differs every turn, so "moved" carries no signal.
                outcome['goal_sync'] = 'skipped_volatile_goal'
                return outcome
            # Both calls below hit the LLM, and pre-llm.py builds every context
            # block under one try, so a provider error must not escape.
            try:
                outcome.update(
                    self._refresh(coscientist, goal, source, existing, refresh, ensure_started))
            except Exception as exc:  # noqa: BLE001 - advisory: never block the turn
                outcome['goal_sync'] = f'failed: {type(exc).__name__}: {exc}'
            return outcome
        finally:
            if started:
                coscientist.stop()

    def _seed(self, coscientist: Any, llm: Any, goal: str, source: str, existing: int, count: int) -> dict[str, Any]:
        knowledge = self._knowledge(goal, llm)
        brief = knowledge.get('brief', '')
        stored = self._compose(goal, source, brief)
        # Always set: Jnana does not restore its in-memory goal from a loaded
        # state, and generation reads that copy, not the persisted one.
        coscientist.set_research_goal(stored)
        coscientist.generate_hypotheses(count=count)
        # Bounded: the ensemble refills its own queue, so an unbounded wait here
        # never returns and the round dies inside the pre-llm hook. Partial
        # seeding is fine -- whatever landed is persisted below and the next
        # turn tops it up.
        drained = coscientist.wait_for_completion(timeout=self._wait_budget())
        coscientist.memory.metadata['loam_knowledge_brief'] = brief
        coscientist.memory.metadata['loam_goal_sha256'] = self._digest(stored)
        coscientist.memory.save()

        result: dict[str, Any] = {
            'status': 'completed' if drained else 'seeded_partial',
            'goal_source': source,
            'generated_hypotheses': len(coscientist.get_all_hypotheses()) - existing,
        }
        if not drained:
            # Say so rather than reporting a clean seed: the pool is whatever
            # landed before the budget ran out, and the next turn tops it up.
            result['seed_wait'] = f'timed out after {self._wait_budget():.0f}s'
        if knowledge:
            result['knowledge'] = {
                key: knowledge.get(key) for key in ('status', 'tool_calls', 'tools_used', 'cached', 'seconds')
            }
            result['knowledge']['brief_chars'] = len(brief)
        return result

    def _refresh(self, coscientist: Any, goal: str, source: str, existing: int, refresh: int,
                 ensure_started: Any = None) -> dict[str, Any]:
        """Re-set a moved goal and generate `refresh` more hypotheses against it.

        The digest is only recorded once the refresh has added something, so a
        refresh that generated nothing is retried on the next turn.
        """
        metadata = coscientist.memory.metadata
        stored = self._compose(goal, source, metadata.get('loam_knowledge_brief', ''))
        digest = self._digest(stored)
        if metadata.get('loam_goal_sha256') == digest:
            return {'goal_sync': 'unchanged'}

        coscientist.set_research_goal(stored)
        outcome: dict[str, Any] = {'goal_sync': 'set'}
        entered = 0
        if refresh:
            # Only this branch needs the agent ensemble running; the caller
            # keeps it stopped otherwise. See the note in _generate().
            if ensure_started is not None:
                ensure_started()
            coscientist.generate_hypotheses(count=refresh)
            # Bounded for the same reason as _seed: this wait used to hold the
            # whole round open indefinitely while the ensemble kept working.
            drained = coscientist.wait_for_completion(timeout=self._wait_budget())
            entered = len(coscientist.get_all_hypotheses()) - existing
            outcome.update(
                refresh_requested=refresh,
                generated_hypotheses=entered,
                refresh_drained=drained,
                status='refreshed' if entered else 'refresh_yielded_nothing',
            )
        if entered or not refresh:
            coscientist.memory.metadata['loam_goal_sha256'] = digest
        coscientist.memory.save()
        return outcome

    def _knowledge(self, goal: str, llm: Any) -> dict[str, Any]:
        """Run the optional knowledge-graph pre-pass; {} when it is switched off."""
        knowledge = self.config.get('knowledge') or {}
        if knowledge.get('enabled') is not True:
            return {}
        # The pre-pass may use a stronger model on the same endpoint.
        client = {'base_url': llm.base_url, 'api_key': llm.api_key, 'model': knowledge.get('model') or llm.model}
        try:
            return knowledge_brief.brief_for(goal, knowledge, client, self.state_file.parent)
        except Exception as exc:  # noqa: BLE001 - advisory: never block generation
            return {'status': f'failed: {type(exc).__name__}: {exc}', 'brief': ''}

    def _report(self, result: Mapping[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        report = {
            'schema_version': 'loam_jnana_tournament_v1',
            'observed_at': now.isoformat(),
            'workflow_id': self.payload.get('workflow_id', ''),
            'loam_profile': self.payload.get('loam_profile', ''),
            'session_id': self.payload.get('session_id', ''),
            'loam_turn': self.payload.get('loam_turn'),
            'result': result,
        }
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f'tournament-{now:%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def _render(self, result: Mapping[str, Any]) -> str:
        rankings = result.get('top_hypotheses')
        if result.get('status') != 'completed' or not rankings:
            return ''

        lines = [
            'Jnana hypothesis tournament update (advisory; model-mediated ranking, not empirical validation):',
            '- Trigger: Loam turn {turn}; ran {completed}/{requested} pairwise matches.'.format(
                turn=self.payload.get('loam_turn', '?'),
                completed=result.get('matches_completed', 0),
                requested=result.get('matches_requested', 0),
            ),
        ]
        generation = result['generation']
        entered = generation.get('generated_hypotheses', 0)
        if generation.get('status') == 'refreshed' and entered:
            # An entrant's 1200.0 is its initial rating, not a ranking.
            lines.append(
                f'- {entered} new hypotheses entered from the last round\'s synthesis. They start at '
                '1200.0 Elo and are unranked until they have played matches, so do not read an '
                'unplayed entrant\'s rating as a verdict.'
            )
        for rank, hypothesis in enumerate(rankings, start=1):
            summary = ' '.join(hypothesis.get('summary', '').split())[:180] or '(no summary)'
            lines.append(
                '- #{rank}: {identifier} — {elo:.1f} Elo; {summary}'.format(
                    rank=rank,
                    identifier=hypothesis.get('hypothesis_id', 'unknown'),
                    elo=hypothesis.get('elo_rating', 0.0),
                    summary=summary,
                )
            )

        lines.append('Prefer top-ranked candidates for refinement, while independently checking evidence and safety.')
        return '\n'.join(lines)

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
