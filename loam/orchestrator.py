"""The orchestrator: plain Python, no LLM. One round is a fixed sequence:

    tournament -> validators (parallel) -> reviewers -> reconcile -> steering -> reasoner -> brief

Everything a round produced is written to state_dir/rounds/round-NNN.json; the highest numeric
round filename is the round counter, so gaps cannot cause a completed round to be overwritten.

Failure policy: only an empty hypothesis pool stops a campaign. A failed tournament falls back
to the pool's current ranking; a failed validator, reviewer or code-generator marks that one
hypothesis untested for the round; a failed reasoner keeps the previous brief.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from . import execution, reconcile, skills, steering
from .agents import Agents, build
from .config import Config
from .execution import Deps
from .memory import Memory, lessons, recall_block
from .memory import from_config as memory_from_config
from .schemas import Hypothesis, ValidationReport
from .synthesis import synthesize
from .tournament import GenerationTimedOut, Tournament
from .workspace import Workspace, write_json

log = logging.getLogger('loam')


class Campaign:
    def __init__(self, cfg: Config, agents: Agents | None = None, tournament: Tournament | None = None,
                 memory: Memory | None = None):
        self.cfg = cfg
        self.skills = skills.discover(cfg.skills_dirs)
        self.agents = agents or build(cfg)
        self.tournament = tournament or Tournament.for_campaign(cfg)
        self.memory = memory if memory is not None else memory_from_config(cfg)
        self.rounds_dir = cfg.state_dir / 'rounds'
        self.ledger = cfg.state_dir / 'ledger.jsonl'
        self.rounds_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, rounds: int = 1) -> None:
        try:
            await self.memory.start()  # unguarded: misconfigured memory stops the campaign before round 1
            await self.retain_latest()
            if self.cfg.execution.mode in {'mcp', 'hpc'}:
                await execution.check_mcp_tools(self.cfg)
            for _ in range(rounds):
                await self.round()
        finally:
            try:
                await self.memory.close()
            except Exception:
                log.exception('memory close failed')

    def completed_rounds(self) -> list[tuple[int, Path]]:
        return [(int(path.stem.removeprefix('round-')), path) for path in self.rounds_dir.glob('round-*.json')
                if path.stem.removeprefix('round-').isdigit()]

    async def retain_latest(self) -> None:
        """Retain the latest completed round's lessons again: a crash between writing a round record and
        retaining its lessons would otherwise lose them. Idempotent: retain replaces by document id."""
        latest = max(self.completed_rounds(), default=None)
        if latest is not None:
            await self.retain(json.loads(latest[1].read_text()))

    async def retain(self, record: dict) -> None:
        try:
            await self.memory.retain(lessons(record))
        except Exception:  # a memory outage never costs a round
            log.exception('round %d: retaining lessons failed', record['round'])

    async def round(self) -> dict:
        completed = self.completed_rounds()
        n = max((number for number, _ in completed), default=0) + 1
        previous = max(completed, default=None)
        brief = json.loads(previous[1].read_text())['brief']['tournament_brief'] if previous else ''
        reconcile.drop_round(self.ledger, n)  # leftovers from a crashed attempt at this round

        log.info('round %d: tournament', n)
        ranked = await self.rank(initial=not completed, brief=brief)
        log.info('round %d: validating %d hypotheses', n, len(ranked))
        gate = asyncio.Semaphore(self.cfg.concurrency)
        threads = await asyncio.gather(*(self.validate(n, h, gate) for h in ranked))

        log.info('round %d: reasoner', n)
        table = self.steering(ranked)
        new_brief = await self.synthesize(threads, table, brief, n)
        record = {'round': n, 'rankings': [h.model_dump() for h in ranked], 'threads': threads,
                  'steering': table, 'brief': new_brief}
        path = self.rounds_dir / f'round-{n:03d}.json'
        write_json(path, record)
        await self.retain(record)
        log.info('round %d: done -> %s', n, path)
        return record

    async def rank(self, *, initial: bool, brief: str) -> list[Hypothesis]:
        """Run the tournament; if Jnana fails, validate the pool's current top_k by Elo instead."""
        try:
            ranked = await asyncio.to_thread(
                self.tournament.run, self.cfg.goal_text(), initial=initial, brief=brief)
            ranked = ranked or self.tournament.top()
        except GenerationTimedOut:
            raise  # generation workers may still own the state; this round must stop here
        except Exception:
            log.exception('tournament failed; falling back to the current pool ranking')
            try:
                ranked = self.tournament.top()
            except Exception:
                log.exception('could not read the current tournament pool')
                ranked = []
        if not ranked:
            raise RuntimeError('No hypotheses: the tournament produced none and the pool is empty')
        return ranked

    async def recite(self, report, deps, ws, n: int, h: Hypothesis):
        """Give the validator one chance to fix claims that cite unowned artifacts.

        Without this the drop is invisible: `reconcile.accepted` runs after both
        the validator and the reviewer have returned, so the agent cannot react
        and the round simply yields nothing. We spend at most one extra turn per
        hypothesis, and only when a claim is already doomed.

        The corrected report is kept only if it carries MORE citable claims, so
        a retry that panics and drops everything cannot make things worse.
        """
        bad = reconcile.unowned_sources(report, ws)
        if not bad:
            return report
        runs = reconcile.citable_runs(ws)
        log.warning('hypothesis %s r%d: %d of %d claim(s) cite artifacts no recorded run owns; '
                    're-citing', h.id, n, len(bad), len(report.claims))
        offenders = '\n'.join(f'  claim {index}: {source}' for index, source in bad)
        if runs:
            citable = '\n'.join(
                f"  {r['run_id']}  ->  {r['run_dir']}"
                + (f"  (batches: {', '.join(r['batches'])})" if r['batches'] else '')
                for r in runs)
        else:
            citable = '  (none -- no run has been opened for this hypothesis yet)'
        prompt = (
            'PROVENANCE CHECK FAILED. These claim sources will be DISCARDED, because no run '
            'recorded for this hypothesis owns them:\n' + offenders
            + '\n\nA claim may cite only a path inside a run directory opened for this '
            'hypothesis, or inside a batch one of those runs submitted. A directory you '
            "created yourself and another round's leftovers are not evidence, however real "
            'the numbers in them are.\n\nRuns you may cite:\n' + citable
            + '\n\nRe-issue your report. For each discarded claim, in order of preference: '
            'cite the owning path if that result already lives in one of the runs above; '
            'else RE-RUN the work so the result lands in a run of your own and cite that; '
            'else drop the claim and lower evidence_score to match the evidence you actually '
            'have. Keep every claim not listed above exactly as it was.')
        try:
            second = await self.agents.run('validator', prompt, deps)
        except Exception:
            log.exception('hypothesis %s r%d: re-cite turn failed; keeping the first report', h.id, n)
            return report
        before = reconcile.owned_claim_count(report, ws)
        after = reconcile.owned_claim_count(second, ws)
        if after > before:
            log.info('hypothesis %s r%d: re-cite recovered %d claim(s)', h.id, n, after - before)
            return second
        log.warning('hypothesis %s r%d: re-cite did not help (%d citable before, %d after); '
                    'keeping the first report', h.id, n, before, after)
        return report

    async def validate(self, n: int, h: Hypothesis, gate: asyncio.Semaphore) -> dict:
        """One hypothesis: validator -> independent reviewer -> reconcile. Never raises."""
        thread: dict = {'hypothesis': h.model_dump(), 'report': None, 'raw_report': None,
                        'review': None, 'accepted': [], 'code_results': [], 'error': None}
        async with gate:
            ws = None
            try:
                ws = Workspace(self.cfg, h)
                deps = Deps(self.cfg, self.skills, ws, n, hypothesis=f'{h.summary}\n\n{h.content}',
                            memory=self.memory)
                thread['code_results'] = deps.code_results  # same list: filled even if the validator dies
                prior = ws.experiment()['rounds']
                prompt = f'Validate this hypothesis (Jnana id {h.id}, Elo {h.elo:.0f}):\n\n{deps.hypothesis}'
                if prior:
                    prompt += f'\n\nPRIOR WORK ON THIS HYPOTHESIS (earlier rounds):\n{json.dumps(prior, indent=1)}'
                prompt += await recall_block(self.memory, h.summary)
                report = await self.agents.run('validator', prompt, deps)
                # Before the reviewer spends a pass on claims that provenance will
                # discard anyway, let the validator correct them.
                report = await self.recite(report, deps, ws, n, h)
                thread['raw_report'] = report.model_dump()
                review = await self.review(report, deps) if report.claims else None
                thread['review'] = review and review.model_dump()
                thread['accepted'] = reconcile.accepted(
                    report, review, unit=f'{h.id}:r{n}', workspace=ws) if report.claims else []
                thread['report'] = thread['raw_report']
                reconcile.record(self.ledger, n, h.id, thread['accepted'])
            except Exception as exc:  # an untested hypothesis, not a contradicted one
                log.exception('hypothesis %s failed', h.id)
                thread['error'] = repr(exc)
            if ws is not None:
                try:
                    r = thread['report'] or {}
                    ws.add_round({'round': n, 'elo': h.elo, 'outcome': r.get('outcome'),
                                  'evidence_score': r.get('evidence_score'), 'next_step': r.get('next_step'),
                                  'pending_batches': r.get('pending_batches'),
                                  'runs': [c['run_id'] for c in thread['code_results']],
                                  'accepted_claims': len(thread['accepted']), 'error': thread['error']})
                except Exception:
                    log.exception('could not record round %d for hypothesis %s', n, h.id)
        return thread

    async def review(self, report: ValidationReport, deps: Deps):
        """The reviewer sees the claims and where the outputs are, never the validator's prose."""
        claims = [{'index': i, **c.model_dump()} for i, c in enumerate(report.claims)]
        prompt = (f'HYPOTHESIS:\n{deps.hypothesis}\n\nCLAIMS TO REVIEW:\n{json.dumps(claims, indent=1)}\n\n'
                  f'Each code-generator run wrote its outputs under {deps.ws.root}/outputs/<run_id>/ -- read them there.')
        return await self.agents.run('reviewer', prompt, deps)

    def steering(self, ranked: list[Hypothesis]) -> list[dict]:
        """The steering table over the whole Jnana pool, with summaries so the reasoner can read it."""
        pool = self.tournament.pool() | {h.id: h for h in ranked}
        table = steering.project(self.ledger, {hid: h.elo for hid, h in pool.items()})
        for row in table:
            row['summary'] = pool[row['hypothesis_id']].summary if row['hypothesis_id'] in pool else ''
        return table

    async def synthesize(self, threads: list[dict], table: list[dict], previous_brief: str, n: int) -> dict:
        digest = []
        for t in threads:
            h, r = t['hypothesis'], t['report'] or {}
            digest.append({
                'hypothesis_id': h['id'], 'summary': h['summary'], 'elo': h['elo'], 'matches': h['matches'],
                'outcome': r.get('outcome'), 'self_reported_evidence_score': r.get('evidence_score'),
                'next_step': r.get('next_step'), 'pending_batches': r.get('pending_batches'),
                'accepted_claims': t['accepted'],
                'claims_not_accepted': len(r.get('claims', [])) - len(t['accepted']),
                'statement_overreach': bool((t['review'] or {}).get('statement_overreach')),
                'error': t['error'],
            })
        try:
            return await synthesize(self.agents, self.cfg, digest, table, previous_brief, n)
        except Exception as exc:  # keep steering from the last good brief rather than losing the round
            log.exception('reasoner failed')
            return {'report': f'reasoner failed: {exc!r}', 'tournament_brief': previous_brief}
