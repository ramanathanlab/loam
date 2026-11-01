"""Orchestrate a round of Jnana-ranked hypothesis validation.

Hermes stays the scheduler: delegated work happens only through `delegate_task`.
This module decides what each round dispatches, builds the tasks it dispatches,
records each child's start and result, and confines what each tier may call.
Model output is recorded as orchestration input, never as validated evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import hypothesis_review
import hypothesis_workspace
import ledger
from campaign import CampaignContext, agent_registry, campaign_context, campaign_field, campaign_text, skills_text
from hypothesis_tournament import Tournament
from parser import Parser

# Must match DISPATCH_TOOL_NAME in .loam/adapters/hermes/plugin/__init__.py.
DISPATCH_TOOL = 'loam_dispatch_round'

# How many evidence reviewers one turn may dispatch. A round produces at most
# top_k candidates, so this binds only the opt-in backlog catch-up -- which
# would otherwise put a campaign's whole accumulated pile of pending candidates
# into a single delegate_task batch.
REVIEW_BATCH = 5


# --- ledger -----------------------------------------------------------------

def _rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return Parser().rows(payload, 'hypothesis-orchestration.jsonl')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session(payload: Mapping[str, Any]) -> str:
    return payload.get('session_id') or payload.get('parent_session_id') or ''


def _start_for(rows: list[dict[str, Any]], session_id: str) -> dict[str, Any] | None:
    """The latest agent_started row for this session, i.e. whether it is a campaign child."""
    if not session_id:
        return None
    return next(
        (row for row in reversed(rows) if row.get('event') == 'agent_started' and row.get('child_session_id') == session_id),
        None,
    )


def _same_child(row: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    return (
        row.get('child_subagent_id') == payload.get('child_subagent_id', '')
        or row.get('child_session_id') == payload.get('child_session_id', '')
    )


def _reasoner_started(rows: list[dict[str, Any]], plan_id: str) -> bool:
    return any(
        row.get('event') == 'agent_started' and row.get('role') == 'reasoner' and row.get('plan_id') == plan_id
        for row in rows
    )


def _plan_finished(rows: list[dict[str, Any]], plan_id: str) -> bool:
    return any(row.get('event') == 'reasoner_completed' and row.get('plan_id') == plan_id for row in rows)


def _latest_brief(rows: list[dict[str, Any]], workflow_id: str) -> str:
    """The last reasoner synthesis for this workflow, which is the next tournament's input."""
    for row in reversed(rows):
        if row.get('event') == 'reasoner_completed' and row.get('workflow_id') == workflow_id:
            return row.get('summary', '').strip()[:25_000]
    return ''


def _marker(value: str, key: str) -> str:
    """The value of a `[key=value]` tag in a child's goal."""
    prefix = f'[{key}='
    start = value.find(prefix)
    if start < 0:
        return ''
    end = value.find(']', start + len(prefix))
    return value[start + len(prefix):end] if end >= 0 else ''


def _strip_tags(value: str) -> str:
    """Drop model-written Loam tags, so a child cannot be relabeled into another plan."""
    return ' '.join(re.sub(r'\[loam-[a-z-]+=[^\]]*\]', ' ', value).split())


def _child_tags(plan_id: str, hypothesis_id: str, run_id: str = '') -> str:
    """The tags record_start reads a child's role, plan and hypothesis from."""
    tags = f'[loam-agent-role=code-generator] [loam-plan-id={plan_id}] [loam-hypothesis-id={hypothesis_id}]'
    return f'{tags} [loam-run-id={run_id}]' if run_id else tags


# --- round state ------------------------------------------------------------

def _active_plan(rows: list[dict[str, Any]], workflow_id: str, run_id: str = '') -> tuple[str, list, list] | None:
    """The latest validator plan of this run whose reasoner has not completed.

    Scoped to the run, so a plan an earlier run left unfinished cannot deadlock this one.
    """
    def in_scope(row: Mapping[str, Any]) -> bool:
        return row.get('workflow_id') == workflow_id and (not run_id or row.get('run_id') == run_id)

    def validator_starts(plan_id: str | None = None) -> list[dict[str, Any]]:
        return [
            row for row in rows
            if row.get('event') == 'agent_started' and row.get('role') == 'hypothesis-validator'
            and in_scope(row) and (plan_id is None or row.get('plan_id') == plan_id)
        ]

    for plan_id in reversed([row['plan_id'] for row in validator_starts()]):
        if any(row.get('event') == 'reasoner_completed' and row.get('plan_id') == plan_id and in_scope(row) for row in rows):
            continue
        completions = [
            row for row in rows
            if row.get('event') == 'validator_completed' and row.get('plan_id') == plan_id and in_scope(row)
        ]
        return plan_id, validator_starts(plan_id), completions
    return None


def _plan_candidates(payload: Mapping[str, Any], rows: list[dict[str, Any]], plan_id: str,
                     backlog: bool = False) -> list[dict[str, Any]]:
    """Pending provenance candidates, each with the plan and hypothesis it came from.

    Candidate rows are written by `ledger.write_stop_and_candidate` and carry
    the producing child's session, not its plan, so the plan is recovered by
    joining on the validator's `agent_started` row.

    `backlog` widens this past the active plan to every pending candidate of
    this workflow. A campaign that ran before review existed has a pile of them
    -- 11 on this one, carrying the refutations that motivated the role -- and
    they are otherwise never reviewed, because the review phase is scoped to the
    round. It is opt-in and bounded by REVIEW_BATCH, so the catch-up is a
    deliberate act spread over turns rather than one enormous fan-out.
    """

    origin = {
        row.get('child_session_id'): (row.get('plan_id', ''), row.get('hypothesis_id', ''))
        for row in rows
        if row.get('event') == 'agent_started' and row.get('role') == 'hypothesis-validator'
        and row.get('child_session_id') and (backlog or row.get('plan_id') == plan_id)
    }
    candidates = []
    for row in Parser().rows(payload, 'candidate-results.jsonl'):
        if row.get('event') != 'candidate_result' or row.get('candidate_status') != 'pending':
            continue
        session_id = row.get('child_session_id')
        if session_id not in origin or not row.get('candidate_id'):
            continue
        candidate_plan, hypothesis_id = origin[session_id]
        candidates.append({**row, 'plan_id': candidate_plan, 'hypothesis_id': hypothesis_id})
    # Oldest first, so a bounded catch-up works through the backlog in order
    # instead of re-picking whichever five the file happens to list first.
    candidates.sort(key=lambda item: str(item.get('observed_at') or ''))
    return candidates


def _review_state(payload: Mapping[str, Any], rows: list[dict[str, Any]], plan_id: str) -> tuple[list[dict[str, Any]], int]:
    """The plan's candidates still awaiting a reviewer, and how many reviewers are in flight.

    A candidate is "awaiting" only until a reviewer is DISPATCHED for it, not
    until a review row appears: a reviewer that returns no usable contract still
    consumed its turn at the claim, and re-dispatching on a missing review would
    put the round in a loop that never reaches its reasoner.
    """

    # Keyed on candidate_id alone, not on the plan: candidate ids are unique,
    # and a backlog candidate's reviewer is tagged with the plan the candidate
    # came from rather than the active one.
    dispatched = {
        row.get('candidate_id')
        for row in rows
        if row.get('event') == 'agent_started' and row.get('role') == 'evidence-reviewer' and row.get('candidate_id')
    }
    finished = {
        row.get('candidate_id')
        for row in rows
        if row.get('event') == 'evidence_review_completed' and row.get('candidate_id')
    }
    reviewed = {str(row.get('candidate_id') or '') for row in Parser().rows(payload, hypothesis_review_file())}
    backlog = payload.get('loam_evidence_review_backlog') is True
    pending = [
        candidate for candidate in _plan_candidates(payload, rows, plan_id, backlog)
        if candidate['candidate_id'] not in dispatched and candidate['candidate_id'] not in reviewed
    ]
    return pending[:REVIEW_BATCH], len(dispatched - finished)


def hypothesis_review_file() -> str:
    """The review JSONL the reconciler reads, whoever authored its rows."""

    return 'hypothesis-reviews.jsonl'


def _plan_id(report_path: Path, payload: Mapping[str, Any]) -> str:
    material = '\0'.join((str(report_path), payload.get('workflow_id', ''), payload.get('session_id', '')))
    return f'plan_{hashlib.sha256(material.encode()).hexdigest()[:16]}'


def _round(payload: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Where the coordinator's round stands.

    phase is one of: reasoner_running, reasoner_due (with completions),
    reviewers_running, reviewers_due (with the candidates to review),
    validators_running, validators_due (with the tournament report), or idle.
    """
    active = _active_plan(rows, payload.get('workflow_id', ''), payload.get('loam_run_id', ''))
    if active is not None:
        plan_id, starts, completions = active
        if _reasoner_started(rows, plan_id):
            return {'phase': 'reasoner_running', 'plan_id': plan_id}
        if len(completions) == len(starts):
            # Review sits between the validators and the reasoner on purpose:
            # the reasoner then reads a brief whose evidence has already been
            # accepted or refused by an agent that did not produce it, instead
            # of synthesizing from candidates nobody checked.
            if payload.get('loam_evidence_review') is True:
                pending, running = _review_state(payload, rows, plan_id)
                if pending:
                    return {'phase': 'reviewers_due', 'plan_id': plan_id, 'candidates': pending,
                            'completions': completions}
                if running:
                    return {'phase': 'reviewers_running', 'plan_id': plan_id}
            return {'phase': 'reasoner_due', 'plan_id': plan_id, 'completions': completions}
        return {'phase': 'validators_running', 'plan_id': plan_id}
    report = Tournament(payload).latest_report()
    if report is None:
        return {'phase': 'idle'}
    plan_id = _plan_id(report[0], payload)
    if _plan_finished(rows, plan_id):
        return {'phase': 'idle'}
    return {'phase': 'validators_due', 'plan_id': plan_id, 'report': report[1]}


def _steering_priorities(payload: Mapping[str, Any]) -> dict[str, float]:
    """Each hypothesis's advisory steering priority, keyed by its Jnana id.

    From the latest `hypothesis-steering.jsonl` snapshot, which
    `project_hypothesis_steering.py` builds out of ACCEPTED evidence (weights
    elo 0.35 / evidence 0.50 / uncertainty 0.15). Empty until a round's claims
    have been reviewed and reconciled.
    """

    snapshot = Parser().snapshot(payload, 'hypothesis-steering.jsonl', 'hypothesis_steering_snapshot')
    if not snapshot:
        return {}
    priorities: dict[str, float] = {}
    for item in snapshot.get('hypotheses') or []:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get('jnana_hypothesis_id') or item.get('hypothesis_id') or '')
        try:
            priority = float(item.get('steering_priority'))
        except (TypeError, ValueError):
            continue
        if key:
            priorities[key] = priority
    return priorities


def _selected_hypotheses(payload: Mapping[str, Any], report: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The ranked hypotheses this round validates, with their full text.

    None when the ranking is empty or any ranked hypothesis is missing from the
    Jnana state: a partial fan-out would silently drop one the tournament chose.

    The tournament's Elo order decides WHICH hypotheses a round takes; accepted
    evidence decides the order they are DISPATCHED in. Nothing is dropped --
    evidence is advisory and a refutation of one claim is not a refutation of a
    hypothesis -- but a hypothesis whose accepted contradictions outweigh its
    support goes last, so it is the first thing a round truncated by a walltime
    boundary fails to reach. `rank` keeps the Elo rank it was selected on, so
    the two signals stay legible as two signals.
    """
    hypotheses = Tournament(payload).hypotheses()
    selected = []
    for rank, ranking in enumerate(report['result'].get('top_hypotheses') or [], start=1):
        hypothesis = hypotheses.get(ranking['hypothesis_id'])
        if not hypothesis:
            return None
        selected.append({'rank': rank, 'elo_rating': ranking.get('elo_rating'), 'hypothesis': hypothesis})
    priorities = _steering_priorities(payload)
    if priorities:
        for item in selected:
            item['steering_priority'] = priorities.get(item['hypothesis']['hypothesis_id'])
        # Stable, so hypotheses with no accepted evidence keep their Elo order
        # among themselves rather than being shuffled by a 0.0 default.
        selected.sort(key=lambda item: -(item.get('steering_priority') or 0.0))
        for position, item in enumerate(selected, start=1):
            item['dispatch_order'] = position
    return selected or None


def _elo_ratings(payload: Mapping[str, Any]) -> dict[str, float]:
    report = Tournament(payload).latest_report()
    if report is None:
        return {}
    return {
        ranking['hypothesis_id']: ranking['elo_rating']
        for ranking in report[1]['result'].get('top_hypotheses', [])
        if ranking.get('elo_rating') is not None
    }


# --- pre-LLM ----------------------------------------------------------------

def context_for(payload: dict[str, Any]) -> str:
    """The coordinator's next scheduling directive, or the latest brief between rounds."""
    rows = _rows(payload)
    state = _round(payload, rows)
    if state['phase'] == 'reasoner_due':
        return _reasoner_directive(state['plan_id'], state['completions'], payload)
    if state['phase'] == 'reviewers_due':
        return _reviewer_directive(payload, len(state['candidates']))
    if state['phase'] == 'validators_due':
        selected = _selected_hypotheses(payload, state['report'])
        return _validator_directive(payload, len(selected)) if selected else ''
    if state['phase'] == 'idle':
        brief = _latest_brief(rows, payload.get('workflow_id', ''))
        return f'Latest reasoner brief for the next tournament turn:\n{brief}' if brief else ''
    return ''


def prepare_payload(payload: dict[str, Any]) -> None:
    """Feed the latest reasoner brief into the next Jnana generation goal."""
    brief = _latest_brief(_rows(payload), payload.get('workflow_id', ''))
    if not brief:
        return
    payload['loam_reasoner_brief'] = brief
    if payload.get('user_message'):
        payload['user_message'] = (
            f"{payload['user_message']}\n\n"
            "Prior reasoner brief (advisory and unvalidated unless independently reconciled):\n"
            f"{brief}"
        )


def is_campaign_child(payload: Mapping[str, Any]) -> bool:
    return _start_for(_rows(payload), payload.get('session_id', '')) is not None


def runbatch_nudge_for(payload: Mapping[str, Any]) -> str:
    """Remind a code-generator looping run_code to batch instead (advisory only).

    Each serial run_code re-sends the child's whole growing context, which is the
    main driver of the token-rate limit; run_batch is one submit plus polling.
    """
    start = _start_for(_rows(payload), payload.get('session_id', ''))
    if start is None or start.get('role') != 'code-generator':
        return ''
    # start['run'] names the workspace run dir (e.g. r2); start['run_id'] is the campaign's uuid.
    if not start.get('workspace') or not start.get('run'):
        return ''
    try:
        record = hypothesis_workspace.read_json(Path(start['workspace']) / 'runs' / start['run'] / hypothesis_workspace.RUN_FILE)
    except (OSError, ValueError):
        return ''  # the run has made no calls yet
    functions = [hypothesis_workspace._hpc_function(call.get('tool', '')) for call in record.get('calls') or []]
    run_code = functions.count('run_code')
    if run_code < 5 or 'run_batch' in functions:
        return ''
    return (
        f"RUN_BATCH REMINDER: this run has {run_code} serial run_code calls and no run_batch. "
        "A design tier's per-candidate fan-out (generate / fold / score many candidates) MUST go "
        "through run_batch -- one JSONL row per candidate, one submit, then poll batch_status until "
        "unfinished == 0 -- NOT a run_code loop. Each serial run_code resends your whole growing "
        "context and is the main cause of the model token-rate-limit failures that abort runs; "
        "run_batch executes the rows on HPC tiles without that per-call cost. Keep run_code only for "
        "setup, a single probe fold, and post-batch aggregation."
    )


def _validator_directive(payload: Mapping[str, Any], count: int) -> str:
    """Announce the round. The task array is not printed: DISPATCH_TOOL hands it to delegate_task."""
    validator = agent_registry(payload)['hypothesis-validator']
    return '\n'.join((
        f"Loam orchestration directive: {count} hypothesis validator"
        f"{'s are' if count != 1 else ' is'} planned for this round. "
        f"Call {DISPATCH_TOOL} to launch {'them' if count != 1 else 'it'} now. It takes no arguments.",
        "Do not compose a delegate_task batch yourself: the selected hypotheses are not in your "
        "context in full, and a batch you write is a substitute for the campaign's work rather "
        "than the campaign's work. Each validator receives its hypothesis, the campaign goal and "
        f"rules, and its role definition ({validator.prompt_path.name}) inlined by the dispatch. "
        "Do not perform experimental code or tool calls yourself.",
        "Every executable action MUST be delegated by that validator to at most 2 leaf code-generator "
        "children; no other child may execute code.",
    ))


def _review_summary(payload: Mapping[str, Any], plan_id: str) -> str:
    """What independent review decided about this round's claims, for the reasoner.

    Empty when review is off, so a campaign without reviewers reads exactly the
    directive it read before.
    """

    if payload.get('loam_evidence_review') is not True:
        return ''
    reviews = [row for row in Parser().rows(payload, hypothesis_review_file()) if row.get('plan_id') == plan_id]
    if not reviews:
        return (
            "Independent review is enabled but produced no review of this round's claims, so nothing "
            "from it has been accepted. Treat every validator claim as unchecked."
        )
    totals = {'accepted': 0, 'rejected': 0, 'unverifiable': 0}
    for review in reviews:
        for verdict, count in (review.get('review_verdicts') or {}).items():
            if verdict in totals:
                totals[verdict] += int(count or 0)
    accepted = sum(len(review.get('evidence') or []) for review in reviews)
    return (
        f"Independent review of this round: {len(reviews)} candidate(s) reviewed by agents that did not "
        f"produce the claims -- {totals['accepted']} claims accepted, {totals['rejected']} rejected, "
        f"{totals['unverifiable']} unverifiable; {accepted} accepted claims entered the evidence ledger. "
        "An unverifiable claim is NOT a refuted one: it means no source or artifact could confirm the "
        "number, which is the normal case for a computed metric. Weigh an accepted claim above an "
        "unverifiable one and a rejected one not at all, and say in your brief which of your "
        "conclusions rest on claims that review could not check."
    )


def _reviewer_directive(payload: Mapping[str, Any], count: int) -> str:
    """Announce the review step. Same shape as the validator round: dispatch, do not compose."""
    reviewer = agent_registry(payload)['evidence-reviewer']
    return '\n'.join((
        f"Loam orchestration directive: this round's validators are terminal and {count} provenance "
        f"candidate{'s are' if count != 1 else ' is'} pending independent review. "
        f"Call {DISPATCH_TOOL} to launch the reviewer{'s' if count != 1 else ''} now. It takes no arguments.",
        "A validator's claims are untrusted model output and carry no weight until an agent that did "
        "not produce them has checked each one against its source or its hashed artifact. Each "
        f"reviewer receives one candidate's claims, the campaign goal and rules, and its role "
        f"definition ({reviewer.prompt_path.name}) inlined by the dispatch -- deliberately NOT the "
        "validator's report, which is the object under review.",
        "Do not review the claims yourself, do not compose a delegate_task batch, and do not accept "
        "or dismiss any candidate on your own reading of a validator's summary.",
    ))


def _reasoner_directive(plan_id: str, completions: list[dict[str, Any]], payload: Mapping[str, Any]) -> str:
    reasoner = agent_registry(payload)['reasoner']
    elo_ratings = _elo_ratings(payload)
    threads = [
        {
            'hypothesis_id': row.get('hypothesis_id', ''),
            'validator_status': row.get('status'),
            # Hermes reports child_status "completed" for a child whose own API
            # calls failed, so validator_status alone cannot distinguish a
            # validation that ran from one that never did. The ledger's verdict
            # can; carry it so the reasoner is not left inferring it from the
            # shape of a 50-character summary.
            'observation_status': row.get('observation_status', 'observed'),
            'failure_class': row.get('failure_class', ''),
            'evidence_score': row.get('evidence_score'),
            'evidence_score_status': row.get('evidence_score_status'),
            'elo_rating': elo_ratings.get(row.get('hypothesis_id', '')),
            'validator_report': row.get('summary', '').strip()[:6_000],
        }
        for row in completions
    ]
    unobserved = [t for t in threads if t['observation_status'] != 'observed']
    parts = (
        "Loam orchestration directive: all selected validator threads are terminal. Delegate exactly one reasoner before responding.",
        f"The reasoner's role definition ({reasoner.prompt_path.name}) is below and Loam itself puts it into the dispatched task as \"agent_definition\"; the reasoner must not execute experimental code. Call delegate_task once for the reasoner -- Loam replaces your arguments with the canonical task, definition included. Do not reproduce the definition yourself, and if this directive was written to a file instead of shown inline, dispatch anyway: you do not need to read that file to comply.",
        (
            f"Dispatch it on the model its definition declares ({reasoner.model}); this synthesis "
            "steers the next tournament and should not run on a weaker tier than the validators it reads."
            if reasoner.model
            else ""
        ),
        json.dumps(
            {**campaign_field(campaign_context(payload), 'reasoner'), 'agent_definition': reasoner.prompt_body},
            ensure_ascii=False,
        ),
        f"Tag its goal with [loam-agent-role=reasoner] [loam-plan-id={plan_id}] so its brief is persisted for the next tournament turn.",
        "Completed validator threads are unvalidated orchestration input; preserve their reported uncertainty and do not turn summaries into accepted evidence.",
        _review_summary(payload, plan_id),
        (
            f"{len(unobserved)} of {len(threads)} threads did not produce an observation "
            f"(observation_status is not 'observed'; failure_class says why). Their hypotheses are "
            "UNTESTED, not refuted: a provider or infrastructure failure is evidence about this run, "
            "not about the science. Do not synthesize their reports as results, do not lower their "
            "standing for failing, and say in your brief that they still need testing so the next "
            "tournament can re-select them."
            + (
                " No thread produced an observation, so there is nothing to synthesize: report that "
                "the round did not run and recommend repeating it rather than writing a brief from "
                "failures."
                if unobserved and len(unobserved) == len(threads)
                else ""
            )
            if unobserved
            else ""
        ),
        json.dumps(threads, ensure_ascii=False, sort_keys=True),
    )
    return '\n'.join(part for part in parts if part)


# --- delegation -------------------------------------------------------------

def canonical_delegation_args(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """The delegate_task arguments the campaign's persisted state calls for.

    Registered as tool_request middleware, so this replaces a model-written
    delegation before it runs. None means there is no canonical delegation for
    this state, notably while a round's validators are still running.
    """
    rows = _rows(payload)
    start = _start_for(rows, _session(payload))
    if start is not None:
        if start['role'] != 'hypothesis-validator':
            return None
        return _child_tasks(payload, start['plan_id'], start['hypothesis_id'])

    state = _round(payload, rows)
    if state['phase'] == 'reasoner_due':
        plan_id = state['plan_id']
        return {
            'goal': (
                "Synthesize validator results. As you commit to each synthesis decision -- which "
                "hypotheses to promote or retire, how to steer the next round -- record it with the "
                "loam_record_decision tool before you finalize your brief; your reasoning is not "
                f"otherwise kept. [loam-agent-role=reasoner] [loam-plan-id={plan_id}]"
            ),
            'context': _reasoner_directive(plan_id, state['completions'], payload),
            'role': 'leaf',
        }
    if state['phase'] == 'reviewers_due':
        return {'tasks': _reviewer_tasks(state['plan_id'], state['candidates'], payload)}
    if state['phase'] == 'validators_due':
        selected = _selected_hypotheses(payload, state['report'])
        return {'tasks': _validator_tasks(state['plan_id'], selected, payload)} if selected else None
    return None


def _reviewer_tags(plan_id: str, hypothesis_id: str, candidate_id: str) -> str:
    return (
        f'[loam-agent-role=evidence-reviewer] [loam-plan-id={plan_id}] '
        f'[loam-hypothesis-id={hypothesis_id}] [loam-candidate-id={candidate_id}]'
    )


def _artifact_manifest(payload: Mapping[str, Any], plan_id: str, hypothesis_id: str,
                       limit: int = 200) -> list[dict[str, Any]]:
    """The hashed output files this hypothesis's completed runs produced.

    This is what makes a computed claim checkable without recomputing the
    science: Loam recorded each file's sha256 when the run closed, so a reviewer
    reading one can attest that a reported number is actually in an artifact of
    a declared run -- the failure the acceptance boundary exists to catch -- and
    cannot be handed a path the campaign never wrote.
    """

    if hypothesis_workspace.settings(payload) is None or not hypothesis_id:
        return []
    try:
        root = hypothesis_workspace.workspace_root(payload, plan_id, hypothesis_id)
    except Exception:
        return []
    entries: list[dict[str, Any]] = []
    runs = root / 'runs'
    if not runs.is_dir():
        return []
    for directory in sorted(runs.iterdir()):
        try:
            run = hypothesis_workspace.read_json(directory / hypothesis_workspace.RUN_FILE)
        except (OSError, ValueError):
            continue
        if run.get('state') != 'completed':
            continue
        for item in run.get('manifest') or []:
            if not isinstance(item, Mapping) or not item.get('path'):
                continue
            entries.append({
                'run': directory.name,
                'manifest_status': run.get('manifest_status', ''),
                'path': item.get('path'),
                'sha256': item.get('sha256', ''),
                'bytes': item.get('bytes'),
            })
            if len(entries) >= limit:
                return entries
    return entries


def _reviewer_tasks(plan_id: str, candidates: list[dict[str, Any]], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    reviewer = agent_registry(payload)['evidence-reviewer']
    execution = payload.get('loam_execution') or {}
    campaign = campaign_context(payload)
    names = hpc_tool_names(execution)
    read_only = sorted({names[key] for key in ('inspect_filesystem', 'batch_status', 'list_batches') if key in names})
    tasks = []
    for candidate in candidates:
        contract = candidate.get('candidate') or {}
        hypothesis_id = candidate.get('hypothesis_id', '')
        # The candidate's OWN plan, which is the active one for a round's own
        # candidates and an earlier one for a backlog catch-up. It has to be the
        # candidate's, or the review would be filed against the wrong round and
        # the artifacts offered would be another plan's.
        candidate_plan = candidate.get('plan_id') or plan_id
        artifacts = _artifact_manifest(payload, candidate_plan, hypothesis_id)
        tasks.append({
            'goal': (
                "Independently review each evidence claim in the attached provenance candidate: check "
                "it against its own source or against the hashed artifact it was computed from, and "
                "return one verdict per claim in the required contract. You did not produce these "
                "claims and you are not reviewing the validator's report, which you have not been "
                "given. "
                f"{_reviewer_tags(candidate_plan, hypothesis_id, candidate['candidate_id'])}"
            ),
            # Ordered, so sort_keys stays off: the claims first, the artifacts
            # they must be checked against next, the role definition last.
            'context': json.dumps(
                {
                    'loam_review_contract': hypothesis_review.CONTRACT_VERSION,
                    'candidate_id': candidate['candidate_id'],
                    'produced_by_unit_id': candidate.get('unit_id', ''),
                    'hypotheses': contract.get('hypotheses') or [],
                    # Indexed, and the review contract keys on the index: a
                    # reviewer copying a 300-character claim back byte for byte
                    # is one normalized dash away from having its verdict
                    # dropped as unmatched.
                    'evidence_candidates': [
                        {'index': position, **item}
                        for position, item in enumerate(contract.get('evidence_candidates') or [])
                    ],
                    'campaign_artifacts': artifacts,
                    # Where to look when there is no manifest: the plan's own
                    # HPC workdir. A file read there still catches a number no
                    # output file contains, which is the failure that matters,
                    # but nothing ties it to a declared run -- so Loam weights
                    # such an acceptance lower, and says so here rather than
                    # letting the reviewer believe it attested a hashed file.
                    'artifact_root': _workdir(execution, candidate_plan, hypothesis_id),
                    'artifact_manifest_available': bool(artifacts),
                    'read_only_tools': read_only,
                    'reviewer_instructions': (
                        "Follow agent_definition above; it is your role specification and is provided in "
                        "full because you cannot read its file. "
                        "You may call ONLY the read-only tools listed in read_only_tools, and you cannot "
                        "execute code: no run_code, no run_batch, no delegation. Return one review item "
                        "per claim in evidence_candidates, identified by its `index` -- do not retype "
                        "the claim text, and do not reorder or merge the claims. "
                        "campaign_artifacts lists every output file of this hypothesis's completed runs "
                        "with the sha256 Loam recorded when the run closed; read a file with "
                        "inspect_filesystem to confirm a reported number is genuinely in it. When that "
                        "list is empty (artifact_manifest_available is false, i.e. this campaign runs "
                        "without the per-hypothesis workspace) list and read under artifact_root "
                        "instead: finding the number in a file there is still worth reporting as "
                        "artifact_attested, and Loam weights it lower on its own because no recorded "
                        "hash ties that file to a declared run. "
                        "'unverifiable' is a real verdict, not a failure: say it whenever you could not "
                        "check a claim, and never accept one you did not actually read a source or an "
                        "artifact for."
                    ),
                    **campaign_field(campaign, 'evidence-reviewer'),
                    'agent_definition': reviewer.prompt_body,
                },
                ensure_ascii=False,
            ),
            'role': 'leaf',
        })
    return tasks


def _validator_tasks(plan_id: str, selected: list[dict[str, Any]], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    validator = agent_registry(payload)['hypothesis-validator']
    execution = payload.get('loam_execution') or {}
    campaign = campaign_context(payload)
    observation = 'batch_status, inspect_filesystem, list_batches'
    tasks = []
    for item in selected:
        hypothesis_id = item['hypothesis']['hypothesis_id']
        tasks.append({
            'goal': (
                "Validate the supplied hypothesis by designing the smallest decisive experiment and delegating "
                "it to at most 2 code-generator leaf children, one at a time, iterating on their actual "
                "results. Do not compute anything yourself; delegate_task is your only executable tool, though "
                f"you may call {observation} to check on and read a child's run. "
                "As you commit to each high-level decision -- the experiment design, what to delegate, "
                "whether the evidence is sufficient -- record it with the loam_record_decision tool before "
                "you act on it; your reasoning is not otherwise kept. "
                "Then return the required report headings, including a bare decimal Evidence Score. "
                f"[loam-agent-role=hypothesis-validator] [loam-plan-id={plan_id}] "
                f"[loam-hypothesis-id={hypothesis_id}]"
            ),
            # Ordered on purpose, so sort_keys stays off: the assignment first, and the
            # long role definition last, where its length cannot bury the hypothesis.
            'context': json.dumps(
                {
                    'loam_validator_contract': 'v1',
                    'selected_hypothesis': item,
                    **campaign_field(campaign, 'hypothesis-validator'),
                    'execution_contract': execution,
                    'child_template': {
                        'role': 'leaf',
                        'goal_suffix': _child_tags(plan_id, hypothesis_id),
                        'max_children': 2,
                        'workdir': _workdir(execution, plan_id, hypothesis_id),
                    },
                    'validator_instructions': (
                        "Follow agent_definition above; it is your role specification and is provided in full "
                        "because you cannot read its file. "
                        "Every code-generator child starts with an empty context and cannot see this payload, "
                        "the tournament state or the campaign files, so write selected_hypothesis into each "
                        "child's goal/context in full -- statement, candidate sequence and the falsifying "
                        "prediction. A hypothesis id is a provenance tag, not a reference a child can resolve. "
                        "Restate the hypothesis as a falsifiable claim and design the smallest decisive experiment. "
                        "Delegate each executable step to a code-generator child: role=leaf, goal = what to compute "
                        "and the acceptance condition, ending with child_template.goal_suffix; context = exact "
                        "inputs, parameter ranges and expected outputs. Children write their own code, can reuse "
                        "registered HPC tools, and run large sweeps as batches. At most max_children children in "
                        "total. Never compute or run code yourself. Do poll and read a child's run yourself with "
                        f"{observation} rather than spending a child on it, and verify a "
                        "child's reported numbers against the files it actually wrote. Then return exactly these headings: "
                        "### Hypothesis; ### Outcome; ### Evidence Ledger; ### Credit Assignment; "
                        "### Evidence Score; ### Next Step. The Evidence Score must be a bare decimal "
                        "in [-1.00, 1.00], for example +0.75, immediately after its heading."
                    ),
                    'agent_definition': validator.prompt_body,
                },
                ensure_ascii=False,
            ),
            'role': 'orchestrator',
        })
    return tasks


def _child_tasks(payload: Mapping[str, Any], plan_id: str, hypothesis_id: str) -> dict[str, Any] | None:
    """Keep each of the validator's instructions, but force the tags, role and contract."""
    execution = payload.get('loam_execution') or {}
    arguments = payload.get('arguments') or {}
    # An empty `tasks` list is single-task mode, as Hermes itself treats it.
    items = arguments.get('tasks') or [arguments]
    workdir = _workdir(execution, plan_id, hypothesis_id)
    hypothesis = Tournament(payload).hypotheses().get(hypothesis_id)
    campaign = campaign_context(payload)

    parsed = []
    for task in items:
        goal = _strip_tags(task.get('goal', ''))
        if not goal:
            return None
        instruction = '\n\n'.join(part for part in (goal, _strip_tags(task.get('context', ''))) if part)
        parsed.append((goal, instruction))

    # Reserved only once every task is valid. The key is the goal as record_start will see it.
    keys = [_strip_tags(_child_goal(hypothesis, goal, plan_id, hypothesis_id)) for goal, _ in parsed]
    run_ids, layout_base = _reserve_child_runs(payload, execution, workdir, plan_id, hypothesis_id, keys)
    remote = execution.get('mode') == 'remote_mcp'
    batch_root = hypothesis_workspace.batch_root_for(execution) if remote else ''
    shared_inputs = tuple((hypothesis_workspace.settings(payload) or {}).get('shared_inputs') or ())
    out = [
        {
            'goal': _child_goal(hypothesis, goal, plan_id, hypothesis_id, run_id),
            'context': _child_context(
                instruction, execution, workdir, hypothesis, campaign,
                _layout_block(layout_base, run_id, remote, batch_root, plan_id, hypothesis_id, shared_inputs),
            ),
            'role': 'leaf',
        }
        for (goal, instruction), run_id in zip(parsed, run_ids)
    ]
    _record_dispatch_edges(payload, plan_id, hypothesis_id, items, out, run_ids)
    return {'tasks': out}


def _child_goal(hypothesis: Mapping[str, Any] | None, validator_goal: str, plan_id: str, hypothesis_id: str,
                run_id: str = '') -> str:
    """Lead the child's goal with the claim, since the model acts on the goal line first."""
    lead = ''
    claim = hypothesis and (hypothesis.get('summary') or hypothesis.get('content'))
    if claim:
        lead = f"Test this binder hypothesis (full text + how-to in your context; do NOT search for it): {claim[:220]} || "
    return f'{lead}{validator_goal[:300]} {_child_tags(plan_id, hypothesis_id, run_id)}'


def _workdir(execution: Mapping[str, Any], plan_id: str, hypothesis_id: str) -> str:
    """The HPC-side folder a remote child writes under: workdir_root/plan/hypothesis."""
    root = execution.get('workdir_root', '')
    return f"{root.rstrip('/')}/{plan_id}/{hypothesis_id}" if root else ''


def _reserve_child_runs(payload: Mapping[str, Any], execution: Mapping[str, Any], workdir: str, plan_id: str,
                        hypothesis_id: str, goals: list[str]) -> tuple[list[str], str]:
    """Reserve one workspace run per child, and the folder the child writes under.

    Idempotent by goal, because tool_request middleware can run more than once
    for one dispatch. A workspace problem never blocks delegation: the children
    then go out without run ids, and record_start allocates runs for them.
    """
    if hypothesis_workspace.settings(payload) is None:
        return [''] * len(goals), ''
    try:
        root = hypothesis_workspace.workspace_root(payload, plan_id, hypothesis_id)
        run_ids = hypothesis_workspace.reserve_runs(root, len(goals), goals)
    except Exception as exc:
        print(f'loam workspace: could not reserve runs: {type(exc).__name__}: {exc}', file=sys.stderr)
        return [''] * len(goals), ''
    if execution.get('mode') == 'remote_mcp':
        return run_ids, workdir or f'<HPC base directory (HPC_BASE_DIR)>/{plan_id}/{hypothesis_id}'
    return run_ids, str(root)


def _record_dispatch_edges(payload: Mapping[str, Any], plan_id: str, hypothesis_id: str,
                           items: list, out: list[dict[str, Any]], run_ids: list[str]) -> None:
    """Record each validator->child edge, hashing what was asked and what the child got."""
    if hypothesis_workspace.settings(payload) is None or not any(run_ids):
        return
    try:
        root = hypothesis_workspace.workspace_root(payload, plan_id, hypothesis_id)
    except Exception as exc:
        print(f'loam workspace: dispatch edge skipped: {type(exc).__name__}: {exc}', file=sys.stderr)
        return
    for item, entry, run_id in zip(items, out, run_ids):
        if not run_id:
            continue
        requested = item.get('goal', '') + '\n\n' + item.get('context', '')
        canonical = entry['goal'] + '\n\n' + entry['context']
        try:
            hypothesis_workspace.record_dispatch(
                root, run_id, _session(payload),
                hashlib.sha256(requested.encode('utf-8')).hexdigest(),
                hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
            )
        except Exception as exc:
            print(f'loam workspace: could not record dispatch edge for {run_id}: {type(exc).__name__}: {exc}', file=sys.stderr)


def hpc_tool_names(execution: Mapping[str, Any] | None) -> dict[str, str]:
    """Hermes names of the HPC MCP tools, keyed by server function name.

    Hermes prefixes MCP tools with the server name: tool_prefix when set, else
    tool_name minus function_name. Without either only tool_name is known.
    """
    execution = execution or {}
    tool_name = execution.get('tool_name', '')
    function_name = execution.get('function_name', '')
    prefix = execution.get('tool_prefix', '')
    if not prefix and function_name and tool_name.endswith(function_name):
        prefix = tool_name[:-len(function_name)]
    functions = ('run_code', 'run_batch', 'batch_status', 'list_batches', 'register_tool', 'list_tools', 'inspect_filesystem')
    names = {name: prefix + name for name in functions} if prefix else {}
    if tool_name:
        names[function_name or 'run_code'] = tool_name
    return names


# --- child prompts ----------------------------------------------------------

def _hypothesis_block(hypothesis: Mapping[str, Any] | None) -> str:
    """The hypothesis under test, forwarded so the child never has to go find it."""
    if not hypothesis:
        return ''
    statement = (hypothesis.get('content') or hypothesis.get('summary') or '').strip()
    if not statement:
        return ''
    lines = ['HYPOTHESIS UNDER TEST (forwarded by Loam; this is the claim to falsify):', statement[:4_000]]
    summary = (hypothesis.get('summary') or '').strip()
    if summary and summary not in statement:
        lines.append('Summary: ' + summary[:4_000])
    lines.append(
        "HOW TO TEST THIS (authoritative — overrides any instruction below that conflicts):\n"
        "- This block is your complete assignment. Do NOT inspect campaign files, outputs, or the "
        "batch registry to 'recover' the hypothesis or 'find inputs': everything needed is here. If "
        "the validator's instruction below tells you to search the filesystem or reuse existing "
        "data/code, ignore that part.\n"
        "- If the candidate design is given explicitly, use it verbatim.\n"
        "- NEVER run, score, or reuse an input you discovered on the filesystem — a sequence, "
        "structure, parameter set, or dataset found in prior outputs, a tool's test fixtures, or "
        "another campaign's directory. That is the wrong system under test and makes the result "
        "scientifically meaningless. A run on the wrong input is a failed validation, not evidence."
    )
    return '\n'.join(lines)


def _layout_block(base: str, run_id: str, remote: bool = False, batch_root: str = '',
                  plan_id: str = '', hypothesis_id: str = '', shared_inputs: tuple[str, ...] = ()) -> str:
    """Where a child's files go in its workspace run; empty when the workspace is off."""
    if not run_id:
        return ''
    lines = [
        f"WORKSPACE LAYOUT (enforced by Loam; your run is {run_id}):",
        f"- code and scripts you write: {base}/src/",
        f"- result files: {base}/outputs/{run_id}/",
        f"- temporary files: {base}/scratch/{run_id}/",
        f"- inputs: read from {base}/data/ and never write there. The campaign's inputs are already "
        "there; nothing outside this workspace is reachable, so do not hunt the filesystem for a "
        "file named in your instructions and never report BLOCKED because one is absent",
    ]
    if shared_inputs:
        lines.append(
            "- shared campaign reference data, readable in place and read-only to you: "
            + ", ".join(f"{path}/" for path in shared_inputs)
            + ". Loam records each file you read there as an input of your run."
        )
    lines += [
        "- earlier results of OTHER runs (this hypothesis's or another's): readable only from their "
        "completed runs' outputs/rK/; Loam records each such read as an input of your run",
        "Write nothing anywhere else, and never /tmp: /tmp is local to one node and the next tool call "
        "may run on another. Other runs' outputs are frozen; do not modify them.",
        "DECISIONS: as you commit to each high-level choice -- the approach, which generator or "
        "folding model, the tier size, a go/stop -- record it with the loam_record_decision tool "
        "(what, why, alternatives, expected, confidence) BEFORE you act on it. Your full reasoning "
        "is not otherwise kept, so an unrecorded decision is lost.",
    ]
    if remote:
        # Always scratch/: the server writes a temp script into the workdir and
        # deletes it afterwards, which would corrupt the outputs/ manifest.
        lines.append(
            f"EVERY run_code call MUST pass workdir={base}/scratch/{run_id} -- always scratch/, "
            f"never outputs/: the server puts a temporary script in the workdir and deletes it "
            f"afterwards, which would corrupt your output manifest. Write results to "
            f"{base}/outputs/{run_id}/ by absolute path from inside your program. "
            "A run_code with no workdir is REFUSED."
        )
    if remote and batch_root:
        prefix = hypothesis_workspace.batch_prefix(hypothesis_id, run_id)
        labels = hypothesis_workspace.batch_labels(plan_id, hypothesis_id, run_id)
        lines += [
            f"RUN_BATCH (if you use it): pass batch_id='{prefix}<name>' and labels={json.dumps(labels)}. "
            "run_batch writes its results to the shared batch folder, not to your outputs. Poll batch_status "
            "until the batch is complete, then copy its results into your outputs by running this program "
            "with run_code (set batch_id first), BEFORE the output manifest step:",
            hypothesis_workspace.harvest_program(batch_root, f"{base}/outputs/{run_id}"),
        ]
    if remote:
        lines += [
            "OUTPUT MANIFEST (required): as your LAST tool call, run this program with run_code, then paste its "
            "complete output verbatim at the end of your report:",
            hypothesis_workspace.manifest_program(f"{base}/outputs/{run_id}"),
        ]
    return '\n'.join(lines)


def _child_context(instruction: str, execution: Mapping[str, Any], workdir: str,
                   hypothesis: Mapping[str, Any] | None = None,
                   campaign: CampaignContext | None = None, layout: str = '') -> str:
    """A code-generator's context: claim and policy first, then the validator's instruction."""
    hypothesis_block = _hypothesis_block(hypothesis)
    campaign_block = campaign_text(campaign, 'code-generator')
    skills_block = skills_text(campaign)
    setup = execution.get('setup', '')
    setup_block = ("ENVIRONMENT SETUP (use exactly this; do not invent venv or interpreter paths):\n" + setup) if setup else ""
    report = (
        "Report the code you ran, every tool call with its arguments, batch_ids, tool names and "
        "versions, and the actual stdout/stderr/returncode or batch_status. A failed or unfinished "
        "run is not evidence; say so instead of guessing results."
    )
    runtime = execution.get('runtime', '')
    if execution.get('mode') != 'remote_mcp':
        return "\n\n".join(part for part in (
            "Write your own Python for the validator's instruction below and execute it locally"
            + (f" with {runtime}." if runtime else "."),
            hypothesis_block,
            campaign_block,
            skills_block,
            setup_block,
            layout,
            "Instruction from validator:\n" + instruction,
            report,
        ) if part)

    names = hpc_tool_names(execution)
    run_code = names.get(execution.get('function_name') or 'run_code', '')
    place = f"under {workdir}" if workdir else "under the HPC base directory (HPC_BASE_DIR)"
    lines = [
        f"- `{run_code}`: run one Python program (`payload`) on a GPU worker; use it for one-off runs "
        "and to write script or JSONL files to disk.",
    ]
    if 'run_batch' in names:
        lines += [
            f"- `{names['list_tools']}`: call this FIRST; reuse a registered tool whose params fit.",
            f"- `{names['register_tool']}`: save a script you will run many times as a named tool "
            "(name, script_path, params as {name: int|float|str|bool|list|dict}, description, example "
            "row). It is smoke-tested on the example; pass skip_smoke only if it needs a GPU to start.",
            f"- `{names['run_batch']}`: many runs at once: tool=<name> (or script_path) and inputs_path, "
            "a JSONL file with one row of params per line. Returns a batch_id at once. gpus is tiles "
            "PER ROW: rows run side by side on all free tiles, so a script using one device (xpu:0) "
            "takes gpus=1; raise it only if one row itself spans several devices.",
            f"- `{names['batch_status']}`: poll a batch. If it is still running when you must report, "
            "report the batch_id and counts as unfinished.",
            f"- `{names['inspect_filesystem']}`: read result files (ls/cat/tail).",
            f"- `{names['list_batches']}`: earlier batches for this plan or hypothesis.",
            "Use run_batch rather than run_code for more than a handful of runs.",
        ]
    # Said out loud: a child told to use no other tool will not try skill_view otherwise.
    others = (
        " You may also call `skill_view` and `skills_list` to read a procedure before you write "
        "code; they only read."
        if campaign is not None and campaign.skills
        else ""
    )
    return "\n\n".join(part for part in (
        "Write your own Python for the validator's instruction below and run it on HPC through the "
        "remote MCP tools listed here; do not execute code locally or use any other tool to run "
        "anything." + others,
        hypothesis_block,
        campaign_block,
        skills_block,
        setup_block,
        "Instruction from validator:\n" + instruction,
        f"Remote target: endpoint={execution.get('endpoint', '')}; runtime={runtime}.",
        "Tools:\n" + "\n".join(lines),
        layout or (
            f"Write scripts and inputs files {place}, never /tmp: /tmp is local to one node and the next "
            "tool call may run on another."
        ),
        report,
    ) if part)


# --- subagent start / stop ---------------------------------------------------

def record_start(payload: Mapping[str, Any]) -> None:
    """Record a role-tagged child's start, and give it its workspace when that is on."""
    goal = payload.get('child_goal', '')
    role = _marker(goal, 'loam-agent-role')
    plan_id = _marker(goal, 'loam-plan-id')
    if role not in ('reasoner', 'hypothesis-validator', 'code-generator', 'evidence-reviewer') or not plan_id:
        return
    hypothesis_id = _marker(goal, 'loam-hypothesis-id')
    candidate_id = _marker(goal, 'loam-candidate-id')
    record = {
        'event': 'agent_started',
        'observed_at': _now(),
        'run_id': payload.get('loam_run_id', ''),
        'workflow_id': payload.get('workflow_id', ''),
        'plan_id': plan_id,
        'role': role,
        'hypothesis_id': hypothesis_id,
        'child_subagent_id': payload.get('child_subagent_id', ''),
        'child_session_id': payload.get('child_session_id', ''),
    }
    if candidate_id:
        # The review step keys on this: a dispatched candidate is not
        # re-dispatched, whatever its reviewer goes on to return.
        record['candidate_id'] = candidate_id
    statement = ''
    if role == 'hypothesis-validator' and hypothesis_workspace.settings(payload) is not None:
        hypothesis = Tournament(payload).hypotheses().get(hypothesis_id) or {}
        statement = hypothesis.get('content') or hypothesis.get('summary') or ''
    record.update(
        hypothesis_workspace.on_agent_started(
            payload, role, plan_id, hypothesis_id,
            run_id=_marker(goal, 'loam-run-id'),
            goal=_strip_tags(goal),
            statement=statement,
        )
    )
    Parser().append(payload, 'hypothesis-orchestration.jsonl', record)
    if role == 'hypothesis-validator':
        Parser().append(payload, 'hypothesis-campaign.jsonl', {**record, 'event': 'hypothesis_validation_started'})


def _ledger_observation(payload: Mapping[str, Any]) -> tuple[str, str] | None:
    """The lifecycle ledger's verdict on this child, as (observation_status, failure_class).

    None when it cannot be determined, so a caller records nothing rather than
    asserting a child was observed. Read through ledger so there is one
    definition of what counts as an observation, not a second copy here.
    """

    status = str(payload.get('child_status') or '')
    if not status:
        return None
    summary = str(payload.get('child_summary') or '')
    return ledger._observation_status(status, summary), ledger._failure_class(status, summary)


def _record_review(payload: Mapping[str, Any], rows: list[dict[str, Any]], start: Mapping[str, Any],
                   summary: str) -> dict[str, Any]:
    """Turn a finished reviewer's contract into the review row the reconciler reads.

    The row goes to `hypothesis-reviews.jsonl` -- the same file an operator
    writes by hand -- so acceptance downstream is unchanged: only its author is.
    Returns the fields describing what happened, for the orchestration record.
    """

    candidate_id = start.get('candidate_id', '')
    if not candidate_id:
        return {'review_status': 'no_candidate_tag'}
    if ledger._observation_status(str(payload.get('child_status') or ''), summary) != 'observed':
        # A provider failure is not a review. Leaving no review row means the
        # candidate stays pending for a later round rather than being silently
        # treated as checked and found wanting.
        return {'review_status': 'unobserved'}
    candidate = next(
        (
            row for row in Parser().rows(payload, 'candidate-results.jsonl')
            if row.get('event') == 'candidate_result' and row.get('candidate_id') == candidate_id
        ),
        None,
    )
    if candidate is None:
        return {'review_status': 'candidate_missing'}
    if candidate.get('child_session_id') == payload.get('child_session_id'):
        # Structurally impossible today, and refused anyway: the whole point of
        # this step is that the acceptor is not the producer.
        return {'review_status': 'self_review_refused'}
    contract, presence = hypothesis_review.extract_contract_object(summary)
    if presence == 'absent' or contract is None:
        return {'review_status': 'contract_absent'}
    hypothesis_id = start.get('hypothesis_id', '')
    keys = [
        row.get('hypothesis_key')
        for row in ((candidate.get('candidate') or {}).get('hypotheses') or [])
        if isinstance(row, Mapping)
    ]
    row, status = hypothesis_review.normalize_review(
        contract,
        candidate,
        # Whether a hash-bearing manifest existed to attest against is Loam's
        # fact about this campaign, not the reviewer's claim about its work.
        manifest_available=bool(_artifact_manifest(payload, start.get('plan_id', ''), hypothesis_id)),
        # The Jnana id is Loam's, not the reviewer's: it is what links accepted
        # evidence to the Elo the tournament assigned, and a model-written id
        # would silently detach the two.
        ranking_hypothesis_ids={key: hypothesis_id for key in keys if key},
    )
    if row is None:
        return {'review_status': status}
    Parser().append(payload, hypothesis_review_file(), {
        'record_version': 'loam_evidence_review_v1',
        'event': 'evidence_review',
        'observed_at': _now(),
        'run_id': payload.get('loam_run_id', ''),
        'workflow_id': payload.get('workflow_id', ''),
        'plan_id': start.get('plan_id', ''),
        'hypothesis_id': hypothesis_id,
        'reviewer_session_id': payload.get('child_session_id', ''),
        'reviewer_unit_id': payload.get('child_subagent_id', ''),
        'reviewed_unit_id': candidate.get('unit_id', ''),
        'acceptance_basis': 'independent_agent_review',
        **row,
    })
    return {
        'review_status': status,
        'review_verdicts': row['review_verdicts'],
        'accepted_evidence': len(row['evidence']),
        'accepted_hypothesis_keys': row['accept_hypothesis_keys'],
    }


def record_stop(payload: Mapping[str, Any]) -> None:
    """Record a tagged child's terminal result as unvalidated orchestration input."""
    rows = _rows(payload)
    start = next((row for row in reversed(rows) if row.get('event') == 'agent_started' and _same_child(row, payload)), None)
    if start is None:
        return
    role = start['role']
    plan_id = start['plan_id']
    hypothesis_id = start['hypothesis_id']
    summary = payload.get('child_summary', '').strip()
    events = {
        'reasoner': 'reasoner_completed',
        'hypothesis-validator': 'validator_completed',
        'evidence-reviewer': 'evidence_review_completed',
    }
    record = {
        'event': events.get(role, 'code_generator_completed'),
        'observed_at': _now(),
        'run_id': payload.get('loam_run_id', ''),
        'workflow_id': payload.get('workflow_id', ''),
        'plan_id': plan_id,
        'role': role,
        'hypothesis_id': hypothesis_id,
        'child_subagent_id': payload.get('child_subagent_id', ''),
        'child_session_id': payload.get('child_session_id', ''),
        'status': payload.get('child_status') or 'error',
        'summary': summary,
    }
    # Hermes reports child_status "completed" for a child whose own API calls
    # failed -- the process exited cleanly and its summary is the provider's
    # error. Carry the ledger's verdict so a round can tell a validation that
    # ran from one that never did; observed 2026-10-31, five validators returned
    # "API call failed after 3 retries" and the round still advanced to its
    # reasoner as though five hypotheses had been tested.
    observation = _ledger_observation(payload)
    if observation is not None:
        record.update(observation_status=observation[0], failure_class=observation[1])
    if role == 'evidence-reviewer':
        record['candidate_id'] = start.get('candidate_id', '')
        record.update(_record_review(payload, rows, start, summary))
    if role == 'hypothesis-validator':
        score = _evidence_score(summary)
        # A score only counts if a code-generator child actually completed an
        # execution for this hypothesis; otherwise the claim is kept for audit.
        backed = any(
            row.get('event') == 'code_generator_completed' and row.get('plan_id') == plan_id
            and row.get('hypothesis_id') == hypothesis_id and row.get('status') == 'completed'
            for row in rows
        )
        if score is not None and not backed:
            record.update(reported_evidence_score=score, evidence_score=None, evidence_score_status='unbacked_no_exec')
        else:
            record.update(evidence_score=score, evidence_score_status='reported' if score is not None else 'missing_or_invalid')
    Parser().append(payload, 'hypothesis-orchestration.jsonl', record)
    if role == 'hypothesis-validator':
        Parser().append(payload, 'hypothesis-campaign.jsonl', {**record, 'event': 'hypothesis_validation_completed'})
    elif role == 'reasoner':
        validations = [row for row in rows if row.get('event') == 'validator_completed' and row.get('plan_id') == plan_id]
        Parser().append(payload, 'hypothesis-campaign.jsonl', {
            **record,
            'event': 'hypothesis_campaign_round_completed',
            'validator_results': validations,
            'synthesis': summary,
        })


def _evidence_score(summary: str) -> float | None:
    """The validator's evidence score: the first signed decimal under its last `Evidence Score` heading.

    The heading must start a line, so an in-prose mention is not read as the score.
    A decimal point is required because the heading line itself can carry numbers
    ("(0-1 scale)"); a bare integer counts only as the first token after the heading.
    """
    headings = list(re.finditer(r'^\s{0,3}#{1,6}\s*Evidence\s+Score\b', summary, re.IGNORECASE | re.MULTILINE))
    if not headings:
        return None
    window = summary[headings[-1].end():headings[-1].end() + 200]
    match = re.search(r'[-+−]?\d*\.\d+', window)
    if match is None:
        _, _, body = window.partition('\n')
        match = re.match(r'\A[^\S\n]*[*`_(\[]*([-+−]?\d+)\b', body.lstrip('\n'))
    if match is None:
        return None
    score = float(match.group(match.lastindex or 0).replace('−', '-'))
    return score if -1.0 <= score <= 1.0 else None


# --- workspace --------------------------------------------------------------

def _session_start(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    return _start_for(_rows(payload), _session(payload))


def workspace_policy(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """The workspace's filesystem boundary for a campaign child (hypothesis_workspace.fs_policy)."""
    if hypothesis_workspace.settings(payload) is None:
        return None
    return hypothesis_workspace.fs_policy(payload, _session_start(payload))


def workspace_capture(payload: Mapping[str, Any]) -> None:
    """Record the code an allowed campaign child runs."""
    if hypothesis_workspace.settings(payload) is not None:
        hypothesis_workspace.capture_code(payload, _session_start(payload))


def workspace_decision(payload: Mapping[str, Any]) -> bool:
    """Record one declared decision for the calling agent; whether one was stored."""
    if hypothesis_workspace.settings(payload) is None:
        return False
    return hypothesis_workspace.record_decision(payload, _session_start(payload))


def workspace_stop(payload: Mapping[str, Any]) -> None:
    """Close a stopped child's workspace run, and log any audit problems to the campaign ledger."""
    if hypothesis_workspace.settings(payload) is None:
        return
    rows = _rows(payload)
    start = next((row for row in reversed(rows) if row.get('event') == 'agent_started' and _same_child(row, payload)), None)
    if start is None:
        return
    completion = next(
        (row for row in reversed(rows) if row.get('event') == 'validator_completed' and _same_child(row, payload)), None
    )
    problems = hypothesis_workspace.on_agent_stopped(payload, start, completion)
    if problems:
        Parser().append(payload, 'hypothesis-campaign.jsonl', {
            'event': 'workspace_violation',
            'observed_at': _now(),
            'run_id': payload.get('loam_run_id', ''),
            'workflow_id': payload.get('workflow_id', ''),
            'plan_id': start['plan_id'],
            'hypothesis_id': start['hypothesis_id'],
            'workspace': start.get('workspace', ''),
            'problems': problems,
        })


# --- pre-tool policy ----------------------------------------------------------

def tool_policy(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """Block tool use that breaks a campaign tier's boundary.

    A raise here lets the call through (the adapter allows on a failed hook), so
    the model-written `arguments` are type-checked before they are read.
    """
    if payload.get('loam_hypothesis_orchestration') is not True:
        return None
    tool_name = payload.get('tool_name', '')
    rows = _rows(payload)
    start = _start_for(rows, _session(payload))

    if start is None:
        # The coordinator. It launches rounds only through DISPATCH_TOOL.
        phase = _round(payload, rows)['phase']
        if phase == 'reasoner_running':
            return _block("The active plan's reasoner is already dispatched; wait for its terminal result before any further tool call.")
        if tool_name == 'delegate_task':
            return _block(
                f"Call {DISPATCH_TOOL} to launch this round instead of composing a delegate_task "
                "batch. It takes no arguments and dispatches the hypothesis validators, or the "
                "reasoner once every validator has reported, from the campaign's persisted plan. "
                "You cannot see the selected hypotheses in full, so a batch you compose yourself "
                "is a substitute for the campaign's work, not the campaign's work."
            )
        if phase in ('validators_due', 'validators_running', 'reviewers_due', 'reviewers_running') and tool_name != DISPATCH_TOOL:
            return _block(
                "The active campaign plan is pending. Do not execute or inspect work as coordinator; "
                f"call {DISPATCH_TOOL} first and let this round's children report."
            )
        return None

    # Recording a decision is provenance, not execution, so every tier may do it.
    if tool_name == hypothesis_workspace.DECISION_TOOL:
        return None
    role = start['role']
    execution = payload.get('loam_execution') or {}
    if role == 'reasoner':
        return _block('reasoner has no tool capability; return the synthesis to the harness')

    if role == 'evidence-reviewer':
        # Read-only by construction. A reviewer that could run code could
        # recompute the number it is supposed to be checking, which turns an
        # independent attestation back into the producer's own claim.
        if tool_name == 'delegate_task':
            return _block('evidence-reviewer must not delegate; return your review contract')
        observation = ('inspect_filesystem', 'batch_status', 'list_batches')
        names = hpc_tool_names(execution)
        allowed = {names[key] for key in observation if key in names}
        if tool_name in allowed or tool_name in ('skills_list', 'skill_view'):
            return None
        return _block(
            "evidence-reviewer may call only the read-only tools "
            f"{sorted(allowed | {'skill_view', 'skills_list'})}: you check claims against sources and "
            "recorded artifacts, and you must not execute, recompute or produce anything"
        )

    if role == 'code-generator':
        if tool_name == 'delegate_task':
            return _block('code-generator must not delegate')
        skill_tools = ('skills_list', 'skill_view')
        if tool_name in skill_tools or execution.get('mode') != 'remote_mcp':
            return None
        allowed = sorted(set(hpc_tool_names(execution).values()))
        if tool_name not in allowed:
            return _block(f'code-generator may use only the HPC MCP tools {allowed} and the read-only skill tools {sorted(skill_tools)}')
        return None

    if role != 'hypothesis-validator':
        return None
    if tool_name != 'delegate_task':
        observation = ('batch_status', 'inspect_filesystem', 'list_batches')
        names = hpc_tool_names(execution)
        if tool_name in {names[key] for key in observation if key in names}:
            return None
        return _block(
            "hypothesis-validator must delegate executable work to code-generator; "
            f"it may call only the read-only tools {sorted(observation)} itself"
        )

    arguments = payload.get('arguments')
    arguments = arguments if isinstance(arguments, Mapping) else {}
    tasks = arguments.get('tasks')
    tasks = tasks if isinstance(tasks, list) else []
    # An empty `tasks` list is single-task mode, as Hermes itself treats it.
    if tasks:
        goals = [task.get('goal') for task in tasks if isinstance(task, Mapping)]
    else:
        goals = [arguments.get(key) for key in ('goal', 'context', 'task') if arguments.get(key)]
    goals = [goal for goal in goals if isinstance(goal, str) and goal.strip()]
    plan_id = start['plan_id']
    if not goals or any(
        _marker(goal, 'loam-agent-role') != 'code-generator' or _marker(goal, 'loam-plan-id') != plan_id
        for goal in goals
    ):
        return _block('hypothesis-validator may delegate only tagged code-generator children in its plan')

    used = sum(
        1 for row in rows
        if row.get('event') == 'agent_started' and row.get('role') == 'code-generator'
        and row.get('plan_id') == plan_id and row.get('hypothesis_id') == start['hypothesis_id']
    )
    if used + (len(tasks) or 1) > 2:
        return _block(f'code-generator budget for this hypothesis is 2 and {used} are used; return your report now.')
    return None


def _block(message: str) -> dict[str, str]:
    return {'action': 'block', 'message': message}
