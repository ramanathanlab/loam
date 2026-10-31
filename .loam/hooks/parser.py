from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Mapping


class Parser:
    def __init__(
        self,
        parent_memory_file: str = 'parent-memory.jsonl',
        memory_file: str = 'hypothesis-memory.jsonl',
        memory_event: str = 'hypothesis_memory_snapshot',
        steering_file: str = 'hypothesis-steering.jsonl',
        steering_event: str = 'hypothesis_steering_snapshot',
        max_findings: int = 12,
        max_hypotheses: int = 6,
        max_steered_hypotheses: int = 4,
        statement_length: int = 1000,
    ):
        self.parent_memory_file = parent_memory_file
        self.memory_file = memory_file
        self.memory_event = memory_event
        self.steering_file = steering_file
        self.steering_event = steering_event
        self.max_findings = max_findings
        self.max_hypotheses = int(os.environ.get('LOAM_RECALL_MAX_HYPOTHESES', max_hypotheses))
        self.max_steered_hypotheses = max_steered_hypotheses
        self.statement_length = statement_length

    def format_path(
        self,
        payload: Mapping[str, Any],
        filename: str,
        override_env: str | None = None,
    ) -> Path:
        """
        Format path using payload directory > env var > override file > root as the order of truth.
        """
        state_dir = payload.get('loam_state_dir')
        runtime_state_dir = os.environ.get('LOAM_STATE_DIR', '').strip()
        if state_dir:
            return Path(state_dir).expanduser() / filename
        elif runtime_state_dir:
            return Path(runtime_state_dir).expanduser() / filename
        elif override_env and os.environ.get(override_env, '').strip():
            return Path(os.environ[override_env].strip()).expanduser()
        return Path(payload.get('repo_root') or '.') / '.loam' / 'state' / filename

    def rows(
        self,
        payload: Mapping[str, Any],
        filename: str,
        override_env: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Every row of a JSONL state file, skipping a torn line left by a writer mid-append.
        """
        path = self.format_path(payload, filename, override_env)
        if not path.is_file():
            return []

        rows = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return rows

    def append(
        self,
        payload: Mapping[str, Any],
        filename: str,
        row: Mapping[str, Any],
    ) -> None:
        path = self.format_path(payload, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + '\n')

    def snapshot(
        self,
        payload: Mapping[str, Any],
        filename: str,
        event: str | None = None,
        override_env: str | None = None,
        match_turn: bool = False,
    ) -> Mapping[str, Any] | None:
        """
        Return the latest `event` row (any row when `event` is None) for this workflow, or for this
        parent session when there is no workflow; `match_turn` also pins the session to its parent turn.
        """
        workflow_id = payload.get('workflow_id', '')
        session_id = payload.get('session_id') or payload.get('parent_session_id')
        turn_id = payload.get('parent_turn_id') if match_turn else None

        latest = None
        for row in self.rows(payload, filename, override_env):
            if event and row.get('event') != event:
                continue

            if workflow_id and row.get('workflow_id') == workflow_id:
                latest = row
            elif (
                not workflow_id
                and session_id
                and row.get('parent_session_id') == session_id
                and (not turn_id or row.get('parent_turn_id') == turn_id)
            ):
                latest = row

        return latest

    def memory_snapshot(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return self.snapshot(
            payload,
            self.parent_memory_file,
            override_env='LOAM_PARENT_MEMORY_PATH',
            match_turn=True,
        )

    def memory_context(self, payload: Mapping[str, Any]) -> str:
        snapshot = self.memory_snapshot(payload)
        if snapshot is None:
            return ''

        findings = snapshot.get('accepted_findings') or []
        if not findings:
            return ''

        lines = ['Validated Loam findings for this parent session (not child summaries):']
        for finding in findings[:self.max_findings]:
            lines.append(
                '- {kind} {accession}: {field} = {value}'.format(
                    kind=finding.get('source_kind', 'source'),
                    accession=finding.get('source_accession', ''),
                    field=finding.get('field', ''),
                    value=finding.get('value', ''),
                )
            )

        lines.append('Do not re-delegate work solely to rediscover an exact finding listed above.')
        return '\n'.join(lines)

    def hypothesis_context(self, payload: Mapping[str, Any]) -> str:
        snapshot = self.snapshot(payload, self.memory_file, self.memory_event)
        if snapshot is None:
            return ''

        hypotheses = snapshot.get('active_hypotheses') or []
        if not hypotheses:
            return ''

        lines = ['Validated hypothesis state (evidence support, tournament ranking, and contribution credit are separate signals):']
        for hypothesis in hypotheses[:self.max_hypotheses]:
            statement = ' '.join(hypothesis.get('statement', '').split())[:self.statement_length]
            lines.append(
                '- {identifier}: support={support}, contradiction={contradict}; {statement}'.format(
                    identifier=hypothesis.get('hypothesis_id', 'unknown'),
                    support=len(hypothesis.get('supporting_evidence', [])),
                    contradict=len(hypothesis.get('contradicting_evidence', [])),
                    statement=statement,
                )
            )

        lines.append('Use only source-backed evidence listed here as epistemic support; tournament Elo is an advisory prioritization signal.')
        return '\n'.join(lines)

    def steering_context(self, payload: Mapping[str, Any]) -> str:
        snapshot = self.snapshot(payload, self.steering_file, self.steering_event)
        if snapshot is None:
            return ''

        hypotheses = snapshot.get('hypotheses') or []
        if not hypotheses:
            return ''

        lines = ['Hypothesis steering state (advisory; evidence acceptance, Elo ranking, and action selection remain distinct):']
        for item in hypotheses[:self.max_steered_hypotheses]:
            lines.append(
                '- {id}: Elo={elo:.1f}, evidence-rating={evidence:.1f}, priority={priority:.2f}; next={action}.'.format(
                    id=item.get('hypothesis_id', 'unknown'),
                    elo=item.get('jnana_elo', 0.0),
                    evidence=item.get('evidence_rating', 0.0),
                    priority=item.get('steering_priority', 0.0),
                    action=item.get('recommended_action', 'review'),
                )
            )

        lines.append('Follow the recommendation only after checking its linked source-backed evidence and current task constraints.')
        return '\n'.join(lines)
