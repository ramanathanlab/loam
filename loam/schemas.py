"""What each agent returns. pydantic-ai validates these and makes the model retry on a bad one."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Hypothesis(AgentOutput):
    id: str
    summary: str
    content: str
    elo: float
    matches: int = 0


class CodeResult(AgentOutput):
    status: Literal['executed', 'submitted', 'blocked']
    code: str = Field(description='The complete snippet that was run or submitted.')
    result: str = Field(description='Verbatim execution output, batch id, or the blocking error.')
    artifacts: list[str] = Field(default=[], description='Paths of output files this run wrote.')


class Claim(AgentOutput):
    claim: str = Field(description='What was measured or what the source states, with the number and any gate.')
    source: str = Field(description='Resolvable URL of the record, or the artifact path that holds the number.')
    stance: Literal['supports', 'contradicts', 'context']


class ValidationReport(AgentOutput):
    outcome: Literal['supported', 'contradicted', 'inconclusive', 'blocked']
    evidence_score: float = Field(ge=-1, le=1)
    ledger: str = Field(description='Markdown evidence ledger: each action, tool/skill used, observation, direction.')
    claims: list[Claim]
    next_step: str
    pending_batches: list[str] = Field(default=[], description='Batch ids still running when you stopped.')


class ClaimVerdict(AgentOutput):
    index: int
    verdict: Literal['accepted', 'rejected', 'unverifiable']
    verification: Literal['source_confirmed', 'artifact_attested'] | None = None
    source_ref: str = Field(default='', description='The record field or artifact path+line you read.')
    checked: str = Field(description='What you actually did and saw.')

    @model_validator(mode='after')
    def valid_verification(self) -> ClaimVerdict:
        if self.verdict != 'accepted':
            self.verification = None  # only an acceptance carries a verification mode
        elif self.verification is None or not self.source_ref.strip() or not self.checked.strip():
            raise ValueError('accepted verdicts require verification, source_ref, and checked details')
        return self


class Review(AgentOutput):
    verdicts: list[ClaimVerdict]
    statement_overreach: bool = Field(default=False, description='The hypothesis claims more than was tested.')

    @model_validator(mode='after')
    def ordered_verdicts(self) -> Review:
        if [verdict.index for verdict in self.verdicts] != list(range(len(self.verdicts))):
            raise ValueError('verdicts must have unique, ordered, zero-based claim indices')
        return self


class Brief(AgentOutput):
    report: str = Field(description='Markdown synthesis: experiment state, thread synthesis, regions, rationale.')
    tournament_brief: str = Field(description='Steering text appended to the next tournament goal.')
