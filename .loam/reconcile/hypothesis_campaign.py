"""Isolated multi-round Jnana exploration campaign orchestration.

A round ranks hypotheses, explores the ranked top three, records one automatic
``unverified_exploration_credit`` per completed exploration, and synthesizes a
compact round result. The next round receives that synthesis in its tournament
input. These credits are operational accounting only: they cannot be consumed
as accepted evidence, evidence rating, contribution credit, or scientific
support.

The caller owns LLM calls, Jnana state locking, and model configuration. This
module owns only deterministic artifact construction and round-to-round data
flow so it can be tested without a live model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

TOP_K = 3
# Jnana's pairwise prompt already includes both full hypothesis bodies. Keep
# prior operational context short enough that it cannot crowd out structured
# ranking output on compact OpenAI-compatible endpoints.
PRIOR_SYNTHESIS_LIMIT = 300
EXPLORATION_SYNTHESIS_LIMIT = 800
CREDIT_VERSION = "loam_unverified_exploration_credit_v1"
ROUND_VERSION = "loam_hypothesis_campaign_round_v1"

TournamentRunner = Callable[[dict[str, Any]], Mapping[str, Any]]
HypothesisLoader = Callable[[], Mapping[str, Mapping[str, Any]]]
Explorer = Callable[[Mapping[str, Any], str], str]
Synthesizer = Callable[[str, list[dict[str, Any]]], str]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _latest_synthesis(rounds_path: Path) -> str:
    if not rounds_path.is_file():
        return ""
    latest = ""
    for line in rounds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get("synthesis"), str):
            latest = row["synthesis"]
    return latest


def _round_goal(research_goal: str, prior_synthesis: str) -> str:
    if not prior_synthesis:
        return research_goal
    bounded_synthesis = prior_synthesis[:PRIOR_SYNTHESIS_LIMIT]
    return (
        f"{research_goal}\n\n"
        "Prior-round exploration synthesis (unverified operational context; do not treat as evidence):\n"
        f"{bounded_synthesis}"
    )


def run_round(
    *,
    campaign_dir: Path,
    workflow_id: str,
    session_id: str,
    research_goal: str,
    turn: int,
    tournament_runner: TournamentRunner,
    hypothesis_loader: HypothesisLoader,
    explorer: Explorer,
    synthesizer: Synthesizer,
) -> dict[str, Any]:
    """Run one isolated round and append its immutable operational artifacts."""

    campaign_dir = Path(campaign_dir)
    rounds_path = campaign_dir / "rounds.jsonl"
    credits_path = campaign_dir / "unverified-exploration-credit.jsonl"
    prior_synthesis = _latest_synthesis(rounds_path)
    tournament_input = {
        "workflow_id": workflow_id,
        "session_id": session_id,
        "loam_turn": turn,
        "user_message": _round_goal(research_goal, prior_synthesis),
    }
    tournament = dict(tournament_runner(tournament_input))
    if tournament.get("status") != "completed":
        raise ValueError(f"incomplete tournament: status={tournament.get('status')!r}")
    requested = tournament.get("matches_requested")
    completed = tournament.get("matches_completed")
    if requested is not None and completed != requested:
        raise ValueError(
            f"incomplete tournament: requested={requested!r}, completed={completed!r}"
        )
    rankings = tournament.get("top_hypotheses")
    if not isinstance(rankings, list):
        rankings = []
    hypotheses = hypothesis_loader()
    ranked_ids = [
        str(item.get("hypothesis_id") or "")
        for item in rankings[:TOP_K]
        if isinstance(item, Mapping)
    ]
    missing = [hypothesis_id for hypothesis_id in ranked_ids if hypothesis_id not in hypotheses]
    if missing:
        raise ValueError(f"ranked hypothesis missing from persistent state: {missing}")
    explored: list[dict[str, Any]] = []
    for rank, item in enumerate(rankings[:TOP_K], start=1):
        if not isinstance(item, Mapping):
            continue
        hypothesis_id = str(item.get("hypothesis_id") or "")
        hypothesis = hypotheses.get(hypothesis_id)
        if not hypothesis_id or not isinstance(hypothesis, Mapping):
            continue
        summary = str(explorer(hypothesis, research_goal) or "").strip()
        if not summary:
            continue
        result = {
            "hypothesis_id": hypothesis_id,
            "rank": rank,
            "elo_rating": item.get("elo_rating"),
            "hypothesis_summary": str(item.get("summary") or hypothesis.get("summary") or ""),
            "exploration_summary": summary,
        }
        explored.append(result)
    if len(explored) != len(ranked_ids):
        completed_ids = {item["hypothesis_id"] for item in explored}
        missing_explorations = [
            hypothesis_id for hypothesis_id in ranked_ids if hypothesis_id not in completed_ids
        ]
        raise ValueError(f"incomplete top-three exploration: {missing_explorations}")
    synthesis_input = [
        {**item, "exploration_summary": item["exploration_summary"][:EXPLORATION_SYNTHESIS_LIMIT]}
        for item in explored
    ]
    synthesis = str(synthesizer(research_goal, synthesis_input) or "").strip()
    if not synthesis:
        raise ValueError("empty round synthesis")
    for item in explored:
        _append(
            credits_path,
            {
                "record_version": CREDIT_VERSION,
                "event": "unverified_exploration_credit_assigned",
                "observed_at": _timestamp(),
                "workflow_id": workflow_id,
                "parent_session_id": session_id,
                "round_turn": turn,
                "hypothesis_id": item["hypothesis_id"],
                "amount": 1.0,
                "basis": "completed_campaign_exploration",
                "epistemic_status": "unverified",
                "eligible_for_evidence_rating": False,
                "eligible_for_contribution_credit": False,
                "exploration_summary": item["exploration_summary"],
            },
        )
    result = {
        "record_version": ROUND_VERSION,
        "event": "hypothesis_campaign_round_completed",
        "observed_at": _timestamp(),
        "workflow_id": workflow_id,
        "parent_session_id": session_id,
        "turn": turn,
        "prior_round_synthesis": prior_synthesis,
        "tournament_input": tournament_input,
        "tournament": tournament,
        "explorations": explored,
        "explored_hypothesis_ids": [item["hypothesis_id"] for item in explored],
        "synthesis": synthesis,
        "credit_semantics": "unverified operational exploration credit only; not evidence or scientific support",
    }
    _append(rounds_path, result)
    return result
