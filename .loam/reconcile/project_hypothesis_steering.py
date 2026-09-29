#!/usr/bin/env python3
"""Project separate Jnana and validated-evidence signals into steering advice.

This is a read-only policy projection. It never accepts evidence, mutates Jnana
state, or schedules a tool call.  The parent agent receives the result as
advisory context and retains control of action selection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

VERSION = "loam_hypothesis_steering_v1"
ELO_PRIOR = 1200.0
ELO_SCALE = 400.0 / math.log(10.0)
WEIGHTS = {"elo": 0.35, "evidence": 0.50, "uncertainty": 0.15}
INDEPENDENCE_DECAY = (1.0, 0.25, 0.10)


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        values.append(value)
    return values


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _z_scores(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
    deviation = math.sqrt(variance)
    return {key: (value - mean) / deviation if deviation else 0.0 for key, value in values.items()}


def _tournament_elos(rows: Iterable[Mapping[str, Any]], workflow_id: str) -> dict[str, float]:
    latest: Mapping[str, Any] | None = None
    for row in rows:
        if row.get("schema_version") != "loam_jnana_tournament_v1":
            continue
        if workflow_id and str(row.get("workflow_id") or "") != workflow_id:
            continue
        latest = row
    if latest is None:
        return {}
    result = latest.get("result")
    rankings = result.get("top_hypotheses") if isinstance(result, Mapping) else None
    if not isinstance(rankings, list):
        return {}
    return {
        str(item.get("hypothesis_id") or ""): _number(item.get("elo_rating"), ELO_PRIOR)
        for item in rankings
        if isinstance(item, Mapping) and str(item.get("hypothesis_id") or "")
    }


def project(
    hypotheses: Iterable[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]], tournament_reports: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return a deterministic, workflow/session-scoped steering snapshot."""

    scopes: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in hypotheses:
        if row.get("event") == "hypothesis_accepted" and row.get("status") == "active" and row.get("hypothesis_id"):
            scopes[(str(row.get("workflow_id") or ""), str(row.get("parent_session_id") or ""))][str(row["hypothesis_id"])] = row

    evidence_by_hypothesis: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_evidence: set[str] = set()
    for row in evidence:
        evidence_id = str(row.get("evidence_id") or "")
        if (
            row.get("event") == "evidence_accepted"
            and row.get("validation_status") == "accepted"
            and evidence_id
            and evidence_id not in seen_evidence
        ):
            seen_evidence.add(evidence_id)
            evidence_by_hypothesis[str(row.get("hypothesis_id") or "")].append(row)

    snapshots = []
    reports = list(tournament_reports)
    for (workflow_id, session_id), active in sorted(scopes.items()):
        elos = _tournament_elos(reports, workflow_id)
        state: dict[str, dict[str, Any]] = {}
        for hypothesis_id, row in active.items():
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for item in evidence_by_hypothesis[hypothesis_id]:
                grouped[str(item.get("independence_group_id") or item.get("source_ref") or item.get("evidence_id"))].append(item)
            log_odds = 0.0
            supports = contradicts = 0
            for group in grouped.values():
                group_delta = 0.0
                for index, item in enumerate(sorted(group, key=lambda item: str(item.get("evidence_id") or ""))):
                    decay = INDEPENDENCE_DECAY[min(index, len(INDEPENDENCE_DECAY) - 1)]
                    group_delta += _number(item.get("evidence_log_odds_delta")) * decay
                log_odds += group_delta
                if group_delta > 0:
                    supports += 1
                elif group_delta < 0:
                    contradicts += 1
            evidence_rating = ELO_PRIOR + ELO_SCALE * log_odds
            state[hypothesis_id] = {
                "hypothesis_id": hypothesis_id,
                "jnana_hypothesis_id": str(row.get("jnana_hypothesis_id") or hypothesis_id),
                "statement": str(row.get("statement") or ""),
                "jnana_elo": elos.get(str(row.get("jnana_hypothesis_id") or hypothesis_id), ELO_PRIOR),
                "evidence_log_odds": log_odds,
                "evidence_rating": evidence_rating,
                "independent_support_count": supports,
                "independent_contradiction_count": contradicts,
                "uncertainty": 1.0 / math.sqrt(1.0 + len(grouped)),
            }
        elo_z = _z_scores({key: item["jnana_elo"] for key, item in state.items()})
        evidence_z = _z_scores({key: item["evidence_rating"] for key, item in state.items()})
        uncertainty_z = _z_scores({key: item["uncertainty"] for key, item in state.items()})
        for hypothesis_id, item in state.items():
            item["rank_evidence_disagreement"] = abs(item["jnana_elo"] - item["evidence_rating"])
            item["steering_priority"] = (
                WEIGHTS["elo"] * elo_z[hypothesis_id]
                + WEIGHTS["evidence"] * evidence_z[hypothesis_id]
                + WEIGHTS["uncertainty"] * uncertainty_z[hypothesis_id]
            )
            if item["rank_evidence_disagreement"] >= 100.0:
                item["recommended_action"] = "resolve_rank_evidence_disagreement"
            elif item["independent_contradiction_count"]:
                item["recommended_action"] = "seek_independent_contradiction_resolution"
            elif item["independent_support_count"] == 0:
                item["recommended_action"] = "seek_independent_source_verification"
            else:
                item["recommended_action"] = "refine_or_test"
        snapshots.append(
            {
                "record_version": VERSION,
                "event": "hypothesis_steering_snapshot",
                "workflow_id": workflow_id,
                "parent_session_id": session_id,
                "scoring_policy": {"weights": WEIGHTS, "elo_prior": ELO_PRIOR, "independence_decay": INDEPENDENCE_DECAY},
                "hypotheses": sorted(state.values(), key=lambda item: (-item["steering_priority"], item["hypothesis_id"])),
            }
        )
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--tournament-reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    if args.tournament_reports.is_dir():
        for path in sorted(args.tournament_reports.glob("tournament-*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                reports.append(value)
    else:
        reports = _rows(args.tournament_reports)
    snapshots = project(_rows(args.hypotheses), _rows(args.evidence), reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in snapshots), encoding="utf-8")
    print(json.dumps({"snapshots_written": len(snapshots), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
