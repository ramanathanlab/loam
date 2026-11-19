"""Steering score: combine tournament Elo with independently accepted evidence (after loam's
project_hypothesis_steering.py). Advisory input to the reasoner; never itself evidence.

Per accepted claim:  delta = polarity * strength * reliability * relevance   (weights by verification mode)
Per independence group, repeats decay 1.0 / 0.25 / 0.10, so one run's five failed gates are not five
independent contradictions.
    evidence_rating = 1200 + (400 / ln 10) * sum(deltas)          (same scale as Elo)
    uncertainty     = 1 / sqrt(1 + independent groups)
    priority        = 0.35 z(Elo) + 0.50 z(evidence_rating) + 0.15 z(uncertainty)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ELO_PRIOR = 1200.0
ELO_SCALE = 400.0 / math.log(10.0)
PRIORITY_WEIGHTS = {'elo': 0.35, 'evidence': 0.50, 'uncertainty': 0.15}
INDEPENDENCE_DECAY = (1.0, 0.25, 0.10)
POLARITY = {'supports': 1.0, 'contradicts': -1.0, 'context': 0.0}

# Weight per accepted claim, by how the reviewer verified it: (strength, reliability, relevance).
WEIGHT_POLICY = {
    'source_confirmed': (0.8, 0.9, 0.8),    # an independent record was read
    'artifact_attested': (0.7, 0.7, 0.8),   # number found in an output file this campaign computed
}


def claim_delta(stance: str, policy: str) -> float:
    return POLARITY[stance] * math.prod(WEIGHT_POLICY[policy])


def read_ledger(ledger: Path) -> list[dict]:
    """Every ledger row; a line cut short by a crash is skipped rather than failing the campaign."""
    rows = []
    if ledger.is_file():
        for line in ledger.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def project(ledger: Path, elos: dict[str, float]) -> list[dict]:
    """One row per hypothesis known to Jnana or the ledger, highest steering priority first."""
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    known = set(elos)
    for row in read_ledger(ledger):
        hid = row['hypothesis_id']
        known.add(hid)
        if row.get('stance') != 'context':
            groups[hid][row['group']].append(row['delta'])

    rows = {}
    for hid in known:
        log_odds, supports, contradicts = 0.0, 0, 0
        for deltas in groups[hid].values():
            ordered = sorted(deltas, key=abs, reverse=True)  # strongest claim counts in full
            g = sum(d * INDEPENDENCE_DECAY[min(i, 2)] for i, d in enumerate(ordered))
            log_odds += g
            supports += g > 0
            contradicts += g < 0
        elo, rating = elos.get(hid, ELO_PRIOR), ELO_PRIOR + ELO_SCALE * log_odds
        rows[hid] = {'hypothesis_id': hid, 'elo': round(elo, 1), 'evidence_rating': round(rating, 1),
                     'independent_supports': supports, 'independent_contradictions': contradicts,
                     'uncertainty': round(1 / math.sqrt(1 + len(groups[hid])), 3),
                     'rank_evidence_disagreement': round(abs(elo - rating), 1)}

    z = {key: _z({h: r[field] for h, r in rows.items()})
         for key, field in [('elo', 'elo'), ('evidence', 'evidence_rating'), ('uncertainty', 'uncertainty')]}
    for hid, r in rows.items():
        r['priority'] = round(sum(w * z[key][hid] for key, w in PRIORITY_WEIGHTS.items()), 3)
        r['recommended_action'] = _action(r)
    return sorted(rows.values(), key=lambda r: (-r['priority'], r['hypothesis_id']))


def _action(r: dict) -> str:
    if r['rank_evidence_disagreement'] >= 100:
        return 'resolve_rank_evidence_disagreement'
    if r['independent_contradictions']:
        return 'seek_independent_contradiction_resolution'
    if not r['independent_supports']:
        return 'seek_independent_verification'
    return 'refine_or_test'


def _z(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values))
    return {k: (v - mean) / sd if sd else 0.0 for k, v in values.items()}
