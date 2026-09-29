"""Tests for loose hypothesis provenance and evidence-only credit."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
RECONCILE = ROOT / ".loam" / "reconcile" / "hypothesis_provenance.py"
PROJECT = ROOT / ".loam" / "reconcile" / "project_hypothesis_memory.py"
STEERING = ROOT / ".loam" / "reconcile" / "project_hypothesis_steering.py"


def load(name: str, path: Path):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HypothesisProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load("hypothesis_contract", HOOKS / "hypothesis_provenance.py")
        self.reconciler = load("hypothesis_reconciler", RECONCILE)
        self.projector = load("hypothesis_projector", PROJECT)
        self.recall = load("hypothesis_recall_test", HOOKS / "hypothesis_recall.py")

    def _candidate(self) -> dict:
        raw = {
            "contract_version": "hypothesis_provenance_v1",
            "hypotheses": [
                {
                    "hypothesis_key": "intervention-x",
                    "statement": "Intervention X improves outcome Y through mechanism Z.",
                    "parent_hypothesis_keys": [],
                }
            ],
            "evidence_candidates": [
                {
                    "hypothesis_key": "intervention-x",
                    "claim": "Study reports an improved Y after X.",
                    "source_url": "https://example.org/study",
                    "stance": "supports",
                },
                {
                    "hypothesis_key": "intervention-x",
                    "claim": "Study reports a confounder for X.",
                    "source_url": "https://example.org/caveat",
                    "stance": "contradicts",
                },
            ],
        }
        candidate, status = self.contract.normalize_candidate(raw)
        self.assertEqual(status, "pending")
        return {
            "record_version": self.contract.CONTRACT_VERSION,
            "candidate_status": "pending",
            "candidate_id": "candidate-1",
            "unit_id": "unit-1",
            "workflow_id": "workflow-1",
            "parent_session_id": "session-1",
            "parent_turn_id": "turn-1",
            "candidate": candidate,
        }

    def test_raw_candidate_cannot_create_credit_without_accepted_review(self) -> None:
        candidate = self._candidate()
        hypotheses, evidence, credit = self.reconciler.reconcile(
            candidate,
            {
                "candidate_id": "candidate-1",
                "accept_hypothesis_keys": ["intervention-x"],
                "evidence": [
                    {
                        "hypothesis_key": "intervention-x",
                        "claim": "Study reports an improved Y after X.",
                        "source_url": "https://example.org/study",
                        "stance": "supports",
                        "validation_status": "pending",
                    }
                ],
            },
        )
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(evidence, [])
        self.assertEqual(credit, [])

    def test_accepted_source_backed_evidence_projects_and_credits_once(self) -> None:
        candidate = self._candidate()
        review = {
            "candidate_id": "candidate-1",
            "accept_hypothesis_keys": ["intervention-x"],
            "evidence": [
                {
                    "hypothesis_key": "intervention-x",
                    "claim": "Study reports an improved Y after X.",
                    "source_url": "https://example.org/study",
                    "stance": "supports",
                    "validation_status": "accepted",
                    "source_ref": "DOI:10.1000/example",
                },
                {
                    "hypothesis_key": "intervention-x",
                    "claim": "Study reports a confounder for X.",
                    "source_url": "https://example.org/caveat",
                    "stance": "contradicts",
                    "validation_status": "accepted",
                    "source_ref": "DOI:10.1000/caveat",
                },
            ],
        }
        hypotheses, evidence, credit = self.reconciler.reconcile(candidate, review)
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(len(credit), 2)
        snapshots = self.projector.project(hypotheses, evidence, credit + credit)
        snapshot = snapshots[0]
        active = snapshot["active_hypotheses"][0]
        self.assertEqual(active["evidence_score"], 0)
        self.assertGreater(active["contribution_credit"], 0.0)
        self.assertEqual(len(active["supporting_evidence"]), 1)
        self.assertEqual(len(active["contradicting_evidence"]), 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypothesis-memory.jsonl"
            path.write_text(__import__("json").dumps(snapshot) + "\n", encoding="utf-8")
            context = self.recall.context_for(
                {"loam_state_dir": tmp, "workflow_id": "workflow-1", "session_id": "session-1"}
            )
        self.assertIn("support=1, contradiction=1", context)
        self.assertIn("tournament Elo is an advisory prioritization signal", context)
    def test_steering_combines_matched_jnana_elo_and_independent_evidence(self) -> None:
        steering = load("hypothesis_steering", STEERING)
        hypotheses = [
            {"event": "hypothesis_accepted", "status": "active", "hypothesis_id": "loam-a", "jnana_hypothesis_id": "jnana-a", "workflow_id": "workflow-1", "parent_session_id": "session-1", "statement": "A"},
            {"event": "hypothesis_accepted", "status": "active", "hypothesis_id": "loam-b", "jnana_hypothesis_id": "jnana-b", "workflow_id": "workflow-1", "parent_session_id": "session-1", "statement": "B"},
        ]
        evidence = [
            {"event": "evidence_accepted", "validation_status": "accepted", "evidence_id": "ev-a", "hypothesis_id": "loam-a", "stance": "supports", "independence_group_id": "study-a", "evidence_log_odds_delta": 0.8},
            {"event": "evidence_accepted", "validation_status": "accepted", "evidence_id": "ev-a-duplicate", "hypothesis_id": "loam-a", "stance": "supports", "independence_group_id": "study-a", "evidence_log_odds_delta": 0.8},
            {"event": "evidence_accepted", "validation_status": "accepted", "evidence_id": "ev-b", "hypothesis_id": "loam-b", "stance": "contradicts", "independence_group_id": "study-b", "evidence_log_odds_delta": -0.8},
        ]
        report = {
            "schema_version": "loam_jnana_tournament_v1",
            "workflow_id": "workflow-1",
            "result": {"top_hypotheses": [{"hypothesis_id": "jnana-a", "elo_rating": 1280}, {"hypothesis_id": "jnana-b", "elo_rating": 1120}]},
        }
        snapshot = steering.project(hypotheses, evidence, [report])[0]
        by_id = {item["hypothesis_id"]: item for item in snapshot["hypotheses"]}
        self.assertEqual(by_id["loam-a"]["jnana_elo"], 1280.0)
        self.assertGreater(by_id["loam-a"]["evidence_rating"], 1200.0)
        self.assertLess(by_id["loam-b"]["evidence_rating"], 1200.0)
        self.assertEqual(by_id["loam-a"]["independent_support_count"], 1)
        self.assertEqual(snapshot["scoring_policy"]["weights"]["evidence"], 0.50)


if __name__ == "__main__":
    unittest.main()
