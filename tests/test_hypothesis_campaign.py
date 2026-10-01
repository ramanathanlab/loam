"""Tests for an isolated, explicitly unverified Jnana exploration campaign."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / ".loam" / "reconcile" / "hypothesis_campaign.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HypothesisCampaignTests(unittest.TestCase):
    def test_round_rejects_missing_completed_exploration_before_credit(self) -> None:
        campaign = load("hypothesis_campaign_incomplete_exploration", CAMPAIGN)
        rankings = [
            {"hypothesis_id": "h-1", "summary": "First", "elo_rating": 1210},
            {"hypothesis_id": "h-2", "summary": "Second", "elo_rating": 1200},
            {"hypothesis_id": "h-3", "summary": "Third", "elo_rating": 1190},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "incomplete top-three exploration"):
                campaign.run_round(
                    campaign_dir=root,
                    workflow_id="campaign-v1",
                    session_id="session-1",
                    research_goal="Explain a GC3 gradient.",
                    turn=1,
                    tournament_runner=lambda _payload: {
                        "status": "completed",
                        "matches_requested": 1,
                        "matches_completed": 1,
                        "top_hypotheses": rankings,
                    },
                    hypothesis_loader=lambda: {item["hypothesis_id"]: item for item in rankings},
                    explorer=lambda hypothesis, _goal: "" if hypothesis["hypothesis_id"] == "h-3" else "done",
                    synthesizer=lambda _goal, _results: "unused",
                )
            self.assertFalse((root / "rounds.jsonl").exists())
            self.assertFalse((root / "unverified-exploration-credit.jsonl").exists())

    def test_round_rejects_missing_ranked_hypothesis_before_credit(self) -> None:
        campaign = load("hypothesis_campaign_missing_hypothesis", CAMPAIGN)
        rankings = [
            {"hypothesis_id": "h-1", "summary": "First", "elo_rating": 1210},
            {"hypothesis_id": "h-2", "summary": "Second", "elo_rating": 1200},
            {"hypothesis_id": "h-3", "summary": "Third", "elo_rating": 1190},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "ranked hypothesis missing"):
                campaign.run_round(
                    campaign_dir=root,
                    workflow_id="campaign-v1",
                    session_id="session-1",
                    research_goal="Explain a GC3 gradient.",
                    turn=1,
                    tournament_runner=lambda _payload: {
                        "status": "completed",
                        "matches_requested": 1,
                        "matches_completed": 1,
                        "top_hypotheses": rankings,
                    },
                    hypothesis_loader=lambda: {"h-1": rankings[0], "h-2": rankings[1]},
                    explorer=lambda _hypothesis, _goal: "unused",
                    synthesizer=lambda _goal, _results: "unused",
                )
            self.assertFalse((root / "rounds.jsonl").exists())
            self.assertFalse((root / "unverified-exploration-credit.jsonl").exists())

    def test_round_rejects_incomplete_tournament_before_exploration_credit(self) -> None:
        campaign = load("hypothesis_campaign_incomplete_tournament", CAMPAIGN)
        explored = []
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "incomplete tournament"):
                campaign.run_round(
                    campaign_dir=Path(temporary),
                    workflow_id="campaign-v1",
                    session_id="session-1",
                    research_goal="Explain a GC3 gradient.",
                    turn=1,
                    tournament_runner=lambda _payload: {
                        "status": "completed",
                        "matches_requested": 1,
                        "matches_completed": 0,
                        "top_hypotheses": [],
                    },
                    hypothesis_loader=lambda: {},
                    explorer=lambda hypothesis, goal: explored.append((hypothesis, goal)) or "unused",
                    synthesizer=lambda _goal, _results: "unused",
                )
        self.assertEqual(explored, [])

    def test_next_round_bounds_prior_synthesis_in_tournament_input(self) -> None:
        campaign = load("hypothesis_campaign_bounded_synthesis", CAMPAIGN)
        prior = "x" * 9000
        with tempfile.TemporaryDirectory() as temporary:
            rounds = Path(temporary) / "rounds.jsonl"
            rounds.write_text(json.dumps({"synthesis": prior}) + "\n", encoding="utf-8")
            injected = campaign._round_goal("Goal.", campaign._latest_synthesis(rounds))

        self.assertIn("Goal.", injected)
        self.assertIn(campaign.PRIOR_SYNTHESIS_LIMIT * "x", injected)
        self.assertNotIn("x" * (campaign.PRIOR_SYNTHESIS_LIMIT + 1), injected)

    def test_round_rejects_empty_synthesis_without_persisting_credit(self) -> None:
        campaign = load("hypothesis_campaign_empty_synthesis", CAMPAIGN)
        rankings = [{"hypothesis_id": "h-1", "summary": "First", "elo_rating": 1210}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "empty round synthesis"):
                campaign.run_round(
                    campaign_dir=root,
                    workflow_id="campaign-v1",
                    session_id="session-1",
                    research_goal="Explain a GC3 gradient.",
                    turn=1,
                    tournament_runner=lambda _payload: {
                        "status": "completed",
                        "matches_requested": 1,
                        "matches_completed": 1,
                        "top_hypotheses": rankings,
                    },
                    hypothesis_loader=lambda: {"h-1": rankings[0]},
                    explorer=lambda _hypothesis, _goal: "Exploration.",
                    synthesizer=lambda _goal, _results: "",
                )
            self.assertFalse((root / "rounds.jsonl").exists())
            self.assertFalse((root / "unverified-exploration-credit.jsonl").exists())

    def test_synthesis_receives_bounded_exploration_summaries(self) -> None:
        campaign = load("hypothesis_campaign_bounded_explorations", CAMPAIGN)
        observed = []
        rankings = [
            {"hypothesis_id": "h-1", "summary": "First", "elo_rating": 1210},
            {"hypothesis_id": "h-2", "summary": "Second", "elo_rating": 1200},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            campaign.run_round(
                campaign_dir=Path(temporary),
                workflow_id="campaign-v1",
                session_id="session-1",
                research_goal="Explain a GC3 gradient.",
                turn=1,
                tournament_runner=lambda _payload: {
                    "status": "completed",
                    "matches_requested": 1,
                    "matches_completed": 1,
                    "top_hypotheses": rankings,
                },
                hypothesis_loader=lambda: {item["hypothesis_id"]: item for item in rankings},
                explorer=lambda _hypothesis, _goal: "x" * 9000,
                synthesizer=lambda _goal, results: observed.extend(results) or "synthesis",
            )
        self.assertEqual(len(observed), 2)
        self.assertTrue(
            all(len(row["exploration_summary"]) == campaign.EXPLORATION_SYNTHESIS_LIMIT for row in observed)
        )

    def test_second_round_injects_prior_synthesis_and_keeps_auto_credit_unverified(self) -> None:
        campaign = load("hypothesis_campaign_test", CAMPAIGN)
        calls = []
        rankings = [
            {"hypothesis_id": "h-1", "summary": "First", "elo_rating": 1210},
            {"hypothesis_id": "h-2", "summary": "Second", "elo_rating": 1200},
            {"hypothesis_id": "h-3", "summary": "Third", "elo_rating": 1190},
        ]
        hypotheses = {
            item["hypothesis_id"]: {"hypothesis_id": item["hypothesis_id"], "content": item["summary"]}
            for item in rankings
        }

        def tournament(payload):
            calls.append(payload)
            return {
                "status": "completed",
                "matches_requested": 1,
                "matches_completed": 1,
                "top_hypotheses": rankings,
            }

        def explore(hypothesis, _goal):
            return f"exploration of {hypothesis['hypothesis_id']}"

        def synthesize(_goal, _results):
            return "Round synthesis: retain h-1, compare h-2 and h-3."

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "campaign_dir": root,
                "workflow_id": "campaign-v1",
                "session_id": "session-1",
                "research_goal": "Explain a GC3 gradient.",
                "tournament_runner": tournament,
                "hypothesis_loader": lambda: hypotheses,
                "explorer": explore,
                "synthesizer": synthesize,
            }
            first = campaign.run_round(turn=1, **common)
            second = campaign.run_round(turn=2, **common)
            credits = [json.loads(line) for line in (root / "unverified-exploration-credit.jsonl").read_text().splitlines()]
            rounds = [json.loads(line) for line in (root / "rounds.jsonl").read_text().splitlines()]

        self.assertEqual(first["explored_hypothesis_ids"], ["h-1", "h-2", "h-3"])
        self.assertEqual(second["prior_round_synthesis"], "Round synthesis: retain h-1, compare h-2 and h-3.")
        self.assertIn("Round synthesis: retain h-1", calls[1]["user_message"])
        self.assertEqual(len(credits), 6)
        self.assertTrue(all(row["event"] == "unverified_exploration_credit_assigned" for row in credits))
        self.assertTrue(all(row["eligible_for_evidence_rating"] is False for row in credits))
        self.assertTrue(all(row["epistemic_status"] == "unverified" for row in credits))
        self.assertEqual(len(rounds), 2)


if __name__ == "__main__":
    unittest.main()
