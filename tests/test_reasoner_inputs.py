"""The reasoner's input contract: Elo alongside evidence, and a brief with room.

`reasoner.md` asks to "assess their evidence and ELO signals together" and treats
a thread missing either score as incomplete metadata. Elo reached the validators
but never the reasoner, so every thread it saw was incomplete by its own rule.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
ORCHESTRATION = HOOKS / "hypothesis_orchestration.py"


def _load(name: str):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, ORCHESTRATION)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _report_payload(root: Path) -> dict:
    reports_dir = root / "reports"
    reports_dir.mkdir()
    (reports_dir / "tournament-001.json").write_text(
        json.dumps(
            {
                "workflow_id": "research-v1",
                "result": {
                    "status": "completed",
                    "top_hypotheses": [
                        {"hypothesis_id": "h-1", "elo_rating": 1210.0},
                        {"hypothesis_id": "h-2", "elo_rating": 1190.5},
                        {"hypothesis_id": "h-3"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "repo_root": str(ROOT),
        "workflow_id": "research-v1",
        "loam_state_dir": str(root / "loam-state"),
        "loam_hypothesis_tournament": {"state_file": str(root / "jnana.json"), "reports_dir": str(reports_dir)},
    }


class ReasonerInputTests(unittest.TestCase):
    def test_elo_is_read_from_the_tournament_report(self) -> None:
        orchestration = _load("loam_reasoner_elo_lookup")
        with tempfile.TemporaryDirectory() as temporary:
            ratings = orchestration._elo_ratings(_report_payload(Path(temporary)))

        self.assertEqual(ratings, {"h-1": 1210.0, "h-2": 1190.5})

    def test_elo_reaches_each_reasoner_thread(self) -> None:
        orchestration = _load("loam_reasoner_elo_threads")
        registry = orchestration.agent_registry({"repo_root": str(ROOT)})
        directive = orchestration._reasoner_directive(
            "plan-1",
            [
                {"hypothesis_id": "h-1", "status": "supported", "evidence_score": 0.5, "summary": "A."},
                {"hypothesis_id": "h-2", "status": "contradicted", "evidence_score": -0.5, "summary": "B."},
            ],
            registry,
            None,
            {"h-1": 1210.0, "h-2": 1190.5},
        )

        threads = json.loads(directive.rsplit("\n", 1)[1])
        self.assertEqual([thread["elo_rating"] for thread in threads], [1210.0, 1190.5])

    def test_a_thread_without_a_rating_keeps_the_key_as_null(self) -> None:
        """Absent must be explicit: the definition's rule is to flag it, not infer it."""

        orchestration = _load("loam_reasoner_elo_missing")
        registry = orchestration.agent_registry({"repo_root": str(ROOT)})
        directive = orchestration._reasoner_directive(
            "plan-1",
            [{"hypothesis_id": "h-9", "status": "inconclusive", "evidence_score": 0.0, "summary": "C."}],
            registry,
            None,
            {},
        )

        threads = json.loads(directive.rsplit("\n", 1)[1])
        self.assertIn("elo_rating", threads[0])
        self.assertIsNone(threads[0]["elo_rating"])

    def test_the_declared_model_tier_is_stated_to_the_dispatcher(self) -> None:
        orchestration = _load("loam_reasoner_model")
        registry = orchestration.agent_registry({"repo_root": str(ROOT)})
        directive = orchestration._reasoner_directive("plan-1", [], registry, None, {})

        self.assertIn(registry["reasoner"].model, directive)

    def test_the_brief_ceiling_leaves_room_for_a_turn_of_synthesis(self) -> None:
        orchestration = _load("loam_reasoner_brief_bound")
        rows = [{"event": "reasoner_completed", "workflow_id": "research-v1", "summary": "s" * 20_000}]
        brief = orchestration._latest_brief(rows, "research-v1")

        self.assertEqual(len(brief), orchestration.MAX_BRIEF_CHARS)
        self.assertGreaterEqual(orchestration.MAX_BRIEF_CHARS, 8_000)


if __name__ == "__main__":
    unittest.main()
