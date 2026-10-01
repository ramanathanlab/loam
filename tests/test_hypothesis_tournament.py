"""Focused tests for the opt-in Jnana pre-LLM tournament integration."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
JNANA_ROOT = ROOT.parent / "jnana"
HOOKS = ROOT / ".loam" / "hooks"
CONFIG = ROOT / ".loam" / "adapters" / "hermes" / "plugin" / "loam_config.py"


def _load(name: str, path: Path):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HypothesisTournamentTests(unittest.TestCase):
    def test_profile_parses_safe_tournament_payload(self) -> None:
        config = _load("loam_config_tournament", CONFIG)
        raw = {
            "enabled": True,
            "active_profile": "research",
            "profiles": {
                "research": {
                    "workflow_id": "research-v1",
                    "state_dir": "/tmp/state",
                    "pre_llm_every": 4,
                    "duplicate_policy": "allow_and_audit",
                    "hypothesis_tournament": {
                        "enabled": True,
                        "jnana_root": "/tmp/jnana",
                        "state_file": "/tmp/state/jnana.json",
                        "reports_dir": "/tmp/state/reports",
                        "initial_hypotheses": 3,
                        "inference_endpoint": "http://example.test/v1",
                        "model": "test-model",
                        "api_key_env": "TEST_JNANA_KEY",
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "loam.yaml"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(config_path)}, clear=False):
                profile = config.load_profile()

        assert profile is not None
        assert profile.pre_llm_every == 4
        assert profile.hypothesis_tournament is not None
        values = config.payload_values(profile)
        tournament = values["loam_hypothesis_tournament"]
        self.assertEqual(tournament["initial_hypotheses"], 3)
        self.assertEqual(tournament["matches_per_run"], 1)
        self.assertEqual(tournament["top_k"], 3)
        self.assertNotIn("api_key", tournament)

    def test_context_for_runs_runner_writes_report_and_renders_compact_ranking(self) -> None:
        hook = _load("loam_hypothesis_tournament", HOOKS / "hypothesis_tournament.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jnana_root = JNANA_ROOT
            state_file = root / "state" / "memory.json"
            reports_dir = root / "reports"
            payload = {
                "workflow_id": "research-v1",
                "loam_profile": "research",
                "session_id": "session-1",
                "loam_turn": 5,
                "loam_hypothesis_tournament": {
                    "enabled": True,
                    "jnana_root": str(jnana_root),
                    "state_file": str(state_file),
                    "reports_dir": str(reports_dir),
                    "matches_per_run": 2,
                    "top_k": 2,
                    "provider": "openai",
                    "inference_endpoint": "http://example.test/v1",
                    "model": "test-model",
                    "api_key_env": "TEST_JNANA_KEY",
                },
            }
            result = {
                "status": "completed",
                "matches_requested": 2,
                "matches_completed": 2,
                "top_hypotheses": [
                    {"hypothesis_id": "h-1", "summary": "First hypothesis", "elo_rating": 1212.25},
                    {"hypothesis_id": "h-2", "summary": "Second hypothesis", "elo_rating": 1187.75},
                ],
            }
            with patch.dict(os.environ, {"TEST_JNANA_KEY": "not-a-real-secret"}, clear=False):
                original_path = list(sys.path)
                sys.path.insert(0, str(jnana_root))
                try:
                    with patch(
                        "jnana.protognosis.periodic_tournament.run_incremental_tournament",
                        return_value=result,
                    ):
                        context = hook.context_for(payload)
                finally:
                    sys.path[:] = original_path

            self.assertIn("ran 2/2 pairwise matches", context)
            self.assertIn("h-1 — 1212.2 Elo", context)
            reports = list(reports_dir.glob("tournament-*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report["result"]["matches_completed"], 2)

    def test_context_for_generates_initial_hypotheses_before_running_tournament(self) -> None:
        hook = _load("loam_hypothesis_tournament_generation", HOOKS / "hypothesis_tournament.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "state" / "memory.json"
            reports_dir = root / "reports"
            payload = {
                "workflow_id": "research-v1",
                "loam_profile": "research",
                "session_id": "session-1",
                "loam_turn": 5,
                "user_message": "Explain why genomic GC content predicts codon usage.",
                "loam_hypothesis_tournament": {
                    "enabled": True,
                    "jnana_root": str(JNANA_ROOT),
                    "state_file": str(state_file),
                    "reports_dir": str(reports_dir),
                    "initial_hypotheses": 3,
                    "matches_per_run": 2,
                    "top_k": 2,
                    "provider": "openai",
                    "inference_endpoint": "http://example.test/v1",
                    "model": "test-model",
                    "json_max_tokens": 1200,
                },
            }
            generated = []
            construction = {}

            class FakeCoScientist:
                def __init__(self, **kwargs):
                    construction.update(kwargs)
                    self.memory = type("Memory", (), {"save": lambda _self: None})()

                def start(self):
                    pass

                def stop(self):
                    pass

                def get_all_hypotheses(self):
                    return generated

                def set_research_goal(self, goal):
                    self.goal = goal

                def generate_hypotheses(self, count):
                    generated.extend(range(count))

                def wait_for_completion(self):
                    pass

            result = {
                "status": "completed",
                "matches_requested": 2,
                "matches_completed": 2,
                "top_hypotheses": [
                    {"hypothesis_id": "h-1", "summary": "First hypothesis", "elo_rating": 1212.25},
                    {"hypothesis_id": "h-2", "summary": "Second hypothesis", "elo_rating": 1187.75},
                ],
            }
            with patch.dict(os.environ, {}, clear=True):
                original_path = list(sys.path)
                sys.path.insert(0, str(JNANA_ROOT))
                try:
                    with patch.object(hook, "CoScientist", FakeCoScientist), patch(
                        "jnana.protognosis.periodic_tournament.run_incremental_tournament",
                        return_value=result,
                    ) as tournament:
                        context = hook.context_for(payload)
                finally:
                    sys.path[:] = original_path

            self.assertEqual(len(generated), 3)
            self.assertEqual(construction["llm_config"].model_adapter["json_max_tokens"], 4096)
            tournament.assert_called_once()
            self.assertIn("ran 2/2 pairwise matches", context)
            report = json.loads(next(reports_dir.glob("tournament-*.json")).read_text())
            self.assertEqual(report["result"]["generation"]["generated_hypotheses"], 3)

    def test_pre_llm_hook_combines_memory_and_tournament_context(self) -> None:
        pre_llm = _load("loam_pre_llm_tournament", HOOKS / "pre-llm.py")
        payload = {"event": "pre_llm_call", "session_id": "session"}
        with patch.object(pre_llm, "memory_context_for", return_value="Validated memory"), patch.object(
            pre_llm, "tournament_context_for", return_value="Tournament ranking"
        ):
            with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
                stdin.read.return_value = json.dumps(payload)
                self.assertEqual(pre_llm.main(), 0)

        emitted = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertIn("Validated memory\\n\\nTournament ranking", emitted)

    def test_endpoint_normalization_and_no_auth_config(self) -> None:
        hook = _load("loam_hypothesis_tournament_endpoint", HOOKS / "hypothesis_tournament.py")
        self.assertEqual(
            hook._base_url(
                {"inference_endpoint": "http://example.test/v1/chat/completions", "endpoint_is_chat_completions": True}
            ),
            "http://example.test/v1",
        )
        payload = {
            "loam_hypothesis_tournament": {
                "enabled": True,
                "jnana_root": "/tmp",
                "state_file": "/tmp/memory.json",
                "reports_dir": "/tmp/reports",
                "provider": "openai",
                "inference_endpoint": "http://example.test/v1",
                "model": "test-model",
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(hook.context_for(payload), "")


if __name__ == "__main__":
    unittest.main()
