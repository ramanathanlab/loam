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
# Jnana is a sibling checkout (acadev/Jnana), not a dependency of this package,
# so the tests that actually drive it are opt-in: without it they would fail at
# `patch("jnana.protognosis...")` rather than report anything useful. Its LLM
# factory additionally imports `openai`, which the no-auth local endpoints this
# repo talks to never need.
HAVE_JNANA = JNANA_ROOT.is_dir()
HAVE_JNANA_LLM = HAVE_JNANA and importlib.util.find_spec("openai") is not None
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


GOAL = "---\nname: fixture-goal\n---\n\nDesign de novo binders against a fixture target.\n"


class _FakeCoScientist:
    """The Jnana surface the hook touches: start/stop, hypotheses, memory."""

    constructed: dict = {}

    def __init__(self, **kwargs):
        type(self).constructed.update(kwargs)
        self.memory = type("Memory", (), {"save": lambda _self: None})()

    def start(self):
        pass

    def stop(self):
        pass

    def get_all_hypotheses(self):
        return []

    def set_research_goal(self, goal):
        pass

    def generate_hypotheses(self, count):
        pass

    def wait_for_completion(self):
        pass


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
                    "hypothesis_orchestration": {"enabled": True},
                    "execution": {
                        "mode": "remote_mcp",
                        "tool_name": "globus_execute",
                        "endpoint": "https://compute.example/mcp",
                        "function_name": "run_python",
                        "input_schema": "run-python-v1",
                        "runtime": "python3.11",
                    },
                    "hypothesis_tournament": {
                        "enabled": True,
                        "jnana_root": "/tmp/jnana",
                        "state_file": "/tmp/static-state/jnana.json",
                        "reports_dir": "/tmp/static-state/reports",
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
        # Absent means off, and the hook must still see the key: a setting the
        # adapter drops is a setting that fails silently.
        self.assertEqual(tournament["refresh_hypotheses"], 0)
        self.assertTrue(values["loam_hypothesis_orchestration"])
        self.assertEqual(values["loam_state_dir"], "/tmp/state")
        self.assertEqual(values["loam_execution"]["mode"], "remote_mcp")
        self.assertEqual(values["loam_execution"]["tool_name"], "globus_execute")

    def test_refresh_hypotheses_reaches_the_hook_and_zero_stays_zero(self) -> None:
        """Zero is the feature off, not a value to clamp up to one."""

        config = _load("loam_config_refresh", CONFIG)
        base = {
            "workflow_id": "research-v1",
            "state_dir": "/tmp/state",
            "hypothesis_tournament": {
                "enabled": True,
                "jnana_root": "/tmp/jnana",
                "state_file": "/tmp/state/memory.json",
                "reports_dir": "/tmp/state/reports",
                "inference_endpoint": "http://example.test/v1",
                "model": "test-model",
            },
        }
        parsed = {}
        with tempfile.TemporaryDirectory() as tmp:
            for label, value in (("set", 25), ("zero", 0), ("junk", "many"), ("negative", -5)):
                raw = {
                    "enabled": True,
                    "active_profile": "research",
                    "profiles": {"research": {**base, "hypothesis_tournament": {
                        **base["hypothesis_tournament"], "refresh_hypotheses": value}}},
                }
                config_path = Path(tmp) / f"loam-{label}.yaml"
                config_path.write_text(json.dumps(raw), encoding="utf-8")
                with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(config_path)}, clear=False):
                    profile = config.load_profile()
                assert profile is not None
                parsed[label] = config.payload_values(profile)["loam_hypothesis_tournament"][
                    "refresh_hypotheses"
                ]

        self.assertEqual(parsed, {"set": 25, "zero": 0, "junk": 0, "negative": 0})

    def test_profile_parses_local_execution_mode(self) -> None:
        config = _load("loam_config_local_execution", CONFIG)
        raw = {
            "enabled": True,
            "active_profile": "research",
            "profiles": {
                "research": {
                    "workflow_id": "research-v1",
                    "state_dir": "/tmp/state",
                    "hypothesis_orchestration": {"enabled": True},
                    "execution": {"mode": "local", "runtime": "python3.11"},
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "loam.yaml"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(config_path)}, clear=False):
                profile = config.load_profile()

        assert profile is not None
        self.assertEqual(config.payload_values(profile)["loam_execution"], {"mode": "local", "tool_name": "", "endpoint": "", "function_name": "", "input_schema": "", "runtime": "python3.11", "setup": "", "code_source": "generated", "max_children": 3, "workdir_root": "", "tool_prefix": ""})

    def test_execution_parses_generated_code_source(self) -> None:
        config = _load("loam_config_generated_execution", CONFIG)
        remote = {"mode": "remote_mcp", "tool_name": "mcp_hpc_run_code", "endpoint": "http://x/mcp",
                  "function_name": "run_code", "input_schema": "v1"}
        generated = config._execution_config(
            {**remote, "code_source": "generated", "workdir_root": " /lus/w ", "tool_prefix": "mcp_hpc_"})
        self.assertEqual((generated.code_source, generated.max_children, generated.workdir_root,
                          generated.tool_prefix), ("generated", 3, "/lus/w", "mcp_hpc_"))
        self.assertEqual(config._execution_config({**remote, "code_source": "generated",
                                                   "max_children": 5}).max_children, 5)
        self.assertEqual(config._execution_config(remote).code_source, "generated")
        self.assertEqual(config._execution_config(remote).max_children, 3)
        fixed = config._execution_config({**remote, "code_source": "fixed"})
        self.assertEqual((fixed.code_source, fixed.max_children), ("fixed", 1))
        # A typo in either field must fail loudly rather than discard the whole
        # execution block: a dropped block degrades a configured HPC campaign to
        # the local arithmetic self-test, or makes load_profile reject the
        # profile outright, with no error anywhere.
        with self.assertRaisesRegex(ValueError, "code_source"):
            config._execution_config({**remote, "code_source": "llm"})
        with self.assertRaisesRegex(ValueError, "mode"):
            config._execution_config({**remote, "mode": "remote-mcp"})
        with self.assertRaisesRegex(ValueError, "mode"):
            config._execution_config({key: value for key, value in remote.items() if key != "mode"})
        # An absent execution block still means "not configured", not an error.
        self.assertIsNone(config._execution_config(None))

    def test_runtime_state_directory_overrides_profile_and_enables_emergency_mode(self) -> None:
        config = _load("loam_config_runtime_state", CONFIG)
        raw = {
            "enabled": True,
            "active_profile": "research",
            "profiles": {
                "research": {
                    "workflow_id": "research-v1",
                    "hypothesis_tournament": {
                        "enabled": True,
                        "jnana_root": "/tmp/jnana",
                        "inference_endpoint": "http://example.test/v1",
                        "model": "test-model",
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "loam.yaml"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            runtime_state = Path(tmp) / "run-state"
            with patch.dict(
                os.environ,
                {"LOAM_CONFIG_PATH": str(config_path), "LOAM_STATE_DIR": str(runtime_state)},
                clear=False,
            ):
                profile = config.effective_profile()

        assert profile is not None
        self.assertEqual(profile.state_dir, runtime_state)
        assert profile.hypothesis_tournament is not None
        self.assertEqual(profile.hypothesis_tournament.state_file, runtime_state / "jnana-memory.json")
        self.assertEqual(profile.hypothesis_tournament.reports_dir, runtime_state / "tournament-reports")

    @unittest.skipUnless(HAVE_JNANA_LLM, "jnana checkout and openai required")
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

    @unittest.skipUnless(HAVE_JNANA, "jnana checkout required")
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

    @unittest.skipUnless(HAVE_JNANA, "jnana checkout required")
    def test_existing_hypotheses_do_not_require_current_user_message(self) -> None:
        """Later pre-LLM turns rank persistent state even without a new prompt."""

        hook = _load("loam_hypothesis_tournament_existing", HOOKS / "hypothesis_tournament.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = [object(), object(), object()]
            construction = {}
            payload = {
                "workflow_id": "research-v1",
                "loam_profile": "research",
                "session_id": "session-1",
                "loam_turn": 6,
                "loam_hypothesis_tournament": {
                    "enabled": True,
                    "jnana_root": str(JNANA_ROOT),
                    "state_file": str(root / "state" / "memory.json"),
                    "reports_dir": str(root / "reports"),
                    "matches_per_run": 1,
                    "top_k": 2,
                    "provider": "openai",
                    "inference_endpoint": "http://example.test/v1",
                    "model": "test-model",
                },
            }

            class FakeCoScientist:
                def __init__(self, **kwargs):
                    construction.update(kwargs)
                    self.memory = type("Memory", (), {"save": lambda _self: None})()

                def start(self):
                    pass

                def stop(self):
                    pass

                def get_all_hypotheses(self):
                    return existing

            result = {
                "status": "completed",
                "matches_requested": 1,
                "matches_completed": 1,
                "top_hypotheses": [{"hypothesis_id": "h-1", "summary": "First", "elo_rating": 1208}],
            }
            with patch.dict(os.environ, {}, clear=True):
                original_path = list(sys.path)
                sys.path.insert(0, str(JNANA_ROOT))
                try:
                    with patch.object(hook, "CoScientist", FakeCoScientist), patch(
                        "jnana.protognosis.periodic_tournament.run_incremental_tournament", return_value=result
                    ) as tournament:
                        context = hook.context_for(payload)
                finally:
                    sys.path[:] = original_path

            self.assertEqual(construction["llm_config"].model_adapter["json_max_tokens"], 4096)
            tournament.assert_called_once()
            self.assertIn("ran 1/1 pairwise matches", context)
            report = json.loads(next((root / "reports").glob("tournament-*.json")).read_text())
            self.assertEqual(report["result"]["generation"], {"status": "not_needed", "generated_hypotheses": 0})

    @unittest.skipUnless(HAVE_JNANA, "jnana checkout required")
    def test_generation_aims_at_the_campaign_goal_not_the_turns_message(self) -> None:
        """The whole point: hypotheses are about the campaign, not about the prompt.

        A campaign's turn message is a round nudge -- "begin the campaign" --
        and generating against it produced a pool of hypotheses about how to run
        research, none of them about the target.
        """

        hook = _load("loam_hypothesis_tournament_campaign_goal", HOOKS / "hypothesis_tournament.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._campaign_payload(root, turn=1, user_message="Begin the campaign.")
            generated: list[object] = []
            goals: list[str] = []

            class FakeCoScientist(_FakeCoScientist):
                def set_research_goal(self, goal):
                    goals.append(goal)

                def get_all_hypotheses(self):
                    return generated

                def generate_hypotheses(self, count):
                    generated.extend(range(count))

            context = self._run(hook, FakeCoScientist, payload)

            self.assertEqual(len(generated), 3)
            self.assertIn("ran 2/2 pairwise matches", context)
            self.assertEqual(len(goals), 1)
            self.assertIn("binders against a fixture target", goals[0])
            self.assertNotIn("Begin the campaign", goals[0])
            report = json.loads(next((root / "reports").glob("tournament-*.json")).read_text())
            self.assertEqual(report["result"]["generation"]["goal_source"], "campaign:fixture-goal")

    @unittest.skipUnless(HAVE_JNANA, "jnana checkout required")
    def test_a_populated_state_is_re_goaled_only_when_the_campaign_file_changes(self) -> None:
        """Generation is one-shot; the goal the pairwise judge reads is not.

        Re-setting it costs an LLM call to re-parse the plan, so it happens on a
        changed goal file and not on every turn.
        """

        hook = _load("loam_hypothesis_tournament_regoal", HOOKS / "hypothesis_tournament.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._campaign_payload(root, turn=7, user_message="Dispatch the next round.")
            goals: list[str] = []
            metadata: dict[str, object] = {}

            class FakeCoScientist(_FakeCoScientist):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.memory.metadata = metadata

                def set_research_goal(self, goal):
                    goals.append(goal)
                    metadata["research_goal"] = goal

                def get_all_hypotheses(self):
                    return [object(), object(), object()]

            self._run(hook, FakeCoScientist, payload)
            first = json.loads(self._latest_report(root).read_text())["result"]["generation"]

            self._run(hook, FakeCoScientist, payload)
            second = json.loads(self._latest_report(root).read_text())["result"]["generation"]

            (root / "campaigns" / "fixture-goal.md").write_text(
                GOAL.replace("a fixture target", "a different fixture target"), encoding="utf-8"
            )
            self._run(hook, FakeCoScientist, payload)
            third = json.loads(self._latest_report(root).read_text())["result"]["generation"]

        self.assertEqual(first["goal_sync"], "set")
        self.assertEqual(second["goal_sync"], "unchanged")
        self.assertEqual(third["goal_sync"], "set")
        self.assertEqual(len(goals), 2)
        self.assertIn("a different fixture target", goals[1])

    @unittest.skipUnless(HAVE_JNANA, "jnana checkout required")
    def test_a_fresh_reasoner_brief_enters_the_pool_as_bounded_new_hypotheses(self) -> None:
        """The reasoner->tournament half of the loop.

        The brief reaches the pairwise judge through the stored goal, and buys
        `refresh_hypotheses` candidates that compete for Elo -- once per brief,
        not once per turn.
        """

        hook = _load("loam_hypothesis_tournament_refresh", HOOKS / "hypothesis_tournament.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._campaign_payload(root, turn=9, user_message="Dispatch the next round.")
            payload["loam_hypothesis_tournament"]["refresh_hypotheses"] = 4
            payload["loam_reasoner_brief"] = "Round 8: the 12-residue binders all fell apart in MD."
            pool = [object(), object(), object()]
            goals: list[str] = []
            metadata: dict[str, object] = {}
            requests: list[int] = []

            class FakeCoScientist(_FakeCoScientist):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.memory.metadata = metadata

                def set_research_goal(self, goal):
                    goals.append(goal)
                    metadata["research_goal"] = goal

                def get_all_hypotheses(self):
                    return pool

                def generate_hypotheses(self, count):
                    requests.append(count)
                    pool.extend(object() for _ in range(count))

            context = self._run(hook, FakeCoScientist, payload)
            entered = json.loads(self._latest_report(root).read_text())["result"]["generation"]

            # Same brief next turn: nothing new to aim at, so nothing generated.
            self._run(hook, FakeCoScientist, payload)
            quiet = json.loads(self._latest_report(root).read_text())["result"]["generation"]

            payload["loam_reasoner_brief"] = "Round 9: the 24-residue scaffold holds."
            self._run(hook, FakeCoScientist, payload)
            next_round = json.loads(self._latest_report(root).read_text())["result"]["generation"]

        self.assertIn("12-residue binders all fell apart", goals[0])
        self.assertIn("binders against a fixture target", goals[0])
        self.assertEqual(entered["status"], "refreshed")
        self.assertEqual(entered["generated_hypotheses"], 4)
        self.assertEqual(quiet, {"status": "not_needed", "generated_hypotheses": 0,
                                 "goal_source": "campaign:fixture-goal", "goal_sync": "unchanged"})
        self.assertEqual(next_round["generated_hypotheses"], 4)
        self.assertEqual(requests, [4, 4])
        self.assertIn("4 new hypotheses entered", context)
        self.assertIn("unranked until they have played matches", context)

    @unittest.skipUnless(HAVE_JNANA, "jnana checkout required")
    def test_a_brief_already_carried_by_the_turns_message_is_not_added_twice(self) -> None:
        """`prepare_payload` appends it there too, and that message is the goal
        for a profile that names no campaign."""

        hook = _load("loam_hypothesis_tournament_dedupe", HOOKS / "hypothesis_tournament.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brief = "Round 8: the 12-residue binders all fell apart in MD."
            payload = self._campaign_payload(root, turn=9, user_message=f"Dispatch the next round.\n\n{brief}")
            payload.pop("loam_campaign")
            payload["loam_reasoner_brief"] = brief
            goals: list[str] = []
            metadata: dict[str, object] = {}

            class FakeCoScientist(_FakeCoScientist):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.memory.metadata = metadata

                def set_research_goal(self, goal):
                    goals.append(goal)
                    metadata["research_goal"] = goal

                def get_all_hypotheses(self):
                    return [object(), object(), object()]

            self._run(hook, FakeCoScientist, payload)
            generation = json.loads(self._latest_report(root).read_text())["result"]["generation"]

        self.assertEqual(generation["goal_source"], "user_message")
        self.assertEqual(goals[0].count("12-residue binders all fell apart"), 1)

    @unittest.skipUnless(HAVE_JNANA, "jnana checkout required")
    def test_without_refresh_configured_a_brief_only_re_goals(self) -> None:
        """Opt-in: the default adds no hypotheses and no generation cost."""

        hook = _load("loam_hypothesis_tournament_norefresh", HOOKS / "hypothesis_tournament.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._campaign_payload(root, turn=9, user_message="Dispatch the next round.")
            payload["loam_reasoner_brief"] = "Round 8: the 12-residue binders all fell apart in MD."
            requests: list[int] = []
            metadata: dict[str, object] = {}

            class FakeCoScientist(_FakeCoScientist):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.memory.metadata = metadata

                def set_research_goal(self, goal):
                    metadata["research_goal"] = goal

                def get_all_hypotheses(self):
                    return [object(), object(), object()]

                def generate_hypotheses(self, count):
                    requests.append(count)

            self._run(hook, FakeCoScientist, payload)
            generation = json.loads(self._latest_report(root).read_text())["result"]["generation"]

        self.assertEqual(generation["status"], "not_needed")
        self.assertEqual(generation["goal_sync"], "set")
        self.assertEqual(requests, [])
        self.assertIn("12-residue binders all fell apart", str(metadata["research_goal"]))

    def _campaign_payload(self, root: Path, *, turn: int, user_message: str) -> dict:
        """A tournament payload whose campaign goal lives in a throwaway directory."""

        campaigns = root / "campaigns"
        campaigns.mkdir(parents=True, exist_ok=True)
        (campaigns / "fixture-goal.md").write_text(GOAL, encoding="utf-8")
        return {
            "workflow_id": "research-v1",
            "loam_profile": "research",
            "session_id": "session-1",
            "loam_turn": turn,
            "user_message": user_message,
            # The throwaway root is also repo_root, so the repo's own invariant
            # rules are out of the picture: this is about the goal.
            "repo_root": str(root),
            "loam_campaign": {"goal": "fixture-goal", "rules": [], "dir": str(root), "skills": []},
            "loam_hypothesis_tournament": {
                "enabled": True,
                "jnana_root": str(JNANA_ROOT),
                "state_file": str(root / "state" / "memory.json"),
                "reports_dir": str(root / "reports"),
                "initial_hypotheses": 3,
                "matches_per_run": 2,
                "top_k": 2,
                "provider": "openai",
                "inference_endpoint": "http://example.test/v1",
                "model": "test-model",
            },
        }

    def _run(self, hook, coscientist, payload: dict) -> str:
        result = {
            "status": "completed",
            "matches_requested": 2,
            "matches_completed": 2,
            "top_hypotheses": [{"hypothesis_id": "h-1", "summary": "First", "elo_rating": 1208}],
        }
        with patch.dict(os.environ, {}, clear=True):
            original_path = list(sys.path)
            sys.path.insert(0, str(JNANA_ROOT))
            try:
                with patch.object(hook, "CoScientist", coscientist), patch(
                    "jnana.protognosis.periodic_tournament.run_incremental_tournament", return_value=result
                ):
                    return hook.context_for(payload)
            finally:
                sys.path[:] = original_path

    def _latest_report(self, root: Path) -> Path:
        return max((root / "reports").glob("tournament-*.json"), key=lambda path: path.name)

    def test_pre_llm_hook_skips_tournament_for_campaign_children(self) -> None:
        pre_llm = _load("loam_pre_llm_campaign_child", HOOKS / "pre-llm.py")
        payload = {
            "event": "pre_llm_call",
            "session_id": "validator-session",
            "loam_hypothesis_orchestration": True,
        }
        with patch.object(pre_llm, "ensure_run_manifest"), patch.object(
            pre_llm, "memory_context_for", return_value="Validated memory"
        ), patch.object(pre_llm, "is_campaign_child", return_value=True), patch.object(
            pre_llm, "tournament_context_for"
        ) as tournament:
            with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
                stdin.read.return_value = json.dumps(payload)
                self.assertEqual(pre_llm.main(), 0)

        tournament.assert_not_called()
        emitted = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
        self.assertEqual(emitted["context"], "Validated memory")

    def test_pre_llm_hook_combines_memory_and_tournament_context(self) -> None:
        pre_llm = _load("loam_pre_llm_tournament", HOOKS / "pre-llm.py")
        payload = {"event": "pre_llm_call", "session_id": "session", "loam_hypothesis_orchestration": True}
        with patch.object(pre_llm, "ensure_run_manifest"), patch.object(
            pre_llm, "memory_context_for", return_value="Validated memory"
        ), patch.object(pre_llm, "tournament_context_for", return_value="Tournament ranking"), patch.object(
            pre_llm, "is_campaign_child", return_value=False
        ):
            with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
                stdin.read.return_value = json.dumps(payload)
                self.assertEqual(pre_llm.main(), 0)

        emitted = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
        self.assertEqual(emitted["context"], "Validated memory\n\nTournament ranking")

    @unittest.skipUnless(HAVE_JNANA_LLM, "jnana checkout and openai required")
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

    def test_role_blocks_override_only_the_keys_they_set(self) -> None:
        """A `generation:`/`ranking:` block retargets one role; the rest inherits."""

        config = _load("loam_config_roles", CONFIG)
        raw = {
            "enabled": True,
            "jnana_root": "/tmp/jnana",
            "state_file": "/tmp/state/jnana.json",
            "reports_dir": "/tmp/state/reports",
            "provider": "openai",
            "inference_endpoint": "http://judge.test/v1",
            "model": "judge-model",
            "json_max_tokens": 16384,
            "generation": {
                "model": "generator-model",
                "inference_endpoint": "http://generator.test/v1",
            },
            "ranking": {"model": "   "},
        }
        parsed = config._tournament_config(raw)
        assert parsed is not None
        # A block of only blank values contributes nothing.
        self.assertEqual(
            parsed.roles,
            {"generation": {"model": "generator-model", "inference_endpoint": "http://generator.test/v1"}},
        )

        hook = _load("loam_hypothesis_tournament_roles", HOOKS / "hypothesis_tournament.py")
        payload = {"loam_hypothesis_tournament": {**raw, "generation": parsed.roles["generation"]}}
        resolved = hook._config(payload)
        assert resolved is not None

        generation = hook._role_config(resolved, "generation")
        self.assertEqual(generation["model"], "generator-model")
        self.assertEqual(hook._base_url(generation), "http://generator.test/v1")
        # Unset keys still come from the shared defaults.
        self.assertEqual(generation["provider"], "openai")
        self.assertEqual(generation["json_max_tokens"], 16384)

        # No usable override for this role: the judge keeps the shared values.
        ranking = hook._role_config(resolved, "ranking")
        self.assertEqual(ranking["model"], "judge-model")
        self.assertEqual(hook._base_url(ranking), "http://judge.test/v1")

    def test_role_overrides_reach_the_hook_payload(self) -> None:
        config = _load("loam_config_role_payload", CONFIG)
        raw = {
            "enabled": True,
            "active_profile": "research",
            "profiles": {
                "research": {
                    "workflow_id": "research-v1",
                    "state_dir": "/tmp/state",
                    "hypothesis_tournament": {
                        "enabled": True,
                        "jnana_root": "/tmp/jnana",
                        "state_file": "/tmp/state/jnana.json",
                        "reports_dir": "/tmp/state/reports",
                        "provider": "openai",
                        "inference_endpoint": "http://judge.test/v1",
                        "model": "judge-model",
                        "generation": {"model": "generator-model"},
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
        values = config.payload_values(profile)
        self.assertEqual(values["loam_hypothesis_tournament"]["model"], "judge-model")
        self.assertEqual(values["loam_hypothesis_tournament"]["generation"], {"model": "generator-model"})
        self.assertNotIn("ranking", values["loam_hypothesis_tournament"])


if __name__ == "__main__":
    unittest.main()
