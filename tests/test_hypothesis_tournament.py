"""Focused tests for the opt-in Jnana pre-LLM tournament integration."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
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
        self.assertEqual(config.payload_values(profile)["loam_execution"], {"mode": "local", "tool_name": "", "endpoint": "", "function_name": "", "input_schema": "", "runtime": "python3.11", "setup": "", "workdir_root": "", "tool_prefix": ""})

    def test_execution_parses_workdir_root_and_tool_prefix(self) -> None:
        config = _load("loam_config_generated_execution", CONFIG)
        remote = {"mode": "remote_mcp", "tool_name": "mcp_hpc_run_code", "endpoint": "http://x/mcp",
                  "function_name": "run_code", "input_schema": "v1"}
        parsed = config._execution_config({**remote, "workdir_root": " /lus/w ", "tool_prefix": "mcp_hpc_"})
        self.assertEqual((parsed.workdir_root, parsed.tool_prefix), ("/lus/w", "mcp_hpc_"))
        # A bad mode must fail loudly rather than discard the whole execution block.
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

    def test_pre_llm_hook_skips_tournament_for_campaign_children(self) -> None:
        pre_llm = _load("loam_pre_llm_campaign_child", HOOKS / "pre-llm.py")
        payload = {
            "event": "pre_llm_call",
            "session_id": "validator-session",
            "loam_hypothesis_orchestration": True,
        }
        with patch.object(pre_llm, "ensure_run_manifest"), patch.object(
            pre_llm.Parser, "memory_context", return_value="Validated memory"
        ), patch.object(pre_llm, "is_campaign_child", return_value=True), patch.object(
            pre_llm.Tournament, "context"
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
            pre_llm.Parser, "memory_context", return_value="Validated memory"
        ), patch.object(pre_llm.Tournament, "context", return_value="Tournament ranking"), patch.object(
            pre_llm, "is_campaign_child", return_value=False
        ):
            with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
                stdin.read.return_value = json.dumps(payload)
                self.assertEqual(pre_llm.main(), 0)

        emitted = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
        self.assertEqual(emitted["context"], "Validated memory\n\nTournament ranking")

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


def _fake_jnana(yields: bool = True):
    """Stand-ins for the Jnana modules the hook imports, sharing one persisted state."""

    state = SimpleNamespace(
        hypotheses=[], metadata={}, goals=[], generated=[], yields=yields, wait_timeouts=[]
    )

    class CoScientist:
        def __init__(self, **kwargs):
            self.research_goal = None
            self.memory = SimpleNamespace(metadata=state.metadata, save=lambda: None)

        def start(self):
            pass

        def stop(self):
            pass

        def get_all_hypotheses(self):
            return state.hypotheses

        def set_research_goal(self, goal):
            self.research_goal = goal
            state.metadata["research_goal"] = goal
            state.goals.append(goal)

        def generate_hypotheses(self, count):
            # Jnana generates against the in-memory goal, which a load never restores.
            assert self.research_goal, "generated without setting the goal this turn"
            state.generated.append(count)
            if state.yields:
                state.hypotheses.extend(range(len(state.hypotheses), len(state.hypotheses) + count))

        def wait_for_completion(self, timeout=None):
            # The real CoScientist.start() runs the whole agent ensemble, and
            # those agents enqueue their own follow-up work, so an unbounded
            # wait here does not return. Every caller in the hook must pass a
            # budget; recorded so a test can assert that it did.
            state.wait_timeouts.append(timeout)
            return True

    ranking = {
        "status": "completed",
        "matches_requested": 2,
        "matches_completed": 2,
        "top_hypotheses": [{"hypothesis_id": "h-1", "summary": "First", "elo_rating": 1208}],
    }
    modules = {name: ModuleType(name) for name in (
        "jnana",
        "jnana.protognosis",
        "jnana.protognosis.core",
        "jnana.protognosis.core.coscientist",
        "jnana.protognosis.core.multi_llm_config",
        "jnana.protognosis.periodic_tournament",
    )}
    modules["jnana.protognosis.core.coscientist"].CoScientist = CoScientist
    modules["jnana.protognosis.core.multi_llm_config"].LLMConfig = SimpleNamespace
    modules["jnana.protognosis.periodic_tournament"].run_incremental_tournament = lambda **_: ranking
    return state, modules


class TournamentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hook = _load("loam_hypothesis_tournament", HOOKS / "hypothesis_tournament.py")
        self.root = Path(tempfile.mkdtemp())
        (self.root / "campaigns").mkdir()
        (self.root / "campaigns" / "fixture-goal.md").write_text(GOAL, encoding="utf-8")

    def _payload(self, *, campaign: bool = True, **extra) -> dict:
        payload = {
            "workflow_id": "research-v1",
            "session_id": "session-1",
            "loam_turn": 1,
            "user_message": "dispatch the next round",
            "repo_root": str(self.root),
            "loam_hypothesis_tournament": {
                "enabled": True,
                "jnana_root": str(self.root / "jnana"),
                "state_file": str(self.root / "state" / "memory.json"),
                "reports_dir": str(self.root / "reports"),
                "initial_hypotheses": 3,
                "refresh_hypotheses": 4,
                "provider": "openai",
                "inference_endpoint": "http://judge.test/v1",
                "model": "judge-model",
            },
        }
        if campaign:
            payload["loam_campaign"] = {"goal": "fixture-goal", "rules": [], "dir": str(self.root), "skills": []}
        payload.update(extra)
        return payload

    def _run(self, modules, payload: dict) -> str:
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True):
            return self.hook.Tournament(payload).context()

    def test_disabled_returns_nothing(self) -> None:
        self.assertEqual(self.hook.Tournament({"loam_hypothesis_tournament": {"enabled": False}}).context(), "")

    def test_first_turn_seeds_against_the_campaign_goal_and_reports(self) -> None:
        state, modules = _fake_jnana()
        context = self._run(modules, self._payload())

        self.assertEqual(state.generated, [3])
        self.assertEqual(state.goals, ["Design de novo binders against a fixture target."])
        self.assertIn("#1: h-1 — 1208.0 Elo; First", context)
        report = json.loads(next((self.root / "reports").glob("tournament-*.json")).read_text())
        self.assertEqual(report["result"]["generation"]["goal_source"], "campaign:fixture-goal")

    def test_an_unchanged_goal_costs_nothing_on_later_turns(self) -> None:
        state, modules = _fake_jnana()
        self._run(modules, self._payload())
        self._run(modules, self._payload(loam_turn=2))

        self.assertEqual(state.generated, [3])
        self.assertEqual(len(state.goals), 1)

    def test_a_new_reasoner_brief_re_goals_and_refreshes_the_pool(self) -> None:
        state, modules = _fake_jnana()
        self._run(modules, self._payload())
        context = self._run(modules, self._payload(loam_turn=2, loam_reasoner_brief="Hotspot K745 matters."))

        self.assertEqual(state.generated, [3, 4])
        self.assertIn(self.hook.Tournament.brief_heading + "\nHotspot K745 matters.", state.goals[-1])
        self.assertIn("4 new hypotheses entered", context)

    def test_a_refresh_that_generated_nothing_is_retried(self) -> None:
        state, modules = _fake_jnana()
        self._run(modules, self._payload())
        state.yields = False
        brief = {"loam_reasoner_brief": "Hotspot K745 matters."}
        self._run(modules, self._payload(loam_turn=2, **brief))
        self._run(modules, self._payload(loam_turn=3, **brief))

        self.assertEqual(state.generated, [3, 4, 4])

    def test_a_turn_message_goal_is_never_re_goaled(self) -> None:
        state, modules = _fake_jnana()
        self._run(modules, self._payload(campaign=False, user_message="Why does X bind Y?"))
        self._run(modules, self._payload(campaign=False, user_message="Something else entirely"))

        self.assertEqual(state.goals, ["Why does X bind Y?"])
        self.assertEqual(state.generated, [3])

    def test_the_knowledge_brief_is_seeded_and_carried_into_a_re_goal(self) -> None:
        state, modules = _fake_jnana()
        payload = self._payload()
        payload["loam_hypothesis_tournament"]["knowledge"] = {"enabled": True, "model": "kg-model"}
        with patch("knowledge_brief.brief_for", return_value={"status": "ok", "brief": "Background: KG facts"}) as brief_for:
            self._run(modules, payload)
        self._run(modules, {**payload, "loam_turn": 2, "loam_reasoner_brief": "New synthesis."})

        self.assertEqual(brief_for.call_count, 1)
        self.assertEqual(brief_for.call_args.args[2]["model"], "kg-model")
        self.assertIn("Background: KG facts", state.goals[0])
        self.assertIn("Background: KG facts", state.goals[-1])

    def test_role_blocks_override_only_the_keys_they_set(self) -> None:
        _, modules = _fake_jnana()
        payload = self._payload()
        payload["loam_hypothesis_tournament"]["generation"] = {
            "model": "generator-model",
            "inference_endpoint": "http://generator.test/v1/chat/completions",
            "endpoint_is_chat_completions": True,
        }
        payload["loam_hypothesis_tournament"]["ranking"] = {"model": "   "}
        tournament = self.hook.Tournament(payload)
        with patch.dict(sys.modules, modules):
            generation = tournament._llm("generation", 4096)
            ranking = tournament._llm("ranking", 1)

        self.assertEqual((generation.model, generation.base_url), ("generator-model", "http://generator.test/v1"))
        self.assertEqual(generation.model_adapter["json_max_tokens"], 4096)
        self.assertEqual((ranking.model, ranking.base_url, ranking.api_key), ("judge-model", "http://judge.test/v1", "local-no-auth"))


if __name__ == "__main__":
    unittest.main()
