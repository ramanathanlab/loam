"""Knowledge-graph pre-pass: caching, fallback, and how the tournament uses it."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
JNANA_ROOT = Path(os.environ.get("JNANA_ROOT", ROOT.parent / "Jnana"))
sys.path.insert(0, str(HOOKS))
sys.path.insert(0, str(ROOT / ".loam" / "adapters" / "hermes" / "plugin"))
import knowledge_brief  # noqa: E402
import loam_config  # noqa: E402

LLM = {"base_url": "http://example.test/v1", "api_key": "k", "model": "m"}
ON = {"enabled": True, "name": "OKN", "mcp_url": "https://kg.test/mcp", "max_tool_calls": 3,
      "timeout_seconds": 10, "max_brief_chars": 600}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BriefTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def test_ok_brief_is_truncated_headed_and_cached(self) -> None:
        calls = []

        async def research(goal, url, allowed, llm, max_calls, seeds=()):
            calls.append((goal, url, sorted(allowed)[:2], max_calls))
            return "X" * 1000, 2, ["list_kgs", "get_schema"]

        with patch.object(knowledge_brief, "_research", research):
            first = knowledge_brief.brief_for("goal A", ON, LLM, self.dir)
            second = knowledge_brief.brief_for("goal A", ON, LLM, self.dir)
        self.assertEqual((first["status"], first["cached"], first["tool_calls"]), ("ok", False, 2))
        self.assertTrue(first["brief"].startswith("Background gathered automatically from the OKN"))
        self.assertIn("must not be treated as established", first["brief"])  # provenance warning
        self.assertEqual(first["brief"].count("X"), 600)  # max_brief_chars
        self.assertTrue(second["cached"])
        self.assertEqual(len(calls), 1)  # second call served from cache
        self.assertEqual(calls[0][1:], ("https://kg.test/mcp", ["describe_kg", "expand_ontology_term"], 3))
        self.assertIn("goal A", (self.dir / knowledge_brief.CACHE_FILE).read_text())

    def test_seeds_reach_the_loop_and_split_the_cache(self) -> None:
        seen = []

        async def research(goal, url, allowed, llm, max_calls, seeds=()):
            seen.append(tuple(seeds))
            return "brief text", 1, ["list_kgs"]

        with patch.object(knowledge_brief, "_research", research):
            knowledge_brief.brief_for("goal S", ON, LLM, self.dir)
            knowledge_brief.brief_for("goal S", {**ON, "seed_terms": ["MONDO:0020499"]}, LLM, self.dir)
            knowledge_brief.brief_for("goal S", {**ON, "seed_terms": ["MONDO:0020499"]}, LLM, self.dir)
        # different seeds are a different brief; identical seeds hit the cache
        self.assertEqual(seen, [(), ("MONDO:0020499",)])

    def test_failure_returns_no_brief_and_is_not_cached(self) -> None:
        async def boom(*_args):
            raise ConnectionError("unreachable")

        with patch.object(knowledge_brief, "_research", boom):
            failed = knowledge_brief.brief_for("goal B", ON, LLM, self.dir)
        self.assertEqual(failed["brief"], "")
        self.assertIn("ConnectionError", failed["status"])
        self.assertFalse((self.dir / knowledge_brief.CACHE_FILE).exists())

    def test_timeout_is_enforced(self) -> None:
        async def slow(*_args):
            await asyncio.sleep(30)

        with patch.object(knowledge_brief, "_research", slow):
            result = knowledge_brief.brief_for("goal D", {**ON, "timeout_seconds": 1}, LLM, self.dir)
        self.assertIn("TimeoutError", result["status"])
        self.assertEqual(result["brief"], "")


class TournamentUsesBriefTests(unittest.TestCase):
    def _run(self, knowledge, brief_result=None):
        hook = _load("loam_tournament_knowledge", HOOKS / "hypothesis_tournament.py")
        seen = {}

        class FakeCoScientist:
            def __init__(self, **kwargs):
                self.memory = type("Memory", (), {"save": lambda _self: None})()
                self.made = []

            def start(self): pass
            def stop(self): pass
            def get_all_hypotheses(self): return self.made
            def set_research_goal(self, goal): seen["goal"] = goal
            def generate_hypotheses(self, count): self.made.extend(range(count))
            def wait_for_completion(self): pass

        tmp = Path(tempfile.mkdtemp())
        config = {"state_file": str(tmp / "memory.json"), "initial_hypotheses": 2}
        if knowledge is not None:
            config["knowledge"] = knowledge
        llm_config = type("C", (), {"base_url": "http://x/v1", "api_key": "k", "model": "m"})()
        with patch.object(hook, "CoScientist", FakeCoScientist), \
                patch("knowledge_brief.brief_for", return_value=brief_result or {}) as brief_for:
            result = hook._ensure_initial_hypotheses(config, {"user_message": "Why does X bind Y?"}, llm_config)
        return seen["goal"], result, brief_for

    def test_knowledge_model_overrides_the_tournament_model(self) -> None:
        _goal, _result, brief_for = self._run({**ON, "model": "inkling-bf16"},
                                              {"status": "ok", "brief": "B", "tool_calls": 1,
                                               "tools_used": [], "cached": False, "seconds": 1.0})
        self.assertEqual(brief_for.call_args.args[2]["model"], "inkling-bf16")

    def test_enabled_appends_brief_to_jnana_goal(self) -> None:
        goal, result, brief_for = self._run(ON, {"status": "ok", "brief": "Background: KG facts", "tool_calls": 4,
                                                 "tools_used": ["list_kgs"], "cached": False, "seconds": 3.0})
        self.assertEqual(goal, "Why does X bind Y?\n\nBackground: KG facts")
        self.assertEqual(result["knowledge"]["status"], "ok")
        self.assertEqual(result["knowledge"]["brief_chars"], len("Background: KG facts"))
        self.assertEqual(brief_for.call_args.args[2], {"base_url": "http://x/v1", "api_key": "k", "model": "m"})

    def test_failed_brief_falls_back_to_plain_goal(self) -> None:
        goal, result, _ = self._run(ON, {"status": "failed: ConnectionError", "brief": ""})
        self.assertEqual(goal, "Why does X bind Y?")
        self.assertEqual(result["knowledge"]["brief_chars"], 0)

    def test_disabled_or_absent_never_calls_the_pre_pass(self) -> None:
        for knowledge in (None, {**ON, "enabled": False}):
            with self.subTest(knowledge=knowledge):
                goal, result, brief_for = self._run(knowledge)
                self.assertEqual(goal, "Why does X bind Y?")
                self.assertNotIn("knowledge", result)
                brief_for.assert_not_called()


class ConfigTests(unittest.TestCase):
    def _profile(self, knowledge):
        tournament = {"enabled": True, "jnana_root": "/j", "inference_endpoint": "http://x/v1", "model": "m",
                      "state_file": "/s/m.json", "reports_dir": "/s/r"}
        if knowledge is not None:
            tournament["knowledge"] = knowledge
        raw = {"enabled": True, "active_profile": "p", "profiles": {"p": {
            "enabled": True, "workflow_id": "w", "state_dir": "/s", "hypothesis_tournament": tournament}}}
        path = Path(tempfile.mkdtemp()) / "loam.yaml"
        path.write_text(json.dumps(raw))
        with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(path)}, clear=False):
            return loam_config.payload_values(loam_config.load_profile())["loam_hypothesis_tournament"]

    def test_enabled_block_is_passed_with_defaults(self) -> None:
        knowledge = self._profile({"enabled": True, "max_tool_calls": 5})["knowledge"]
        self.assertEqual(knowledge, {"enabled": True, "name": "OKN", "mcp_url": "https://apps.okn.us/okn-mcp-dev/mcp",
                                     "max_tool_calls": 5, "timeout_seconds": 900, "max_brief_chars": 4000})

    def test_seed_terms_are_passed_through(self) -> None:
        self.assertNotIn("seed_terms", self._profile({"enabled": True})["knowledge"])
        self.assertEqual(self._profile({"enabled": True, "seed_terms": ["MONDO:0020499"]})
                         ["knowledge"]["seed_terms"], ["MONDO:0020499"])

    def test_model_override_is_optional(self) -> None:
        self.assertNotIn("model", self._profile({"enabled": True})["knowledge"])
        self.assertEqual(self._profile({"enabled": True, "model": "inkling-bf16"})["knowledge"]["model"],
                         "inkling-bf16")

    def test_missing_or_disabled_block_passes_nothing(self) -> None:
        for knowledge in (None, {"enabled": False}, {"enabled": "yes"}):
            with self.subTest(knowledge=knowledge):
                self.assertNotIn("knowledge", self._profile(knowledge))


if __name__ == "__main__":
    unittest.main()
