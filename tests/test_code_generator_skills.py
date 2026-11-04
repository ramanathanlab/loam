"""A code-generator may read its campaign's skills before writing code.

The tier that writes the code was confined to the HPC execution tools, so it
wrote GPU code from recollection while a reviewed recipe sat in the campaign's
skill library unreadable. Three things had to change together: the veto had to
allow reading, the child's context had to say which skills are its, and
`code-generator.md` had to stop implying that consulting a procedure is a
detour from producing code.

Reading is not executing -- the same argument that let a validator poll its own
batch rather than spend a child on it.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
ADAPTER_DIR = ROOT / ".loam" / "adapters" / "hermes" / "plugin"

REMOTE_GENERATED = {
    "mode": "remote_mcp",
    "tool_name": "mcp__hpc__run_code",
    "function_name": "run_code",
    "endpoint": "https://compute.example/mcp",
    "input_schema": "run-python-v1",
    "runtime": "python3.12",
    "max_children": 3,
}


def _load(name: str, filename: str = "hypothesis_orchestration.py"):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, HOOKS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_config(name: str):
    if str(ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(ADAPTER_DIR))
    spec = importlib.util.spec_from_file_location(name, ADAPTER_DIR / "loam_config.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _child_start(orchestration, payload: dict, state_dir: Path) -> dict:
    """State in which a code-generator child's tool calls are judged."""

    orchestration.record_start(
        {
            **payload,
            "child_session_id": "codegen-1",
            "child_subagent_id": "sa-codegen-1",
            "child_goal": (
                "Compute it [loam-agent-role=code-generator] "
                "[loam-plan-id=plan_1] [loam-hypothesis-id=h-1]"
            ),
        }
    )
    return {**payload, "session_id": "codegen-1"}


class ToolPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestration = _load("loam_skills_policy")

    def _verdict(self, tool_name: str) -> dict | None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            payload = {
                "repo_root": str(ROOT),
                "workflow_id": "research-v1",
                "loam_run_id": "run-1",
                "loam_state_dir": str(state_dir),
                "loam_hypothesis_orchestration": True,
                "loam_execution": REMOTE_GENERATED,
            }
            child = _child_start(self.orchestration, payload, state_dir)
            return self.orchestration.tool_policy({**child, "tool_name": tool_name, "arguments": {}})

    def test_reading_a_skill_is_allowed(self) -> None:
        for tool_name in ("skills_list", "skill_view"):
            with self.subTest(tool=tool_name):
                self.assertIsNone(self._verdict(tool_name))

    def test_authoring_a_skill_is_not(self) -> None:
        """The child consults skills; it does not write them."""

        verdict = self._verdict("skill_manage")
        assert verdict is not None
        self.assertEqual(verdict["action"], "block")

    def test_execution_is_still_confined(self) -> None:
        """Allowing a read must not widen what the child may run."""

        verdict = self._verdict("terminal")
        assert verdict is not None
        self.assertEqual(verdict["action"], "block")
        self.assertIn("mcp__hpc__run_code", verdict["message"])
        # The block message names the read-only exception, so a child that
        # guesses a tool name learns what it may actually call.
        self.assertIn("skill_view", verdict["message"])

    def test_delegation_is_still_refused(self) -> None:
        verdict = self._verdict("delegate_task")
        assert verdict is not None
        self.assertEqual(verdict["action"], "block")


class ChildContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestration = _load("loam_skills_context")

    def _context(self, skills: tuple[str, ...]) -> str:
        campaign = self.orchestration.CampaignContext(
            goal_name="g", goal="Design something.", rules=(), skills=skills
        )
        return self.orchestration._child_context(
            "Fold the candidate.", REMOTE_GENERATED, "/work", None, campaign
        )

    def test_named_skills_reach_the_child_with_their_authority_stated(self) -> None:
        context = self._context(("folding", "inverse-folding"))

        self.assertIn("CAMPAIGN SKILLS", context)
        self.assertIn("`folding`", context)
        self.assertIn("`inverse-folding`", context)
        self.assertIn("skill_view", context)
        # The point of the change: the recipe outranks what the model recalls.
        self.assertIn("authoritative over your recollection", context)

    def test_the_no_other_tool_line_stops_contradicting_the_permission(self) -> None:
        """A child told to use nothing else will not try a tool it may call."""

        context = self._context(("folding",))
        self.assertNotIn("do not execute code locally or use any other tool.", context)
        self.assertIn("You may also call `skill_view`", context)

    def test_a_campaign_naming_no_skills_says_nothing_about_them(self) -> None:
        context = self._context(())
        self.assertNotIn("CAMPAIGN SKILLS", context)
        self.assertNotIn("skill_view", context)
        # And the original confinement wording is intact for that campaign.
        self.assertIn("use any other tool to run anything.", context)

    def test_skills_land_after_policy_and_before_the_environment(self) -> None:
        """Goal, then which procedure, then how to run it."""

        rule = sys.modules["campaign"].RuleSpec(
            name="r", scopes=frozenset(("code-generator",)), body="Fetch, do not recall."
        )
        campaign = self.orchestration.CampaignContext(
            goal_name="g", goal="Design something.", rules=(rule,), skills=("folding",)
        )
        context = self.orchestration._child_context(
            "Fold it.", {**REMOTE_GENERATED, "setup": "source /venv/bin/activate"}, "/work", None, campaign
        )

        self.assertLess(context.index("CAMPAIGN RULES"), context.index("CAMPAIGN SKILLS"))
        self.assertLess(context.index("CAMPAIGN SKILLS"), context.index("ENVIRONMENT SETUP"))
        self.assertLess(context.index("CAMPAIGN SKILLS"), context.index("Instruction from validator"))


class ConfigTests(unittest.TestCase):
    def test_skills_are_bare_names_and_reach_the_payload(self) -> None:
        config = _load_config("loam_skills_config")
        profile = config._campaign_config(
            {"goal": "my-campaign", "skills": ["folding", "inverse-folding"]}
        )

        assert profile is not None
        self.assertEqual(profile.skills, ("folding", "inverse-folding"))

    def test_a_path_is_rejected_as_a_skill_name(self) -> None:
        """Same validation as rules: a name, never a path."""

        config = _load_config("loam_skills_config_traversal")
        with self.assertRaises(ValueError):
            config._campaign_config({"goal": "my-campaign", "skills": ["../../etc/passwd"]})

    def test_a_repeated_skill_is_rejected(self) -> None:
        config = _load_config("loam_skills_config_repeat")
        with self.assertRaises(ValueError):
            config._campaign_config({"goal": "my-campaign", "skills": ["folding", "folding"]})

    def test_skills_are_optional(self) -> None:
        config = _load_config("loam_skills_config_absent")
        profile = config._campaign_config({"goal": "my-campaign"})

        assert profile is not None
        self.assertEqual(profile.skills, ())


class AgentDefinitionTests(unittest.TestCase):
    def test_the_definition_tells_the_child_to_consult_before_writing(self) -> None:
        """The rules used to push the other way: "generate one self-contained
        snippet" and "prefer standard library" read as "write it from memory"."""

        body = (ROOT / ".loam" / "agents" / "code-generator.md").read_text(encoding="utf-8")
        self.assertIn("CAMPAIGN SKILLS", body)
        self.assertIn("skill_view", body)
        self.assertIn("authoritative", body)
        # The deliverable is unchanged -- a skill is not something to summarize.
        self.assertIn("the deliverable is still the code", body)
        self.assertLess(body.index("skill_view"), body.index("## Generation Rules"))


if __name__ == "__main__":
    unittest.main()
