"""A campaign names a task or a tool; both tiers see the tools they can use.

The validator designed experiments from recollection because the skill list
reached only the code-generator, and a configured task folder
(`structure-prediction`) resolved to nothing at all: it holds no SKILL.md, so
`skill_view` answered 404 and children hunted the filesystem for an invocation.

So a task name expands to its tools, a tool name pins one, and the catalog
reaches the validator as well -- which is the tier that chooses.
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
ADAPTER_DIR = ROOT / ".loam" / "adapters" / "hermes" / "plugin"


def _load(name: str, filename: str):
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


def _library(root: Path) -> Path:
    """A two-level library: tasks holding tools, plus one standalone skill."""

    def skill(path: Path, name: str, description: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\nBody.\n", encoding="utf-8"
        )

    skill(root / "structure-prediction" / "chai", "chai", "Co-fold a complex. Reports ipTM.")
    skill(root / "structure-prediction" / "boltz", "boltz", "Predict structure and affinity.")
    skill(root / "inverse-folding" / "proteinmpnn", "proteinmpnn", "Design a sequence for a backbone.")
    skill(root / "hpc-deployment", "hpc-deployment", "Stand up the execution server.")
    # Support folders are a skill's own files, never further skills.
    (root / "structure-prediction" / "chai" / "references").mkdir()
    (root / "structure-prediction" / "chai" / "references" / "SKILL.md").write_text(
        "---\nname: nope\ndescription: not a skill\n---\n", encoding="utf-8"
    )
    return root


class ResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = _load("loam_skill_catalog", "skill_catalog.py")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = _library(Path(self.temporary.name) / "skills")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _resolve(self, *names: str):
        return self.catalog.resolve(names, [self.root])

    def test_a_task_expands_to_every_tool_in_it(self) -> None:
        """The fix for the 404s: the validator may name the stage, not the tool."""

        resolution = self._resolve("structure-prediction")

        self.assertEqual(resolution.names, ("boltz", "chai"))
        self.assertEqual(resolution.missing, ())
        # Each entry carries the task it came from, so the catalog can group.
        self.assertEqual({skill.task for skill in resolution.skills}, {"structure-prediction"})

    def test_a_tool_name_pins_exactly_that_tool(self) -> None:
        """How a campaign keeps 150 validators comparable on one folding model."""

        self.assertEqual(self._resolve("chai").names, ("chai",))

    def test_a_task_slash_tool_name_resolves_to_the_tool(self) -> None:
        resolution = self._resolve("structure-prediction/chai")

        self.assertEqual(resolution.names, ("chai",))
        self.assertEqual(resolution.skills[0].task, "structure-prediction")

    def test_a_standalone_skill_resolves_to_itself(self) -> None:
        resolution = self._resolve("hpc-deployment")

        self.assertEqual(resolution.names, ("hpc-deployment",))
        self.assertEqual(resolution.skills[0].task, "")

    def test_an_unresolvable_name_is_reported_not_silently_dropped(self) -> None:
        resolution = self._resolve("chai", "folding")

        self.assertEqual(resolution.names, ("chai",))
        self.assertEqual(resolution.missing, ("folding",))
        self.assertTrue(any("folding" in problem for problem in resolution.errors()))

    def test_support_folders_are_not_skills(self) -> None:
        """A `references/SKILL.md` must not become a dispatchable tool name."""

        self.assertEqual(self._resolve("nope").missing, ("nope",))

    def test_order_is_the_configured_order_without_repeats(self) -> None:
        resolution = self._resolve("chai", "structure-prediction", "chai")

        self.assertEqual(resolution.names, ("chai", "boltz"))

    def test_the_catalog_groups_tools_under_their_task(self) -> None:
        resolution = self._resolve("structure-prediction", "hpc-deployment")
        lines = self.catalog.catalog_lines(resolution.skills)

        self.assertEqual(lines[0], "- structure-prediction:")
        self.assertIn("`boltz`", lines[1])
        self.assertIn("Predict structure and affinity.", lines[1])
        # A standalone skill is listed flat, with no task heading of its own.
        self.assertIn("`hpc-deployment`", lines[-1])
        self.assertEqual(
            self.catalog.catalog_mapping(resolution.skills)["structure-prediction"]["chai"],
            "Co-fold a complex. Reports ipTM.",
        )


class ContextTests(unittest.TestCase):
    """Both tiers get the catalog: the validator to choose, the child to follow."""

    def setUp(self) -> None:
        self.orchestration = _load("loam_catalog_orchestration", "hypothesis_orchestration.py")
        self.campaign_module = sys.modules["campaign"]
        self.temporary = tempfile.TemporaryDirectory()
        self.root = _library(Path(self.temporary.name) / "skills")
        self.catalog = _load("loam_catalog_for_context", "skill_catalog.py")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _campaign(self, *names: str):
        resolution = self.catalog.resolve(names, [self.root])
        return self.campaign_module.CampaignContext(
            goal_name="g", goal="Design a binder.", rules=(), skills=names, catalog=resolution.skills
        )

    def test_the_validator_is_given_the_catalog_and_told_to_choose(self) -> None:
        field = self.campaign_module.validator_skills_field(self._campaign("structure-prediction"))

        self.assertEqual(sorted(field["campaign_skills"]["structure-prediction"]), ["boltz", "chai"])
        instructions = field["campaign_skills_instructions"]
        # Choose the task, then the tool -- and tell the child which it chose.
        self.assertIn("task first", instructions)
        self.assertIn("skill: <tool name>", instructions)

    def test_a_campaign_naming_no_skills_gives_the_validator_no_such_key(self) -> None:
        self.assertEqual(self.campaign_module.validator_skills_field(self._campaign()), {})

    def test_the_child_sees_the_tools_grouped_with_the_skill_as_authority(self) -> None:
        context = self.orchestration._child_context(
            "Fold the candidate.",
            {"mode": "remote_mcp", "tool_name": "mcp__hpc__run_code", "function_name": "run_code"},
            "/work",
            None,
            self._campaign("structure-prediction", "proteinmpnn"),
        )

        self.assertIn("CAMPAIGN SKILLS", context)
        self.assertIn("- structure-prediction:", context)
        self.assertIn("`proteinmpnn`", context)
        self.assertIn("authoritative over your recollection", context)
        # The step's named skill is what it reads, not the whole library.
        self.assertIn("Do not browse `skills_list`", context)

    def test_the_registry_no_longer_outranks_the_skills(self) -> None:
        """`list_tools` was advertised as "call this FIRST", above the recipe."""

        execution = {
            "mode": "remote_mcp",
            "tool_name": "mcp__hpc__run_code",
            "function_name": "run_code",
            "tool_prefix": "mcp__hpc__",
        }
        context = self.orchestration._child_context(
            "Fold it.", execution, "/work", None, self._campaign("chai")
        )

        self.assertNotIn("list_tools", context)
        self.assertNotIn("register_tool", context)
        self.assertIn("mcp__hpc__write_files", context)


class ValidatorPolicyTests(unittest.TestCase):
    """Reading a recipe is looking, like polling a batch."""

    def setUp(self) -> None:
        self.orchestration = _load("loam_catalog_policy", "hypothesis_orchestration.py")

    def _verdict(self, tool_name: str):
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                "repo_root": str(ROOT),
                "workflow_id": "research-v1",
                "loam_run_id": "run-1",
                "loam_state_dir": temporary,
                "loam_hypothesis_orchestration": True,
                "loam_execution": {
                    "mode": "remote_mcp",
                    "tool_name": "mcp__hpc__run_code",
                    "function_name": "run_code",
                    "max_children": 1,
                },
            }
            self.orchestration.record_start({
                **payload,
                "child_session_id": "validator-1",
                "child_subagent_id": "sa-validator-1",
                "child_goal": (
                    "Validate [loam-agent-role=hypothesis-validator] "
                    "[loam-plan-id=plan-1] [loam-hypothesis-id=h-1]"
                ),
            })
            return self.orchestration.tool_policy(
                {**payload, "session_id": "validator-1", "tool_name": tool_name, "arguments": {}}
            )

    def test_the_validator_may_read_a_skill_before_choosing_a_tool(self) -> None:
        for tool_name in ("skill_view", "skills_list"):
            with self.subTest(tool=tool_name):
                self.assertIsNone(self._verdict(tool_name))

    def test_the_validator_still_may_not_execute(self) -> None:
        verdict = self._verdict("mcp__hpc__run_code")

        assert verdict is not None
        self.assertEqual(verdict["action"], "block")
        # The refusal names the reads it does allow, so it is actionable.
        self.assertIn("skill_view", verdict["message"])


class ConfigTests(unittest.TestCase):
    def test_a_task_slash_tool_name_is_accepted(self) -> None:
        config = _load_config("loam_catalog_config")
        campaign = config._campaign_config(
            {"goal": "my-campaign", "skills": ["structure-prediction/chai", "analysis"]}
        )

        assert campaign is not None
        self.assertEqual(campaign.skills, ("structure-prediction/chai", "analysis"))

    def test_traversal_and_deeper_nesting_are_still_rejected(self) -> None:
        config = _load_config("loam_catalog_config_traversal")
        for name in ("../../etc/passwd", "a/b/c", "/abs/path", "..", "structure-prediction/"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    config._campaign_config({"goal": "my-campaign", "skills": [name]})


class ProvenanceTests(unittest.TestCase):
    """`declared_skills` must compare like with like against what a run read."""

    def test_a_declared_task_is_recorded_as_the_tools_it_offers(self) -> None:
        workspace = _load("loam_catalog_workspace", "hypothesis_workspace.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = _library(Path(temporary) / "skills")
            declared = workspace._declared_skills({
                "repo_root": str(root.parent),
                "loam_campaign": {"skills": ["structure-prediction", "folding"]},
            })

        # The tools, so a run that read `chai` shows as following the recipe --
        # plus the unresolvable name, so a misconfiguration stays visible.
        self.assertEqual(declared, ["boltz", "chai", "folding"])


class PreflightTests(unittest.TestCase):
    def test_launch_is_refused_when_a_name_resolves_to_nothing(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "loam_catalog_preflight", ROOT / "scripts" / "campaign_preflight.py"
        )
        assert spec and spec.loader
        preflight = importlib.util.module_from_spec(spec)
        sys.modules["loam_catalog_preflight"] = preflight
        spec.loader.exec_module(preflight)

        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            _library(home / "skills")
            config = Path(temporary) / "loam.yaml"

            def write(skills: list[str]) -> None:
                config.write_text(
                    json.dumps({
                        "enabled": True,
                        "active_profile": "p",
                        "profiles": {"p": {"campaign": {"goal": "g", "skills": skills}}},
                    }),
                    encoding="utf-8",
                )

            write(["structure-prediction"])
            report = preflight.check_skills(ROOT, home, config)
            self.assertEqual(report["resolved"], ["boltz", "chai"])

            write(["structure-prediction", "folding"])
            with self.assertRaises(RuntimeError) as caught:
                preflight.check_skills(ROOT, home, config)
            self.assertIn("folding", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
