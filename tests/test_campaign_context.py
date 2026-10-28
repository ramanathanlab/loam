"""Loading the campaign goal and rules files named by a profile's `campaign:` block."""

from __future__ import annotations

import importlib.util
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


def _repo(root: Path, goal: str | None = None, rules: dict[str, str] | None = None) -> dict:
    """Build a throwaway repo_root holding the campaign files under test."""

    if goal is not None:
        campaigns = root / ".loam" / "campaigns"
        campaigns.mkdir(parents=True, exist_ok=True)
        (campaigns / "fixture-goal.md").write_text(goal, encoding="utf-8")
    for name, text in (rules or {}).items():
        rules_dir = root / ".loam" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        (rules_dir / f"{name}.md").write_text(text, encoding="utf-8")
    return {"repo_root": str(root)}


GOAL = "---\nname: fixture-goal\n---\n\nDesign de novo binders against a fixture target.\n"


def _rule(name: str, scope: str, body: str = "Do not fabricate a reference sequence.", extra: str = "") -> str:
    return f"---\nname: {name}\nscope: {scope}\n{extra}---\n\n{body}\n"


class CampaignContextTests(unittest.TestCase):
    def test_absent_block_loads_nothing(self) -> None:
        orchestration = _load("loam_campaign_absent")
        self.assertIsNone(orchestration.campaign_context({"repo_root": str(ROOT)}))

    def test_goal_and_rules_load_from_the_repo(self) -> None:
        orchestration = _load("loam_campaign_load")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **_repo(Path(temporary), goal=GOAL, rules={"design-integrity": _rule("design-integrity", "all")}),
                "loam_campaign": {"goal": "fixture-goal", "rules": ["design-integrity"]},
            }
            context = orchestration.campaign_context(payload)

        assert context is not None
        self.assertEqual(context.goal_name, "fixture-goal")
        self.assertIn("de novo binders", context.goal)
        self.assertNotIn("---", context.goal)  # front matter stripped
        self.assertEqual(len(context.rules), 1)
        self.assertEqual(context.rules[0].scopes, frozenset(orchestration.RULE_TIERS))

    def test_the_checked_in_campaign_file_is_well_formed(self) -> None:
        """Guards the real `.loam/campaigns/` fixture, not just a synthetic one."""

        orchestration = _load("loam_campaign_real_file")
        context = orchestration.campaign_context(
            {"repo_root": str(ROOT), "loam_campaign": {"goal": "egfr-binders"}}
        )

        assert context is not None
        self.assertIn("EGFR", context.goal)
        self.assertEqual(context.rules, ())

    def test_scope_selects_the_tiers_a_rule_reaches(self) -> None:
        orchestration = _load("loam_campaign_scope")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **_repo(
                    Path(temporary),
                    goal=GOAL,
                    rules={
                        "validator-only": _rule("validator-only", "validator"),
                        "child-only": _rule("child-only", "code-generator"),
                        "two-tiers": _rule("two-tiers", "validator, reasoner"),
                    },
                ),
                "loam_campaign": {
                    "goal": "fixture-goal",
                    "rules": ["validator-only", "child-only", "two-tiers"],
                },
            }
            context = orchestration.campaign_context(payload)

        assert context is not None
        self.assertEqual(
            [rule.name for rule in context.rules_for("validator")], ["validator-only", "two-tiers"]
        )
        self.assertEqual([rule.name for rule in context.rules_for("code-generator")], ["child-only"])
        self.assertEqual([rule.name for rule in context.rules_for("reasoner")], ["two-tiers"])

    def test_enforced_by_is_carried_when_declared(self) -> None:
        orchestration = _load("loam_campaign_enforced_by")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **_repo(
                    Path(temporary),
                    goal=GOAL,
                    rules={"budget": _rule("budget", "validator", extra="enforced_by: tool_policy max_children\n")},
                ),
                "loam_campaign": {"goal": "fixture-goal", "rules": ["budget"]},
            }
            context = orchestration.campaign_context(payload)

        assert context is not None
        self.assertEqual(context.rules[0].enforced_by, "tool_policy max_children")

    def test_a_missing_named_file_raises(self) -> None:
        """Named-but-absent policy must fail loudly, not leave the run ungoverned."""

        orchestration = _load("loam_campaign_missing")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                orchestration.campaign_context(
                    {**_repo(Path(temporary)), "loam_campaign": {"goal": "fixture-goal"}}
                )
            with self.assertRaises(FileNotFoundError):
                orchestration.campaign_context(
                    {
                        **_repo(Path(temporary), goal=GOAL),
                        "loam_campaign": {"goal": "fixture-goal", "rules": ["nope"]},
                    }
                )

    def test_a_name_mismatch_raises(self) -> None:
        orchestration = _load("loam_campaign_mismatch")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **_repo(Path(temporary), goal="---\nname: other\n---\n\nBody.\n"),
                "loam_campaign": {"goal": "fixture-goal"},
            }
            with self.assertRaises(ValueError):
                orchestration.campaign_context(payload)

    def test_a_rule_must_declare_a_known_scope(self) -> None:
        orchestration = _load("loam_campaign_bad_scope")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                **_repo(
                    root,
                    goal=GOAL,
                    rules={
                        "no-scope": "---\nname: no-scope\n---\n\nBody.\n",
                        "bad-scope": _rule("bad-scope", "both"),
                    },
                ),
                "loam_campaign": {"goal": "fixture-goal", "rules": ["no-scope"]},
            }
            with self.assertRaises(ValueError):
                orchestration.campaign_context(payload)
            payload["loam_campaign"] = {"goal": "fixture-goal", "rules": ["bad-scope"]}
            with self.assertRaises(ValueError):
                orchestration.campaign_context(payload)

    def test_bodies_are_bounded(self) -> None:
        orchestration = _load("loam_campaign_bounds")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **_repo(
                    Path(temporary),
                    goal="---\nname: fixture-goal\n---\n\n" + "g" * 10_000,
                    rules={"long": _rule("long", "validator", body="r" * 10_000)},
                ),
                "loam_campaign": {"goal": "fixture-goal", "rules": ["long"]},
            }
            context = orchestration.campaign_context(payload)

        assert context is not None
        self.assertEqual(len(context.goal), orchestration.MAX_CAMPAIGN_CHARS)
        self.assertEqual(len(context.rules[0].body), orchestration.MAX_RULES_CHARS)


if __name__ == "__main__":
    unittest.main()
