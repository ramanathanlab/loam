"""Loading the campaign goal and rules files named by a profile's `campaign:` block."""

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


def _orchestration_payload(root: Path, campaign: dict | None = None) -> dict:
    """A payload shaped like a live turn, against the repo's real campaign files."""

    state_file = root / "jnana.json"
    reports_dir = root / "reports"
    reports_dir.mkdir()
    state_file.write_text(
        json.dumps(
            {"hypotheses": [{"hypothesis_id": "h-1", "content": "A candidate binder engages the target interface."}]}
        ),
        encoding="utf-8",
    )
    (reports_dir / "tournament-001.json").write_text(
        json.dumps(
            {
                "workflow_id": "research-v1",
                "result": {
                    "status": "completed",
                    "top_hypotheses": [{"hypothesis_id": "h-1", "elo_rating": 1210.0}],
                },
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "repo_root": str(ROOT),
        "workflow_id": "research-v1",
        "session_id": "session-1",
        "loam_state_dir": str(root / "loam-state"),
        "loam_hypothesis_tournament": {"state_file": str(state_file), "reports_dir": str(reports_dir)},
        "loam_execution": {
            "mode": "remote_mcp",
            "tool_name": "globus_execute",
            "endpoint": "https://compute.example/mcp",
            "function_name": "run_python",
            "input_schema": "run-python-v1",
            "runtime": "python3.11",
        },
    }
    if campaign is not None:
        payload["loam_campaign"] = campaign
    return payload


CAMPAIGN = {"goal": "egfr-binders", "rules": ["design-integrity", "experiment-discipline", "binder-cofold"]}


def _validator_context(directive: str) -> str:
    """The first validator task's context, as the child will receive it.

    `_validator_directive` ends with the ready-to-run task array, and each task's
    context is itself a JSON string, so the keys appear escaped in the directive
    text. Ordering must be asserted inside this payload, not against the
    surrounding prose -- which mentions the same key names.
    """

    tasks = json.loads(directive.rsplit("\n", 1)[1])
    return tasks[0]["context"]


class CampaignInjectionTests(unittest.TestCase):
    def test_validator_payload_carries_the_goal_ahead_of_the_role_definition(self) -> None:
        """Guards the 2026-10-21 regression: policy must not land behind 8.6 KB of boilerplate."""

        orchestration = _load("loam_campaign_validator_inject")
        with tempfile.TemporaryDirectory() as temporary:
            directive = orchestration.context_for(_orchestration_payload(Path(temporary), CAMPAIGN))

        context = _validator_context(directive)
        self.assertIn("EGFR", context)
        self.assertLess(context.index('"campaign"'), context.index('"validator_instructions"'))
        self.assertLess(context.index('"campaign"'), context.index('"agent_definition"'))
        # The whole point of the ordering: policy lands where the model reads it.
        self.assertLess(context.index('"campaign"'), 2_000)

    def test_validator_payload_omits_a_child_scoped_rule(self) -> None:
        orchestration = _load("loam_campaign_validator_scope")
        with tempfile.TemporaryDirectory() as temporary:
            directive = orchestration.context_for(_orchestration_payload(Path(temporary), CAMPAIGN))

        self.assertIn("authoritative source", directive)  # design-integrity, scope: all
        self.assertNotIn("ONE decisive measurement", directive)  # experiment-discipline, code-generator
        self.assertNotIn("complex co-fold", directive)  # binder-cofold, code-generator

    def test_no_campaign_block_leaves_the_directive_unchanged(self) -> None:
        orchestration = _load("loam_campaign_validator_absent")
        with tempfile.TemporaryDirectory() as temporary:
            directive = orchestration.context_for(_orchestration_payload(Path(temporary)))

        self.assertIn("Loam orchestration directive", directive)
        self.assertNotIn('"campaign"', directive)

    def test_child_context_carries_goal_and_rules_before_the_instruction(self) -> None:
        orchestration = _load("loam_campaign_child_inject")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **_orchestration_payload(Path(temporary), CAMPAIGN),
                "arguments": {"tasks": [{"goal": "Co-fold the candidate.", "context": "Inputs follow."}]},
            }
            tasks = orchestration._generated_child_tasks(payload, "plan-1", "h-1")

        assert tasks is not None
        context = tasks["tasks"][0]["context"]
        self.assertIn("CAMPAIGN GOAL", context)
        self.assertIn("ONE decisive measurement", context)  # discipline layer
        self.assertIn("complex co-fold", context)  # assay layer
        self.assertLess(context.index("CAMPAIGN RULES"), context.index("Instruction from validator"))

    def test_reasoner_directive_carries_the_campaign(self) -> None:
        """reasoner.md requires an objective; until now nothing supplied one."""

        orchestration = _load("loam_campaign_reasoner_inject")
        registry = orchestration.agent_registry({"repo_root": str(ROOT)})
        context = orchestration.campaign_context({"repo_root": str(ROOT), "loam_campaign": CAMPAIGN})
        directive = orchestration._reasoner_directive(
            "plan-1",
            [{"hypothesis_id": "h-1", "status": "supported", "evidence_score": 0.5, "summary": "Report."}],
            registry,
            context,
            {"h-1": 1210.0},
        )

        self.assertIn("EGFR", directive)
        # Asserted inside the JSON block, because the surrounding prose names
        # "agent_definition" before it appears as a key.
        block = next(line for line in directive.splitlines() if '"agent_definition"' in line and line.startswith("{"))
        self.assertLess(block.index('"campaign"'), block.index('"agent_definition"'))
        self.assertNotIn("ONE decisive measurement", directive)  # code-generator scope only


class HypothesisBlockExtractionTests(unittest.TestCase):
    def test_campaign_specific_protocol_moved_out_of_the_hook(self) -> None:
        orchestration = _load("loam_campaign_extraction")
        block = orchestration._hypothesis_block({"hypothesis_id": "h-1", "content": "A claim."})

        self.assertIn("A claim.", block)
        # Kept: campaign-independent, and backed by _SCRUBBERS.
        self.assertIn("NEVER run, score, or reuse", block)
        self.assertIn("complete assignment", block)
        # Moved to the assay rules under .loam/rules/.
        self.assertNotIn("ONE decisive measurement", block)
        self.assertNotIn("ipTM", block)
        self.assertNotIn("co-fold", block)


class RuleLayeringTests(unittest.TestCase):
    """Loam must serve any domain; only the assay layer may name one.

    Binder design and enzyme engineering are the near-term campaigns, and the
    first version of these rules read as binder-only -- the decisive experiment
    was "a co-fold", the metrics were ipTM/ipSAE/pDockQ, and the hook itself
    named one campaign's scratch directory. An enzyme campaign would have had to
    rewrite the invariant rules to say anything at all. See .loam/rules/README.md.
    """

    # Tool and metric names: checkable, and exactly what leaked last time. A
    # domain example in prose is allowed; a named instrument or number is not.
    DOMAIN_TERMS = (
        "co-fold", "cofold", "iptm", "ipsae", "pdockq", "plddt", "chai",
        "fasta", ".cif", ".pdb", "proteinmpnn", "batch_status",
    )

    def _assert_domain_neutral(self, name: str) -> None:
        body = (ROOT / ".loam" / "rules" / f"{name}.md").read_text(encoding="utf-8").lower()
        for term in self.DOMAIN_TERMS:
            self.assertNotIn(term, body, f"{name}.md names {term!r}; that belongs in an assay rule")

    def test_the_integrity_rule_is_domain_neutral(self) -> None:
        self._assert_domain_neutral("design-integrity")

    def test_the_discipline_rule_is_domain_neutral(self) -> None:
        self._assert_domain_neutral("experiment-discipline")

    def test_the_hook_itself_is_domain_neutral(self) -> None:
        """Inline text reaches every child of every campaign, configured or not."""

        orchestration = _load("loam_rule_layering")
        block = orchestration._hypothesis_block({"hypothesis_id": "h-1", "content": "A claim."}).lower()
        for term in self.DOMAIN_TERMS:
            self.assertNotIn(term, block, f"_hypothesis_block names {term!r}; it must serve any domain")

    def test_the_assay_rule_is_where_the_domain_lives(self) -> None:
        """The negative tests above are only meaningful if something carries it."""

        body = (ROOT / ".loam" / "rules" / "binder-cofold.md").read_text(encoding="utf-8")
        self.assertIn("co-fold", body)
        self.assertIn("ipSAE", body)


if __name__ == "__main__":
    unittest.main()
