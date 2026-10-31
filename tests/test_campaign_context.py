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
TIERS = ("reasoner", "hypothesis-validator", "code-generator")
sys.path.insert(0, str(HOOKS))
import campaign as policy  # noqa: E402


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
    def test_the_invariant_rules_load_without_a_campaign_block(self) -> None:
        """They were unconditional while inline in `_hypothesis_block`.

        Putting the text in a file is where it lives, not a reason to make it
        optional: a profile should not have to remember that an experiment must
        finish before it is reported.
        """

        orchestration = _load("loam_campaign_absent")
        context = orchestration.campaign_context({"repo_root": str(ROOT)})

        assert context is not None
        self.assertEqual(context.goal_name, "")
        self.assertEqual([rule.name for rule in context.rules], ["evidence-integrity", "experiment-discipline"])

    def test_an_empty_rules_directory_loads_nothing(self) -> None:
        orchestration = _load("loam_campaign_empty")
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(orchestration.campaign_context(_repo(Path(temporary))))

    def test_an_assay_rule_is_ignored_until_a_profile_names_it(self) -> None:
        """The default direction: the wrong decisive experiment beats a missing one."""

        orchestration = _load("loam_campaign_when_named")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _repo(
                Path(temporary),
                rules={
                    "base": _rule("base", "all", "Fetch, do not recall.", extra="applies: always\n"),
                    "assay": _rule("assay", "code-generator", "Run the wrong-domain assay."),
                },
            )
            unnamed = orchestration.campaign_context(payload)
            named = orchestration.campaign_context({**payload, "loam_campaign": {"rules": ["assay"]}})

        assert unnamed is not None and named is not None
        self.assertEqual([rule.name for rule in unnamed.rules], ["base"])
        # Base first, so the assay rule is the most specific text a child reads.
        self.assertEqual([rule.name for rule in named.rules], ["base", "assay"])

    def test_naming_an_invariant_rule_anyway_does_not_duplicate_it(self) -> None:
        orchestration = _load("loam_campaign_dedupe")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _repo(
                Path(temporary),
                rules={"base": _rule("base", "all", "Fetch, do not recall.", extra="applies: always\n")},
            )
            context = orchestration.campaign_context({**payload, "loam_campaign": {"rules": ["base"]}})

        assert context is not None
        self.assertEqual([rule.name for rule in context.rules], ["base"])

    def test_a_malformed_unnamed_rule_does_not_break_an_unrelated_campaign(self) -> None:
        """A draft left in the directory must not take down a running campaign."""

        orchestration = _load("loam_campaign_malformed_draft")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _repo(
                Path(temporary),
                rules={
                    "base": _rule("base", "all", "Fetch, do not recall.", extra="applies: always\n"),
                    "draft": "no front matter at all\n",
                },
            )
            context = orchestration.campaign_context(payload)

        assert context is not None
        self.assertEqual([rule.name for rule in context.rules], ["base"])

    def test_a_malformed_invariant_rule_still_raises(self) -> None:
        orchestration = _load("loam_campaign_malformed_base")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _repo(Path(temporary), rules={"base": "---\nname: base\napplies: always\n---\n\nNo scope.\n"})
            with self.assertRaises(ValueError):
                orchestration.campaign_context(payload)

    def test_goal_and_rules_load_from_the_repo(self) -> None:
        orchestration = _load("loam_campaign_load")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **_repo(Path(temporary), goal=GOAL, rules={"evidence-integrity": _rule("evidence-integrity", "all")}),
                "loam_campaign": {"goal": "fixture-goal", "rules": ["evidence-integrity"]},
            }
            context = orchestration.campaign_context(payload)

        assert context is not None
        self.assertEqual(context.goal_name, "fixture-goal")
        self.assertIn("de novo binders", context.goal)
        self.assertNotIn("---", context.goal)  # front matter stripped
        self.assertEqual(len(context.rules), 1)
        self.assertEqual(context.rules[0].scopes, frozenset(TIERS))

    def test_the_repo_ships_no_campaign_content(self) -> None:
        """The framework must not carry one field's goal or protocol.

        A checked-in goal made this repo quietly about protein binders, and the
        test suite asserted against it, so the coupling was load-bearing in both
        directions. Campaign files now come from `campaign.dir`.
        """

        orchestration = _load("loam_campaign_no_content")
        self.assertFalse((ROOT / ".loam" / "campaigns").exists())
        shipped = {
            path.stem
            for path in (ROOT / ".loam" / "rules").glob("*.md")
            if path.name != "README.md"
        }
        self.assertEqual(shipped, {"evidence-integrity", "experiment-discipline", "design-provenance"})
        # And what ships loads with no campaign configured at all.
        context = orchestration.campaign_context({"repo_root": str(ROOT)})
        assert context is not None
        self.assertEqual([rule.name for rule in context.rules], ["evidence-integrity", "experiment-discipline"])

    def test_a_user_directory_supplies_the_goal_and_protocol(self) -> None:
        orchestration = _load("loam_campaign_user_dir")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = _campaign(root)
            context = orchestration.campaign_context({"repo_root": str(ROOT), "loam_campaign": campaign})

        assert context is not None
        self.assertIn("FIXTURE SUBJECT", context.goal)
        # The user's protocol rule, after the invariant rules the repo ships.
        self.assertEqual(
            [rule.name for rule in context.rules],
            ["evidence-integrity", "experiment-discipline", "fixture-protocol"],
        )

    def test_a_user_rule_shadows_a_shipped_one_of_the_same_name(self) -> None:
        orchestration = _load("loam_campaign_shadow")
        with tempfile.TemporaryDirectory() as temporary:
            policy = Path(temporary) / "policy"
            (policy / "rules").mkdir(parents=True)
            (policy / "rules" / "design-provenance.md").write_text(
                "---\nname: design-provenance\nscope: all\n---\n\nMy own provenance policy.\n",
                encoding="utf-8",
            )
            context = orchestration.campaign_context(
                {
                    "repo_root": str(ROOT),
                    "loam_campaign": {"dir": str(policy), "rules": ["design-provenance"]},
                }
            )

        assert context is not None
        body = next(rule.body for rule in context.rules if rule.name == "design-provenance")
        self.assertEqual(body, "My own provenance policy.")

    def test_a_missing_file_names_every_directory_searched(self) -> None:
        orchestration = _load("loam_campaign_missing_both")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError) as raised:
                orchestration.campaign_context(
                    {
                        "repo_root": str(ROOT),
                        "loam_campaign": {"dir": temporary, "rules": ["no-such-rule"]},
                    }
                )

        self.assertIn(temporary, str(raised.exception))
        self.assertIn(str(ROOT / ".loam" / "rules"), str(raised.exception))

    def test_scope_selects_the_tiers_a_rule_reaches(self) -> None:
        orchestration = _load("loam_campaign_scope")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **_repo(
                    Path(temporary),
                    goal=GOAL,
                    rules={
                        "validator-only": _rule("validator-only", "hypothesis-validator"),
                        "child-only": _rule("child-only", "code-generator"),
                        "two-tiers": _rule("two-tiers", "hypothesis-validator, reasoner"),
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
            [rule.name for rule in context.rules_for("hypothesis-validator")], ["validator-only", "two-tiers"]
        )
        self.assertEqual([rule.name for rule in context.rules_for("code-generator")], ["child-only"])
        self.assertEqual([rule.name for rule in context.rules_for("reasoner")], ["two-tiers"])

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
                    rules={"long": _rule("long", "hypothesis-validator", body="r" * 10_000)},
                ),
                "loam_campaign": {"goal": "fixture-goal", "rules": ["long"]},
            }
            context = orchestration.campaign_context(payload)

        assert context is not None
        self.assertEqual(len(context.goal), 4_000)
        self.assertEqual(len(context.rules[0].body), 4_000)


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


FIXTURE_GOAL = (
    "---\nname: fixture-goal\n---\n\nEstablish whether the FIXTURE SUBJECT behaves as predicted.\n"
)
FIXTURE_PROTOCOL = (
    "---\nname: fixture-protocol\nscope: code-generator\n---\n\n"
    "- The decisive check is the FIXTURE MEASUREMENT.\n"
)


def _campaign(root: Path) -> dict:
    """A campaign in a directory the user owns, which is where they now live.

    Loam ships no goal and no protocol rule, so these tests supply their own
    rather than asserting against checked-in campaign content -- which is the
    point of `campaign.dir`, and also what kept this suite coupled to one
    protein target.
    """

    policy = root / "campaign-policy"
    (policy / "campaigns").mkdir(parents=True, exist_ok=True)
    (policy / "rules").mkdir(parents=True, exist_ok=True)
    (policy / "campaigns" / "fixture-goal.md").write_text(FIXTURE_GOAL, encoding="utf-8")
    (policy / "rules" / "fixture-protocol.md").write_text(FIXTURE_PROTOCOL, encoding="utf-8")
    return {"dir": str(policy), "goal": "fixture-goal", "rules": ["fixture-protocol"]}


def _validator_context(orchestration, payload: dict) -> str:
    """The first validator task's context, as the child will receive it.

    Read from the arguments the dispatch tool hands to delegate_task, which is
    where the payload now travels. It used to be parsed off the end of the
    directive, because the directive printed the task array for the coordinator
    to copy; it no longer does, and asserting against prose was always a proxy
    for asserting against the arguments.
    """

    arguments = orchestration.canonical_delegation_args(payload)
    assert arguments is not None, "no delegation pending for this payload"
    return arguments["tasks"][0]["context"]


class CampaignInjectionTests(unittest.TestCase):
    def test_validator_payload_carries_the_goal_ahead_of_the_role_definition(self) -> None:
        """Guards the 2026-10-21 regression: policy must not land behind 8.6 KB of boilerplate."""

        orchestration = _load("loam_campaign_validator_inject")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _orchestration_payload(Path(temporary), _campaign(Path(temporary)))
            orchestration.context_for(dict(payload))
            context = _validator_context(orchestration, payload)
        self.assertIn("FIXTURE SUBJECT", context)
        self.assertLess(context.index('"campaign"'), context.index('"validator_instructions"'))
        self.assertLess(context.index('"campaign"'), context.index('"agent_definition"'))
        # The whole point of the ordering: policy lands where the model reads it.
        self.assertLess(context.index('"campaign"'), 2_000)

    def test_validator_payload_omits_a_child_scoped_rule(self) -> None:
        orchestration = _load("loam_campaign_validator_scope")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _orchestration_payload(Path(temporary), _campaign(Path(temporary)))
            orchestration.context_for(dict(payload))
            context = _validator_context(orchestration, payload)

        # Scope is asserted against what the validator actually receives. The
        # directive itself no longer carries any of it.
        self.assertIn("authoritative source", context)  # evidence-integrity, scope: all
        self.assertNotIn("smallest action", context)  # experiment-discipline, code-generator
        self.assertNotIn("FIXTURE MEASUREMENT", context)  # fixture-protocol, code-generator

    def test_no_campaign_block_still_carries_the_invariant_rules(self) -> None:
        """Asserted against the dispatch arguments, not the directive.

        An earlier version looked for `"campaign"` in the whole directive and so
        could only ever pass -- inside the task context the key was escaped as
        `\\"campaign\\"`. Reading the arguments removes the trap entirely.
        """

        orchestration = _load("loam_campaign_validator_absent")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _orchestration_payload(Path(temporary))
            directive = orchestration.context_for(dict(payload))
            context = _validator_context(orchestration, payload)

        self.assertIn("Loam orchestration directive", directive)
        self.assertIn('"campaign"', context)
        self.assertIn("authoritative source", context)  # evidence-integrity
        # No profile named a goal, so none is claimed.
        self.assertNotIn('"goal"', context)
        self.assertNotIn('"goal_name"', context)

    def test_child_context_carries_goal_and_rules_before_the_instruction(self) -> None:
        orchestration = _load("loam_campaign_child_inject")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **_orchestration_payload(Path(temporary), _campaign(Path(temporary))),
                "arguments": {"tasks": [{"goal": "Co-fold the candidate.", "context": "Inputs follow."}]},
            }
            tasks = orchestration._child_tasks(payload, "plan-1", "h-1")

        assert tasks is not None
        context = tasks["tasks"][0]["context"]
        self.assertIn("CAMPAIGN GOAL", context)
        self.assertIn("smallest action", context)  # discipline layer
        self.assertIn("FIXTURE MEASUREMENT", context)  # the user's protocol rule
        self.assertLess(context.index("CAMPAIGN RULES"), context.index("Instruction from validator"))

    def test_reasoner_directive_carries_the_campaign(self) -> None:
        """reasoner.md requires an objective; until now nothing supplied one."""

        orchestration = _load("loam_campaign_reasoner_inject")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {"repo_root": str(ROOT), "loam_campaign": _campaign(Path(temporary))}
            directive = orchestration._reasoner_directive(
                "plan-1",
                [{"hypothesis_id": "h-1", "status": "supported", "evidence_score": 0.5, "summary": "Report."}],
                payload,
            )

        self.assertIn("FIXTURE SUBJECT", directive)
        # Asserted inside the JSON block, because the surrounding prose names
        # "agent_definition" before it appears as a key.
        block = next(line for line in directive.splitlines() if '"agent_definition"' in line and line.startswith("{"))
        self.assertLess(block.index('"campaign"'), block.index('"agent_definition"'))
        self.assertNotIn("smallest action", directive)  # code-generator scope only


class HypothesisBlockExtractionTests(unittest.TestCase):
    def test_campaign_specific_protocol_moved_out_of_the_hook(self) -> None:
        orchestration = _load("loam_campaign_extraction")
        block = orchestration._hypothesis_block({"hypothesis_id": "h-1", "content": "A claim."})

        self.assertIn("A claim.", block)
        # Kept: campaign-independent.
        self.assertIn("NEVER run, score, or reuse", block)
        self.assertIn("complete assignment", block)
        # Moved to the assay rules under .loam/rules/.
        self.assertNotIn("smallest action", block)
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

    # Instruments and metrics: checkable, and what leaked the first time.
    DOMAIN_TERMS = (
        "co-fold", "cofold", "iptm", "ipsae", "pdockq", "plddt", "chai",
        "fasta", ".cif", ".pdb", "proteinmpnn", "batch_status",
        "protein", "uniprot", "residue", "binder", "enzyme", "molecule",
    )
    # Campaign shape: the stricter bar. An always-on rule may not assume the
    # campaign designs anything, since one testing an existing claim does not.
    SHAPE_TERMS = ("design", "candidate", "sequence", "variant", "target")

    def _assert_invariant(self, name: str) -> None:
        body = (ROOT / ".loam" / "rules" / f"{name}.md").read_text(encoding="utf-8")
        body = body.split("---", 2)[2].lower()  # front matter may name enforcement machinery
        for term in self.DOMAIN_TERMS:
            self.assertNotIn(term, body, f"{name}.md names {term!r}; that belongs in a protocol rule")
        for term in self.SHAPE_TERMS:
            self.assertNotIn(term, body, f"{name}.md assumes a campaign shape via {term!r}")

    def test_the_integrity_rule_is_invariant(self) -> None:
        self._assert_invariant("evidence-integrity")

    def test_the_discipline_rule_is_invariant(self) -> None:
        self._assert_invariant("experiment-discipline")

    def test_every_always_on_rule_is_invariant(self) -> None:
        """Catches a rule added later, which is when this bar gets forgotten."""

        names = policy._base_rule_names({"repo_root": str(ROOT)})
        self.assertTrue(names)
        for name in names:
            self._assert_invariant(name)

    def test_the_hook_itself_is_domain_neutral(self) -> None:
        """Inline text reaches every child of every campaign, configured or not."""

        orchestration = _load("loam_rule_layering")
        block = orchestration._hypothesis_block({"hypothesis_id": "h-1", "content": "A claim."}).lower()
        for term in self.DOMAIN_TERMS:
            self.assertNotIn(term, block, f"_hypothesis_block names {term!r}; it must serve any domain")

    def test_the_shape_rule_is_named_rather_than_always_on(self) -> None:
        """The negative tests above are only meaningful if something is excluded.

        Paste-what-was-designed is campaign shape, not an invariant: a campaign
        testing an existing claim designs nothing, so it must not arrive unasked.
        """

        shape = (ROOT / ".loam" / "rules" / "design-provenance.md").read_text(encoding="utf-8")
        self.assertIn("Paste what was designed", shape)
        self.assertNotIn("applies:", shape)
        self.assertNotIn("design-provenance", policy._base_rule_names({"repo_root": str(ROOT)}))


class StarterKitTests(unittest.TestCase):
    """`examples/campaign-starter/` is copied and launched, so it must load.

    The kit's files are read by the hook on the first turn, long after the
    preflight has passed, so a malformed template does not surface until an
    allocation is already running. Loading them here is the only cheap check.
    """

    KIT = ROOT / "examples" / "campaign-starter"

    def test_the_shipped_goal_and_rules_load(self) -> None:
        orchestration = _load("loam_starter_kit")
        payload = {
            "repo_root": str(ROOT),
            "loam_campaign": {
                "goal": "example-campaign",
                "rules": ["example-protocol", "example-constraint"],
                "dir": str(self.KIT),
                "skills": ["chai", "proteinmpnn"],
            },
        }
        context = orchestration.campaign_context(payload)
        self.assertEqual(context.goal_name, "example-campaign")
        self.assertTrue(context.goal, "the goal body is empty")
        names = [rule.name for rule in context.rules]
        self.assertIn("example-protocol", names)
        self.assertIn("example-constraint", names)
        for tier in TIERS:
            self.assertTrue(context.rules_for(tier), f"no rule reaches {tier}")

    def test_the_profile_names_the_files_the_kit_ships(self) -> None:
        """A rename that touched only one of the three places would pass above."""

        profile = (self.KIT / "loam.yaml").read_text(encoding="utf-8")
        self.assertIn("goal: example-campaign", profile)
        self.assertIn("rules: [example-protocol, example-constraint]", profile)
        self.assertTrue((self.KIT / "campaigns" / "example-campaign.md").is_file())


if __name__ == "__main__":
    unittest.main()
