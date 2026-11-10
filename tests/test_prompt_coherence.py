"""The prompts and the hooks that inject text beside them must not disagree.

Every contradiction this guards against was live at once, because each layer was
patched after its own incident with no check that the others still agreed:

- the required report headings drifted apart (`### Workflow` in the prompt, absent
  from the injected list), so a validator could not satisfy both;
- the validator wrote a Credit Assignment table summing to 1.00 that nothing read
  -- `reconcile()` derives credit mechanically from reviewed evidence;
- `reasoner.md` lost its synthesis method and the first half of its report
  template to an unrelated edit, and nothing noticed;
- the batch floor had three different answers across prompt, refusal and nudge;
- the validator was told to re-delegate a child to poll, which `tool_policy`
  refuses at the budget the profile actually sets.

These are cheap text assertions on purpose: the failure mode is drift between
files, which only a test that reads more than one file can catch.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
AGENTS = ROOT / ".loam" / "agents"


def _orchestration():
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(
        "loam_prompt_coherence", HOOKS / "hypothesis_orchestration.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["loam_prompt_coherence"] = module
    spec.loader.exec_module(module)
    return module


def _body(name: str) -> str:
    return (AGENTS / f"{name}.md").read_text(encoding="utf-8")


def _section(body: str, heading: str) -> str:
    """One `## ` section's text, stopping at the next one.

    Splitting on the heading alone runs into the sections below it -- the
    validator's provenance contract has `###` headings of its own that are not
    report sections.
    """

    after = body.split(heading, 1)[1]
    return re.split(r"^## ", after, maxsplit=1, flags=re.MULTILINE)[0]


def _validator_task_text() -> str:
    """The instruction text Loam actually puts into a validator's task."""

    orchestration = _orchestration()
    tasks = orchestration._validator_tasks(
        "plan-1",
        [{"hypothesis": {"hypothesis_id": "h-1", "content": "A claim."}}],
        {
            "repo_root": str(ROOT),
            "loam_execution": {
                "mode": "remote_mcp",
                "tool_name": "mcp__hpc__run_code",
                "function_name": "run_code",
                "max_children": 1,
            },
        },
    )
    return tasks[0]["goal"] + "\n" + tasks[0]["context"]


class ReportHeadingTests(unittest.TestCase):
    def test_the_prompt_and_the_injected_list_require_the_same_sections(self) -> None:
        """The drift that made the contract unsatisfiable."""

        injected = _validator_task_text()
        # The headings Loam names in the task, in the order it names them.
        named = re.findall(r"### ([A-Z][A-Za-z ]+?)(?=[;.]|$)", injected, re.MULTILINE)
        # The headings the role definition's report template declares.
        template = _section(_body("hypothesis-validator"), "## Required Report to Parent")
        declared = re.findall(r"^### (.+)$", template, re.MULTILINE)

        self.assertEqual(declared, [heading.strip() for heading in named])

    def test_the_score_heading_the_parser_looks_for_is_one_of_them(self) -> None:
        """The one heading a hook actually parses, so it cannot be renamed freely."""

        orchestration = _orchestration()
        template = _section(_body("hypothesis-validator"), "## Required Report to Parent")
        self.assertIn("### Evidence Score", template)
        self.assertEqual(
            orchestration._evidence_score("### Evidence Score\n+0.75\n"), 0.75
        )


class CreditTests(unittest.TestCase):
    def test_no_tier_is_asked_to_assign_credit_by_hand(self) -> None:
        """`reconcile()` computes credit from reviewed evidence; a self-reported
        table fed nothing and competed with the Evidence Score for attention."""

        for name in ("hypothesis-validator", "reasoner", "evidence-reviewer"):
            with self.subTest(agent=name):
                self.assertNotIn("### Credit Assignment", _body(name))
                self.assertNotIn("must total `1.00`", _body(name))
        self.assertNotIn("Credit Assignment", _validator_task_text())

    def test_credit_is_still_derived_mechanically(self) -> None:
        """Removing the table must not remove the ledger it was confused with."""

        provenance = (ROOT / ".loam" / "reconcile" / "hypothesis_provenance.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("credit_assigned", provenance)
        self.assertIn("accepted_evidence_rating_delta", provenance)


class ReasonerTests(unittest.TestCase):
    """f62c127 deleted these while rewriting the memory section beside them."""

    def setUp(self) -> None:
        self.body = _body("reasoner")

    def test_the_synthesis_method_is_present(self) -> None:
        self.assertIn("## Synthesis Workflow", self.body)
        # The distinction the whole role turns on.
        self.assertIn("ELO never substitutes for evidence", self.body)
        self.assertIn("ELO score** is comparative tournament performance", self.body)

    def test_the_report_template_is_whole_and_not_nested_under_memory(self) -> None:
        self.assertIn("## Required Report to Parent", self.body)
        for section in (
            "### Experiment State",
            "### Thread Synthesis",
            "### Productive Regions",
            "### Excluded or Deprioritized Regions",
            "### Exploration Priorities",
            "### Next Jnana Tournament Inputs",
            "### Memory Update",
            "### Decision Rationale",
        ):
            with self.subTest(section=section):
                self.assertIn(section, self.body)
                # Every report section must sit under the report heading, not
                # orphaned inside "### Where durable memory goes".
                self.assertGreater(
                    self.body.index(section),
                    self.body.index("## Required Report to Parent"),
                )

    def test_an_unobserved_thread_is_untested_not_refuted(self) -> None:
        """The directive tells the reasoner this; the prompt has to agree."""

        self.assertIn("UNTESTED", self.body)


class BatchFloorTests(unittest.TestCase):
    """Prompt, refusal message and nudge all describe one floor."""

    def _nudge_and_refusal(self) -> list[str]:
        source = (HOOKS / "hypothesis_orchestration.py").read_text(encoding="utf-8")
        return [source]

    def test_nothing_promises_what_the_server_refuses(self) -> None:
        texts = [_body("code-generator"), *self._nudge_and_refusal()]
        for text in texts:
            for blessing in (
                "one-row batch is fine",
                "a one-row batch is correct",
                "even for a single candidate",
            ):
                with self.subTest(blessing=blessing):
                    self.assertNotIn(blessing, text)

    def test_the_floor_is_stated_as_a_principle_without_a_number(self) -> None:
        """HPC_MIN_BATCH_ROWS is configurable, so a literal here would drift."""

        body = _body("code-generator")
        self.assertIn("refuses a degenerate submission", body)
        self.assertNotIn("32 rows", body)

    def test_staging_is_write_files_everywhere(self) -> None:
        """run_code was advertised as the file writer beside the line replacing it."""

        orchestration = _orchestration()
        context = orchestration._child_context(
            "Fold it.",
            {
                "mode": "remote_mcp",
                "tool_name": "mcp__hpc__run_code",
                "function_name": "run_code",
                "tool_prefix": "mcp__hpc__",
            },
            "/work",
        )
        self.assertIn("write_files", context)
        self.assertNotIn("also used to write the", context)
        # The unadvertised registry must not reappear in the run_batch signature.
        self.assertNotIn("tool=<name>", context)


class WaitingTests(unittest.TestCase):
    def test_the_validator_polls_its_own_batch_instead_of_spending_a_child(self) -> None:
        """With max_children=1 and no replacement for a child that reported,
        re-delegating the poll is a blocked call."""

        body = _body("hypothesis-validator")
        self.assertIn("Poll it yourself", body)
        self.assertNotIn("re-delegate a fresh child to CONTINUE POLLING", body)
        # And the rule that makes it so is stated, not left to be discovered.
        self.assertIn("DIED without reporting", body)


if __name__ == "__main__":
    unittest.main()
