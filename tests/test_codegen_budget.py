"""One code-generator per validator, with a replacement only if one dies.

The budget is a LIFETIME TOTAL, not a concurrency limit: tool_policy counts
`agent_started` ledger events and never decrements them when a child finishes.
That surprises people, so it is pinned here.

It was hardcoded to 2 in three independent places -- tool_policy's comparison,
the validator's child_template, and the coordinator's directive -- while
`execution.max_children` was read by loam_config and DROPPED, so a profile
asking for 1 was silently ignored and the three could disagree.

With max_children=1 a validator gets ONE child and must give it the whole sweep
up front. A child that goes DOWN (crash, provider abort, dead endpoint) earns
one replacement, because otherwise an infrastructure failure silently costs the
round that hypothesis. A child that finished AND reported -- including an honest
`blocked` -- earns nothing: it produced evidence.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"

PLAN = "plan_c0ffee"
HYPOTHESIS = "11111111-2222-3333-4444-555555555555"
WORKDIR_ROOT = "/flare/PROJ/user/hpc-run/work"


def _load(name: str):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, HOOKS / "hypothesis_orchestration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CodegenBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.o = _load("loam_codegen_budget")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name)

    def _payload(self, max_children: int) -> dict:
        return {
            "repo_root": str(ROOT),
            "workflow_id": "research-v1",
            "loam_run_id": "run-1",
            "loam_state_dir": str(self.state),
            "loam_hypothesis_orchestration": True,
            "loam_execution": {
                "mode": "remote_mcp",
                "tool_name": "mcp__hpc__run_code",
                "function_name": "run_code",
                "workdir_root": WORKDIR_ROOT,
                "max_children": max_children,
            },
        }

    def _start(self, payload: dict, session: str, role: str) -> None:
        self.o.record_start({
            **payload,
            "child_session_id": session,
            "child_subagent_id": f"sa-{session}",
            "child_goal": (f"work [loam-agent-role={role}] [loam-plan-id={PLAN}] "
                           f"[loam-hypothesis-id={HYPOTHESIS}]"),
        })

    def _stop(self, payload: dict, session: str, status: str, summary: str) -> None:
        self.o.record_stop({
            **payload,
            "child_session_id": session,
            "child_subagent_id": f"sa-{session}",
            "child_status": status,
            "child_summary": summary,
        })

    def _delegate(self, payload: dict, session: str) -> dict | None:
        return self.o.tool_policy({
            **payload, "session_id": session, "tool_name": "delegate_task",
            "arguments": {"tasks": [{
                "goal": (f"Run the sweep [loam-agent-role=code-generator] "
                         f"[loam-plan-id={PLAN}] [loam-hypothesis-id={HYPOTHESIS}]"),
                "context": f"Write outputs under {WORKDIR_ROOT}/{PLAN}/{HYPOTHESIS}.",
            }]},
        })

    # --- the budget is a total, not a concurrency limit --------------------

    def test_one_child_is_allowed_then_the_budget_is_spent(self) -> None:
        payload = self._payload(1)
        self._start(payload, "validator-1", "hypothesis-validator")
        self.assertIsNone(self._delegate(payload, "validator-1"), "the first child must be allowed")
        self._start(payload, "codegen-1", "code-generator")
        verdict = self._delegate(payload, "validator-1")
        assert verdict is not None, "a second child must be refused at max_children=1"
        self.assertEqual(verdict["action"], "block")
        self.assertIn("budget for this hypothesis is 1", verdict["message"])

    def test_a_finished_child_does_not_free_its_slot(self) -> None:
        # The surprising property: completion does NOT return the slot.
        payload = self._payload(1)
        self._start(payload, "validator-1", "hypothesis-validator")
        self._start(payload, "codegen-1", "code-generator")
        self._stop(payload, "codegen-1", "ok", "### Result\nfolded 2000 candidates")
        verdict = self._delegate(payload, "validator-1")
        assert verdict is not None, "a reported child must not earn a replacement"
        self.assertIn("return your report now", verdict["message"])

    def test_the_refusal_says_tool_calls_are_not_limited(self) -> None:
        # A validator that reads "no more children" must not conclude "no more
        # work": one child may run the whole sweep over many sequential calls.
        payload = self._payload(1)
        self._start(payload, "validator-1", "hypothesis-validator")
        self._start(payload, "codegen-1", "code-generator")
        verdict = self._delegate(payload, "validator-1")
        assert verdict is not None
        message = verdict["message"]
        self.assertIn("limits CHILDREN, not tool calls", message)
        self.assertIn("run_batch", message)
        self.assertIn("whole sweep", message)

    # --- replacement, only for a child that went down ----------------------

    def test_a_child_that_died_without_reporting_earns_one_replacement(self) -> None:
        payload = self._payload(1)
        self._start(payload, "validator-1", "hypothesis-validator")
        self._start(payload, "codegen-1", "code-generator")
        self._stop(payload, "codegen-1", "error", "")      # crashed, said nothing
        self.assertIsNone(self._delegate(payload, "validator-1"),
                          "a dead child must earn a replacement")

    def test_only_one_replacement_however_many_die(self) -> None:
        # Without a ceiling a crash loop would mint a child every time and one
        # hypothesis could eat the round's whole agent budget.
        payload = self._payload(1)
        self._start(payload, "validator-1", "hypothesis-validator")
        for n in (1, 2):
            self._start(payload, f"codegen-{n}", "code-generator")
            self._stop(payload, f"codegen-{n}", "error", "")
        verdict = self._delegate(payload, "validator-1")
        assert verdict is not None, "the second death must not earn a second replacement"
        self.assertIn("budget for this hypothesis is 2", verdict["message"])

    def test_an_honest_blocked_report_is_not_replaceable(self) -> None:
        payload = self._payload(1)
        self._start(payload, "validator-1", "hypothesis-validator")
        self._start(payload, "codegen-1", "code-generator")
        self._stop(payload, "codegen-1", "blocked", "### Status\nblocked: UniProt unreachable")
        verdict = self._delegate(payload, "validator-1")
        assert verdict is not None, "a child that reported `blocked` produced evidence"
        self.assertIn("return your report now", verdict["message"])

    def test_a_crash_with_a_summary_is_not_replaceable(self) -> None:
        # It told us something, so the validator can reason about it.
        payload = self._payload(1)
        self._start(payload, "validator-1", "hypothesis-validator")
        self._start(payload, "codegen-1", "code-generator")
        self._stop(payload, "codegen-1", "error", "### Result\npartial: 12 of 32 rows folded")
        self.assertIsNotNone(self._delegate(payload, "validator-1"))

    # --- the config drives it ---------------------------------------------

    def test_a_larger_budget_from_config_is_honoured(self) -> None:
        payload = self._payload(3)
        self._start(payload, "validator-1", "hypothesis-validator")
        for n in (1, 2):
            self._start(payload, f"codegen-{n}", "code-generator")
        self.assertIsNone(self._delegate(payload, "validator-1"),
                          "max_children=3 must allow a third child")

    def test_a_missing_or_zero_max_children_falls_back_to_one(self) -> None:
        for value in (None, 0):
            with self.subTest(max_children=value):
                payload = self._payload(1)
                execution = dict(payload["loam_execution"])
                if value is None:
                    execution.pop("max_children")
                else:
                    execution["max_children"] = value
                payload["loam_execution"] = execution
                self._start(payload, "validator-1", "hypothesis-validator")
                self._start(payload, "codegen-1", "code-generator")
                verdict = self._delegate(payload, "validator-1")
                assert verdict is not None, "an unset budget must not mean unlimited"
                self.assertIn("budget for this hypothesis is 1", verdict["message"])

    def test_another_hypothesis_budget_is_independent(self) -> None:
        # The count is scoped to plan+hypothesis, so one validator spending its
        # child must not starve another's.
        payload = self._payload(1)
        self._start(payload, "validator-1", "hypothesis-validator")
        self._start(payload, "codegen-1", "code-generator")
        other = dict(payload)
        self.o.record_start({
            **other,
            "child_session_id": "validator-2",
            "child_subagent_id": "sa-validator-2",
            "child_goal": ("work [loam-agent-role=hypothesis-validator] "
                           f"[loam-plan-id={PLAN}] [loam-hypothesis-id=99999999-aaaa-bbbb-cccc-dddddddddddd]"),
        })
        verdict = self.o.tool_policy({
            **other, "session_id": "validator-2", "tool_name": "delegate_task",
            "arguments": {"tasks": [{
                "goal": ("Run the sweep [loam-agent-role=code-generator] "
                         f"[loam-plan-id={PLAN}] [loam-hypothesis-id=99999999-aaaa-bbbb-cccc-dddddddddddd]"),
                "context": f"Write outputs under {WORKDIR_ROOT}/{PLAN}/99999999-aaaa-bbbb-cccc-dddddddddddd.",
            }]},
        })
        self.assertIsNone(verdict, "a second validator must have its own budget")


if __name__ == "__main__":
    unittest.main()
