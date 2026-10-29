"""Only the tagged reasoner may use FALDA tools, only in its own pool and workflow."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
for directory in (HOOKS, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import falda_campaign  # noqa: E402
import falda_memory  # noqa: E402
from falda_fake import POOL, FakeTransport, falda_block  # noqa: E402

REMEMBER = "mcp__falda__falda_remember"
RECALL = "mcp__falda__falda_recall"
GOOD = {"content": "Float checks mislead at the 2.4 boundary.", "type": "fact", "tags": ["workflow:wf-1", "negative"]}


def _load_orchestration(name: str):
    spec = importlib.util.spec_from_file_location(name, HOOKS / "hypothesis_orchestration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FaldaToolGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = Path(temporary.name)
        self.orchestration = _load_orchestration(f"orchestration_guard_{self._testMethodName}")
        self.payload = {
            "repo_root": str(ROOT),
            "workflow_id": "wf-1",
            "loam_run_id": "run-1",
            "loam_state_dir": str(self.state),
            "loam_hypothesis_orchestration": True,
            "loam_falda": falda_block(),
        }
        for role, session in (("reasoner", "rs-1"), ("hypothesis-validator", "vs-1"), ("code-generator", "cs-1")):
            self.orchestration.record_start(
                {
                    **self.payload,
                    "child_subagent_id": f"{role}-child",
                    "child_session_id": session,
                    "child_goal": f"Work [loam-agent-role={role}] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]",
                }
            )

    def _call(self, session: str, tool: str, arguments, **overrides):
        return self.orchestration.tool_policy(
            {**self.payload, **overrides, "session_id": session, "tool_name": tool, "arguments": arguments}
        )

    def _calls(self) -> list[dict]:
        return [json.loads(line) for line in (self.state / "falda-calls.jsonl").read_text().splitlines()]

    def test_reasoner_may_write_with_the_workflow_tag_and_it_is_logged(self) -> None:
        self.assertIsNone(self._call("rs-1", REMEMBER, GOOD))
        row = self._calls()[-1]
        self.assertEqual(
            (row["op"], row["tool"], row["role"], row["plan_id"], row["allowed"]),
            ("mcp_tool", REMEMBER, "reasoner", "plan-1", True),
        )

    def test_other_roles_and_the_coordinator_are_blocked(self) -> None:
        for session in ("vs-1", "cs-1", "coordinator-1"):
            with self.subTest(session=session):
                result = self._call(session, REMEMBER, GOOD)
                self.assertEqual(result["action"], "block")
                self.assertIn("Only the campaign's tagged reasoner", result["message"])
        self.assertEqual([row["allowed"] for row in self._calls()], [False, False, False])

    def test_coordinator_is_blocked_even_without_orchestration(self) -> None:
        result = self._call("coordinator-1", REMEMBER, GOOD, loam_hypothesis_orchestration=False)
        self.assertEqual(result["action"], "block")

    def test_write_without_the_workflow_tag_is_blocked_with_a_fix(self) -> None:
        for tags in (["negative"], "workflow:wf-1", None, ["workflow:WF-1"]):
            with self.subTest(tags=tags):
                result = self._call("rs-1", REMEMBER, {**GOOD, "tags": tags})
                self.assertEqual(result["action"], "block")
                self.assertIn('Add the tag "workflow:wf-1"', result["message"])

    def test_write_tagged_for_another_workflow_is_blocked_with_a_fix(self) -> None:
        result = self._call("rs-1", REMEMBER, {**GOOD, "tags": ["workflow:wf-1", "workflow:wf-2"]})
        self.assertEqual(result["action"], "block")
        self.assertIn('Remove the tag "workflow:wf-2"', result["message"])

    def test_write_with_hook_reserved_tags_is_blocked_with_a_fix(self) -> None:
        for reserved in ("source:validator-report", "run:run-9", "plan:plan-9", "hypothesis:h-9"):
            with self.subTest(tag=reserved):
                result = self._call("rs-1", REMEMBER, {**GOOD, "tags": ["workflow:wf-1", reserved]})
                self.assertEqual(result["action"], "block")
                self.assertIn(f'Remove the tag "{reserved}"', result["message"])

    def test_disallowed_tool_message_names_all_allowed_tools(self) -> None:
        result = self._call("rs-1", "mcp__falda__falda_forget", {})
        for name in ("falda_remember", "falda_recall", "falda_mark_used"):
            self.assertIn(name, result["message"])

    def test_pinned_writes_are_blocked(self) -> None:
        result = self._call("rs-1", REMEMBER, {**GOOD, "pinned": True})
        self.assertEqual(result["action"], "block")
        self.assertIn("pinned", result["message"])
        self.assertIsNone(self._call("rs-1", REMEMBER, {**GOOD, "pinned": False}))

    def test_pool_must_be_the_campaign_pool(self) -> None:
        blocked = self._call("rs-1", REMEMBER, {**GOOD, "pool": "self"})
        self.assertEqual(blocked["action"], "block")
        self.assertIn(POOL, blocked["message"])
        self.assertIsNone(self._call("rs-1", REMEMBER, {**GOOD, "pool": POOL.upper()}))

    def test_recall_must_be_scoped_to_the_workflow(self) -> None:
        self.assertEqual(self._call("rs-1", RECALL, {"query": "q"})["action"], "block")
        self.assertIsNone(self._call("rs-1", RECALL, {"query": "q", "require_tags": ["workflow:wf-1"]}))

    def test_tools_outside_the_allow_list_are_blocked(self) -> None:
        for tool in ("falda_forget", "falda_distill", "falda_stream_add"):
            with self.subTest(tool=tool):
                result = self._call("rs-1", f"mcp__falda__{tool}", {"tags": ["workflow:wf-1"]})
                self.assertIn("not permitted in a Loam campaign", result["message"])

    def test_disabled_tools_or_no_block_blocks_everyone(self) -> None:
        for falda in (falda_block(reasoner_tools=False), None):
            with self.subTest(falda=falda):
                result = self._call("rs-1", REMEMBER, GOOD, loam_falda=falda)
                self.assertIn("not enabled", result["message"])

    def test_reasoner_keeps_no_other_tool(self) -> None:
        result = self._call("rs-1", "terminal", {"command": "ls"})
        self.assertEqual(result["message"], "reasoner has no tool capability; return the synthesis to the harness")

    def test_torn_state_line_does_not_unblock_or_crash_the_guard(self) -> None:
        # develop's row reader skips torn lines, so the reasoner's start row still decides.
        with (self.state / "hypothesis-orchestration.jsonl").open("a") as handle:
            handle.write("not json\n")
        self.assertIsNone(self._call("rs-1", REMEMBER, GOOD))
        self.assertEqual(self._call("coordinator-1", REMEMBER, GOOD)["action"], "block")

    def test_guard_failure_blocks_falda_calls(self) -> None:
        with patch.object(self.orchestration, "falda_tool_policy", side_effect=RuntimeError("boom")):
            result = self._call("rs-1", REMEMBER, GOOD)
        self.assertEqual(result["action"], "block")
        self.assertIn("could not verify", result["message"])

    def test_pre_tool_script_blocks_falda_and_leaves_other_tools(self) -> None:
        (self.state / "run-manifest.json").write_text("{", encoding="utf-8")  # bookkeeping failure is isolated
        for tool, expect_block in ((REMEMBER, True), ("terminal", False)):
            with self.subTest(tool=tool):
                completed = subprocess.run(
                    [sys.executable, str(HOOKS / "pre-tool.py")],
                    input=json.dumps({**self.payload, "event": "pre_tool_call", "session_id": "coordinator-1", "tool_name": tool, "arguments": GOOD}),
                    text=True,
                    capture_output=True,
                    cwd=str(ROOT),
                    check=False,
                )
                self.assertEqual(completed.returncode, 0)
                if expect_block:
                    self.assertIn("FALDA", json.loads(completed.stdout)["message"])
                else:
                    self.assertNotIn("FALDA", completed.stdout)  # other tools follow develop's own policy

    def test_pre_tool_fails_closed_for_falda_when_the_policy_raises(self) -> None:
        spec = importlib.util.spec_from_file_location("pre_tool_falda", HOOKS / "pre-tool.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def _boom(payload):
            raise RuntimeError("state unreadable")

        module.tool_policy = _boom
        for tool, expect_block in ((REMEMBER, True), ("terminal", False)):
            with self.subTest(tool=tool):
                request = {**self.payload, "event": "pre_tool_call", "session_id": "rs-1", "tool_name": tool, "arguments": GOOD}
                with patch.object(sys, "stdin", io.StringIO(json.dumps(request))), patch.object(sys, "stdout", io.StringIO()) as out:
                    self.assertEqual(module.main(), 0)
                if expect_block:
                    self.assertEqual(json.loads(out.getvalue())["message"], "Loam could not verify this FALDA call; continue without FALDA tools.")
                else:
                    self.assertNotIn("FALDA", out.getvalue())

    def test_one_tag_is_used_for_writes_recalls_and_guards(self) -> None:
        workflow_id = "Wf-Mixed.Case_1"
        payload = {**self.payload, "workflow_id": workflow_id, "loam_execution": {"mode": "local"}}
        tag = f"workflow:{workflow_id}"
        atom, _ = falda_campaign.outcome_atom(
            payload,
            {"plan_id": "plan-1", "run_id": "run-1", "hypothesis_id": "h-1", "evidence_score": 0.6, "summary": "### Outcome\nsupported"},
        )
        self.assertIn(tag, atom["tags"])
        fake = FakeTransport()
        falda_memory.set_transport(fake)
        self.addCleanup(falda_memory.set_transport, None)
        with patch.dict(os.environ, {"FALDA_TOKEN": "falda_test"}):
            falda_memory.recall(payload, "q", consumer="t", budget=100)
        self.assertEqual(fake.bodies("/recall")[0]["require_tags"], [tag])
        reasoner = {"role": "reasoner", "plan_id": "plan-1"}
        self.assertIsNone(falda_campaign.falda_tool_policy(payload, REMEMBER, {**GOOD, "tags": [tag]}, reasoner))
        self.assertIsNotNone(falda_campaign.falda_tool_policy(payload, REMEMBER, {**GOOD, "tags": [tag.lower()]}, reasoner))


if __name__ == "__main__":
    unittest.main()
