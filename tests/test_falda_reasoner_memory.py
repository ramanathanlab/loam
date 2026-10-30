"""The reasoner directive carries a memory interface, recalled once per plan."""

from __future__ import annotations

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
for directory in (HOOKS, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import falda_memory  # noqa: E402
from falda_fake import POOL, FakeTransport, falda_block  # noqa: E402

RECALLED = "## What did not work\n[m1] prior negative result"


def _load_orchestration(name: str):
    spec = importlib.util.spec_from_file_location(name, HOOKS / "hypothesis_orchestration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReasonerMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {"FALDA_TOKEN": "falda_test"})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(falda_memory.set_transport, None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.orchestration = _load_orchestration(f"orchestration_reasoner_{self._testMethodName}")

    def _payload(self, falda=None) -> dict:
        state_file = self.root / "jnana.json"
        reports = self.root / "reports"
        reports.mkdir(exist_ok=True)
        state_file.write_text(
            json.dumps({"hypotheses": [{"hypothesis_id": "h-1", "content": "A"}, {"hypothesis_id": "h-2", "content": "B"}]}),
            encoding="utf-8",
        )
        (reports / "tournament-001.json").write_text(
            json.dumps(
                {
                    "workflow_id": "research-v1",
                    "result": {
                        "status": "completed",
                        "top_hypotheses": [{"hypothesis_id": "h-1", "elo_rating": 1210.0}, {"hypothesis_id": "h-2", "elo_rating": 1190.0}],
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "repo_root": str(ROOT),
            "workflow_id": "research-v1",
            "session_id": "coordinator-1",
            "user_message": "Investigate codon bias.",
            "loam_run_id": "run-1",
            "loam_state_dir": str(self.root / "loam"),
            "loam_hypothesis_tournament": {"state_file": str(state_file), "reports_dir": str(reports)},
            "loam_execution": {"mode": "local", "runtime": "python3"},
            "loam_falda": falda or falda_block(),
        }

    def _complete_validators(self, payload: dict) -> str:
        self.orchestration.context_for(payload)
        plan_id = self.orchestration._plan_id(self.orchestration._latest_report(payload)[0], payload)
        for index, hypothesis_id in enumerate(("h-1", "h-2"), start=1):
            child = {
                **payload,
                "child_subagent_id": f"validator-{index}",
                "child_session_id": f"validator-session-{index}",
                "child_goal": (
                    f"Validate [loam-agent-role=hypothesis-validator] [loam-plan-id={plan_id}] "
                    f"[loam-hypothesis-id={hypothesis_id}]"
                ),
            }
            self.orchestration.record_start(child)
            self.orchestration.record_stop(
                {
                    **child,
                    "child_status": "completed",
                    "child_summary": f"### Hypothesis\n{hypothesis_id} claim\n### Outcome\nsupported\n### Evidence Score\n+0.75 — ok",
                }
            )
        return plan_id

    def test_directive_carries_the_memory_block_before_the_threads(self) -> None:
        fake = FakeTransport(recall=[(200, {"recall_id": "r-1", "context": RECALLED, "items": [{"id": "a-1"}]})])
        falda_memory.set_transport(fake)
        payload = self._payload()
        plan_id = self._complete_validators(payload)
        directive = self.orchestration.context_for(payload)
        self.assertIn("Delegate exactly one reasoner", directive)
        self.assertLess(directive.index("Memory interface:\n- Shared FALDA pool"), directive.index('"validator_report"'))
        for expected in (POOL, "workflow:research-v1", "mcp__falda__falda_remember", "[m1] prior negative result"):
            self.assertIn(expected, directive)
        [body] = fake.bodies("/recall")
        self.assertEqual(body["require_tags"], ["workflow:research-v1"])
        self.assertIn("h-1 claim — supported", body["query"])
        row = [json.loads(line) for line in (self.root / "loam" / "falda-calls.jsonl").read_text().splitlines()][-1]
        self.assertEqual((row["consumer"], row["plan_id"]), ("reasoner", plan_id))

    def test_reasoner_recall_is_cached_per_plan_even_on_failure(self) -> None:
        fake = FakeTransport(recall=[(503, {"error": "unavailable"})])
        falda_memory.set_transport(fake)
        payload = self._payload()
        self._complete_validators(payload)
        first = self.orchestration.context_for(payload)
        second = self.orchestration.context_for(payload)
        canonical = self.orchestration.canonical_delegation_args(payload)
        self.assertEqual(len(fake.bodies("/recall")), 1)
        # A failed recall adds nothing: no "unavailable" notice in the reasoner's prompt.
        self.assertNotIn("Recall was unavailable", first)
        self.assertNotIn("Recalled campaign memory", first)
        self.assertEqual(first, second)
        self.assertIn("Memory interface:", canonical["context"])

    def test_the_reasoner_task_carries_no_coordinator_dispatch_instructions(self) -> None:
        # The reasoner read "call loam_dispatch_round now" in its own task, tried it, and returned
        # "Dispatched the reasoner" instead of a synthesis (Aurora, 2026-10-31).
        falda_memory.set_transport(FakeTransport())
        payload = self._payload()
        self._complete_validators(payload)
        directive = self.orchestration.context_for(payload)
        task = self.orchestration.canonical_delegation_args(payload)["context"]
        self.assertIn(self.orchestration.DISPATCH_TOOL, directive)
        for coordinator_only in (self.orchestration.DISPATCH_TOOL, "Delegate exactly one reasoner", "Tag its goal"):
            self.assertNotIn(coordinator_only, task)
        for expected in ('"agent_definition"', "h-1 claim", "Memory interface:", "unvalidated orchestration input"):
            self.assertIn(expected, task)

    def test_corrupt_cache_file_does_not_lose_the_directive(self) -> None:
        falda_memory.set_transport(FakeTransport())
        payload = self._payload()
        self._complete_validators(payload)
        cache = self.root / "loam" / "falda-reasoner-memory.jsonl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"\xff\xfe\x80 not utf-8\n")
        self.assertIn("Delegate exactly one reasoner", self.orchestration.context_for(payload))

    def test_recalled_context_is_cut_to_the_budget_on_a_line_boundary(self) -> None:
        context = "## What did not work\n" + "\n".join(f"[m{i}] {'x' * 40}" for i in range(50))
        falda_memory.set_transport(FakeTransport(recall=[(200, {"recall_id": "r-1", "context": context, "items": []})]))
        payload = self._payload(falda_block(recall_budget_chars=200))
        self._complete_validators(payload)
        directive = self.orchestration.context_for(payload)
        lines = [line for line in directive.splitlines() if line.startswith("[m")]
        self.assertTrue(lines)
        self.assertLess(len(lines), 50)
        self.assertLessEqual(len("## What did not work\n" + "\n".join(lines)), 200)
        self.assertTrue(all(line.endswith("x" * 40) for line in lines))

    def test_both_switches_off_means_no_block_and_no_recall(self) -> None:
        fake = FakeTransport()
        falda_memory.set_transport(fake)
        payload = self._payload(falda_block(recall_reasoner=False, reasoner_tools=False))
        self._complete_validators(payload)
        directive = self.orchestration.context_for(payload)
        self.assertNotIn("Memory interface:\n- Shared FALDA pool", directive)
        self.assertEqual(fake.bodies("/recall"), [])

    def test_tools_without_recall_names_the_pool_but_does_not_recall(self) -> None:
        fake = FakeTransport()
        falda_memory.set_transport(fake)
        payload = self._payload(falda_block(recall_reasoner=False))
        self._complete_validators(payload)
        directive = self.orchestration.context_for(payload)
        self.assertIn("You may write", directive)
        # Hermes may have failed to connect the FALDA server at launch; three calls to a missing tool
        # stop a subagent, so the reasoner is told to skip rather than retry.
        self.assertIn("If mcp__falda__falda_remember is not in your tool list, skip FALDA", directive)
        self.assertNotIn("Recalled campaign memory", directive)
        self.assertEqual(fake.bodies("/recall"), [])

    def test_recall_without_tools_says_writing_is_disabled(self) -> None:
        falda_memory.set_transport(FakeTransport(recall=[(200, {"context": RECALLED, "items": [{"id": "a"}]})]))
        payload = self._payload(falda_block(reasoner_tools=False))
        self._complete_validators(payload)
        directive = self.orchestration.context_for(payload)
        self.assertIn("Writing is disabled for this run", directive)
        self.assertIn("[m1] prior negative result", directive)

    def test_empty_recall_without_tools_adds_no_block(self) -> None:
        falda_memory.set_transport(FakeTransport())
        payload = self._payload(falda_block(reasoner_tools=False))
        self._complete_validators(payload)
        directive = self.orchestration.context_for(payload)
        self.assertNotIn("Memory interface:\n- Shared FALDA pool", directive)
        self.assertNotIn("nothing recorded yet", directive)

    def test_reasoner_block_omits_recall_on_timeout(self) -> None:
        import time

        def _hang(url, headers, body, timeout):
            time.sleep(5)
            return 200, json.dumps({"context": RECALLED}).encode()

        falda_memory.set_transport(_hang)
        payload = self._payload()
        self._complete_validators(payload)
        start = time.monotonic()
        directive = self.orchestration.context_for(payload)
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertIn("Memory interface:", directive)
        self.assertNotIn("[m1]", directive)
        self.assertNotIn("Recalled campaign memory", directive)

    def test_no_mcp_key_at_launch_means_no_tool_lines(self) -> None:
        falda_memory.set_transport(FakeTransport())
        payload = self._payload()
        self._complete_validators(payload)
        with patch.dict(os.environ, {"LOAM_FALDA_MCP": "0"}):
            directive = self.orchestration.context_for(payload)
        self.assertNotIn("You may write", directive)
        self.assertNotIn("Memory interface:\n- Shared FALDA pool", directive)

    def test_cairn_delegation_decoration_keeps_the_falda_block(self) -> None:
        import cairn_brief

        falda_memory.set_transport(FakeTransport(recall=[(200, {"context": RECALLED, "items": [{"id": "a"}]})]))
        payload = self._payload()
        self._complete_validators(payload)
        args = self.orchestration.canonical_delegation_args(payload)
        cairn = {
            "enabled": True, "base_url": "https://cairn.invalid", "project": "loam-test-project", "read_side": False,
            "rate_agent_reports": True, "rate_roles": ["reasoner"], "timeout_seconds": 2.0,
            "executor": "mcp://hpc#run_code", "entities": {"mcp://hpc#run_code": {"patterns": [], "task_tags": ["code_exec"]}},
        }
        decorated = cairn_brief.decorate_delegation(dict(args), {**payload, "loam_cairn": cairn})
        self.assertIn("Memory interface:", json.dumps(decorated))
        self.assertIn("[m1] prior negative result", json.dumps(decorated))

    def test_recall_preamble_is_dropped_and_citation_is_requested(self) -> None:
        context = (
            "Memories below come from earlier sessions. Verify current state with your tools before acting on it.\n"
            "Filtered to memories tagged workflow:research-v1.\n\n" + RECALLED
        )
        falda_memory.set_transport(FakeTransport(recall=[(200, {"context": context, "items": [{"id": "a"}]})]))
        payload = self._payload()
        self._complete_validators(payload)
        directive = self.orchestration.context_for(payload)
        self.assertNotIn("with your tools", directive)
        self.assertIn("[m1] prior negative result", directive)
        self.assertIn("cite the recalled items you used by their [mN] tags", directive)

    def test_reasoner_prompt_has_the_stewardship_section(self) -> None:
        text = (ROOT / ".loam" / "agents" / "reasoner.md").read_text(encoding="utf-8")
        self.assertIn("## Shared campaign memory", text)
        for expected in ("falda_remember", "`negative`", "`best-known`", "`hypothesis`", "`coverage`", "only what is new", "never count as an existing", "Claim only what was tested", "Do not pin"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
