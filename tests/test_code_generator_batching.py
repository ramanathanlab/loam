"""A code-generator must batch the science instead of running it serially.

Across 2026-10-31, 11-01 and 11-06, 62 code-generator children made 524
`run_code` calls and 15 `run_batch` calls; only five children ever batched, so a
64-node allocation ran at about 1% of its tiles. `code-generator.md` had told
them to batch from the start.

The reason it never took is in the instruction itself: `run_batch` takes a
script_path and an inputs_path that must already exist on the cluster, and no
HPC tool writes a file, so the only way to create them is to run code that
does. "`run_code` ONLY as a diagnostic AFTER a batch has errored" could not be
obeyed by anyone who had to stage first, so it was disregarded whole.

So the rule now names staging as the expected first step, and the budget below
enforces the part that was always the point: the science does not run inside
`run_code`.
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

REMOTE_GENERATED = {
    "mode": "remote_mcp",
    "tool_name": "mcp__hpc__run_code",
    "function_name": "run_code",
    "endpoint": "https://compute.example/mcp",
    "input_schema": "run-python-v1",
    "runtime": "python3.12",
}

RUN_CODE = "mcp__hpc__run_code"
RUN_BATCH = "mcp__hpc__run_batch"


def _load(name: str, filename: str = "hypothesis_orchestration.py"):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, HOOKS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RunCodeBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestration = _load("loam_batching_policy")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_dir = Path(self._tmp.name)
        self.payload = {
            "repo_root": str(ROOT),
            "workflow_id": "research-v1",
            "loam_run_id": "run-1",
            "loam_state_dir": str(self.state_dir),
            "loam_hypothesis_orchestration": True,
            "loam_execution": REMOTE_GENERATED,
        }
        self._start("codegen-1")

    def _start(self, session: str) -> None:
        self.orchestration.record_start(
            {
                **self.payload,
                "child_session_id": session,
                "child_subagent_id": f"sa-{session}",
                "child_goal": (
                    "Compute it [loam-agent-role=code-generator] "
                    "[loam-plan-id=plan_1] [loam-hypothesis-id=h-1]"
                ),
            }
        )

    def _already_called(self, tool_name: str, times: int = 1, session: str = "codegen-1") -> None:
        """Put calls on disk the way the audit recorder would have."""

        path = self.state_dir / "tool-calls.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for _ in range(times):
                handle.write(json.dumps({
                    "record_version": "loam_tool_call_v1",
                    "session_id": session,
                    "tool_name": tool_name,
                    "arguments": {},
                }) + "\n")

    def _verdict(self, tool_name: str, session: str = "codegen-1") -> dict | None:
        return self.orchestration.tool_policy(
            {**self.payload, "session_id": session, "tool_name": tool_name, "arguments": {}}
        )

    def test_staging_calls_are_allowed(self) -> None:
        # Writing the script and the inputs file is the expected first step.
        for used in range(self.orchestration._RUN_CODE_STAGING):
            with self.subTest(used=used):
                self.assertIsNone(self._verdict(RUN_CODE))
            self._already_called(RUN_CODE)

    def test_serial_execution_past_the_staging_allowance_is_refused(self) -> None:
        self._already_called(RUN_CODE, self.orchestration._RUN_CODE_STAGING)
        verdict = self._verdict(RUN_CODE)
        assert verdict is not None
        self.assertEqual(verdict["action"], "block")
        # The refusal has to say what to do instead, or it just stalls the round.
        self.assertIn("run_batch", verdict["message"])
        self.assertIn("write_files", verdict["message"])
        # And it must not contradict the floor the server enforces. This message
        # used to end "a one-row batch is fine", which is what an agent read at
        # the moment it was deciding -- then the cluster refused the batch.
        self.assertNotIn("one-row batch is fine", verdict["message"])
        self.assertIn("too narrow", verdict["message"])

    def test_submitting_a_batch_reopens_an_allowance_for_diagnosis(self) -> None:
        self._already_called(RUN_CODE, self.orchestration._RUN_CODE_STAGING)
        self.assertIsNotNone(self._verdict(RUN_CODE))  # blocked before batching
        self._already_called(RUN_BATCH)
        self.assertIsNone(self._verdict(RUN_CODE), "a failed row must still be diagnosable")

    def test_the_diagnosis_allowance_is_also_bounded(self) -> None:
        self._already_called(RUN_BATCH)
        self._already_called(
            RUN_CODE,
            self.orchestration._RUN_CODE_STAGING + self.orchestration._RUN_CODE_AFTER_BATCH,
        )
        verdict = self._verdict(RUN_CODE)
        assert verdict is not None
        self.assertEqual(verdict["action"], "block")
        self.assertIn("batch_status", verdict["message"])

    def test_the_batch_tools_are_never_budgeted(self) -> None:
        # Whatever a child has spent on run_code, the path out must stay open.
        self._already_called(RUN_CODE, 50)
        for tool in (RUN_BATCH, "mcp__hpc__batch_status", "mcp__hpc__list_batches",
                     "mcp__hpc__inspect_filesystem"):
            with self.subTest(tool=tool):
                self.assertIsNone(self._verdict(tool))

    def test_one_child_cannot_spend_another_s_allowance(self) -> None:
        self._start("codegen-2")
        self._already_called(RUN_CODE, 20, session="codegen-2")
        self.assertIsNone(self._verdict(RUN_CODE, session="codegen-1"))

    def test_no_audit_file_yet_is_not_an_exhausted_budget(self) -> None:
        self.assertFalse((self.state_dir / "tool-calls.jsonl").exists())
        self.assertIsNone(self._verdict(RUN_CODE))

    def test_a_torn_audit_line_does_not_void_the_budget(self) -> None:
        self._already_called(RUN_CODE, self.orchestration._RUN_CODE_STAGING)
        with (self.state_dir / "tool-calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"session_id": "codegen-1", "tool_na')  # writer mid-append
        self.assertIsNotNone(self._verdict(RUN_CODE))


class CodeGeneratorInstructionTests(unittest.TestCase):
    """The prompt must describe a sequence the child can actually perform."""

    def setUp(self) -> None:
        self.body = (ROOT / ".loam" / "agents" / "code-generator.md").read_text(encoding="utf-8")

    def test_staging_is_named_as_the_expected_first_step(self) -> None:
        self.assertIn("`write_files` to stage", self.body)
        self.assertIn("required first step", self.body)

    def test_staging_does_not_require_an_interpreter(self) -> None:
        # write_files closes the gap that made the batch-only rule impossible:
        # creating script_path/inputs_path no longer means running code.
        self.assertIn("It executes nothing", self.body)
        self.assertLess(self.body.index("`write_files` to stage"),
                        self.body.index("**`run_batch` once**"))

    def test_the_science_is_still_confined_to_run_batch(self) -> None:
        self.assertIn("The science never runs inside `run_code`", self.body)

    def test_a_single_job_is_not_an_excuse_to_skip_the_batch(self) -> None:
        # The commonest rationalisation, answered in the prompt itself.
        self.assertIn('because it is "only one job" is still wrong', self.body)

    def test_the_prompt_does_not_bless_a_degenerate_batch(self) -> None:
        """It must not promise what the server refuses.

        This assertion used to be `assertIn("a one-row batch is correct")`, and
        it passed only because that sentence survived as a QUOTE inside a note
        explaining it was wrong -- so the test was pinning the retracted wording
        rather than the rule. The server enforces a floor on rows and on
        distinct rows, so the prompt states the floor exists.
        """

        self.assertNotIn("a one-row batch is correct", self.body)
        self.assertIn("DISTINCT", self.body)
        self.assertIn("refuses a degenerate submission", self.body)

    def test_the_prompt_warns_that_the_budget_is_enforced(self) -> None:
        # A child that is refused should recognise the refusal, not fight it.
        self.assertIn("Loam counts your `run_code` calls", self.body)

    def test_the_impossible_absolute_is_gone(self) -> None:
        self.assertNotIn("is permitted ONLY as a short ad-hoc diagnostic AFTER", self.body)


if __name__ == "__main__":
    unittest.main()
