"""An invented project directory must not be inherited by a child.

On 2026-11-08, during the first full round on a 128-node Aurora block, a
hypothesis-validator wrote this into its child's `context`, three times, as a
statement of fact:

    The workspace is /flare/SHAREIDP/msinclair/hpc-run/work/plan_8ed4bb77.../13f00530-.../

The campaign's project is FRAME-IDP. `/flare/SHAREIDP` does not exist. The same
validator had used the correct path minutes earlier, so this was not a
misconfiguration -- it hallucinated the project segment mid-turn and passed it
down as settled. Six write_files calls went to it and failed; the bad root then
came back up in the child's summary, where the parent would read it as real.

`_safe_write_path` on the server already refuses the write, so nothing unsafe
happened. What was missing was anything to stop the root propagating, which is
what these tests cover.
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

WORKDIR_ROOT = "/flare/FRAME-IDP/msinclair/hpc-run/work"
BATCH_ROOT = "/flare/FRAME-IDP/msinclair/hpc-run/batches"

REMOTE = {
    "mode": "remote_mcp",
    "tool_name": "mcp__hpc__run_code",
    "function_name": "run_code",
    "workdir_root": WORKDIR_ROOT,
    "batch_root": BATCH_ROOT,
    # 2, not the production default of 1, because this module's fixture starts a
    # code-generator in setUp. With a budget of 1 that child spends the whole
    # budget and the validator's delegation is refused for BUDGET reasons, so
    # the foreign-root assertions below would pass or fail for the wrong cause.
    # Budget behaviour is tested in test_hypothesis_orchestration.
    "max_children": 2,
}

PLAN = "plan_8ed4bb773a77577c"
HYPOTHESIS = "13f00530-3dd9-461d-b65d-92b8dd1c19be"
GOOD = f"{WORKDIR_ROOT}/{PLAN}/{HYPOTHESIS}"
BAD = f"/flare/SHAREIDP/msinclair/hpc-run/work/{PLAN}/{HYPOTHESIS}"


def _load(name: str):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, HOOKS / "hypothesis_orchestration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ForeignRootDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.o = _load("loam_foreign_paths")

    def test_the_observed_invented_root_is_caught(self) -> None:
        found = self.o._foreign_hpc_paths(f"write under {BAD}.", REMOTE)
        self.assertEqual(len(found), 1)
        self.assertIn("SHAREIDP", found[0])

    def test_the_real_workdir_is_not_flagged(self) -> None:
        self.assertEqual(self.o._foreign_hpc_paths(f"write under {GOOD}.", REMOTE), [])

    def test_the_lus_spelling_of_the_real_tree_is_accepted(self) -> None:
        # /flare and /lus/flare/projects are the same filesystem and the agents
        # use them interchangeably; HPC_INSPECT_ROOTS lists both for that reason.
        alias = GOOD.replace("/flare/", "/lus/flare/projects/", 1)
        self.assertEqual(self.o._foreign_hpc_paths(alias, REMOTE), [])

    def test_siblings_of_workdir_root_are_accepted(self) -> None:
        # batches/ and tools/ live beside work/ under HPC_BASE_DIR.
        for path in (f"{BATCH_ROOT}/b-2f95dbc8caab/inputs.jsonl",
                     "/flare/FRAME-IDP/msinclair/hpc-run/tools/x.py"):
            with self.subTest(path=path):
                self.assertEqual(self.o._foreign_hpc_paths(path, REMOTE), [])

    def test_a_different_project_is_not_reachable_via_the_base(self) -> None:
        # Allowing HPC_BASE_DIR must not widen to the whole /flare/FRAME-IDP tree.
        self.assertTrue(self.o._foreign_hpc_paths("/flare/FRAME-IDP/someone-else/x", REMOTE))

    def test_unconfigured_execution_fails_open(self) -> None:
        # Local mode has no workdir_root; there is nothing to compare against,
        # and blocking every path would break non-remote campaigns.
        self.assertEqual(self.o._foreign_hpc_paths(BAD, {}), [])

    def test_prefix_collision_is_not_treated_as_containment(self) -> None:
        # A path that merely SHARES a prefix with an allowed root is outside it.
        # (`<base>/work-evil` is deliberately NOT the case to test: it really is
        # inside HPC_BASE_DIR, which HPC_INSPECT_ROOTS allows wholesale.)
        for path in ("/flare/FRAME-IDP-evil/msinclair/hpc-run/work/x",
                     "/flare/FRAME-IDPX/msinclair/hpc-run/work/x"):
            with self.subTest(path=path):
                self.assertTrue(self.o._foreign_hpc_paths(path, REMOTE))

    def test_findings_are_deduplicated_and_bounded(self) -> None:
        blob = " ".join([BAD] * 3 + [f"/flare/NOPE{i}/x" for i in range(6)])
        found = self.o._foreign_hpc_paths(blob, REMOTE)
        self.assertEqual(len(found), len(set(found)))
        self.assertLessEqual(len(found), 4)

    def test_a_trailing_quote_or_bracket_is_not_part_of_the_path(self) -> None:
        found = self.o._foreign_hpc_paths(json.dumps({"p": BAD}), REMOTE)
        self.assertEqual(found, [BAD], "JSON quoting must not be captured into the path")


class ForeignRootPolicyTests(unittest.TestCase):
    """The block has to fire at the delegation, not only at the write."""

    def setUp(self) -> None:
        self.o = _load("loam_foreign_paths_policy")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name)
        self.payload = {
            "repo_root": str(ROOT),
            "workflow_id": "research-v1",
            "loam_run_id": "run-1",
            "loam_state_dir": str(self.state),
            "loam_hypothesis_orchestration": True,
            "loam_execution": REMOTE,
        }
        self._start("validator-1", "hypothesis-validator")
        self._start("codegen-1", "code-generator")

    def _start(self, session: str, role: str) -> None:
        self.o.record_start({
            **self.payload,
            "child_session_id": session,
            "child_subagent_id": f"sa-{session}",
            "child_goal": (
                f"work [loam-agent-role={role}] [loam-plan-id={PLAN}] "
                f"[loam-hypothesis-id={HYPOTHESIS}]"
            ),
        })

    def _policy(self, session: str, tool_name: str, arguments: dict) -> dict | None:
        return self.o.tool_policy({
            **self.payload, "session_id": session,
            "tool_name": tool_name, "arguments": arguments,
        })

    def _delegation(self, workdir: str) -> dict:
        return {"tasks": [{
            "goal": (f"Compute SASA [loam-agent-role=code-generator] "
                     f"[loam-plan-id={PLAN}] [loam-hypothesis-id={HYPOTHESIS}]"),
            "context": f"The workspace is {workdir}. Write all outputs there.",
        }]}

    def test_a_validator_cannot_hand_a_child_an_invented_root(self) -> None:
        verdict = self._policy("validator-1", "delegate_task", self._delegation(BAD))
        assert verdict is not None, "the delegation that caused this bug must be refused"
        self.assertEqual(verdict["action"], "block")
        self.assertIn("SHAREIDP", verdict["message"])
        # The refusal must name the right path, or the validator just guesses again.
        self.assertIn(GOOD, verdict["message"])

    def test_a_correct_delegation_still_passes(self) -> None:
        self.assertIsNone(self._policy("validator-1", "delegate_task", self._delegation(GOOD)))

    def test_a_code_generator_write_to_an_invented_root_is_refused(self) -> None:
        verdict = self._policy("codegen-1", "mcp__hpc__write_files",
                               {"files": {f"{BAD}/probe.txt": "probe"}})
        assert verdict is not None
        self.assertEqual(verdict["action"], "block")

    def test_a_code_generator_write_to_the_real_root_is_allowed(self) -> None:
        self.assertIsNone(self._policy("codegen-1", "mcp__hpc__write_files",
                                       {"files": {f"{GOOD}/probe.txt": "probe"}}))

    def test_the_guard_does_not_consume_the_run_code_budget(self) -> None:
        # A refused call must not count against staging, or one bad path would
        # exhaust the child's allowance and deadlock it.
        for _ in range(self.o._RUN_CODE_STAGING + 2):
            self._policy("codegen-1", "mcp__hpc__run_code", {"payload": f"open('{BAD}/x')"})
        self.assertIsNone(
            self._policy("codegen-1", "mcp__hpc__run_code", {"payload": f"open('{GOOD}/x')"}),
            "the guard must not have been recorded as budget use",
        )


if __name__ == "__main__":
    unittest.main()
