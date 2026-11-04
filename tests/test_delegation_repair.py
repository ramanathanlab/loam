"""Which states the delegation rewrite repairs, and which it leaves to the veto.

`canonicalize_campaign_delegation` (the adapter's `tool_request` middleware)
replaces a model-written `delegate_task` call with the arguments
`canonical_delegation_args` reconstructs from persisted state. That seam is the
reason the coordinator's transcription mostly does not matter -- but it does not
cover every state, and the uncovered one is why `tool_policy` still vetoes
malformed delegations.

These tests pin the coverage map that `.loam/manifest.yaml` documents. They are
expected to change when the hook dispatches delegation itself: at that point the
coordinator never writes these arguments in any state, and the branches that
check its transcription can go.
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

# What a coordinator looks like when it ignores the injected array and
# substitutes work of its own.
SUBSTITUTE_BATCH = {"tasks": [{"goal": "do the analysis myself", "context": "none", "role": "leaf"}]}


def _load(name: str):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, HOOKS / "hypothesis_orchestration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _payload(root: Path) -> dict:
    """A live-looking turn with two ranked hypotheses and a persisted report."""

    state_file = root / "jnana.json"
    reports_dir = root / "reports"
    reports_dir.mkdir()
    state_file.write_text(
        json.dumps(
            {
                "hypotheses": [
                    {"hypothesis_id": "h-1", "content": "First hypothesis."},
                    {"hypothesis_id": "h-2", "content": "Second hypothesis."},
                ]
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "tournament-001.json").write_text(
        json.dumps(
            {
                "workflow_id": "research-v1",
                "result": {
                    "status": "completed",
                    "top_hypotheses": [
                        {"hypothesis_id": "h-1", "elo_rating": 1210.0},
                        {"hypothesis_id": "h-2", "elo_rating": 1190.0},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "repo_root": str(ROOT),
        "workflow_id": "research-v1",
        "session_id": "coordinator",
        "loam_run_id": "run-1",
        "loam_state_dir": str(root / "loam-state"),
        "loam_hypothesis_orchestration": True,
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


def _plan_id(orchestration, payload: dict) -> str:
    return orchestration._plan_id(orchestration.Tournament(payload).latest_report()[0], payload)


def _launch_validator(orchestration, payload: dict, plan_id: str, hypothesis_id: str) -> None:
    """Record a validator start the way `subagent-start.py` does.

    `record_start` reads `child_goal`, not `goal`; a payload using the wrong key
    records nothing and silently leaves the plan looking un-launched.
    """

    orchestration.record_start(
        {
            **payload,
            "child_session_id": f"child-{hypothesis_id}",
            "child_subagent_id": f"sa-{hypothesis_id}",
            "child_goal": (
                f"Validate [loam-agent-role=hypothesis-validator] "
                f"[loam-plan-id={plan_id}] [loam-hypothesis-id={hypothesis_id}]"
            ),
        }
    )


def _complete_validator(orchestration, payload: dict, plan_id: str, hypothesis_id: str) -> None:
    orchestration.record_stop(
        {
            **payload,
            "child_session_id": f"child-{hypothesis_id}",
            "child_subagent_id": f"sa-{hypothesis_id}",
            "child_goal": (
                f"Validate [loam-agent-role=hypothesis-validator] "
                f"[loam-plan-id={plan_id}] [loam-hypothesis-id={hypothesis_id}]"
            ),
            "child_status": "completed",
            "child_summary": "### Outcome\nsupported\n\n### Evidence Score\n+0.75\n",
        }
    )


class RepairedStatesTests(unittest.TestCase):
    def test_a_fresh_round_is_repaired(self) -> None:
        """The coordinator's substitute batch is replaced by the real plan."""

        orchestration = _load("loam_repair_fresh")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            orchestration.context_for(dict(payload))
            repaired = orchestration.canonical_delegation_args({**payload, "arguments": SUBSTITUTE_BATCH})

        assert repaired is not None
        self.assertEqual(len(repaired["tasks"]), 2)
        self.assertNotIn("do the analysis myself", json.dumps(repaired))

    def test_a_validator_delegating_its_child_is_repaired(self) -> None:
        orchestration = _load("loam_repair_validator")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            orchestration.context_for(dict(payload))
            plan_id = _plan_id(orchestration, payload)
            _launch_validator(orchestration, payload, plan_id, "h-1")
            repaired = orchestration.canonical_delegation_args(
                {**payload, "session_id": "child-h-1", "arguments": SUBSTITUTE_BATCH}
            )

        assert repaired is not None
        self.assertIn("code-generator", json.dumps(repaired))

    def test_a_completed_round_is_repaired_into_the_reasoner(self) -> None:
        orchestration = _load("loam_repair_reasoner")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            orchestration.context_for(dict(payload))
            plan_id = _plan_id(orchestration, payload)
            for hypothesis_id in ("h-1", "h-2"):
                _launch_validator(orchestration, payload, plan_id, hypothesis_id)
                _complete_validator(orchestration, payload, plan_id, hypothesis_id)
            repaired = orchestration.canonical_delegation_args({**payload, "arguments": SUBSTITUTE_BATCH})

        assert repaired is not None
        self.assertIn("loam-agent-role=reasoner", repaired["goal"])


class UnrepairedStateTests(unittest.TestCase):
    """The gap the veto covers. Deleting those branches without closing it would
    let a coordinator's substitute batch execute as written."""

    def test_mid_round_is_not_repaired_and_the_veto_is_what_stops_it(self) -> None:
        orchestration = _load("loam_repair_midround")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            orchestration.context_for(dict(payload))
            plan_id = _plan_id(orchestration, payload)
            for hypothesis_id in ("h-1", "h-2"):
                _launch_validator(orchestration, payload, plan_id, hypothesis_id)

            repaired = orchestration.canonical_delegation_args({**payload, "arguments": SUBSTITUTE_BATCH})
            verdict = orchestration.tool_policy(
                {**payload, "tool_name": "delegate_task", "arguments": SUBSTITUTE_BATCH}
            )

        self.assertIsNone(repaired)
        assert verdict is not None
        self.assertEqual(verdict["action"], "block")

    def test_a_dispatched_plan_is_never_re_offered(self) -> None:
        """The repair fails closed rather than duplicating a fan-out.

        Worth pinning because the alternative would be the worst failure in the
        system: the fresh-round branch reconstructs the FULL batch, so if it
        were reachable once validators are already running, a repair would
        dispatch a second validator per hypothesis on top of the live ones.
        Two things prevent it -- `_active_plan` matches the launched plan even
        with the run id absent, and `context_for` stops offering a task array
        once the plan is dispatched.
        """

        orchestration = _load("loam_repair_unmatched")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            orchestration.context_for(dict(payload))
            plan_id = _plan_id(orchestration, payload)
            for hypothesis_id in ("h-1", "h-2"):
                _launch_validator(orchestration, payload, plan_id, hypothesis_id)
            unmatched = {key: value for key, value in payload.items() if key != "loam_run_id"}

            self.assertNotIn(
                "Required delegate_task tasks payload:", orchestration.context_for(dict(unmatched))
            )
            self.assertIsNone(
                orchestration.canonical_delegation_args({**unmatched, "arguments": SUBSTITUTE_BATCH})
            )


if __name__ == "__main__":
    unittest.main()
