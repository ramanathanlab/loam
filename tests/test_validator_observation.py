"""A hypothesis-validator may look at results, but still may not compute.

Allowing it nothing but `delegate_task` made every status poll cost an
experiment slot out of `max_children`, and left its evidence ledger entirely
downstream of a child's self-reported numbers.
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
ORCHESTRATION = HOOKS / "hypothesis_orchestration.py"

EXECUTION = {
    "mode": "remote_mcp",
    "tool_name": "hpc_run_code",
    "endpoint": "https://compute.example/mcp",
    "function_name": "run_code",
    "input_schema": "run-python-v1",
    "runtime": "python3.11",
    "tool_prefix": "hpc_",
}


def _load(name: str):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, ORCHESTRATION)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validator_payload(root: Path, tool_name: str) -> dict:
    """A payload from inside a running, role-tagged validator child."""

    state_dir = root / "loam-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "hypothesis-orchestration.jsonl").write_text(
        json.dumps(
            {
                "event": "agent_started",
                "child_session_id": "validator-session",
                "role": "hypothesis-validator",
                "plan_id": "plan-1",
                "hypothesis_id": "h-1",
                "workflow_id": "research-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "repo_root": str(ROOT),
        "workflow_id": "research-v1",
        "session_id": "validator-session",
        "loam_state_dir": str(state_dir),
        "loam_hypothesis_orchestration": True,
        "loam_execution": EXECUTION,
        "tool_name": tool_name,
        "arguments": {},
    }


class ValidatorObservationTests(unittest.TestCase):
    def test_read_only_tools_are_allowed(self) -> None:
        orchestration = _load("loam_validator_observation_allow")
        for tool in ("hpc_batch_status", "hpc_inspect_filesystem", "hpc_list_batches"):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(tool=tool):
                verdict = orchestration.tool_policy(_validator_payload(Path(temporary), tool))
                self.assertIsNone(verdict)

    def test_executing_tools_are_still_blocked(self) -> None:
        """The invariant: the validator never computes, however convenient."""

        orchestration = _load("loam_validator_observation_block")
        for tool in ("hpc_run_code", "hpc_run_batch", "hpc_register_tool"):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(tool=tool):
                verdict = orchestration.tool_policy(_validator_payload(Path(temporary), tool))
                assert verdict is not None
                self.assertEqual(verdict["action"], "block")
                self.assertIn("must delegate executable work", verdict["message"])

    def test_an_unrelated_tool_is_still_blocked(self) -> None:
        orchestration = _load("loam_validator_observation_unrelated")
        with tempfile.TemporaryDirectory() as temporary:
            verdict = orchestration.tool_policy(_validator_payload(Path(temporary), "web_search"))

        assert verdict is not None
        self.assertEqual(verdict["action"], "block")

    def test_observation_names_follow_the_configured_prefix(self) -> None:
        orchestration = _load("loam_validator_observation_names")
        names = orchestration.hpc_tool_names(EXECUTION)
        self.assertEqual(
            {names[key] for key in ("batch_status", "inspect_filesystem", "list_batches")},
            {"hpc_batch_status", "hpc_inspect_filesystem", "hpc_list_batches"},
        )
        # No prefix derivable, so no observation tools are granted by name.
        self.assertEqual(orchestration.hpc_tool_names({"mode": "local"}), {})
        self.assertEqual(orchestration.hpc_tool_names(None), {})

    def test_a_code_generator_is_unaffected(self) -> None:
        orchestration = _load("loam_validator_observation_child")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _validator_payload(Path(temporary), "hpc_run_code")
            state = Path(payload["loam_state_dir"]) / "hypothesis-orchestration.jsonl"
            state.write_text(
                json.dumps(
                    {
                        "event": "agent_started",
                        "child_session_id": "validator-session",
                        "role": "code-generator",
                        "plan_id": "plan-1",
                        "hypothesis_id": "h-1",
                        "workflow_id": "research-v1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            verdict = orchestration.tool_policy(payload)

        self.assertIsNone(verdict)


if __name__ == "__main__":
    unittest.main()
