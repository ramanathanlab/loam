"""Behavior tests for Jnana-ranked Loam subagent orchestration."""

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


class HypothesisOrchestrationTests(unittest.TestCase):
    def test_coordinator_rejects_untagged_batch_while_validator_plan_is_pending(self) -> None:
        orchestration = _load("loam_orchestration_coordinator_batch_guard")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            blocked = orchestration.tool_policy(
                {
                    **payload,
                    "loam_hypothesis_orchestration": True,
                    "tool_name": "delegate_task",
                    "arguments": {"tasks": [{"goal": "Validate the arithmetic claim.", "role": "orchestrator"}]},
                }
            )

        self.assertEqual(blocked["action"], "block")
        self.assertIn("tagged hypothesis-validator", blocked["message"])

    def test_coordinator_rejects_untagged_single_validator_delegation(self) -> None:
        orchestration = _load("loam_orchestration_coordinator_single_guard")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            blocked = orchestration.tool_policy(
                {
                    **payload,
                    "loam_hypothesis_orchestration": True,
                    "tool_name": "delegate_task",
                    "arguments": {"goal": "Validate the arithmetic claim.", "role": "orchestrator"},
                }
            )

        self.assertEqual(blocked["action"], "block")
        self.assertIn("tagged hypothesis-validator", blocked["message"])

    def test_coordinator_cannot_execute_while_validator_plan_is_pending(self) -> None:
        orchestration = _load("loam_orchestration_coordinator_execution_guard")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            blocked = orchestration.tool_policy(
                {
                    **payload,
                    "loam_hypothesis_orchestration": True,
                    "tool_name": "terminal",
                    "arguments": {"command": "python3 -c 'print(2.5 > 2.4)'"},
                }
            )

        self.assertEqual(blocked["action"], "block")
        self.assertIn("delegate tagged hypothesis-validator", blocked["message"])

    def test_validator_batch_requires_per_task_orchestrator_role(self) -> None:
        orchestration = _load("loam_orchestration_validator_batch_role")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            goal = (
                "Validate [loam-agent-role=hypothesis-validator] "
                "[loam-plan-id=plan-1] [loam-hypothesis-id=h-1]"
            )
            missing = orchestration.tool_policy(
                {
                    **payload,
                    "loam_hypothesis_orchestration": True,
                    "tool_name": "delegate_task",
                    "arguments": {"tasks": [{"goal": goal}]},
                }
            )
            accepted = orchestration.tool_policy(
                {
                    **payload,
                    "loam_hypothesis_orchestration": True,
                    "tool_name": "delegate_task",
                    "arguments": {
                        "tasks": [
                            {
                                "goal": goal,
                                "role": "orchestrator",
                                "context": '{"loam_validator_contract": "v1"}',
                            }
                        ]
                    },
                }
            )

        self.assertEqual(missing["action"], "block")
        self.assertIn("per-task role=orchestrator", missing["message"])
        self.assertIsNone(accepted)

    def test_validator_batch_requires_complete_injected_contract(self) -> None:
        orchestration = _load("loam_orchestration_validator_contract")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            goal = (
                "Validate [loam-agent-role=hypothesis-validator] "
                "[loam-plan-id=plan-1] [loam-hypothesis-id=h-1]"
            )
            blocked = orchestration.tool_policy(
                {
                    **payload,
                    "loam_hypothesis_orchestration": True,
                    "tool_name": "delegate_task",
                    "arguments": {
                        "tasks": [{"goal": goal, "role": "orchestrator", "context": "{}"}]
                    },
                }
            )

        self.assertEqual(blocked["action"], "block")
        self.assertIn("loam_validator_contract v1", blocked["message"])

    def test_evidence_score_accepts_markdown_code_span(self) -> None:
        orchestration = _load("loam_orchestration_markdown_evidence_score")

        self.assertEqual(orchestration._evidence_score("### Evidence Score\n`0.00` — inconclusive"), 0.0)

    def test_validator_directive_renders_ready_tagged_task_payload(self) -> None:
        orchestration = _load("loam_orchestration_ready_validator_tasks")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            directive = orchestration.context_for(payload)

        tasks = json.loads(directive.split("Required delegate_task tasks payload:\n", 1)[1])
        self.assertEqual(len(tasks), 2)
        self.assertEqual({task["role"] for task in tasks}, {"orchestrator"})
        self.assertTrue(all("[loam-agent-role=hypothesis-validator]" in task["goal"] for task in tasks))
        self.assertEqual(
            {"h-1", "h-2"},
            {task["goal"].split("[loam-hypothesis-id=", 1)[1].split("]", 1)[0] for task in tasks},
        )
        contexts = [json.loads(task["context"]) for task in tasks]
        self.assertTrue(
            all("[loam-agent-role=code-generator]" in context["required_child"]["goal"] for context in contexts)
        )
        self.assertTrue(all(context["required_child"]["role"] == "leaf" for context in contexts))
        self.assertTrue(all("Fraction" in context["required_child"]["context"] for context in contexts))
        self.assertTrue(all("### Evidence Score" in context["validator_instructions"] for context in contexts))

    def test_canonical_delegation_reconstructs_validator_code_generator(self) -> None:
        orchestration = _load("loam_orchestration_canonical_delegation")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            validator = {
                **payload,
                "child_subagent_id": "validator-1",
                "child_session_id": "validator-session-1",
                "child_goal": (
                    "Validate [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] "
                    "[loam-hypothesis-id=h-1]"
                ),
            }
            orchestration.record_start(validator)
            canonical = orchestration.canonical_delegation_args(
                {**payload, "session_id": "validator-session-1"}
            )

        self.assertEqual(canonical["tasks"][0]["role"], "leaf")
        self.assertIn("[loam-agent-role=code-generator]", canonical["tasks"][0]["goal"])
        self.assertIn("Fraction", canonical["tasks"][0]["context"])

    def test_canonical_remote_delegation_requires_the_configured_mcp_tool(self) -> None:
        orchestration = _load("loam_orchestration_canonical_remote_delegation")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                **self._payload(Path(temporary)),
                "loam_execution": {
                    "mode": "remote_mcp",
                    "tool_name": "mcp_hpc_execution_run_code",
                    "endpoint": "http://127.0.0.1:8000/mcp",
                    "function_name": "run_code",
                    "input_schema": "hpc-run-code-v1",
                    "runtime": "python3.12",
                },
            }
            validator = {
                **payload,
                "child_subagent_id": "validator-1",
                "child_session_id": "validator-session-1",
                "child_goal": (
                    "Validate [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] "
                    "[loam-hypothesis-id=h-1]"
                ),
            }
            orchestration.record_start(validator)
            canonical = orchestration.canonical_delegation_args(
                {**payload, "session_id": "validator-session-1"}
            )

        context = canonical["tasks"][0]["context"]
        self.assertIn("mcp_hpc_execution_run_code", context)
        self.assertIn("do not execute code locally", context)
        self.assertIn("pbs_jobid", context)
        self.assertNotIn("Use local python3.11", context)

    def _payload(self, root: Path) -> dict:
        state_file = root / "jnana.json"
        reports_dir = root / "reports"
        reports_dir.mkdir()
        hypotheses = [
            {"hypothesis_id": "h-1", "content": "First complete hypothesis."},
            {"hypothesis_id": "h-2", "content": "Second complete hypothesis."},
        ]
        state_file.write_text(json.dumps({"hypotheses": hypotheses}), encoding="utf-8")
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
            "session_id": "session-1",
            "user_message": "Investigate codon bias.",
            "loam_state_dir": str(root / "loam-state"),
            "loam_hypothesis_tournament": {
                "state_file": str(state_file),
                "reports_dir": str(reports_dir),
            },
            "loam_execution": {
                "mode": "remote_mcp",
                "tool_name": "globus_execute",
                "endpoint": "https://compute.example/mcp",
                "function_name": "run_python",
                "input_schema": "run-python-v1",
                "runtime": "python3.11",
            },
        }

    def test_registry_enforces_validator_execution_boundary(self) -> None:
        orchestration = _load("loam_orchestration_registry")
        registry = orchestration.agent_registry({"repo_root": str(ROOT)})

        self.assertEqual(registry["hypothesis-validator"].allowed_children, ("code-generator",))
        self.assertFalse(registry["hypothesis-validator"].direct_execution)
        self.assertTrue(registry["code-generator"].direct_execution)
        self.assertFalse(registry["reasoner"].direct_execution)
    def test_tagged_roles_enforce_tool_boundaries(self) -> None:
        orchestration = _load("loam_orchestration_tool_policy")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            validator = {
                **payload,
                "child_session_id": "validator-session",
                "child_subagent_id": "validator-1",
                "child_goal": (
                    "Validate [loam-agent-role=hypothesis-validator] "
                    "[loam-plan-id=plan-1] [loam-hypothesis-id=h-1]"
                ),
            }
            orchestration.record_start(validator)
            base = {**payload, "loam_hypothesis_orchestration": True, "session_id": "validator-session"}
            blocked = orchestration.tool_policy({**base, "tool_name": "terminal", "arguments": {}})
            allowed = orchestration.tool_policy(
                {
                    **base,
                    "tool_name": "delegate_task",
                    "arguments": {"goal": "Execute [loam-agent-role=code-generator] [loam-plan-id=plan-1]"},
                }
            )
            allowed_batch = orchestration.tool_policy(
                {
                    **base,
                    "tool_name": "delegate_task",
                    "arguments": {
                        "tasks": [
                            {
                                "goal": "Execute [loam-agent-role=code-generator] [loam-plan-id=plan-1]",
                                "role": "leaf",
                            }
                        ]
                    },
                }
            )
            wrong_child = orchestration.tool_policy(
                {**base, "tool_name": "delegate_task", "arguments": {"goal": "Delegate a researcher."}}
            )
            orchestration.record_start(
                {
                    **payload,
                    "child_session_id": "code-generator-session",
                    "child_subagent_id": "code-generator-1",
                    "child_goal": "Execute [loam-agent-role=code-generator] [loam-plan-id=plan-1]",
                }
            )
            code_generator = {**base, "session_id": "code-generator-session"}
            wrong_execution_tool = orchestration.tool_policy(
                {**code_generator, "tool_name": "terminal", "arguments": {}}
            )
            remote_execution = orchestration.tool_policy(
                {**code_generator, "tool_name": "globus_execute", "arguments": {}}
            )

        self.assertEqual(blocked["action"], "block")
        self.assertIsNone(allowed)
        self.assertIsNone(allowed_batch)
        self.assertEqual(wrong_child["action"], "block")
        self.assertEqual(wrong_execution_tool["action"], "block")
        self.assertIsNone(remote_execution)

    def test_local_execution_mode_allows_local_tools(self) -> None:
        orchestration = _load("loam_orchestration_local_execution")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {**self._payload(Path(temporary)), "loam_execution": {"mode": "local"}}
            orchestration.record_start(
                {
                    **payload,
                    "child_session_id": "code-generator-session",
                    "child_subagent_id": "code-generator-1",
                    "child_goal": "Execute [loam-agent-role=code-generator] [loam-plan-id=plan-1]",
                }
            )
            result = orchestration.tool_policy(
                {
                    **payload,
                    "loam_hypothesis_orchestration": True,
                    "session_id": "code-generator-session",
                    "tool_name": "terminal",
                    "arguments": {},
                }
            )

        self.assertIsNone(result)


    def test_completed_validators_request_reasoner_and_next_turn_uses_brief(self) -> None:
        orchestration = _load("loam_orchestration_lifecycle")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            first = orchestration.context_for(payload)
            self.assertIn("launch every selected hypothesis validator", first)
            self.assertIn('"role": "orchestrator"', first)
            plan_id = first.split("[loam-plan-id=", 1)[1].split("]", 1)[0]

            for index, hypothesis_id in enumerate(("h-1", "h-2"), start=1):
                child = {
                    **payload,
                    "child_subagent_id": f"validator-{index}",
                    "child_session_id": f"validator-session-{index}",
                    "child_goal": (
                        "Validate a hypothesis "
                        f"[loam-agent-role=hypothesis-validator] [loam-plan-id={plan_id}] "
                        f"[loam-hypothesis-id={hypothesis_id}]"
                    ),
                }
                orchestration.record_start(child)
                orchestration.record_stop(
                    {
                        **child,
                        "child_status": "completed",
                        "child_summary": (
                            f"### Hypothesis\n{hypothesis_id}\n### Outcome\nsupported\n"
                            "### Evidence Score\n+0.75 — reliable support"
                        ),
                    }
                )

            reasoner = orchestration.context_for(payload)
            self.assertIn("Delegate exactly one reasoner", reasoner)
            self.assertIn('"hypothesis_id": "h-1"', reasoner)
            self.assertIn('"hypothesis_id": "h-2"', reasoner)
            rows = [
                json.loads(line)
                for line in (Path(payload["loam_state_dir"]) / "hypothesis-orchestration.jsonl").read_text().splitlines()
            ]
            scores = {row["hypothesis_id"]: row["evidence_score"] for row in rows if row["event"] == "validator_completed"}
            self.assertEqual(scores, {"h-1": 0.75, "h-2": 0.75})
            self.assertIn('"evidence_score": 0.75', reasoner)
            self.assertIn('\\"mode\\": \\"remote_mcp\\"', first)
            self.assertIsNone(
                orchestration.tool_policy(
                    {
                        **payload,
                        "loam_hypothesis_orchestration": True,
                        "tool_name": "delegate_task",
                        "arguments": {"goal": f"Synthesize [loam-agent-role=reasoner] [loam-plan-id={plan_id}]"},
                    }
                )
            )
            campaign_rows = [
                json.loads(line)
                for line in (Path(payload["loam_state_dir"]) / "hypothesis-campaign.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in campaign_rows],
                ["hypothesis_validation_started", "hypothesis_validation_completed", "hypothesis_validation_started", "hypothesis_validation_completed"],
            )

            reasoner_child = {
                **payload,
                "child_subagent_id": "reasoner-1",
                "child_session_id": "reasoner-session-1",
                "child_goal": f"Synthesize [loam-agent-role=reasoner] [loam-plan-id={plan_id}]",
            }
            orchestration.record_start(reasoner_child)
            orchestration.record_stop(
                {
                    **reasoner_child,
                    "child_status": "completed",
                    "child_summary": "Prioritize a discriminating comparative test.",
                }
            )
            orchestration.prepare_payload(payload)

            self.assertIn("Prior reasoner brief", payload["user_message"])
            self.assertIn("Prioritize a discriminating comparative test.", payload["user_message"])
            self.assertIn("Latest reasoner brief", orchestration.context_for(payload))
            campaign_rows = [
                json.loads(line)
                for line in (Path(payload["loam_state_dir"]) / "hypothesis-campaign.jsonl").read_text().splitlines()
            ]
            self.assertEqual(campaign_rows[-1]["event"], "hypothesis_campaign_round_completed")
            self.assertEqual(campaign_rows[-1]["synthesis"], "Prioritize a discriminating comparative test.")

    def test_validator_without_a_valid_score_records_missing_status(self) -> None:
        orchestration = _load("loam_orchestration_missing_score")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            child = {
                **payload,
                "child_subagent_id": "validator-1",
                "child_session_id": "validator-session-1",
                "child_goal": (
                    "Validate [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] "
                    "[loam-hypothesis-id=h-1]"
                ),
            }
            orchestration.record_start(child)
            orchestration.record_stop(
                {**child, "child_status": "completed", "child_summary": "### Evidence Score\n1.25 — invalid"}
            )
            row = json.loads(
                (Path(payload["loam_state_dir"]) / "hypothesis-orchestration.jsonl").read_text().splitlines()[-1]
            )

        self.assertIsNone(row["evidence_score"])
        self.assertEqual(row["evidence_score_status"], "missing_or_invalid")


if __name__ == "__main__":
    unittest.main()
