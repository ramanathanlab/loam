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
            "loam_execution": {"mode": "remote_mcp", "tool_name": "globus_execute"},
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
            self.assertIn('"mode": "remote_mcp"', first)
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
