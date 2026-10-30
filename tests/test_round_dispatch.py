"""The coordinator launches a round by calling a tool, not by transcribing one.

The hook computes each round's delegation from persisted state, so the
authoritative arguments exist before the model writes anything. They used to be
printed into the coordinator's context with instructions to copy them verbatim,
and the copying is what failed: a validator at char ~9,000 of an 11 KB payload
told its child the hypothesis "is not visible in this message" while holding it
(2026-10-21). `loam_dispatch_round` hands those arguments to `delegate_task`
from inside the adapter instead.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOAM = ROOT / ".loam"
HOOKS = LOAM / "hooks"
ADAPTER = LOAM / "adapters" / "hermes" / "plugin" / "__init__.py"
MANIFEST = LOAM / "manifest.yaml"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_delegation_repair import _payload  # noqa: E402


def _load(name: str):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, HOOKS / "hypothesis_orchestration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ToolNameAgreementTests(unittest.TestCase):
    """The hook names the tool in a block message; the adapter registers it.

    A drift between the two tells the coordinator to call something that does
    not exist, and the only symptom is a campaign that never launches.
    """

    def test_hook_and_adapter_agree_on_the_name(self) -> None:
        orchestration = _load("loam_dispatch_name")
        self.assertIn(
            f'DISPATCH_TOOL_NAME = "{orchestration.DISPATCH_TOOL}"',
            ADAPTER.read_text(),
        )

    def test_the_manifest_declares_the_tool(self) -> None:
        orchestration = _load("loam_dispatch_manifest")
        declared = yaml.safe_load(MANIFEST.read_text()).get("tools") or {}
        self.assertIn(orchestration.DISPATCH_TOOL, declared)
        spec = declared[orchestration.DISPATCH_TOOL]
        # The toolset matters: a plugin's own toolset is off until configured,
        # so declaring it wrong here is how the tool goes invisible.
        self.assertEqual(spec["toolset"], "delegation")
        self.assertIn(f'toolset="{spec["toolset"]}"', ADAPTER.read_text())
        self.assertIn(f"def {spec['handler']}(", ADAPTER.read_text())


class DirectiveTests(unittest.TestCase):
    def test_the_directive_announces_the_tool_and_prints_no_task_array(self) -> None:
        orchestration = _load("loam_dispatch_directive")
        with tempfile.TemporaryDirectory() as temporary:
            directive = orchestration.context_for(_payload(Path(temporary)))

        self.assertIn(orchestration.DISPATCH_TOOL, directive)
        self.assertNotIn("Required delegate_task tasks payload", directive)
        self.assertNotIn('"role": "orchestrator"', directive)
        # The point of the change: the coordinator's context stops carrying a
        # payload it cannot use and used to mis-copy. Two validators' tasks ran
        # ~26 KB; the announcement is a few hundred characters.
        self.assertLess(len(directive), 2_000)

    def test_everything_the_array_carried_still_reaches_the_validator(self) -> None:
        """The payload moved into the dispatch; it was not dropped."""

        orchestration = _load("loam_dispatch_payload")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            orchestration.context_for(dict(payload))
            arguments = orchestration.canonical_delegation_args(payload)

        assert arguments is not None
        self.assertEqual(len(arguments["tasks"]), 2)
        self.assertEqual({task["role"] for task in arguments["tasks"]}, {"orchestrator"})
        first = arguments["tasks"][0]["context"]
        self.assertIn("First hypothesis.", first)
        context = json.loads(first)
        self.assertEqual(context["loam_validator_contract"], "v1")
        self.assertIn("campaign", context)
        self.assertGreater(len(context["agent_definition"]), 1_000)
        self.assertIn("[loam-agent-role=hypothesis-validator]", arguments["tasks"][0]["goal"])


class CoordinatorBoundaryTests(unittest.TestCase):
    def test_the_coordinator_cannot_compose_delegation_in_any_shape(self) -> None:
        orchestration = _load("loam_dispatch_boundary")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {**_payload(Path(temporary)), "loam_hypothesis_orchestration": True}
            orchestration.context_for(dict(payload))
            blocked = orchestration.tool_policy(
                {**payload, "tool_name": "delegate_task", "arguments": {"tasks": [{"goal": "mine"}]}}
            )
            allowed = orchestration.tool_policy(
                {**payload, "tool_name": orchestration.DISPATCH_TOOL, "arguments": {}}
            )

        assert blocked is not None
        self.assertEqual(blocked["action"], "block")
        self.assertIn(orchestration.DISPATCH_TOOL, blocked["message"])
        self.assertIsNone(allowed, "the dispatch tool must stay callable; it is the only legal move")

    def test_a_validator_may_still_delegate_its_own_child(self) -> None:
        """Only the coordinator hop changed. The validator hop is unaffected."""

        orchestration = _load("loam_dispatch_validator_hop")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {**_payload(Path(temporary)), "loam_hypothesis_orchestration": True}
            orchestration.context_for(dict(payload))
            plan_id = orchestration._plan_id(orchestration._latest_report(payload)[0], payload)
            orchestration.record_start(
                {
                    **payload,
                    "child_session_id": "child-h-1",
                    "child_subagent_id": "sa-h-1",
                    "child_goal": (
                        f"Validate [loam-agent-role=hypothesis-validator] "
                        f"[loam-plan-id={plan_id}] [loam-hypothesis-id=h-1]"
                    ),
                }
            )
            verdict = orchestration.tool_policy(
                {
                    **payload,
                    "session_id": "child-h-1",
                    "tool_name": "delegate_task",
                    "arguments": {
                        "tasks": [
                            {
                                "goal": (
                                    "Compute it [loam-agent-role=code-generator] "
                                    f"[loam-plan-id={plan_id}] [loam-hypothesis-id=h-1]"
                                ),
                                "role": "leaf",
                            }
                        ]
                    },
                }
            )

        self.assertIsNone(verdict)


if __name__ == "__main__":
    unittest.main()


class ReasonerDirectiveTests(unittest.TestCase):
    def test_the_reasoner_directive_points_at_the_tool_the_policy_allows(self) -> None:
        # The coordinator's delegate_task is blocked in every shape (above), so a directive that says
        # "call delegate_task" for the reasoner leaves it blocked with no reasoner run (Aurora, 2026-10-30).
        orchestration = _load("loam_dispatch_reasoner_directive")
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            directive = orchestration._reasoner_directive(
                "plan-1", [], orchestration.agent_registry(payload), payload=payload
            )

        self.assertIn(f"Call {orchestration.DISPATCH_TOOL}", directive)
        self.assertNotIn("Call delegate_task", directive)
