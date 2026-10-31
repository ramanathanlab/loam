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
    def test_coordinator_must_use_the_dispatch_tool_not_delegate_task(self) -> None:
        """The coordinator no longer composes delegation in any shape.

        This replaces four tests that each checked one way an LLM transcribed
        the injected array badly -- an untagged batch, an untagged single task,
        a missing per-task role=orchestrator, a dropped loam_validator_contract.
        Each was an approximation of "did it copy correctly". The hook now
        dispatches the round itself, so the whole hop is refused and the shape
        stops mattering.
        """

        orchestration = _load("loam_orchestration_coordinator_hop")
        shapes = {
            "untagged batch": {"tasks": [{"goal": "Validate the arithmetic claim.", "role": "orchestrator"}]},
            "untagged single": {"goal": "Validate the arithmetic claim.", "role": "orchestrator"},
            "tagged but hand-built": {
                "tasks": [
                    {
                        "goal": (
                            "Validate [loam-agent-role=hypothesis-validator] "
                            "[loam-plan-id=p] [loam-hypothesis-id=h-1]"
                        ),
                        "context": '{"loam_validator_contract": "v1"}',
                        "role": "orchestrator",
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            for label, arguments in shapes.items():
                with self.subTest(shape=label):
                    blocked = orchestration.tool_policy(
                        {
                            **payload,
                            "loam_hypothesis_orchestration": True,
                            "tool_name": "delegate_task",
                            "arguments": arguments,
                        }
                    )
                    self.assertEqual(blocked["action"], "block")
                    self.assertIn(orchestration.DISPATCH_TOOL, blocked["message"])

    def test_the_dispatch_tool_is_the_coordinator_s_one_legal_move(self) -> None:
        """Blocking it while a plan is pending would leave no legal move at all."""

        orchestration = _load("loam_orchestration_dispatch_allowed")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            verdict = orchestration.tool_policy(
                {
                    **payload,
                    "loam_hypothesis_orchestration": True,
                    "tool_name": orchestration.DISPATCH_TOOL,
                    "arguments": {},
                }
            )

        self.assertIsNone(verdict)

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
        self.assertIn(orchestration.DISPATCH_TOOL, blocked["message"])

    def test_evidence_score_accepts_markdown_code_span(self) -> None:
        orchestration = _load("loam_orchestration_markdown_evidence_score")

        self.assertEqual(orchestration._evidence_score("### Evidence Score\n`0.00` — inconclusive"), 0.0)

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

    def test_empty_tasks_array_is_single_task_not_a_blocked_batch(self) -> None:
        """A tagged goal beside tasks:[] must dispatch, not block.

        Live, gpt-6.1-sol filled every optional field and emitted tasks:[] next
        to a correctly tagged top-level goal. Hermes runs that as single mode;
        Loam was reading the empty list as a zero-task batch and blocking every
        dispatch, so no code-generator ever ran. Both the gate and the repair
        must now treat an empty list as single mode.
        """

        orchestration = _load("loam_orch_empty_tasks")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            orchestration.record_start({**payload, "child_session_id": "validator-session",
                "child_subagent_id": "validator-1",
                "child_goal": "Validate [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]"})
            base = {**payload, "loam_hypothesis_orchestration": True, "session_id": "validator-session"}
            tagged_goal = "Audit prerequisites [loam-agent-role=code-generator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]"
            args = {"goal": tagged_goal, "context": "", "tasks": [], "role": "leaf", "background": True}
            # The gate must allow it (not "may delegate only tagged ...").
            self.assertIsNone(orchestration.tool_policy({**base, "tool_name": "delegate_task", "arguments": args}))
            # The repair must build exactly one child, not an empty batch.
            repaired = orchestration.canonical_delegation_args({**base, "arguments": args})
            self.assertIsNotNone(repaired)
            self.assertEqual(len(repaired["tasks"]), 1)
            self.assertIn("[loam-agent-role=code-generator]", repaired["tasks"][0]["goal"])

    def test_every_role_may_record_a_decision(self) -> None:
        """loam_record_decision is provenance, not execution: no role gate may block it.

        A live campaign showed tool_policy refusing every validator's
        loam_record_decision call, so no decision was ever recorded though the
        validators were calling it with real content. This goes through
        tool_policy, the boundary the workspace unit tests never cross.
        """

        orchestration = _load("loam_orchestration_decision_tool")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            roles = {
                "validator-session": "Validate [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]",
                "codegen-session": "Execute [loam-agent-role=code-generator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]",
                "reasoner-session": "Synthesize [loam-agent-role=reasoner] [loam-plan-id=plan-1]",
            }
            for session, goal in roles.items():
                orchestration.record_start({**payload, "child_session_id": session,
                                            "child_subagent_id": session, "child_goal": goal})
            base = {**payload, "loam_hypothesis_orchestration": True}
            for session in roles:
                verdict = orchestration.tool_policy({
                    **base, "session_id": session, "tool_name": "loam_record_decision",
                    "arguments": {"what": "x", "why": "y"},
                })
                self.assertIsNone(verdict, f"{session} was blocked from recording a decision")

    def test_runbatch_nudge_fires_only_on_serial_run_code_without_a_batch(self) -> None:
        """L2 warn-only: a code-generator looping run_code with no run_batch gets
        an advisory nudge; it stays silent under the threshold, once a batch
        exists, and for other roles."""
        module = _load("ho_nudge")
        import hypothesis_workspace as hw
        RC, RB, SK = "mcp__hpc__run_code", "mcp__hpc__run_batch", "skill_view"

        def nudge(role, tools):
            with tempfile.TemporaryDirectory() as tmp:
                sd = Path(tmp) / "state"
                sd.mkdir(parents=True)
                # Match the real agent_started schema: the run DIR is named by
                # "run" ("r1"), while "run_id" is the campaign uuid and must NOT
                # name the dir. A distinct run_id here fails the nudge if it ever
                # reverts to reading run_id instead of run.
                ws = Path(tmp) / "ws"
                (ws / "runs" / "r1").mkdir(parents=True)
                row = {"event": "agent_started", "role": role, "child_session_id": "S1",
                       "run": "r1", "run_id": "campaign-run-uuid-9999",
                       "workspace": str(ws), "plan_id": "p1"}
                (sd / "hypothesis-orchestration.jsonl").write_text(json.dumps(row) + "\n")
                calls = [{"tool": t, "ts": "x"} for t in tools]
                (ws / "runs" / "r1" / hw.RUN_FILE).write_text(json.dumps({"calls": calls}))
                return module.runbatch_nudge_for({"session_id": "S1", "loam_state_dir": str(sd)})

        self.assertIn("RUN_BATCH REMINDER", nudge("code-generator", [SK, SK, RC, RC, RC, RC, RC]))
        self.assertEqual(nudge("code-generator", [RC, RC, RC, RC]), "")
        self.assertEqual(nudge("code-generator", [RC, RC, RC, RB, RC, RC, RC]), "")
        self.assertEqual(nudge("hypothesis-validator", [RC, RC, RC, RC, RC, RC]), "")

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
            self.assertIn(orchestration.DISPATCH_TOOL, first)
            # Shape lives in the dispatch arguments; the directive only announces.
            planned = orchestration.canonical_delegation_args(payload)["tasks"]
            self.assertEqual({task["role"] for task in planned}, {"orchestrator"})
            self.assertIn('"mode": "remote_mcp"', planned[0]["context"])
            plan_id = orchestration._plan_id(orchestration.Tournament(payload).latest_report()[0], payload)

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
                # C2: a trustworthy score requires a completed code-generator
                # execution for this plan+hypothesis. Stage one, as a real
                # validator would delegate, before recording the validator stop.
                code_gen = {
                    **payload,
                    "child_subagent_id": f"codegen-{index}",
                    "child_session_id": f"codegen-session-{index}",
                    "child_goal": (
                        "Execute [loam-agent-role=code-generator] "
                        f"[loam-plan-id={plan_id}] [loam-hypothesis-id={hypothesis_id}]"
                    ),
                }
                orchestration.record_start(code_gen)
                orchestration.record_stop({**code_gen, "child_status": "completed", "child_summary": "ran"})
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
            # The reasoner hop goes through the dispatch tool like the validator
            # hop: once every validator has reported, `canonical_delegation_args`
            # returns the reasoner task, so the coordinator has one move for both.
            reasoner_hop = orchestration.tool_policy(
                {
                    **payload,
                    "loam_hypothesis_orchestration": True,
                    "tool_name": "delegate_task",
                    "arguments": {"goal": f"Synthesize [loam-agent-role=reasoner] [loam-plan-id={plan_id}]"},
                }
            )
            self.assertEqual(reasoner_hop["action"], "block")
            self.assertIn(orchestration.DISPATCH_TOOL, reasoner_hop["message"])
            self.assertIsNone(
                orchestration.tool_policy(
                    {
                        **payload,
                        "loam_hypothesis_orchestration": True,
                        "tool_name": orchestration.DISPATCH_TOOL,
                        "arguments": {},
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

    def test_validator_score_without_execution_is_unbacked(self) -> None:
        # C2: a well-formed score is rejected as unbacked when NO code-generator
        # child completed an execution for this plan+hypothesis.
        orchestration = _load("loam_orchestration_unbacked_score")
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
                {**child, "child_status": "completed", "child_summary": "### Evidence Score\n0.00 — inconclusive"}
            )
            row = json.loads(
                (Path(payload["loam_state_dir"]) / "hypothesis-orchestration.jsonl").read_text().splitlines()[-1]
            )
        self.assertIsNone(row["evidence_score"])
        self.assertEqual(row["evidence_score_status"], "unbacked_no_exec")
        self.assertEqual(row["reported_evidence_score"], 0.0)

    # -- generated code-generator mode ---------------------------------------

    def _generated_payload(self, root: Path, **execution) -> dict:
        payload = self._payload(root)
        payload["loam_execution"] = {
            "mode": "remote_mcp",
            "tool_name": "mcp_hpc_run_code",
            "endpoint": "http://127.0.0.1:8000/mcp",
            "function_name": "run_code",
            "input_schema": "hpc-run-code-v1",
            "runtime": "python3.12",
            "workdir_root": "/lus/campaign/work",
            **execution,
        }
        return payload

    def _start_validator(self, orchestration, payload: dict) -> None:
        orchestration.record_start(
            {
                **payload,
                "child_subagent_id": "validator-1",
                "child_session_id": "validator-session-1",
                "child_goal": (
                    "Validate [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] "
                    "[loam-hypothesis-id=h-1]"
                ),
            }
        )

    def test_hpc_tool_names_derive_from_the_run_tool(self) -> None:
        orchestration = _load("loam_orchestration_hpc_tool_names")
        names = orchestration.hpc_tool_names(
            {"tool_name": "mcp_hpc_run_code", "function_name": "run_code"}
        )
        self.assertEqual(names["run_batch"], "mcp_hpc_run_batch")
        self.assertEqual(names["register_tool"], "mcp_hpc_register_tool")
        explicit = orchestration.hpc_tool_names(
            {"tool_name": "execute", "function_name": "run_code", "tool_prefix": "hpc__"}
        )
        self.assertEqual((explicit["run_code"], explicit["list_tools"]), ("execute", "hpc__list_tools"))
        bare = orchestration.hpc_tool_names({"tool_name": "execute", "function_name": "run_code"})
        self.assertEqual(bare, {"run_code": "execute"})

    def test_generated_directive_lets_validator_design_children(self) -> None:
        orchestration = _load("loam_orchestration_generated_directive")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._generated_payload(Path(temporary))
            directive = orchestration.context_for(dict(payload))
            tasks = orchestration.canonical_delegation_args(payload)["tasks"]
        self.assertIn("at most 2 leaf code-generator children", directive)
        context = json.loads(tasks[0]["context"])
        self.assertNotIn("required_child", context)
        self.assertNotIn("Fraction", tasks[0]["context"])
        self.assertEqual(context["loam_validator_contract"], "v1")
        self.assertEqual(context["child_template"]["max_children"], 2)
        self.assertIn("[loam-agent-role=code-generator]", context["child_template"]["goal_suffix"])
        self.assertTrue(context["child_template"]["workdir"].startswith("/lus/campaign/work/plan_"))
        self.assertIn("### Evidence Score", context["validator_instructions"])

    def test_validator_definition_is_injected_not_referenced_as_a_path(self) -> None:
        """A validator may call only delegate_task, so its prompt must be inlined.

        Handing it `prompt_path` to read fails closed in tool_policy, which left
        the validator running with no role instructions at all.
        """

        orchestration = _load("loam_orchestration_injected_definition")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._generated_payload(Path(temporary))
            directive = orchestration.context_for(payload)
            registry = orchestration.agent_registry(payload)
            tasks = orchestration.canonical_delegation_args(payload)["tasks"]

        body = registry["hypothesis-validator"].prompt_body
        self.assertTrue(body, "validator prompt body should be loaded")
        self.assertFalse(body.startswith("---"), "front matter should be stripped")
        self.assertNotIn("name: hypothesis-validator", body)
        self.assertLessEqual(len(body), 24_000)

        raw_context = tasks[0]["context"]
        context = json.loads(raw_context)
        self.assertEqual(context["agent_definition"], body)
        # The directive must not tell the child to read a file it cannot read.
        self.assertNotIn("MUST read and follow", directive)

        # The assignment must precede the role boilerplate in the serialized
        # payload. sort_keys=True silently ordered `agent_definition` first and
        # pushed `selected_hypothesis` ~9000 chars in, so assert on byte
        # positions rather than dict keys, which carry no order through JSON.
        self.assertLess(
            raw_context.index('"selected_hypothesis"'),
            raw_context.index('"agent_definition"'),
            "selected_hypothesis must come before the long agent_definition",
        )
        self.assertLess(
            raw_context.index('"validator_instructions"'),
            raw_context.index('"agent_definition"'),
        )
        # tool_policy matches this exact substring; serialization must keep it.
        self.assertIn('"loam_validator_contract": "v1"', raw_context)
        # The validator must be told to pass the hypothesis text down, because a
        # child cannot resolve a bare id.
        self.assertIn("provenance tag", context["validator_instructions"])

    def test_generated_canonical_keeps_instruction_and_forces_tags(self) -> None:
        orchestration = _load("loam_orchestration_generated_canonical")
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._generated_payload(Path(temporary))
            self._start_validator(orchestration, payload)
            canonical = orchestration.canonical_delegation_args(
                {
                    **payload,
                    "session_id": "validator-session-1",
                    "arguments": {
                        "tasks": [
                            {
                                "goal": "Sweep T from 300 to 900 K [loam-plan-id=other-plan]",
                                "context": "Use 500 points; report the peak.",
                                "role": "orchestrator",
                            }
                        ]
                    },
                }
            )
            missing = orchestration.canonical_delegation_args(
                {**payload, "session_id": "validator-session-1", "arguments": {"tasks": [{"goal": " "}]}}
            )

        task = canonical["tasks"][0]
        self.assertEqual(task["role"], "leaf")
        self.assertNotIn("other-plan", task["goal"])
        # The goal now leads with the hypothesis (so the child can't miss it),
        # then preserves the validator's instruction, then the forced tags.
        self.assertIn("Test this binder hypothesis", task["goal"])
        self.assertIn("Sweep T from 300 to 900 K", task["goal"])
        self.assertIn("[loam-agent-role=code-generator]", task["goal"])
        self.assertIn("[loam-plan-id=plan-1] [loam-hypothesis-id=h-1]", task["goal"])
        self.assertIn("Use 500 points; report the peak.", task["context"])
        self.assertIn("mcp_hpc_list_tools", task["context"])
        self.assertIn("mcp_hpc_run_batch", task["context"])
        self.assertIn("/lus/campaign/work/plan-1/h-1", task["context"])
        self.assertIn("never /tmp", task["context"])
        self.assertNotIn("Fraction", task["context"])
        self.assertIsNone(missing)

    def test_generated_policy_caps_children_and_allows_hpc_tools(self) -> None:
        orchestration = _load("loam_orchestration_generated_policy")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {**self._generated_payload(Path(temporary)), "loam_hypothesis_orchestration": True}
            self._start_validator(orchestration, payload)
            validator = {**payload, "session_id": "validator-session-1", "tool_name": "delegate_task"}
            goal = "Run [loam-agent-role=code-generator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]"
            first = orchestration.tool_policy({**validator, "arguments": {"goal": goal}})
            too_many = orchestration.tool_policy(
                {**validator, "arguments": {"tasks": [{"goal": goal}, {"goal": goal}, {"goal": goal}]}}
            )
            for index in range(2):
                orchestration.record_start(
                    {
                        **payload,
                        "child_subagent_id": f"code-generator-{index}",
                        "child_session_id": f"code-generator-session-{index}",
                        "child_goal": goal,
                    }
                )
            exhausted = orchestration.tool_policy({**validator, "arguments": {"goal": goal}})
            code_generator = {**payload, "session_id": "code-generator-session-0"}
            batch = orchestration.tool_policy({**code_generator, "tool_name": "mcp_hpc_run_batch", "arguments": {}})
            register = orchestration.tool_policy(
                {**code_generator, "tool_name": "mcp_hpc_register_tool", "arguments": {}}
            )
            terminal = orchestration.tool_policy({**code_generator, "tool_name": "terminal", "arguments": {}})

        self.assertIsNone(first)
        self.assertEqual(too_many["action"], "block")
        self.assertIn("budget", exhausted["message"])
        self.assertIsNone(batch)
        self.assertIsNone(register)
        self.assertEqual(terminal["action"], "block")

class EvidenceScoreTests(unittest.TestCase):
    """M2 (evidence-score parsing) and H2 (fs-recovery neutralizer)."""

    def setUp(self) -> None:
        self.orch = _load("loam_orchestration_m2_h2")

    def test_score_requires_a_heading_not_an_in_prose_mention(self) -> None:
        # M2 false positive: a recap sentence must not be read as the score.
        self.assertIsNone(self.orch._evidence_score(
            "Earlier the prior Evidence Score was 0.00, but we re-ran it."))

    def test_score_tolerates_markup_and_a_lead_in(self) -> None:
        # M2 false negatives: bold markup and a lead-in line before the number.
        self.assertEqual(self.orch._evidence_score(
            "### Evidence Score\nsupported\n**+0.75** — ipTM 0.75"), 0.75)
        self.assertEqual(self.orch._evidence_score("## Evidence Score\n\n0.18"), 0.18)

    def test_last_heading_wins(self) -> None:
        # A recap heading then the final verdict -> the final verdict's number.
        self.assertEqual(self.orch._evidence_score(
            "### Evidence Score\n0.10 (prior)\n\n### Evidence Score\n+0.90"), 0.90)

    def test_score_ignores_a_number_in_the_heading_lead_in(self) -> None:
        # H1: `start` is the end of the "Evidence Score" match, so the rest of
        # the heading line was still in the window: the `0` of "(0-1" was read
        # as a neutral 0.0 score and recorded as "reported".
        self.assertEqual(self.orch._evidence_score(
            "### Evidence Score (0-1 scale)\n\n0.42"), 0.42)
        self.assertEqual(self.orch._evidence_score(
            "### Evidence Score\n\nBased on 3 folds, 0.42"), 0.42)
        # A bare integer is still read when it is the first token of the body.
        self.assertEqual(self.orch._evidence_score("### Evidence Score\n\n0"), 0.0)

if __name__ == "__main__":
    unittest.main()
