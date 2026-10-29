"""Validator outcomes become tagged FALDA atoms; writing never disturbs orchestration."""

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

import falda_campaign  # noqa: E402
import falda_flush  # noqa: E402
import falda_memory  # noqa: E402
from falda_fake import POOL, FakeTransport, falda_block  # noqa: E402

PAYLOAD = {"workflow_id": "wf-1", "loam_execution": {"mode": "local"}}
COMMON = ["workflow:wf-1", "run:run-1", "plan:plan-1", "hypothesis:h-1", "source:validator-report"]
SUFFIX = "(validator report, unreconciled; plan plan-1, run run-1)"


def _load_orchestration(name: str):
    spec = importlib.util.spec_from_file_location(name, HOOKS / "hypothesis_orchestration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _report(
    outcome: str = "contradicted",
    score: str = "-0.80",
    ledger: bool = True,
    next_step: str = "none",
    hypothesis: str = "The mean of [1, 2, 3, 4] exceeds 2.4.",
) -> str:
    lines = ["### Hypothesis", hypothesis, "### Outcome", f"`{outcome}` — reason given", "### Evidence Ledger"]
    if ledger:
        lines += [
            "| ID | Action and source | Observation | Direction | Reliability / confounders |",
            "| --- | --- | --- | --- | --- |",
            "| E1 | python3 Fraction check | mean=5/2 | supports | exact |",
            "| E2 | float check | mean=2.4999 | contradicts | rounding |",
        ]
    lines += ["### Credit Assignment", "| Evidence ID | Contribution | Credit |", "### Evidence Score", f"{score} — calibrated"]
    lines += ["### Next Step", next_step]
    return "\n".join(lines)


def _record(summary: str, score):
    return {"plan_id": "plan-1", "run_id": "run-1", "hypothesis_id": "h-1", "summary": summary, "evidence_score": score}


class OutcomeAtomTests(unittest.TestCase):
    def test_contradicted_is_negative_with_decisive_contradicting_row(self) -> None:
        atom, reason = falda_campaign.outcome_atom(PAYLOAD, _record(_report(), -0.8))
        self.assertEqual(reason, "")
        self.assertEqual(atom["tags"], ["negative", *COMMON])
        self.assertEqual((atom["type"], atom["confidence"]), ("fact", "medium"))
        self.assertTrue(atom["content"].startswith("Contradicted (evidence score -0.80): The mean of [1, 2, 3, 4] exceeds 2.4. "))
        self.assertIn("Decisive evidence: float check: mean=2.4999.", atom["content"])
        self.assertIn("Conditions: execution mode local; actions python3 Fraction check; float check.", atom["content"])
        self.assertTrue(atom["content"].endswith(SUFFIX))

    def test_contradiction_without_a_contradicting_row_records_no_decisive_evidence(self) -> None:
        summary = _report().replace("| E2 | float check | mean=2.4999 | contradicts | rounding |\n", "")
        summary = summary.replace("| E2 | float check | mean=2.4999 | contradicts | rounding |", "")
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(summary, -0.8))
        self.assertIn("Decisive evidence: none recorded.", atom["content"])
        self.assertNotIn("Fraction check: mean", atom["content"].split("Conditions")[0])

    def test_blocked_is_negative_and_names_the_missing_prerequisite(self) -> None:
        summary = _report("blocked", "0.00", next_step="Install Chai-1 weights on the host.")
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(summary, 0.0))
        self.assertEqual(atom["tags"], ["negative", *COMMON])
        self.assertIn("Missing prerequisite: Install Chai-1 weights on the host.", atom["content"])
        self.assertEqual(atom["confidence"], "low")

    def test_blocked_without_a_score_is_still_written_low(self) -> None:
        atom, reason = falda_campaign.outcome_atom(PAYLOAD, _record(_report("blocked"), None))
        self.assertEqual((reason, atom["confidence"]), ("", "low"))

    def test_inconclusive_is_coverage_and_lists_what_was_tried(self) -> None:
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(_report("inconclusive", "+0.10"), 0.1))
        self.assertEqual(atom["tags"], ["coverage", *COMMON])
        self.assertTrue(atom["content"].startswith("Tested, inconclusive (+0.10): "))
        self.assertIn("What was tried: python3 Fraction check; float check.", atom["content"])
        self.assertEqual(atom["confidence"], "low")

    def test_supported_has_no_category_tag(self) -> None:
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(_report("supported", "+0.75"), 0.75))
        self.assertEqual(atom["tags"], COMMON)
        self.assertTrue(atom["content"].startswith("Supported (+0.75) for what was tested: reason given. "))
        self.assertIn("Proposed claim, supported only as far as that test reaches: The mean of [1, 2, 3, 4] exceeds 2.4.", atom["content"])
        self.assertIn("Decisive evidence: python3 Fraction check: mean=5/2.", atom["content"])

    def test_supported_scope_comes_from_the_outcome_reason(self) -> None:
        summary = _report("supported", "+1.00", hypothesis="For every n >= 1 the mean of n..n+3 exceeds 2.4.").replace(
            "`supported` — reason given", "**supported** – the run confirmed that for {1, 2, 3, 4} the mean is 5/2 > 12/5."
        )
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(summary, 1.0))
        self.assertTrue(
            atom["content"].startswith("Supported (+1.00) for what was tested: the run confirmed that for {1, 2, 3, 4} the mean is 5/2 > 12/5. ")
        )
        self.assertIn("supported only as far as that test reaches: For every n >= 1", atom["content"])

    def test_supported_without_a_reason_keeps_the_short_form(self) -> None:
        summary = _report("supported", "+0.75").replace("`supported` — reason given", "supported")
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(summary, 0.75))
        self.assertTrue(atom["content"].startswith("Supported (+0.75): The mean of [1, 2, 3, 4] exceeds 2.4. Decisive evidence:"))

    def test_missing_outcome_or_score_is_skipped(self) -> None:
        self.assertEqual(falda_campaign.outcome_atom(PAYLOAD, _record("no headings at all", 0.5)), (None, "missing_outcome"))
        self.assertEqual(falda_campaign.outcome_atom(PAYLOAD, _record(_report("supported"), None)), (None, "missing_score"))

    def test_report_variants_parse(self) -> None:
        for line in ("contradicted", "**Contradicted** — x", "`contradicted`: x", "Contradicted."):
            with self.subTest(line=line):
                summary = _report().replace("`contradicted` — reason given", line)
                atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(summary, -0.6))
                self.assertEqual(atom["tags"][0], "negative")
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(_report(ledger=False), -0.6))
        self.assertIn("Decisive evidence: none recorded.", atom["content"])

    def test_outcome_keyword_later_in_the_section_is_accepted_when_unambiguous(self) -> None:
        summary = _report().replace("`contradicted` — reason given", "Remote execution shows the claim is **supported**: mean 5/2 > 12/5.")
        atom, reason = falda_campaign.outcome_atom(PAYLOAD, _record(summary, 0.9))
        self.assertEqual(reason, "")
        self.assertEqual(atom["tags"], COMMON)

    def test_prose_outcome_without_a_keyword_or_with_two_is_skipped(self) -> None:
        for line in (
            "The remote execution confirms the hypothesis: mean = 5/2 > 2.4.",
            "Not contradicted by the float check, so supported overall.",
        ):
            with self.subTest(line=line):
                summary = _report().replace("`contradicted` — reason given", line)
                self.assertEqual(falda_campaign.outcome_atom(PAYLOAD, _record(summary, 0.9)), (None, "missing_outcome"))

    def test_bullet_ledger_uses_its_code_block_as_decisive_evidence(self) -> None:
        ledger = (
            "- **Remote hostname:** `x4710c5s7b0n0`\n- **Return code:** `0`\n- **Stdout:**\n\n"
            "```json\n{\n  \"exceeds\": true,\n  \"mean\": \"5/2\"\n}\n```\n"
        )
        summary = _report(outcome="supported", ledger=False).replace("### Evidence Ledger\n", "### Evidence Ledger\n" + ledger)
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(summary, 0.9))
        self.assertIn('Decisive evidence: {"exceeds": true, "mean": "5/2"}.', atom["content"])

    def test_bullet_ledger_without_code_uses_its_first_bullets(self) -> None:
        ledger = "- **Remote hostname:** `x1`\n- **Return code:** `0`\n- **Stderr:** *(empty)*\n- **Extra:** four\n"
        summary = _report(outcome="supported", ledger=False).replace("### Evidence Ledger\n", "### Evidence Ledger\n" + ledger)
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(summary, 0.9))
        self.assertIn("Decisive evidence: Remote hostname: x1; Return code: 0; Stderr: (empty).", atom["content"])

    def test_content_is_capped_and_keeps_its_provenance_suffix(self) -> None:
        atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(_report(hypothesis="x" * 5000), -0.9))
        self.assertLessEqual(len(atom["content"]), 1200)
        self.assertTrue(atom["content"].endswith(SUFFIX))

    def test_confidence_is_never_high(self) -> None:
        for score in (-1.0, -0.5, -0.49, 0.0, 0.49, 0.5, 1.0):
            with self.subTest(score=score):
                atom, _ = falda_campaign.outcome_atom(PAYLOAD, _record(_report("supported"), score))
                self.assertEqual(atom["confidence"], "medium" if abs(score) >= 0.5 else "low")


class ValidatorOutcomeWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {"FALDA_TOKEN": "falda_test"})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(falda_memory.set_transport, None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = Path(temporary.name)

    def _flush(self) -> dict:
        """Run what the detached flusher would (spawn_flush is recorded, not run, in tests)."""

        return falda_flush.flush(self.state, falda_block())

    def _stop(self, name: str, falda, transport, role="hypothesis-validator", summary=None):
        orchestration = _load_orchestration(name)
        falda_memory.set_transport(transport)
        payload = {
            "repo_root": str(ROOT),
            "workflow_id": "wf-1",
            "loam_run_id": "run-1",
            "loam_state_dir": str(self.state),
            "loam_execution": {"mode": "local"},
        }
        if falda is not None:
            payload["loam_falda"] = falda
        child = {
            **payload,
            "child_subagent_id": "child-1",
            "child_session_id": "child-session-1",
            "child_goal": f"Work [loam-agent-role={role}] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]",
        }
        # develop keeps a validator's score only when a code-generator child completed for the same plan
        # and hypothesis (unbacked scores become None); back it, as a real generated run would.
        executor = {
            **payload,
            "child_subagent_id": "exec-1",
            "child_session_id": "exec-session-1",
            "child_goal": "Run [loam-agent-role=code-generator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]",
        }
        orchestration.record_start(executor)
        orchestration.record_stop({**executor, "child_status": "completed", "child_summary": "ran"})
        orchestration.record_start(child)
        orchestration.record_stop({**child, "child_status": "completed", "child_summary": summary or _report()})

    def _rows(self, name: str) -> list[dict]:
        path = self.state / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []

    def test_validator_stop_writes_one_tagged_atom_to_the_pool(self) -> None:
        fake = FakeTransport()
        self._stop("orchestration_falda_write", falda_block(), fake)
        self.assertEqual(fake.requests, [], "the hook itself must not touch the network")
        self.assertEqual(len(falda_flush.spawned), 1)
        [row] = self._rows(falda_flush.SPOOL)
        self.assertEqual((row["plan_id"], row["hypothesis_id"]), ("plan-1", "h-1"))
        self._flush()
        [body] = fake.bodies("/atoms/upsert")
        self.assertEqual(body["pool"], POOL)
        self.assertEqual(body["atoms"][0]["tags"], ["negative", *COMMON])
        self.assertEqual(self._rows("falda-calls.jsonl")[-1]["consumer"], "validator_outcome")

    def test_switch_off_or_no_block_writes_nothing(self) -> None:
        for name, falda in (("orchestration_falda_off", falda_block(write_validator_outcomes=False)), ("orchestration_falda_none", None)):
            with self.subTest(name=name):
                fake = FakeTransport()
                self._stop(name, falda, fake)
                self.assertEqual(fake.requests, [])
                self.assertEqual(self._rows(falda_flush.SPOOL), [])

    def test_reasoner_and_code_generator_stops_write_nothing(self) -> None:
        for role in ("reasoner", "code-generator"):
            with self.subTest(role=role):
                fake = FakeTransport()
                self._stop(f"orchestration_falda_{role}", falda_block(), fake, role=role)
                self.assertEqual(self._rows(falda_flush.SPOOL), [])

    def test_missing_outcome_logs_a_skip_and_sends_nothing(self) -> None:
        fake = FakeTransport()
        self._stop("orchestration_falda_skip", falda_block(), fake, summary="### Evidence Score\n0.5 — x")
        self.assertEqual(fake.requests, [])
        self.assertEqual(self._rows(falda_flush.SPOOL), [])
        skip = self._rows("falda-calls.jsonl")[-1]
        self.assertEqual((skip["event"], skip["reason"], skip["plan_id"]), ("falda_write_skipped", "missing_outcome", "plan-1"))

    def test_falda_failure_leaves_orchestration_records_unchanged(self) -> None:
        self._stop("orchestration_falda_down", falda_block(), FakeTransport(error=ConnectionRefusedError()))
        self._flush()
        self.assertEqual(
            [row["event"] for row in self._rows("hypothesis-orchestration.jsonl")],
            ["agent_started", "code_generator_completed", "agent_started", "validator_completed"],
        )
        self.assertEqual(self._rows("hypothesis-orchestration.jsonl")[-1]["evidence_score"], -0.8)
        self.assertEqual(
            [row["event"] for row in self._rows("hypothesis-campaign.jsonl")],
            ["hypothesis_validation_started", "hypothesis_validation_completed"],
        )
        self.assertFalse(self._rows("falda-calls.jsonl")[-1]["ok"])

    def test_unbacked_validator_score_is_not_written(self) -> None:
        orchestration = _load_orchestration("orchestration_falda_unbacked")
        payload = {
            "repo_root": str(ROOT), "workflow_id": "wf-1", "loam_run_id": "run-1",
            "loam_state_dir": str(self.state), "loam_execution": {"mode": "local"}, "loam_falda": falda_block(),
        }
        child = {
            **payload, "child_subagent_id": "child-1", "child_session_id": "child-session-1",
            "child_goal": "Work [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]",
        }
        orchestration.record_start(child)
        orchestration.record_stop({**child, "child_status": "completed", "child_summary": _report()})
        self.assertEqual(self._rows(falda_flush.SPOOL), [])
        self.assertEqual(self._rows("falda-calls.jsonl")[-1]["reason"], "missing_score")

    def test_record_stop_returns_fast_when_falda_hangs(self) -> None:
        import time

        def _hang(url, headers, body, timeout):
            time.sleep(5)
            return 200, b"{}"

        start = time.monotonic()
        self._stop("orchestration_falda_hang", falda_block(), _hang)
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual(len(self._rows(falda_flush.SPOOL)), 1)

    def test_blank_workflow_id_falls_back_to_the_experiment_tag(self) -> None:
        orchestration = _load_orchestration("orchestration_falda_blank_wf")
        fake = FakeTransport()
        falda_memory.set_transport(fake)
        payload = {
            "repo_root": str(ROOT),
            "workflow_id": "",
            "loam_run_id": "run-1",
            "loam_state_dir": str(self.state),
            "loam_execution": {"mode": "local"},
            "loam_falda": falda_block(),
        }
        child = {
            **payload,
            "child_subagent_id": "child-1",
            "child_session_id": "child-session-1",
            "child_goal": "Work [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]",
        }
        executor = {
            **payload,
            "child_subagent_id": "exec-1",
            "child_session_id": "exec-session-1",
            "child_goal": "Run [loam-agent-role=code-generator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]",
        }
        orchestration.record_start(executor)
        orchestration.record_stop({**executor, "child_status": "completed", "child_summary": "ran"})
        orchestration.record_start(child)
        with patch.dict(os.environ, {"CAIRN_PROJECT": "exp-7"}):
            orchestration.record_stop({**child, "child_status": "completed", "child_summary": _report()})
        self.assertEqual(fake.requests, [])
        [row] = self._rows(falda_flush.SPOOL)
        self.assertIn("workflow:exp-7", row["atom"]["tags"])


if __name__ == "__main__":
    unittest.main()
