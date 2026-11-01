"""Behavior tests for independent evidence review and autonomous acceptance.

The acceptance boundary protects independence of the acceptor from the
producer. These tests pin that property rather than the mechanics: that a
reviewer is dispatched for a claim it did not make, that it is never handed the
producing validator's prose, that it cannot weight its own acceptance, and that
a campaign which has not opted in reads exactly the round it read before.
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


def _load(name: str, filename: str = "hypothesis_orchestration.py"):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, HOOKS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CLAIM_COMPUTED = (
    "Cross-conformer local Calpha RMSD for the domain-II/III component was 11.71 A against a "
    "predeclared <=3.0 A gate, computed from report.json; the component failed."
)
CLAIM_SOURCE = "PDB 1NQL is an inactive EGFR ectodomain complex."


def _candidate(unit_id: str = "unit-1", child_session_id: str = "validator-session") -> dict:
    return {
        "event": "candidate_result",
        "record_version": "hypothesis_provenance_v1",
        "candidate_status": "pending",
        "candidate_id": "candidate-1",
        "unit_id": unit_id,
        "run_id": "run-1",
        "workflow_id": "research-v1",
        "parent_session_id": "session-1",
        "parent_turn_id": "turn-1",
        "child_session_id": child_session_id,
        "child_subagent_id": "validator-1",
        "candidate": {
            "contract_version": "hypothesis_provenance_v1",
            "hypotheses": [{"hypothesis_key": "h1", "statement": "The DII/III component is robust.", "parent_hypothesis_keys": []}],
            "evidence_candidates": [
                {"hypothesis_key": "h1", "claim": CLAIM_COMPUTED,
                 "source_url": "https://www.rcsb.org/structure/1NQL", "stance": "contradicts"},
                {"hypothesis_key": "h1", "claim": CLAIM_SOURCE,
                 "source_url": "https://www.rcsb.org/structure/1NQL", "stance": "context"},
            ],
        },
    }


def _review_contract(**overrides) -> dict:
    item = {
        "hypothesis_key": "h1",
        "claim": CLAIM_COMPUTED,
        "source_url": "https://www.rcsb.org/structure/1NQL",
        "stance": "contradicts",
        "verdict": "accepted",
        "verification": "artifact_attested",
        "source_ref": "outputs/r1/report.json#rmsd",
        "checked": "cat shows 11.71 in outputs/r1/report.json",
    }
    item.update(overrides)
    return {
        "contract_version": "hypothesis_review_v1",
        "candidate_id": "candidate-1",
        "accept_hypothesis_keys": ["h1"],
        "reviews": [item],
    }


class ReviewContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.review = _load("loam_review_contract", "hypothesis_review.py")

    def test_an_acceptance_cannot_set_its_own_weight(self) -> None:
        """Strength, reliability and relevance come from the verification mode.

        A reviewer that could weight its own acceptance could turn a weak
        attestation into a strong one with nothing downstream able to see the
        difference, so values in its output are ignored outright.
        """

        contract = _review_contract(strength=1.0, reliability=1.0, relevance=1.0)
        row, status = self.review.normalize_review(contract, _candidate())
        self.assertEqual(status, "reviewed")
        accepted = row["evidence"][0]
        self.assertEqual(
            {key: accepted[key] for key in ("strength", "reliability", "relevance")},
            self.review.WEIGHT_POLICY["artifact_attested"],
        )

    def test_an_attested_metric_carries_less_than_a_confirmed_source(self) -> None:
        attested = self.review.WEIGHT_POLICY["artifact_attested"]
        confirmed = self.review.WEIGHT_POLICY["source_confirmed"]
        self.assertLess(attested["reliability"], confirmed["reliability"])

    def test_an_acceptance_that_names_no_check_is_recorded_unverifiable(self) -> None:
        """Accepted without a verification mode or a source_ref is an assertion.

        It is downgraded rather than dropped, so the audit row says the reviewer
        tried to accept and could not say on what basis.
        """

        for missing in ({"verification": ""}, {"source_ref": ""}):
            with self.subTest(missing=missing):
                row, _ = self.review.normalize_review(_review_contract(**missing), _candidate())
                self.assertEqual(row["evidence"], [])
                self.assertEqual(row["review_verdicts"]["unverifiable"], 1)
                self.assertIn("downgraded", row["review_audit"][0]["checked"])

    def test_an_edited_claim_is_dropped_rather_than_accepted(self) -> None:
        """A review item must match its candidate item exactly.

        A reviewer cannot restate a claim into one it is willing to accept: the
        reconciler refuses an unmatched item, so this drops it here and counts
        it as unreviewed.
        """

        row, _ = self.review.normalize_review(
            _review_contract(claim=CLAIM_COMPUTED.replace("11.71", "1.71")), _candidate()
        )
        self.assertEqual(row["dropped_unmatched"], 1)
        self.assertEqual(row["evidence"], [])

    def test_a_claim_review_of_another_candidate_is_refused(self) -> None:
        contract = _review_contract()
        contract["candidate_id"] = "candidate-elsewhere"
        row, status = self.review.normalize_review(contract, _candidate())
        self.assertIsNone(row)
        self.assertEqual(status, "candidate_id_mismatch")

    def test_a_hypothesis_needs_both_approval_and_surviving_evidence(self) -> None:
        contract = _review_contract()
        contract["accept_hypothesis_keys"] = []
        row, _ = self.review.normalize_review(contract, _candidate())
        self.assertEqual(row["accept_hypothesis_keys"], [])
        self.assertEqual(row["evidence"], [], "evidence of an unaccepted hypothesis must not be accepted")

    def test_one_units_claims_share_an_independence_group(self) -> None:
        """Five failed gates from one validator are not five independent contradictions.

        `project_hypothesis_steering.py` decays repeats within a group, so
        grouping artifact-attested claims by their producing unit is what stops
        a single run from counting as a consensus.
        """

        group = self.review.independence_group("artifact_attested", "https://example.org/a", "unit-1")
        other = self.review.independence_group("artifact_attested", "https://example.org/b", "unit-1")
        self.assertEqual(group, other)
        self.assertEqual(
            self.review.independence_group("source_confirmed", "https://example.org/a", "unit-1"),
            "source:https://example.org/a",
        )


class ReviewPhaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestration = _load("loam_review_phase")

    def _payload(self, root: Path, review: bool = True) -> dict:
        state = root / "loam-state"
        state.mkdir(parents=True, exist_ok=True)
        reports = root / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "tournament-001.json").write_text(
            json.dumps({
                "workflow_id": "research-v1",
                "result": {"status": "completed", "top_hypotheses": [{"hypothesis_id": "h-1", "elo_rating": 1210.0}]},
            }),
            encoding="utf-8",
        )
        (root / "jnana.json").write_text(
            json.dumps({"hypotheses": [{"hypothesis_id": "h-1", "content": "First hypothesis."}]}), encoding="utf-8"
        )
        (state / "candidate-results.jsonl").write_text(json.dumps(_candidate()) + "\n", encoding="utf-8")
        (state / "hypothesis-orchestration.jsonl").write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in (
                    {"event": "agent_started", "role": "hypothesis-validator", "plan_id": "plan-1",
                     "hypothesis_id": "h-1", "workflow_id": "research-v1", "run_id": "run-1",
                     "child_session_id": "validator-session", "child_subagent_id": "validator-1"},
                    {"event": "validator_completed", "role": "hypothesis-validator", "plan_id": "plan-1",
                     "hypothesis_id": "h-1", "workflow_id": "research-v1", "run_id": "run-1",
                     "child_session_id": "validator-session", "child_subagent_id": "validator-1",
                     "status": "completed", "summary": "### Evidence Score\n-0.80"},
                )
            ),
            encoding="utf-8",
        )
        return {
            "repo_root": str(ROOT),
            "workflow_id": "research-v1",
            "session_id": "session-1",
            "loam_run_id": "run-1",
            "loam_state_dir": str(state),
            "loam_hypothesis_orchestration": True,
            "loam_evidence_review": review,
            "loam_hypothesis_tournament": {"state_file": str(root / "jnana.json"), "reports_dir": str(reports)},
            "loam_execution": {
                "mode": "remote_mcp",
                "tool_name": "hpc_run_code",
                "tool_prefix": "hpc_",
                "function_name": "run_code",
                "endpoint": "https://compute.example/mcp",
                "runtime": "python3.11",
            },
        }

    def test_review_runs_between_the_validators_and_the_reasoner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            state = self.orchestration._round(payload, self.orchestration._rows(payload))
            self.assertEqual(state["phase"], "reviewers_due")
            self.assertEqual([item["candidate_id"] for item in state["candidates"]], ["candidate-1"])

    def test_a_campaign_that_has_not_opted_in_keeps_its_round_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary), review=False)
            state = self.orchestration._round(payload, self.orchestration._rows(payload))
            self.assertEqual(state["phase"], "reasoner_due")

    def test_the_reviewer_is_not_given_the_validators_report(self) -> None:
        """The producing validator's prose is the object under review.

        An agent handed a claim plus the argument for it tends to confirm it, so
        the dispatch carries the claims, their sources and the campaign's hashed
        artifacts -- and nothing the validator wrote about them.
        """

        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            tasks = self.orchestration.canonical_delegation_args(payload)["tasks"]
            self.assertEqual(len(tasks), 1)
            context = json.loads(tasks[0]["context"])
            self.assertEqual(context["candidate_id"], "candidate-1")
            self.assertEqual(len(context["evidence_candidates"]), 2)
            self.assertNotIn("Evidence Score", tasks[0]["context"])
            self.assertNotIn("validator_report", context)
            self.assertIn("[loam-agent-role=evidence-reviewer]", tasks[0]["goal"])
            self.assertIn("[loam-candidate-id=candidate-1]", tasks[0]["goal"])
            self.assertEqual(sorted(context["read_only_tools"]),
                             ["hpc_batch_status", "hpc_inspect_filesystem", "hpc_list_batches"])

    def test_a_reviewer_that_returned_nothing_usable_does_not_stall_the_round(self) -> None:
        """Re-dispatch keys on dispatch, not on a review row appearing.

        A reviewer whose own API failed, or which emitted no contract, has still
        had its turn at the claim; keying on the review row would put the round
        in a loop that never reaches its reasoner.
        """

        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            path = Path(payload["loam_state_dir"]) / "hypothesis-orchestration.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                for row in (
                    {"event": "agent_started", "role": "evidence-reviewer", "plan_id": "plan-1",
                     "hypothesis_id": "h-1", "workflow_id": "research-v1", "run_id": "run-1",
                     "candidate_id": "candidate-1", "child_session_id": "reviewer-session"},
                    {"event": "evidence_review_completed", "role": "evidence-reviewer", "plan_id": "plan-1",
                     "workflow_id": "research-v1", "run_id": "run-1", "candidate_id": "candidate-1",
                     "child_session_id": "reviewer-session", "status": "completed",
                     "review_status": "contract_absent"},
                ):
                    handle.write(json.dumps(row) + "\n")
            state = self.orchestration._round(payload, self.orchestration._rows(payload))
            self.assertEqual(state["phase"], "reasoner_due")

    def test_a_reviewer_may_read_but_not_execute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            path = Path(payload["loam_state_dir"]) / "hypothesis-orchestration.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "agent_started", "role": "evidence-reviewer", "plan_id": "plan-1",
                    "hypothesis_id": "h-1", "workflow_id": "research-v1", "run_id": "run-1",
                    "candidate_id": "candidate-1", "child_session_id": "reviewer-session",
                }) + "\n")
            reviewer = {**payload, "session_id": "reviewer-session"}
            for tool, blocked in (
                ("hpc_inspect_filesystem", False),
                ("hpc_batch_status", False),
                ("skill_view", False),
                ("hpc_run_code", True),
                ("hpc_run_batch", True),
                ("delegate_task", True),
            ):
                with self.subTest(tool=tool):
                    verdict = self.orchestration.tool_policy({**reviewer, "tool_name": tool, "arguments": {}})
                    self.assertEqual(verdict is not None, blocked)

    def test_the_reasoner_is_told_what_review_decided(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._payload(Path(temporary))
            (Path(payload["loam_state_dir"]) / "hypothesis-reviews.jsonl").write_text(
                json.dumps({
                    "event": "evidence_review", "plan_id": "plan-1", "candidate_id": "candidate-1",
                    "workflow_id": "research-v1",
                    "review_verdicts": {"accepted": 1, "rejected": 1, "unverifiable": 3},
                    "evidence": [{"hypothesis_key": "h1"}],
                }) + "\n",
                encoding="utf-8",
            )
            summary = self.orchestration._review_summary(payload, "plan-1")
            self.assertIn("1 claims accepted, 1 rejected, 3 unverifiable", summary)
            self.assertIn("NOT a refuted one", summary)


class AcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.acceptance = _load("loam_acceptance", "hypothesis_acceptance.py")
        self.review = _load("loam_acceptance_review", "hypothesis_review.py")

    def _state(self, root: Path) -> dict:
        state = root / "loam-state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "candidate-results.jsonl").write_text(json.dumps(_candidate()) + "\n", encoding="utf-8")
        payload = {
            "repo_root": str(ROOT),
            "workflow_id": "research-v1",
            "session_id": "session-1",
            "loam_state_dir": str(state),
        }
        row, _ = self.review.normalize_review(
            _review_contract(), _candidate(), ranking_hypothesis_ids={"h1": "h-1"}
        )
        (state / "hypothesis-reviews.jsonl").write_text(
            json.dumps({"event": "evidence_review", "plan_id": "plan-1", **row}) + "\n", encoding="utf-8"
        )
        return payload

    def test_an_accepted_claim_reaches_the_ledgers_and_both_projections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._state(Path(temporary))
            counts = self.acceptance.reconcile(payload)
            self.assertEqual(counts["reconciled"], 1)
            self.assertEqual((counts["hypotheses"], counts["evidence"], counts["credit"]), (1, 1, 1))
            state = Path(payload["loam_state_dir"])
            evidence = json.loads((state / "evidence-ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(evidence["stance"], "contradicts")
            self.assertLess(evidence["evidence_log_odds_delta"], 0.0)
            memory = json.loads((state / "hypothesis-memory.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(len(memory["active_hypotheses"][0]["contradicting_evidence"]), 1)
            steering = json.loads((state / "hypothesis-steering.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(steering["hypotheses"][0]["independent_contradiction_count"], 1)

    def test_the_jnana_id_links_accepted_evidence_to_its_elo(self) -> None:
        """Without it the evidence and the tournament rank describe different things."""

        with tempfile.TemporaryDirectory() as temporary:
            payload = self._state(Path(temporary))
            self.acceptance.reconcile(payload)
            row = json.loads(
                (Path(payload["loam_state_dir"]) / "hypothesis-ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(row["jnana_hypothesis_id"], "h-1")

    def test_acceptance_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._state(Path(temporary))
            self.acceptance.reconcile(payload)
            again = self.acceptance.reconcile(payload)
            self.assertEqual(again["reconciled"], 0)
            evidence = (Path(payload["loam_state_dir"]) / "evidence-ledger.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len(evidence.strip().splitlines()), 1)

    def test_a_review_the_reconciler_refuses_is_not_retried_forever(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._state(Path(temporary))
            state = Path(payload["loam_state_dir"])
            (state / "hypothesis-reviews.jsonl").write_text(
                json.dumps({
                    "event": "evidence_review", "candidate_id": "candidate-1",
                    "accept_hypothesis_keys": ["h1"], "ranking_hypothesis_ids": {},
                    "evidence": [{"hypothesis_key": "h1", "claim": "not a candidate claim",
                                  "source_url": "https://example.org", "stance": "supports",
                                  "validation_status": "accepted", "source_ref": "x"}],
                }) + "\n",
                encoding="utf-8",
            )
            first = self.acceptance.reconcile(payload)
            self.assertEqual(first["failed"], 1)
            self.assertEqual(self.acceptance.reconcile(payload)["failed"], 0)


if __name__ == "__main__":
    unittest.main()
