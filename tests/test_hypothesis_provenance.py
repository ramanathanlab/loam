"""Tests for loose hypothesis provenance and evidence-only credit."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
RECONCILE = ROOT / ".loam" / "reconcile" / "hypothesis_provenance.py"
PROJECT = ROOT / ".loam" / "reconcile" / "project_hypothesis_memory.py"
STEERING = ROOT / ".loam" / "reconcile" / "project_hypothesis_steering.py"


def load(name: str, path: Path):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HypothesisProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load("hypothesis_contract", HOOKS / "hypothesis_provenance.py")
        self.reconciler = load("hypothesis_reconciler", RECONCILE)
        self.projector = load("hypothesis_projector", PROJECT)
        self.parser = load("hypothesis_parser_test", HOOKS / "parser.py").Parser()

    def _candidate(self) -> dict:
        raw = {
            "contract_version": "hypothesis_provenance_v1",
            "hypotheses": [
                {
                    "hypothesis_key": "intervention-x",
                    "statement": "Intervention X improves outcome Y through mechanism Z.",
                    "parent_hypothesis_keys": [],
                }
            ],
            "evidence_candidates": [
                {
                    "hypothesis_key": "intervention-x",
                    "claim": "Study reports an improved Y after X.",
                    "source_url": "https://example.org/study",
                    "stance": "supports",
                },
                {
                    "hypothesis_key": "intervention-x",
                    "claim": "Study reports a confounder for X.",
                    "source_url": "https://example.org/caveat",
                    "stance": "contradicts",
                },
            ],
        }
        candidate, status = self.contract.normalize_candidate(raw)
        self.assertEqual(status, "pending")
        return {
            "record_version": self.contract.CONTRACT_VERSION,
            "candidate_status": "pending",
            "candidate_id": "candidate-1",
            "unit_id": "unit-1",
            "workflow_id": "workflow-1",
            "parent_session_id": "session-1",
            "parent_turn_id": "turn-1",
            "candidate": candidate,
        }

    def test_raw_candidate_cannot_create_credit_without_accepted_review(self) -> None:
        candidate = self._candidate()
        hypotheses, evidence, credit = self.reconciler.reconcile(
            candidate,
            {
                "candidate_id": "candidate-1",
                "accept_hypothesis_keys": ["intervention-x"],
                "evidence": [
                    {
                        "hypothesis_key": "intervention-x",
                        "claim": "Study reports an improved Y after X.",
                        "source_url": "https://example.org/study",
                        "stance": "supports",
                        "validation_status": "pending",
                    }
                ],
            },
        )
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(evidence, [])
        self.assertEqual(credit, [])

    def test_accepted_source_backed_evidence_projects_and_credits_once(self) -> None:
        candidate = self._candidate()
        review = {
            "candidate_id": "candidate-1",
            "accept_hypothesis_keys": ["intervention-x"],
            "evidence": [
                {
                    "hypothesis_key": "intervention-x",
                    "claim": "Study reports an improved Y after X.",
                    "source_url": "https://example.org/study",
                    "stance": "supports",
                    "validation_status": "accepted",
                    "source_ref": "DOI:10.1000/example",
                },
                {
                    "hypothesis_key": "intervention-x",
                    "claim": "Study reports a confounder for X.",
                    "source_url": "https://example.org/caveat",
                    "stance": "contradicts",
                    "validation_status": "accepted",
                    "source_ref": "DOI:10.1000/caveat",
                },
            ],
        }
        hypotheses, evidence, credit = self.reconciler.reconcile(candidate, review)
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(len(credit), 2)
        snapshots = self.projector.project(hypotheses, evidence, credit + credit)
        snapshot = snapshots[0]
        active = snapshot["active_hypotheses"][0]
        self.assertEqual(active["evidence_score"], 0)
        self.assertGreater(active["contribution_credit"], 0.0)
        self.assertEqual(len(active["supporting_evidence"]), 1)
        self.assertEqual(len(active["contradicting_evidence"]), 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypothesis-memory.jsonl"
            path.write_text(__import__("json").dumps(snapshot) + "\n", encoding="utf-8")
            context = self.parser.hypothesis_context(
                {"loam_state_dir": tmp, "workflow_id": "workflow-1", "session_id": "session-1"}
            )
        self.assertIn("support=1, contradiction=1", context)
        self.assertIn("tournament Elo is an advisory prioritization signal", context)
    def test_steering_combines_matched_jnana_elo_and_independent_evidence(self) -> None:
        steering = load("hypothesis_steering", STEERING)
        hypotheses = [
            {"event": "hypothesis_accepted", "status": "active", "hypothesis_id": "loam-a", "jnana_hypothesis_id": "jnana-a", "workflow_id": "workflow-1", "parent_session_id": "session-1", "statement": "A"},
            {"event": "hypothesis_accepted", "status": "active", "hypothesis_id": "loam-b", "jnana_hypothesis_id": "jnana-b", "workflow_id": "workflow-1", "parent_session_id": "session-1", "statement": "B"},
        ]
        evidence = [
            {"event": "evidence_accepted", "validation_status": "accepted", "evidence_id": "ev-a", "hypothesis_id": "loam-a", "stance": "supports", "independence_group_id": "study-a", "evidence_log_odds_delta": 0.8},
            {"event": "evidence_accepted", "validation_status": "accepted", "evidence_id": "ev-a-duplicate", "hypothesis_id": "loam-a", "stance": "supports", "independence_group_id": "study-a", "evidence_log_odds_delta": 0.8},
            {"event": "evidence_accepted", "validation_status": "accepted", "evidence_id": "ev-b", "hypothesis_id": "loam-b", "stance": "contradicts", "independence_group_id": "study-b", "evidence_log_odds_delta": -0.8},
        ]
        report = {
            "schema_version": "loam_jnana_tournament_v1",
            "workflow_id": "workflow-1",
            "result": {"top_hypotheses": [{"hypothesis_id": "jnana-a", "elo_rating": 1280}, {"hypothesis_id": "jnana-b", "elo_rating": 1120}]},
        }
        snapshot = steering.project(hypotheses, evidence, [report])[0]
        by_id = {item["hypothesis_id"]: item for item in snapshot["hypotheses"]}
        self.assertEqual(by_id["loam-a"]["jnana_elo"], 1280.0)
        self.assertGreater(by_id["loam-a"]["evidence_rating"], 1200.0)
        self.assertLess(by_id["loam-b"]["evidence_rating"], 1200.0)
        self.assertEqual(by_id["loam-a"]["independent_support_count"], 1)
        self.assertEqual(snapshot["scoring_policy"]["weights"]["evidence"], 0.50)


class ParentHypothesisKeyTests(unittest.TestCase):
    """`parent_hypothesis_keys` names keys in THIS contract, nothing else."""

    def setUp(self) -> None:
        self.contract = load("hypothesis_contract", HOOKS / "hypothesis_provenance.py")

    def _raw(self, parents: list[str], extra: list[dict] | None = None) -> dict:
        hypotheses = [{"hypothesis_key": "h1", "statement": "A claim.", "parent_hypothesis_keys": parents}]
        hypotheses.extend(extra or [])
        return {
            "contract_version": "hypothesis_provenance_v1",
            "hypotheses": hypotheses,
            "evidence_candidates": [
                {"hypothesis_key": "h1", "claim": "A source says so.",
                 "source_url": "https://example.org/record", "stance": "supports"}
            ],
        }

    def test_parent_naming_a_key_in_the_same_contract_is_accepted(self) -> None:
        raw = self._raw(
            ["h0"],
            [{"hypothesis_key": "h0", "statement": "The broader claim.", "parent_hypothesis_keys": []}],
        )
        candidate, status = self.contract.normalize_candidate(raw)
        self.assertEqual(status, "pending")
        assert candidate is not None
        by_key = {item["hypothesis_key"]: item for item in candidate["hypotheses"]}
        self.assertEqual(by_key["h1"]["parent_hypothesis_keys"], ["h0"])

    def test_parent_declared_after_its_child_still_resolves(self) -> None:
        # Membership is checked once every key is known, so declaration order
        # inside the contract does not decide validity.
        raw = self._raw(
            ["h2"],
            [{"hypothesis_key": "h2", "statement": "Declared later.", "parent_hypothesis_keys": []}],
        )
        self.assertEqual(self.contract.normalize_candidate(raw)[1], "pending")

    def test_parent_naming_no_key_in_the_contract_is_rejected(self) -> None:
        candidate, status = self.contract.normalize_candidate(self._raw(["h9"]))
        self.assertIsNone(candidate)
        self.assertEqual(status, "unknown_parent_hypothesis_key")

    def test_assigned_hypothesis_uuid_is_rejected_whatever_its_first_digit(self) -> None:
        # The bug this pins: `_KEY` fullmatches a UUID whenever its leading hex
        # digit is a-f, so the SAME validator mistake was accepted for
        # `b89b1a61-...` and rejected for `4d1c2b46-...` in one round, and the
        # acceptance stored a parent resolving to nothing. Both must now fail,
        # and report the same thing -- the diagnosis cannot depend on which
        # character the id happens to start with.
        for uuid in (
            "b89b1a61-fad3-4220-86be-34d525d70165",  # leading 'b': passed the old shape check
            "4d1c2b46-50d8-4ec2-927a-928fdd737f54",  # leading '4': failed it
        ):
            with self.subTest(uuid=uuid):
                candidate, status = self.contract.normalize_candidate(self._raw([uuid]))
                self.assertIsNone(candidate)
                self.assertEqual(status, "unknown_parent_hypothesis_key")

    def test_a_malformed_parent_key_reports_the_same_unresolved_key(self) -> None:
        # Membership subsumes the pattern: a key that cannot match `_KEY` can
        # never be in `seen_keys`, so it is reported as what it is.
        self.assertEqual(
            self.contract.normalize_candidate(self._raw(["Not A Key"]))[1],
            "unknown_parent_hypothesis_key",
        )

    def test_a_non_string_parent_is_a_type_error(self) -> None:
        for parents in ([None], [3], "h0", {"h0": True}):
            with self.subTest(parents=parents):
                self.assertEqual(
                    self.contract.normalize_candidate(self._raw(parents))[1],
                    "invalid_parent_hypothesis_keys",
                )


class TruncatedSummarySpillTests(unittest.TestCase):
    """A contract cut in half by the parent's context budget is still read."""

    CONTRACT = (
        '```json\n{"contract_version": "hypothesis_provenance_v1", '
        '"hypotheses": [{"hypothesis_key": "h1", "statement": "A claim.", '
        '"parent_hypothesis_keys": []}], "evidence_candidates": []}\n```'
    )

    def setUp(self) -> None:
        self.contract = load("hypothesis_contract", HOOKS / "hypothesis_provenance.py")

    def _truncated(self) -> str:
        # What the parent actually sees: head + tail, with the middle of the
        # fenced block gone, exactly as _trim_summary_with_footer leaves it.
        full = f"Final review below.\n\n{self.CONTRACT}\n\nIssues: none.\n"
        return full[:40] + "\n[... middle omitted — see footer ...]\n" + full[-20:]

    def test_truncated_summary_alone_yields_nothing(self) -> None:
        self.assertEqual(
            self.contract.extract_contract_object(self._truncated())[1], "absent"
        )

    def test_spill_file_recovers_the_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spill = Path(tmp) / "subagent-summary-0.txt"
            spill.write_text(f"Final review below.\n\n{self.CONTRACT}\n", encoding="utf-8")
            value, presence = self.contract.extract_contract_object(
                self._truncated(), str(spill)
            )
        self.assertEqual(presence, "present")
        assert value is not None
        self.assertEqual(value["hypotheses"][0]["hypothesis_key"], "h1")

    def test_an_untruncated_summary_never_reads_the_file(self) -> None:
        # The common case must do no I/O, so an unreadable path is harmless.
        value, presence = self.contract.extract_contract_object(
            f"Report.\n{self.CONTRACT}", "/nonexistent/spill.txt"
        )
        self.assertEqual(presence, "present")
        assert value is not None

    def test_a_missing_or_empty_spill_path_degrades_to_absent(self) -> None:
        for path in ("", None, "/nonexistent/spill.txt"):
            with self.subTest(path=path):
                self.assertEqual(
                    self.contract.extract_contract_object(self._truncated(), path)[1],
                    "absent",
                )

    def test_an_oversized_spill_file_is_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spill = Path(tmp) / "huge.txt"
            spill.write_text("x" * (self.contract._MAX_SPILL_BYTES + 1), encoding="utf-8")
            self.assertEqual(
                self.contract.extract_contract_object(self._truncated(), str(spill))[1],
                "absent",
            )

    def test_the_reviewer_contract_shares_the_same_recovery(self) -> None:
        # One extractor serves both contract types; the reviewer is the role
        # that actually loses its contract to truncation, so pin it here too.
        review = load("hypothesis_review_test", HOOKS / "hypothesis_review.py")
        body = (
            '```json\n{"contract_version": "hypothesis_review_v1", '
            '"candidate_id": "candidate-1", "accept_hypothesis_keys": [], '
            '"reviews": [{"index": 0, "verdict": "unverifiable", '
            '"verification": "none", "source_ref": "", "checked": "MCP unreachable."}]}\n```'
        )
        with tempfile.TemporaryDirectory() as tmp:
            spill = Path(tmp) / "review.txt"
            spill.write_text(body, encoding="utf-8")
            value, presence = review.extract_contract_object(
                "Confirmed unreachable.\n[... middle omitted — see footer ...]\n", str(spill)
            )
        self.assertEqual(presence, "present")
        assert value is not None
        self.assertEqual(value["reviews"][0]["verdict"], "unverifiable")


if __name__ == "__main__":
    unittest.main()
