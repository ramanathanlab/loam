"""Smoke tests for Loam's observe-only subagent lifecycle ledger."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
START = HOOKS / "subagent-start.py"
STOP = HOOKS / "subagent-stop.py"
MEMORY_PROJECTOR = ROOT / ".loam" / "reconcile" / "project_parent_memory.py"
MEMORY_RECALL = HOOKS / "memory_recall.py"


def _load_module(name: str, path: Path):
    hook_path = str(HOOKS)
    if hook_path not in sys.path:
        sys.path.insert(0, hook_path)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_memory_projector():
    return _load_module("loam_memory_projector", MEMORY_PROJECTOR)


class SubagentLedgerTests(unittest.TestCase):
    def _run_hook(self, script: Path, payload: dict, ledger: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["LOAM_LEDGER_PATH"] = str(ledger)
        return subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

    def test_start_and_stop_records_share_a_unit_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            common = {
                "repo_root": str(ROOT),
                "parent_session_id": "parent-session",
                "parent_turn_id": "turn-7",
                "child_session_id": "child-session",
                "child_subagent_id": "child-42",
                "child_role": "leaf",
            }
            start = self._run_hook(
                START,
                {**common, "event": "subagent_start", "child_goal": "Inspect one source."},
                ledger,
            )
            stop = self._run_hook(
                STOP,
                {
                    **common,
                    "event": "subagent_stop",
                    "child_status": "completed",
                    "duration_ms": 123,
                    "child_summary": "Unvalidated audit text.",
                },
                ledger,
            )
            self.assertEqual(start.returncode, 0, start.stderr)
            self.assertEqual(stop.returncode, 0, stop.stderr)
            records = [json.loads(line) for line in ledger.read_text().splitlines()]

        self.assertEqual([record["event"] for record in records], ["subagent_start", "subagent_stop"])
        self.assertEqual(records[0]["unit_id"], records[1]["unit_id"])
        self.assertEqual(records[1]["observation_status"], "observed")
        self.assertEqual(records[1]["finding_ids"], [])
        self.assertEqual(records[1]["cost"]["elapsed_ms"], 123)
    def test_tagged_validator_lifecycle_records_orchestration_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "ledger.jsonl"
            state_dir = root / "state"
            common = {
                "repo_root": str(ROOT),
                "workflow_id": "research-v1",
                "loam_state_dir": str(state_dir),
                "parent_session_id": "parent-session",
                "parent_turn_id": "turn-7",
                "child_session_id": "validator-session",
                "child_subagent_id": "validator-42",
                "child_role": "orchestrator",
                "child_goal": (
                    "Validate ranked hypothesis "
                    "[loam-agent-role=hypothesis-validator] "
                    "[loam-plan-id=plan-1] [loam-hypothesis-id=hypothesis-1]"
                ),
            }
            start = self._run_hook(START, {**common, "event": "subagent_start"}, ledger)
            # C2: a backed score requires a completed code-generator execution
            # for this plan+hypothesis. Stage one, as the validator would.
            code_common = {
                **common,
                "child_session_id": "codegen-session",
                "child_subagent_id": "codegen-1",
                "child_goal": (
                    "Execute [loam-agent-role=code-generator] "
                    "[loam-plan-id=plan-1] [loam-hypothesis-id=hypothesis-1]"
                ),
            }
            self._run_hook(START, {**code_common, "event": "subagent_start"}, ledger)
            self._run_hook(STOP, {**code_common, "event": "subagent_stop", "child_status": "completed", "child_summary": "ran"}, ledger)
            stop = self._run_hook(
                STOP,
                {
                    **common,
                    "event": "subagent_stop",
                    "child_status": "completed",
                    "child_summary": "Validator report.\n### Evidence Score\n-0.50 — contradictory result",
                },
                ledger,
            )
            rows = [
                json.loads(line)
                for line in (state_dir / "hypothesis-orchestration.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(start.returncode, 0, start.stderr)
        self.assertEqual(stop.returncode, 0, stop.stderr)
        self.assertEqual(
            [row["event"] for row in rows],
            ["agent_started", "agent_started", "code_generator_completed", "validator_completed"],
        )
        self.assertEqual(rows[-1]["hypothesis_id"], "hypothesis-1")
        self.assertEqual(rows[-1]["summary"], "Validator report.\n### Evidence Score\n-0.50 — contradictory result")
        self.assertEqual(rows[-1]["evidence_score"], -0.5)
        self.assertEqual(rows[-1]["evidence_score_status"], "reported")

    def test_runtime_state_directory_routes_direct_hook_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "deployment-42"
            common = {
                "repo_root": str(ROOT),
                "workflow_id": "research-v1",
                "parent_session_id": "parent-session",
                "parent_turn_id": "turn-7",
                "child_session_id": "validator-session",
                "child_subagent_id": "validator-42",
                "child_role": "orchestrator",
                "child_goal": (
                    "Validate ranked hypothesis [loam-agent-role=hypothesis-validator] "
                    "[loam-plan-id=plan-1] [loam-hypothesis-id=hypothesis-1]"
                ),
            }
            env = dict(os.environ)
            env["LOAM_STATE_DIR"] = str(state_dir)
            start = subprocess.run(
                [sys.executable, str(START)], cwd=str(ROOT), input=json.dumps({**common, "event": "subagent_start"}),
                text=True, capture_output=True, env=env, check=False,
            )
            # C2: stage a completed code-generator execution so the score is backed.
            code_common = {
                **common,
                "child_session_id": "codegen-session",
                "child_subagent_id": "codegen-1",
                "child_goal": (
                    "Execute [loam-agent-role=code-generator] "
                    "[loam-plan-id=plan-1] [loam-hypothesis-id=hypothesis-1]"
                ),
            }
            for ev, hook in (("subagent_start", START), ("subagent_stop", STOP)):
                subprocess.run(
                    [sys.executable, str(hook)], cwd=str(ROOT),
                    input=json.dumps({**code_common, "event": ev, "child_status": "completed", "child_summary": "ran"}),
                    text=True, capture_output=True, env=env, check=False,
                )
            stop = subprocess.run(
                [sys.executable, str(STOP)], cwd=str(ROOT),
                input=json.dumps(
                    {
                        **common,
                        "event": "subagent_stop",
                        "child_status": "completed",
                        "child_summary": "### Evidence Score\n+0.75 — reliable support",
                    }
                ),
                text=True, capture_output=True, env=env, check=False,
            )
            rows = [
                json.loads(line)
                for line in (state_dir / "hypothesis-orchestration.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(start.returncode, 0, start.stderr)
            self.assertEqual(stop.returncode, 0, stop.stderr)
            self.assertTrue((state_dir / "subagent-ledger.jsonl").is_file())
            self.assertEqual(rows[-1]["hypothesis_id"], "hypothesis-1")
            self.assertEqual(rows[-1]["evidence_score"], 0.75)

    def test_hooks_create_and_update_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "deployment-42"
            common = {
                "repo_root": str(ROOT),
                "loam_state_dir": str(state_dir),
                "loam_run_id": "run-42",
                "loam_profile": "research",
                "workflow_id": "research-v1",
                "loam_execution": {"mode": "local", "runtime": "python3.11"},
                "parent_session_id": "parent-session",
                "parent_turn_id": "turn-7",
                "child_session_id": "validator-session",
                "child_subagent_id": "validator-42",
                "child_role": "orchestrator",
                "child_goal": (
                    "Validate [loam-agent-role=hypothesis-validator] "
                    "[loam-plan-id=plan-1] [loam-hypothesis-id=hypothesis-1]"
                ),
            }
            start = self._run_hook(START, {**common, "event": "subagent_start"}, state_dir / "ledger.jsonl")
            stop = self._run_hook(
                STOP,
                {
                    **common,
                    "event": "subagent_stop",
                    "child_status": "completed",
                    "child_summary": "### Evidence Score\n0.00 — inconclusive",
                },
                state_dir / "ledger.jsonl",
            )
            manifest = json.loads((state_dir / "run-manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(start.returncode, 0, start.stderr)
        self.assertEqual(stop.returncode, 0, stop.stderr)
        self.assertEqual(manifest["identity"]["run_id"], "run-42")
        self.assertEqual(manifest["identity"]["execution"]["mode"], "local")
        self.assertEqual(manifest["lifecycle"]["event_counts"], {"subagent_start": 1, "subagent_stop": 1})


    def test_interrupted_child_is_excluded_not_zero_yield(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            completed = self._run_hook(
                STOP,
                {
                    "event": "subagent_stop",
                    "repo_root": str(ROOT),
                    "parent_session_id": "parent",
                    "parent_turn_id": "turn",
                    "child_session_id": "child",
                    "child_subagent_id": "child-id",
                    "child_status": "interrupted",
                },
                ledger,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            record = json.loads(ledger.read_text())

        self.assertEqual(record["observation_status"], "excluded")
        self.assertEqual(record["finding_ids"], [])
    def test_completed_opt_in_contract_creates_pending_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            summary = '''Candidate output:
```json
{
  "contract_version": "hypothesis_provenance_v1",
  "hypotheses": [
    {"hypothesis_key": "h1", "statement": "The candidate engages the target interface.", "parent_hypothesis_keys": []}
  ],
  "evidence_candidates": [
    {"hypothesis_key": "h1", "claim": "The record reports the interface residues.",
     "source_url": "https://example.org/record/1", "stance": "supports"}
  ]
}
```'''
            completed = self._run_hook(
                STOP,
                {
                    "event": "subagent_stop",
                    "repo_root": str(ROOT),
                    "parent_session_id": "parent",
                    "parent_turn_id": "turn",
                    "child_session_id": "child",
                    "child_subagent_id": "child-id",
                    "child_status": "completed",
                    "child_summary": summary,
                },
                ledger,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            candidate_file = ledger.with_name("candidate-results.jsonl")
            candidate = json.loads(candidate_file.read_text())

        self.assertEqual(candidate["candidate_status"], "pending")
        self.assertEqual(candidate["record_version"], "hypothesis_provenance_v1")
        self.assertEqual(candidate["candidate"]["hypotheses"][0]["hypothesis_key"], "h1")
        self.assertEqual(
            candidate["candidate"]["evidence_candidates"][0]["source_url"],
            "https://example.org/record/1",
        )
        # Extraction is not acceptance: no finding until explicit reconciliation.
        self.assertNotIn("finding_id", candidate)

    def test_completed_plain_fence_contract_creates_pending_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            summary = (
                "Verified record:\n\n```\n"
                '{"contract_version":"hypothesis_provenance_v1",'
                '"hypotheses":[{"hypothesis_key":"h1","statement":"A claim.","parent_hypothesis_keys":[]}],'
                '"evidence_candidates":[{"hypothesis_key":"h1","claim":"The source states it.",'
                '"source_url":"https://example.org/record/1","stance":"supports"}]}'
                "\n```"
            )
            completed = self._run_hook(
                STOP,
                {
                    "event": "subagent_stop",
                    "repo_root": str(ROOT),
                    "parent_session_id": "parent",
                    "parent_turn_id": "turn",
                    "child_session_id": "child",
                    "child_subagent_id": "child-id",
                    "child_status": "completed",
                    "child_summary": summary,
                },
                ledger,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            candidate = json.loads(ledger.with_name("candidate-results.jsonl").read_text())

        self.assertEqual(candidate["candidate_status"], "pending")
        self.assertEqual(candidate["candidate"]["hypotheses"][0]["statement"], "A claim.")

    def test_parent_memory_uses_only_accepted_deduplicated_findings(self) -> None:
        projector = _load_memory_projector()
        decisions = [
            {
                "parent_session_id": "parent",
                "parent_turn_id": "turn",
                "unit_id": "unit-a",
                "source_kind": "uniprot",
                "source_accession": "P00918",
                "field": "gene_name",
                "authoritative_value": "CA2",
                "source_url": "https://rest.uniprot.org/uniprotkb/P00918.json",
                "validation_status": "accepted",
                "finding_id": "finding-ca2",
            },
            {
                "parent_session_id": "parent",
                "parent_turn_id": "turn",
                "unit_id": "unit-b",
                "source_kind": "uniprot",
                "source_accession": "P00918",
                "field": "gene_name",
                "authoritative_value": "CA2",
                "source_url": "https://rest.uniprot.org/uniprotkb/P00918.json",
                "validation_status": "accepted",
                "finding_id": "finding-ca2",
            },
            {
                "parent_session_id": "parent",
                "parent_turn_id": "turn",
                "unit_id": "unit-c",
                "validation_status": "claim_mismatch",
                "finding_id": "",
            },
        ]
        snapshots = projector.project(decisions)

        self.assertEqual(len(snapshots), 1)
        snapshot = snapshots[0]
        self.assertEqual(snapshot["accepted_finding_ids"], ["finding-ca2"])
        self.assertEqual(snapshot["accepted_findings"][0]["unit_ids"], ["unit-a", "unit-b"])
        self.assertEqual(snapshot["validation_failure_statuses"], ["claim_mismatch"])
        self.assertEqual(snapshot["attempted_unit_ids"], ["unit-a", "unit-b", "unit-c"])

    def test_recall_injects_validated_memory_and_audits_exact_duplicate(self) -> None:
        recall = _load_module("loam_memory_recall", MEMORY_RECALL)
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "parent-memory.jsonl"
            audit_path = Path(tmp) / "subagent-ledger.jsonl"
            snapshot = {
                "parent_session_id": "parent",
                "parent_turn_id": "turn",
                "accepted_findings": [
                    {
                        "finding_id": "finding-ca2",
                        "source_kind": "uniprot",
                        "source_accession": "P00918",
                        "field": "gene_name",
                        "value": "CA2",
                    }
                ],
            }
            memory_path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
            old_memory = os.environ.get("LOAM_PARENT_MEMORY_PATH")
            old_ledger = os.environ.get("LOAM_LEDGER_PATH")
            os.environ["LOAM_PARENT_MEMORY_PATH"] = str(memory_path)
            os.environ["LOAM_LEDGER_PATH"] = str(audit_path)
            try:
                payload = {
                    "session_id": "parent",
                    "parent_session_id": "parent",
                    "parent_turn_id": "turn",
                    "tool_name": "delegate_task",
                    "arguments": {"goal": "Verify the UniProt P00918 gene name."},
                }
                context = recall.context_for(payload)
                matches = recall.audit_duplicates(payload)
            finally:
                if old_memory is None:
                    os.environ.pop("LOAM_PARENT_MEMORY_PATH", None)
                else:
                    os.environ["LOAM_PARENT_MEMORY_PATH"] = old_memory
                if old_ledger is None:
                    os.environ.pop("LOAM_LEDGER_PATH", None)
                else:
                    os.environ["LOAM_LEDGER_PATH"] = old_ledger
            audit_rows = [
                json.loads(line)
                for line in audit_path.with_name("duplicate-candidates.jsonl").read_text().splitlines()
            ]

        self.assertIn("uniprot P00918: gene_name = CA2", context)
        self.assertEqual(matches, None)
        self.assertEqual(audit_rows[0]["policy_action"], "allow_and_audit")

    def test_exact_duplicate_block_is_opt_in(self) -> None:
        recall = _load_module("loam_memory_recall_block", MEMORY_RECALL)
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "parent-memory.jsonl"
            ledger_path = Path(tmp) / "subagent-ledger.jsonl"
            memory_path.write_text(
                json.dumps(
                    {
                        "workflow_id": "ca2",
                        "accepted_findings": [
                            {
                                "finding_id": "finding-ca2",
                                "source_kind": "uniprot",
                                "source_accession": "P00918",
                                "field": "gene_name",
                                "value": "CA2",
                            }
                        ],
                    }
                ) + "\n",
                encoding="utf-8",
            )
            old_memory = os.environ.get("LOAM_PARENT_MEMORY_PATH")
            old_ledger = os.environ.get("LOAM_LEDGER_PATH")
            os.environ["LOAM_PARENT_MEMORY_PATH"] = str(memory_path)
            os.environ["LOAM_LEDGER_PATH"] = str(ledger_path)
            try:
                blocked = recall.audit_duplicates(
                    {
                        "session_id": "new-session",
                        "workflow_id": "ca2",
                        "loam_duplicate_policy": "block_exact_duplicate",
                        "tool_name": "delegate_task",
                        "arguments": {"goal": "Verify the UniProt P00918 gene name."},
                    }
                )
                distinct = recall.audit_duplicates(
                    {
                        "session_id": "new-session",
                        "workflow_id": "ca2",
                        "loam_duplicate_policy": "block_exact_duplicate",
                        "tool_name": "delegate_task",
                        "arguments": {"goal": "Verify the UniProt P00918 sequence length."},
                    }
                )
            finally:
                if old_memory is None:
                    os.environ.pop("LOAM_PARENT_MEMORY_PATH", None)
                else:
                    os.environ["LOAM_PARENT_MEMORY_PATH"] = old_memory
                if old_ledger is None:
                    os.environ.pop("LOAM_LEDGER_PATH", None)
                else:
                    os.environ["LOAM_LEDGER_PATH"] = old_ledger

        self.assertEqual(blocked["action"], "block")
        self.assertEqual(blocked["reason"], "accepted_exact_duplicate")
        self.assertIsNone(distinct)


if __name__ == "__main__":
    unittest.main()
