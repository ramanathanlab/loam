"""Read validated Loam parent memory and audit exact duplicate task candidates.

This module deliberately matches only explicit, structured source/field requests.
It does not use embeddings, LLM classification, source-name similarity, or raw
child summaries to decide that a proposed delegation is redundant.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from ledger import append_record, ledger_path

DEFAULT_MEMORY_FILENAME = "parent-memory.jsonl"
DUPLICATE_AUDIT_FILENAME = "duplicate-candidates.jsonl"
_FIELD_PATTERNS = {
    "gene_name": r"\bgene\s+name\b",
    "sequence_length": r"\b(sequence\s+length|length\s+of\s+(the\s+)?canonical\s+sequence)\b",
    "recommended_name": r"\b(recommended\s+protein\s+name|recommended\s+name)\b",
    "experimental_method": r"\b(experimental\s+method|method)\b",
    "resolution_angstrom": r"\b(resolution|resolution_angstrom)\b",
    "chain_ids": r"\b(chain\s+ids?|chains?)\b",
    "ligand_component_ids": r"\b(ligand\s+(component\s+)?ids?|component\s+ids?)\b",
}


def memory_path(payload: Mapping[str, Any]) -> Path:
    """Resolve parent memory from the selected profile, then legacy override."""

    state_dir = payload.get("loam_state_dir")
    if isinstance(state_dir, str) and state_dir.strip():
        return Path(state_dir).expanduser() / DEFAULT_MEMORY_FILENAME
    configured = os.environ.get("LOAM_PARENT_MEMORY_PATH", "").strip()
    return Path(configured).expanduser() if configured else ledger_path(payload).with_name(DEFAULT_MEMORY_FILENAME)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def latest_snapshot(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the latest exact-session parent snapshot, if one exists."""

    session_id = str(payload.get("session_id") or payload.get("parent_session_id") or "")
    turn_id = str(payload.get("parent_turn_id") or "")
    workflow_id = str(payload.get("workflow_id") or "")
    rows = list(_read_jsonl(memory_path(payload)))
    if workflow_id:
        matches = [row for row in rows if str(row.get("workflow_id") or "") == workflow_id]
    else:
        matches = [
            row
            for row in rows
            if str(row.get("parent_session_id") or "") == session_id
            and (not turn_id or str(row.get("parent_turn_id") or "") == turn_id)
        ]
    return matches[-1] if matches else None


def context_for(payload: Mapping[str, Any]) -> str:
    snapshot = latest_snapshot(payload)
    if snapshot is None:
        return ""
    findings = snapshot.get("accepted_findings")
    if not isinstance(findings, list) or not findings:
        return ""
    lines = ["Validated Loam findings for this parent session (not child summaries):"]
    for finding in findings[:12]:
        if not isinstance(finding, Mapping):
            continue
        lines.append(
            "- {kind} {accession}: {field} = {value}".format(
                kind=str(finding.get("source_kind") or "source"),
                accession=str(finding.get("source_accession") or ""),
                field=str(finding.get("field") or ""),
                value=str(finding.get("value") or ""),
            )
        )
    lines.append("Do not re-delegate work solely to rediscover an exact finding listed above.")
    return "\n".join(lines)


def _task_strings(arguments: Mapping[str, Any]) -> list[str]:
    values = []
    goal = arguments.get("goal")
    if isinstance(goal, str):
        values.append(goal)
    tasks = arguments.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, Mapping) and isinstance(task.get("goal"), str):
                values.append(task["goal"])
    return values


def _explicit_source_kind(text: str, source_kind: str) -> bool:
    lowered = text.casefold()
    return (source_kind == "uniprot" and "uniprot" in lowered) or (
        source_kind == "rcsb_pdb" and ("rcsb" in lowered or "pdb" in lowered)
    )


def exact_duplicates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Match only explicit source accession plus one explicit supported field."""

    snapshot = latest_snapshot(payload)
    arguments = payload.get("arguments")
    if snapshot is None or not isinstance(arguments, Mapping):
        return []
    findings = snapshot.get("accepted_findings")
    if not isinstance(findings, list):
        return []
    duplicates = []
    for task in _task_strings(arguments):
        lowered = task.casefold()
        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            source_kind = str(finding.get("source_kind") or "")
            accession = str(finding.get("source_accession") or "")
            field = str(finding.get("field") or "")
            pattern = _FIELD_PATTERNS.get(field)
            if not accession or pattern is None:
                continue
            if (
                accession.casefold() in lowered
                and _explicit_source_kind(task, source_kind)
                and re.search(pattern, lowered)
            ):
                duplicates.append(
                    {
                        "task": task,
                        "finding_id": str(finding.get("finding_id") or ""),
                        "source_kind": source_kind,
                        "source_accession": accession,
                        "field": field,
                        "value": finding.get("value"),
                    }
                )
    return duplicates


def audit_duplicates(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Audit exact matches and optionally block only under explicit profile policy."""

    matches = exact_duplicates(payload)
    if not matches:
        return None
    policy = str(payload.get("loam_duplicate_policy") or "allow_and_audit")
    action = "block" if policy == "block_exact_duplicate" else "allow_and_audit"
    path = ledger_path(payload).with_name(DUPLICATE_AUDIT_FILENAME)
    for match in matches:
        record = {
            "record_version": "loam_duplicate_candidate_v1",
            "event": "duplicate_candidate",
            "workflow_id": str(payload.get("workflow_id") or ""),
            "loam_profile": str(payload.get("loam_profile") or ""),
            "parent_session_id": str(payload.get("session_id") or ""),
            "parent_turn_id": str(payload.get("parent_turn_id") or ""),
            "tool_name": str(payload.get("tool_name") or ""),
            "match": match,
            "policy_action": action,
        }
        append_record(path, record)
    if action == "block":
        first = matches[0]
        return {
            "action": "block",
            "reason": "accepted_exact_duplicate",
            "message": (
                "Loam blocked an exact duplicate: {kind} {accession} {field} "
                "was already accepted as finding {finding_id}."
            ).format(
                kind=first["source_kind"],
                accession=first["source_accession"],
                field=first["field"],
                finding_id=first["finding_id"],
            ),
        }
    return None
