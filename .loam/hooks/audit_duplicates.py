"""Audit delegated tasks that would rediscover an exact finding already in parent memory.

This module deliberately matches only explicit, structured source/field requests.
It does not use embeddings, LLM classification, source-name similarity, or raw
child summaries to decide that a proposed delegation is redundant.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from ledger import append_record, ledger_path
from parser import Parser

DUPLICATE_AUDIT_FILENAME = "duplicate-candidates.jsonl"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")


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


def _field_pattern(field: str) -> "re.Pattern[str] | None":
    """A word-boundary pattern for a finding's field name, built from the name.

    This was a table of seven field names -- `gene_name`, `resolution_angstrom`,
    `ligand_component_ids` and so on -- from the retired protein-metadata pilot.
    A finding whose field was not one of those seven could never match, so the
    audit was silently inert for every other kind of claim, and "support a new
    campaign" read as "add more field names to a list in a hook".

    Deriving the pattern from the name instead makes any field a reasoner
    records matchable, with no vocabulary to maintain. Separators are optional
    so `sequence_length` matches "sequence length", "sequence-length" and
    "sequencelength". The cost is the pilot's hand-written synonyms: "length of
    the canonical sequence" no longer matches `sequence_length`, and a bare
    "method" no longer matches `experimental_method`. That trade buys generality
    and loses recall on seven names that no longer have a producer.
    """

    name = field.casefold()
    if not _IDENTIFIER.fullmatch(name):
        return None
    words = [re.escape(word) for word in name.split("_") if word]
    if not words or len("".join(words)) < 4:
        return None
    return re.compile(r"\b" + r"[\s_-]*".join(words) + r"\b")


def _mentions_source_kind(text: str, source_kind: str) -> bool:
    """Whether the task names the source itself, not merely the accession.

    Any word of the source kind counts, which is how the pilot's hand-written
    check behaved: `rcsb_pdb` was satisfied by either "rcsb" or "pdb". A finding
    recorded without a source kind never matches -- requiring the task to name
    the source explicitly is what keeps this from blocking an unrelated call
    that happens to quote an accession.
    """

    words = [word for word in source_kind.casefold().split("_") if len(word) >= 3]
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def exact_duplicates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Match a task that names an accepted finding's accession, source and field.

    All three must be explicit in the task text. "Supported field" no longer
    means a name from a fixed list: any field recorded in parent memory is
    matchable (see `_field_pattern`).
    """

    snapshot = Parser().memory_snapshot(payload)
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
            pattern = _field_pattern(field)
            if not accession or pattern is None:
                continue
            if (
                accession.casefold() in lowered
                and _mentions_source_kind(lowered, source_kind)
                and pattern.search(lowered)
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
