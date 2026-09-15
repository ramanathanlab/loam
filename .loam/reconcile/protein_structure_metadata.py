#!/usr/bin/env python3
"""Reconcile Loam protein-structure candidates against RCSB and UniProt.

This is an explicit, manual acceptance step. It reads unvalidated candidate JSONL
records and writes one immutable validation decision per claimed field. It never
modifies lifecycle or candidate records and it never treats a source-fetch error
as a rejected scientific claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

HOOKS = Path(__file__).resolve().parents[1] / "hooks"
sys.path.insert(0, str(HOOKS))
from ledger import append_record  # noqa: E402
from protein_structure_metadata import (  # noqa: E402
    CONTRACT_VERSION,
    SOURCE_RCSB_PDB,
    SOURCE_UNIPROT,
    stable_id,
)

DECISION_VERSION = "protein_structure_validation_v1"
RCSB_URL = "https://data.rcsb.org/rest/v1/core/entry/{accession}"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"


def _fetch_json(url: str, timeout: float) -> tuple[dict[str, Any] | None, str]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"source_fetch_failed:{type(exc).__name__}"
    return value if isinstance(value, dict) else None, "ok"


def _rcsb_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
    info_value = entry.get("rcsb_entry_info")
    info: Mapping[str, Any] = info_value if isinstance(info_value, Mapping) else {}
    exptl_value = entry.get("exptl")
    exptl: list[Any] = exptl_value if isinstance(exptl_value, list) else []
    methods = sorted(
        {
            str(item.get("method", "")).strip().upper()
            for item in exptl
            if isinstance(item, Mapping) and item.get("method")
        }
    )
    resolution_value = info.get("resolution_combined")
    resolution: list[Any] = resolution_value if isinstance(resolution_value, list) else []
    return {
        "experimental_method": methods[0] if len(methods) == 1 else methods,
        "resolution_angstrom": float(resolution[0]) if resolution else None,
        # Entry-core data does not expose all polymer-instance chains or nonpolymer
        # component IDs in one inexpensive endpoint. Leave those unavailable rather
        # than infer them from a summary or accept an unverified child assertion.
        "chain_ids": None,
        "ligand_component_ids": None,
    }


def _uniprot_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    protein_value = record.get("proteinDescription")
    protein: Mapping[str, Any] = protein_value if isinstance(protein_value, Mapping) else {}
    recommended_value = protein.get("recommendedName")
    recommended: Mapping[str, Any] = (
        recommended_value if isinstance(recommended_value, Mapping) else {}
    )
    full_name_value = recommended.get("fullName")
    full_name: Mapping[str, Any] = full_name_value if isinstance(full_name_value, Mapping) else {}
    genes_value = record.get("genes")
    genes: list[Any] = genes_value if isinstance(genes_value, list) else []
    gene_name = ""
    if genes and isinstance(genes[0], Mapping):
        primary = genes[0].get("geneName")
        if isinstance(primary, Mapping):
            gene_name = str(primary.get("value") or "")
    sequence = record.get("sequence") if isinstance(record.get("sequence"), Mapping) else {}
    return {
        "recommended_name": str(full_name.get("value") or ""),
        "gene_name": gene_name,
        "sequence_length": sequence.get("length"),
    }


def _same(claimed: Any, authoritative: Any) -> bool:
    if isinstance(claimed, float) and isinstance(authoritative, (float, int)):
        return abs(claimed - float(authoritative)) <= 0.01
    if isinstance(claimed, list) and isinstance(authoritative, list):
        return sorted(str(value).upper() for value in claimed) == sorted(str(value).upper() for value in authoritative)
    return str(claimed).strip().casefold() == str(authoritative).strip().casefold()


def decisions(candidate_record: Mapping[str, Any], timeout: float) -> Iterable[dict[str, Any]]:
    candidate = candidate_record.get("candidate")
    if candidate_record.get("candidate_status") != "pending" or not isinstance(candidate, Mapping):
        return []
    source_kind = candidate["source_kind"]
    accession = candidate["source_accession"]
    url = RCSB_URL.format(accession=accession) if source_kind == SOURCE_RCSB_PDB else UNIPROT_URL.format(accession=accession)
    source, source_status = _fetch_json(url, timeout)
    fields = _rcsb_fields(source) if source_kind == SOURCE_RCSB_PDB and source else (_uniprot_fields(source) if source else {})
    rows = []
    for field, claimed in candidate["claims"].items():
        actual = fields.get(field)
        if source is None:
            status = source_status
        elif actual is None or actual == "":
            status = "authoritative_field_unavailable"
        elif _same(claimed, actual):
            status = "accepted"
        else:
            status = "claim_mismatch"
        material = {
            "candidate_id": candidate_record["candidate_id"],
            "field": field,
            "claimed": claimed,
            "authoritative": actual,
            "status": status,
        }
        rows.append(
            {
                "record_version": DECISION_VERSION,
                "event": "validation_decision",
                "decision_id": stable_id("validation", material),
                "candidate_id": candidate_record["candidate_id"],
                "unit_id": candidate_record["unit_id"],
                "workflow_id": candidate_record.get("workflow_id", ""),
                "loam_profile": candidate_record.get("loam_profile", ""),
                "parent_session_id": candidate_record.get("parent_session_id", ""),
                "parent_turn_id": candidate_record.get("parent_turn_id", ""),
                "source_kind": source_kind,
                "source_accession": accession,
                "source_url": url,
                "field": field,
                "claimed_value": claimed,
                "authoritative_value": actual,
                "validation_status": status,
                "finding_id": (
                    stable_id("finding", {"source_kind": source_kind, "source_accession": accession, "field": field, "value": actual})
                    if status == "accepted"
                    else ""
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    if not args.candidates.is_file():
        parser.error(f"candidate file does not exist: {args.candidates}")
    count = 0
    for line in args.candidates.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for decision in decisions(record, args.timeout):
            append_record(args.out, decision)
            count += 1
    print(json.dumps({"decisions_written": count, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
