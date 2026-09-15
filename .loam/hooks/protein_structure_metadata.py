"""Typed candidate contract for the first Loam scientific validation pilot.

`protein_structure_metadata_v1` is intentionally narrow.  It accepts claims
about one RCSB PDB entry or one UniProtKB protein record.  Candidate records are
untrusted; this module only parses and validates their shape.  The reconciler is
the acceptance boundary that retrieves and compares authoritative source data.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

CONTRACT_VERSION = "protein_structure_metadata_v1"
SOURCE_RCSB_PDB = "rcsb_pdb"
SOURCE_UNIPROT = "uniprot"
_SOURCE_KINDS = {SOURCE_RCSB_PDB, SOURCE_UNIPROT}
_FENCED_JSON = re.compile(
    r"```(?:json|JSON)?[ \t]*\r?\n?\s*(\{.*?\})\s*```",
    re.DOTALL,
)

_RCSB_FIELDS = {
    "experimental_method",
    "resolution_angstrom",
    "chain_ids",
    "ligand_component_ids",
}
_UNIPROT_FIELDS = {"recommended_name", "gene_name", "sequence_length"}


def stable_id(kind: str, material: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"contract_version": CONTRACT_VERSION, "kind": kind, **dict(material)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"{kind}_{hashlib.sha256(encoded).hexdigest()}"


def extract_contract_object(summary: str) -> tuple[dict[str, Any] | None, str]:
    """Return the one opt-in contract object embedded in a child summary.

    A normal prose response returns ``(None, "absent")``.  This avoids treating
    every generic delegated child as a malformed scientific candidate.
    """

    for match in _FENCED_JSON.finditer(summary or ""):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("contract_version") == CONTRACT_VERSION:
            return value, "present"
    return None, "absent"


def normalize_candidate(value: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Validate and normalize a candidate without accepting any claim."""

    if value.get("contract_version") != CONTRACT_VERSION:
        return None, "unsupported_contract"
    source_kind = value.get("source_kind")
    if source_kind not in _SOURCE_KINDS:
        return None, "invalid_source_kind"
    accession = value.get("source_accession")
    if not isinstance(accession, str) or not accession.strip():
        return None, "invalid_source_accession"
    claims = value.get("claims")
    if not isinstance(claims, Mapping) or not claims:
        return None, "invalid_claims"
    allowed = _RCSB_FIELDS if source_kind == SOURCE_RCSB_PDB else _UNIPROT_FIELDS
    if not set(claims).issubset(allowed):
        return None, "unsupported_claim_field"
    if source_kind == SOURCE_RCSB_PDB:
        accession = accession.strip().upper()
        if not re.fullmatch(r"[0-9A-Z]{4}", accession):
            return None, "invalid_pdb_id"
    else:
        accession = accession.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{6,10}", accession):
            return None, "invalid_uniprot_accession"
    normalized_claims: dict[str, Any] = {}
    for name, raw in claims.items():
        if name in {"chain_ids", "ligand_component_ids"}:
            if not isinstance(raw, list) or not raw or not all(isinstance(v, str) and v.strip() for v in raw):
                return None, f"invalid_{name}"
            normalized_claims[name] = sorted({v.strip().upper() for v in raw})
        elif name == "resolution_angstrom":
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
                return None, "invalid_resolution_angstrom"
            normalized_claims[name] = float(raw)
        elif name == "sequence_length":
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                return None, "invalid_sequence_length"
            normalized_claims[name] = raw
        elif isinstance(raw, str) and raw.strip():
            normalized_claims[name] = raw.strip()
        else:
            return None, f"invalid_{name}"
    source_url = value.get("source_url")
    if source_url is not None and not isinstance(source_url, str):
        return None, "invalid_source_url"
    uncertainties = value.get("uncertainties", [])
    if not isinstance(uncertainties, list) or not all(isinstance(v, str) for v in uncertainties):
        return None, "invalid_uncertainties"
    return {
        "contract_version": CONTRACT_VERSION,
        "source_kind": source_kind,
        "source_accession": accession,
        "source_url": source_url.strip() if isinstance(source_url, str) else "",
        "claims": normalized_claims,
        "uncertainties": uncertainties,
    }, "pending"
