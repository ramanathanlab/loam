"""Layer 0 provenance: observe where an execution's scientific inputs came from.

This is the first, domain-blind layer of the provenance design (C4). It does NOT
prohibit hardcoding and it does NOT block any tool call. For every *execution*
tool call (run_code / run_batch / run_python and their MCP spellings) it reads
the call's argument text and records which input-origin signals are present:

  * fetched         -- the payload pulls data at runtime (HTTP client, or a
                       known reference host such as UniProt / RCSB / PubChem).
  * filesystem_read -- the payload reads data off disk (open/read_text/np.load).
  * derived_ref     -- the payload references a registered batch/tool artifact.
  * inline_literal  -- the payload carries a large inline data-like literal that
                       is not, on its face, traceable to any of the above.

The classifier is deliberately blind to WHAT the datum is (protein, nucleotide,
SMILES, a number): it reports origin signals and a coarse `verdict`, with
`inline_only` marking the case the GLP1R target hit -- a long literal pasted in
with no fetch/file/derivation. `looks_like` on a literal is a best-effort hint,
never authoritative. Later layers (consistency checking, authority verifiers)
build on these records; Layer 0 only observes and logs.

Observer only: every failure is swallowed, so this can never block or alter a
tool call.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

FILENAME = "provenance.jsonl"

# A tool call is an "execution" if its name carries one of these tokens (covers
# run_code, run_batch, run_python and MCP spellings like mcp__hpc__run_code), or
# if it matches the profile's configured execution tool name.
_EXECUTION_TOKENS = ("run_code", "run_batch", "run_python")

# Only literals at least this long are worth flagging; shorter strings are
# overwhelmingly ordinary arguments, flags, and identifiers.
LITERAL_MIN_CHARS = 24

_FETCH_PATTERNS = (
    r"requests\.(?:get|post|put|request)",
    r"urllib\.request|urlopen",
    r"\bhttpx\b|\baiohttp\b",
    r"\bcurl\s|\bwget\s",
    r"https?://",
    r"uniprot\.org",
    r"rcsb\.org|files\.rcsb",
    r"ebi\.ac\.uk",
    r"ncbi\.nlm\.nih\.gov|eutils",
    r"pubchem",
    r"alphafold",
    r"ensembl",
)
_FS_PATTERNS = (
    r"\bopen\s*\(",
    r"\.read_text\s*\(|\.read_bytes\s*\(",
    r"\bnp\.load\b|numpy\.load",
    r"\bpd\.read_[a-z]+\s*\(|pandas\.read_",
    r"\bjson\.load\s*\(",
    r"\bPath\s*\([^)]*\)\.(?:open|read)",
)
# A registered artifact reference: an explicit batch_id/tool= keyword, or an
# id-like token produced by run_batch (b-<hex>) / the tool registry.
_DERIVED_PATTERNS = (
    r"\bbatch_id\b",
    r"\btool\s*=",
    r"\brun_batch\b",
    r"\bregister_tool\b",
    r"\bb-[0-9a-f]{6,}\b",
)

_FETCH_RE = re.compile("|".join(_FETCH_PATTERNS), re.IGNORECASE)
_FS_RE = re.compile("|".join(_FS_PATTERNS))
_DERIVED_RE = re.compile("|".join(_DERIVED_PATTERNS), re.IGNORECASE)

# Quoted string literals: triple-quoted first, then single-line single/double.
_LITERAL_RE = re.compile(
    r"\"\"\"(.*?)\"\"\"|'''(.*?)'''|\"([^\"\n]*)\"|'([^'\n]*)'",
    re.DOTALL,
)
_PROTEIN_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYX]{20,}$")
_NUCLEOTIDE_RE = re.compile(r"^[ACGTUN]{20,}$", re.IGNORECASE)
_SMILES_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#/\\%.]{6,200}$")
_DATALIKE_RE = re.compile(r"^[A-Za-z0-9/+=._\-]+$")


def _audit_path(payload: Mapping[str, Any]) -> Path | None:
    state_dir = payload.get("loam_state_dir")
    if not isinstance(state_dir, str) or not state_dir.strip():
        state_dir = os.environ.get("LOAM_STATE_DIR", "").strip()
    if not state_dir:
        return None
    return Path(state_dir).expanduser() / FILENAME


def _is_execution_tool(payload: Mapping[str, Any]) -> bool:
    name = str(payload.get("tool_name") or "")
    if not name:
        return False
    if any(token in name for token in _EXECUTION_TOKENS):
        return True
    execution = payload.get("loam_execution")
    if isinstance(execution, Mapping) and str(execution.get("tool_name") or "") == name:
        return True
    return False


def _argument_text(arguments: Any) -> str:
    """Concatenate every string in the arguments, tool-blind.

    run_code carries its code in `payload`/`code`; run_batch carries script and
    input paths. Flattening all string values lets one classifier serve any
    execution tool without knowing its argument schema.
    """
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(arguments)
    return "\n".join(parts)


_DATA_KINDS = {"protein_sequence_like", "nucleotide_like", "smiles_like"}


def _looks_like(literal: str) -> str:
    stripped = literal.strip()
    if _PROTEIN_RE.match(stripped):
        return "protein_sequence_like"
    if _NUCLEOTIDE_RE.match(stripped):
        return "nucleotide_like"
    # SMILES may carry /\ (stereo bonds), so test it before path/URL detection.
    if " " not in stripped and any(c in stripped for c in "=#[]()@") and _SMILES_RE.match(stripped):
        return "smiles_like"
    if "://" in stripped:
        return "url_like"
    if stripped.startswith(("~", ".", "/")) or "/" in stripped or "\\" in stripped:
        return "path_like"
    if _DATALIKE_RE.match(stripped):
        return "token_like"
    return "unknown"


def _inline_literals(text: str) -> list[dict[str, Any]]:
    """Flag inline data literals, excluding paths, URLs, prose, and short tokens.

    Recognized data kinds (protein/nucleotide/SMILES) are flagged at or above
    the length floor -- with a lower bar for structural SMILES, since small
    molecules are often short. A long opaque contiguous blob (e.g. an unlabeled
    numeric/encoded datum) is flagged too, so the layer is not limited to the
    kinds it can name. Paths and URLs are origins in their own right, not data.
    """
    found: list[dict[str, Any]] = []
    for match in _LITERAL_RE.finditer(text):
        literal = next((group for group in match.groups() if group is not None), "")
        stripped = literal.strip()
        kind = _looks_like(stripped)
        if kind in _DATA_KINDS:
            special = sum(stripped.count(c) for c in "=#[]()@")
            smiles_short = kind == "smiles_like" and len(stripped) >= 8 and special >= 2
            if len(stripped) < LITERAL_MIN_CHARS and not smiles_short:
                continue
        elif kind == "token_like" and len(stripped) >= LITERAL_MIN_CHARS * 2:
            kind = "opaque_data_like"  # unnamed but large and contiguous
        else:
            continue  # path_like, url_like, prose/unknown, short token
        preview = stripped if len(stripped) <= 40 else f"{stripped[:20]}...{stripped[-12:]}"
        found.append({"length": len(stripped), "looks_like": kind, "preview": preview})
        if len(found) >= 25:
            break
    return found


def classify(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a Layer 0 lineage record for an execution call, or None."""

    if not _is_execution_tool(payload):
        return None
    text = _argument_text(payload.get("arguments"))
    if not text:
        return None

    fetched = sorted({m.group(0) for m in _FETCH_RE.finditer(text)})[:10]
    filesystem = bool(_FS_RE.search(text))
    derived = bool(_DERIVED_RE.search(text))
    literals = _inline_literals(text)

    origins: list[str] = []
    if fetched:
        origins.append("fetched")
    if filesystem:
        origins.append("filesystem_read")
    if derived:
        origins.append("derived_ref")
    if literals:
        origins.append("inline_literal")

    # The load-bearing signal: a data-like literal with no fetch/file/derivation
    # behind it (the hardcoded-target case). Everything else is informational.
    if literals and not (fetched or filesystem or derived):
        verdict = "inline_only"
    elif not origins:
        verdict = "none"
    elif len(origins) == 1:
        verdict = origins[0]
    else:
        verdict = "mixed"

    return {
        "verdict": verdict,
        "origins": origins,
        "fetched": fetched,
        "filesystem_read": filesystem,
        "derived_ref": derived,
        "inline_literals": literals,
    }


def record(payload: Mapping[str, Any]) -> None:
    """Append one Layer 0 provenance record for an execution call. Never raises."""

    try:
        path = _audit_path(payload)
        if path is None:
            return
        lineage = classify(payload)
        if lineage is None:
            return
        row = {
            "record_version": "loam_provenance_v0",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": str(payload.get("loam_run_id") or ""),
            "session_id": str(payload.get("session_id") or ""),
            "agent_role": str(payload.get("agent_role") or ""),
            "tool_name": str(payload.get("tool_name") or ""),
            **lineage,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.chmod(path, 0o600)
    except Exception:  # noqa: BLE001 - an observer must never affect a tool call
        return
