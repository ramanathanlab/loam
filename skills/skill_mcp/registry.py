"""skill_mcp.registry — typed artifact registry with provenance (design doc §11).

Runs WORKER-SIDE (it touches /flare): harvest a completed batch's outputs into
typed artifacts, and query them by type to route the next tool. Append-only JSONL
per hypothesis under HPC_BASE_DIR/artifacts/<hypothesis_id>.jsonl, flock-guarded so
the parallel code-generators can't corrupt it (same discipline as experiments-index).

Pure stdlib. Importable, and runnable as `python registry.py '<json-op>'`:
  {"op":"harvest","base_dir":..,"batch_id":..,"tool":..,"action":..,
   "produces":"structure","expects":["out/*.cif"][,"hypothesis_id":..]}
  {"op":"query","base_dir":..,"hypothesis_id":..,"type":"structure"
   [,"unused_only":true][,"limit":N][,"producer_tool":..]}
Both print a JSON result.
"""
from __future__ import annotations

import fcntl
import glob
import json
import os
import sys
import time
from pathlib import Path


def _registry_path(base_dir: str, hypothesis_id: str) -> Path:
    d = Path(base_dir, "artifacts")
    d.mkdir(parents=True, exist_ok=True)
    # keep the filename filesystem-safe
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in hypothesis_id)
    return d / f"{safe}.jsonl"


def _read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            rows = [json.loads(l) for l in f if l.strip()]
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return rows


def _append(path: Path, entries: list[dict]) -> None:
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _batch_label(base_dir: str, batch_id: str) -> dict:
    """The batch's frozen registry entry (labels, etc.), or {}."""
    p = Path(base_dir, "batches", "registry", f"{batch_id}.json")
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def harvest(base_dir: str, batch_id: str, tool: str, action: str, produces: str,
            expects: list[str], hypothesis_id: str | None = None) -> dict:
    """Register one typed artifact per completed run of a batch.

    Idempotent: a run whose artifact_id is already in the registry is skipped.
    """
    batch_dir = Path(base_dir, "batches", batch_id)
    if not batch_dir.is_dir():
        return {"error": f"batch dir not found: {batch_dir}", "registered": 0}
    label = _batch_label(base_dir, batch_id)
    hyp = hypothesis_id or (label.get("labels") or {}).get("hypothesis_id") or "default"
    reg = _registry_path(base_dir, hyp)
    existing = {e["artifact_id"] for e in _read_all(reg)}

    new: list[dict] = []
    for run in sorted(batch_dir.glob("run_*")):
        if not run.is_dir() or not (run / "done").exists():
            continue
        paths = []
        for pat in (expects or []):
            paths += [str(p) for p in run.glob(pat) if p.is_file()]
        if not paths:
            continue  # a run with no promoted output is not a ready artifact
        row = {}
        ip = run / "input.json"
        if ip.exists():
            try: row = json.loads(ip.read_text())
            except Exception: row = {}
        row_id = str(row.get("id", run.name[len("run_"):]))
        art_id = f"{batch_id}:{row_id}"
        if art_id in existing:
            continue
        parents = row.get("parents") or ([row["parent"]] if row.get("parent") else [])
        params = {k: row[k] for k in ("seed", "binder_len", "hotspots", "sampling_temp")
                  if k in row}
        new.append({
            "artifact_id": art_id, "type": produces, "paths": sorted(paths),
            "hypothesis_id": hyp, "producer_tool": tool, "producer_action": action,
            "producer_batch": batch_id, "row_id": row_id, "parents": parents,
            "params": params, "round": int(row.get("round", 1)),
            "status": "ready", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    if new:
        _append(reg, new)
    return {"registry": str(reg), "hypothesis_id": hyp, "type": produces,
            "registered": len(new), "already_present": len(existing),
            "artifacts": new}


def query(base_dir: str, hypothesis_id: str, type: str, unused_only: bool = True,
          limit: int | None = None, producer_tool: str | None = None) -> dict:
    """Return artifacts of a type for a hypothesis, newest first."""
    reg = _registry_path(base_dir, hypothesis_id)
    rows = _read_all(reg)
    used = set()
    for e in rows:
        used.update(e.get("parents") or [])
    out = []
    for e in rows:
        if e.get("type") != type:
            continue
        if producer_tool and e.get("producer_tool") != producer_tool:
            continue
        if unused_only and e["artifact_id"] in used:
            continue
        out.append(e)
    out.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    if limit:
        out = out[:limit]
    return {"registry": str(reg), "type": type, "count": len(out), "artifacts": out}


# A venv python that has gemmi, for structure format conversion (worker-side).
_GEMMI_PY = os.environ.get(
    "LOAM_GEMMI_PY", "/flare/FRAME-IDP/msinclair/software/Protenix/.venv/bin/python")


def _first_design_seq(fa_path: str) -> str | None:
    """First DESIGNED sequence from a ProteinMPNN .fa (record 0 is native)."""
    import re
    recs, cur, hdr = [], [], None
    for line in open(fa_path, errors="replace"):
        if line.startswith(">"):
            if hdr is not None:
                recs.append((hdr, "".join(cur)))
            hdr, cur = line, []
        else:
            cur.append(line.strip())
    if hdr is not None:
        recs.append((hdr, "".join(cur)))
    # skip the native (first) record; take the first sample
    designs = [s for h, s in recs[1:] if s]
    return designs[0] if designs else (recs[0][1] if recs else None)


def materialize(artifact: dict, want: str, workdir: str | None = None) -> dict:
    """Return the artifact in the format a consumer needs, converting if needed.

    want ∈ {pdb, cif, fasta, sequence}. Structure conversions use gemmi; sequence
    extraction reads a ProteinMPNN .fa. Converted files are cached next to the
    source (or under workdir). Returns {ok, want, path?/value?, error?}.
    """
    import subprocess
    paths = artifact.get("paths") or []
    src = paths[0] if paths else None
    if want in ("pdb", "cif"):
        if not src:
            return {"ok": False, "error": "no source path"}
        if src.endswith("." + want):
            return {"ok": True, "want": want, "path": src}
        out = (src.rsplit(".", 1)[0]) + "." + want
        if not os.path.isfile(out):
            conv = ("import sys,gemmi;st=gemmi.read_structure(sys.argv[1]);"
                    "st.setup_entities();"
                    + ("st.write_pdb(sys.argv[2])" if want == "pdb"
                       else "doc=st.make_pdbx_document();doc.write_file(sys.argv[2])"))
            r = subprocess.run([_GEMMI_PY, "-c", conv, src, out],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.isfile(out):
                return {"ok": False, "error": (r.stderr or r.stdout)[-300:]}
        return {"ok": True, "want": want, "path": out}
    if want in ("sequence", "fasta"):
        seq = None
        if src and src.endswith((".fa", ".fasta")):
            seq = _first_design_seq(src)
        if not seq:
            return {"ok": False, "error": f"no sequence extractable from {src}"}
        if want == "sequence":
            return {"ok": True, "want": want, "value": seq}
        name = artifact.get("row_id", "seq")
        out = os.path.join(workdir or os.path.dirname(src or "."), f"{name}.fasta")
        open(out, "w").write(f">protein|name={name}\n{seq}\n")
        return {"ok": True, "want": want, "path": out, "value": seq}
    return {"ok": False, "error": f"unknown format {want!r}"}


def route(base_dir: str, hypothesis_id: str, type: str, want: str,
          limit: int | None = None, unused_only: bool = False,
          producer_tool: str | None = None) -> dict:
    """query + materialize in one worker call: artifacts of a type, each rendered
    into `want` format, ready to drop into a downstream tool's rows."""
    q = query(base_dir, hypothesis_id, type, unused_only=unused_only,
              limit=limit, producer_tool=producer_tool)
    mats = []
    for a in q["artifacts"]:
        m = materialize(a, want)
        if m.get("ok"):
            e = {"artifact_id": a["artifact_id"], "producer_tool": a.get("producer_tool")}
            if "path" in m: e["path"] = m["path"]
            if "value" in m: e["value"] = m["value"]
            mats.append(e)
    return {"type": type, "want": want, "n_seeds": q["count"],
            "n_materialized": len(mats), "materialized": mats}


if __name__ == "__main__":
    op = json.loads(sys.argv[1])
    kind = op.pop("op")
    fn = {"harvest": harvest, "query": query, "materialize": materialize, "route": route}[kind]
    print(json.dumps(fn(**op)))
