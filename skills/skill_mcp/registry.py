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


if __name__ == "__main__":
    op = json.loads(sys.argv[1])
    kind = op.pop("op")
    fn = {"harvest": harvest, "query": query}[kind]
    print(json.dumps(fn(**op)))
