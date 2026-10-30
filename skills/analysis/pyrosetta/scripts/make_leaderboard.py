#!/usr/bin/env python
"""Build a leaderboard.json so the repo's compute_rosetta_energy.py can score ANY
directory of 2-chain (A=target, B=binder) PDBs.

Usage: python make_leaderboard.py /path/to/pdb_dir [--glob '*.pdb']
Writes /path/to/pdb_dir/leaderboard.json with entries
  {"binder_id": <file stem>, "pdb": <abs path>, "score": 0.0}
(the repo script only reads binder_id, pdb (basename is re-joined to the dir) and score).
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("pdb_dir")
ap.add_argument("--glob", default="*.pdb")
a = ap.parse_args()
d = Path(a.pdb_dir).resolve()
entries = [{"binder_id": p.stem, "pdb": str(p), "score": 0.0} for p in sorted(d.glob(a.glob))]
out = d / "leaderboard.json"
if out.exists():
    raise SystemExit(f"{out} exists; refusing to overwrite")
json.dump(entries, open(out, "w"), indent=1)
print(f"wrote {len(entries)} entries to {out}")
