#!/usr/bin/env python3
"""Loam run_batch row runner for PyRosetta (skill-MCP route).

`run_batch` executes `python pyrosetta_run_row.py input.json` once per JSONL row,
in a per-row folder. PyRosetta is CPU-ONLY (never needs a GPU); EL still pins the
row to one tile slot and may set ZE_AFFINITY_MASK, which the runner ignores for
compute. This is "the function that takes inputs": it reads one row, resolves the
pyrosetta_energy venv WORKER-SIDE (site manifest or built-in defaults mirroring the
SKILL.md Environment block), and subprocesses the committed, verified CLI
`rosetta_tool.py` under that venv (one `pyrosetta.init` per process), writing
outputs into the per-row cwd.

Although PyRosetta is pure-python and could be imported in-process, the runner
subprocesses into the skill venv for consistency with the chai C2 pattern and so
EL's per-row process/tile pinning holds (see docs/SKILL_MCP_DESIGN.md §4.C).

Input row (JSONL):
  id         : row identity
  pdb        : ABSOLUTE path to the input PDB (worker-visible /flare)   [required]
  action     : "score" (VERIFIED) | "relax" (VERIFIED) | "interface" (VERIFIED)
                                                                   (default "score")
  sfxn       : score function                                     (default "ref2015")
  rounds     : FastRelax repeats (action="relax")                 (default 1)
  max_iter   : minimizer cap, 0 = unset (action="relax")          (default 0)
  constrain  : coordinate-constrain to start coords (relax)       (default False)
  partners   : docking partners string, e.g. "A_B" (interface)    (default "A_B")
  pack_input, pack_separated : repack flags (interface)           (default False)

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B):
  python   : site["pyrosetta"]["python"]   -> else built-in default
  launcher : site["pyrosetta"]["launcher"] -> else $HPC_LOAM_HOME/<launcher_rel>
             -> else rosetta_tool.py beside this runner
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.

All three actions were verified on an Aurora debug node (2026-10-29). SKILL.md marks
only the PBS `qsub` submission UNVERIFIED; this runner does not use qsub (EL submits),
so there is no untested path here. `init` uses `extra_options=` (NOT a positional
string), which the SKILL.md warns silently changes packing/relax/interface results.
"""
import glob
import json
import os
import socket
import subprocess
import sys
import time

SKILL = "pyrosetta"  # site-manifest key

_DEFAULTS = {
    "python": "/flare/FRAME-IDP/msinclair/software/pyrosetta_energy/.venv/bin/python",
    "launcher_rel": "skills/analysis/pyrosetta/scripts/rosetta_tool.py",
}


def _load_site_entry():
    """This skill's entry from the worker-side site manifest, or {} if none."""
    loam = os.environ.get("HPC_LOAM_HOME", "")
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("LOAM_SITE_MANIFEST"),
        os.path.join(here, "site.aurora.json"),
        os.path.join(loam, "skills", "site.aurora.json") if loam else None,
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return (json.load(open(path)) or {}).get(SKILL, {}) or {}
            except Exception as exc:
                print(f"[pyrosetta_run_row] WARN bad site manifest {path}: {exc}", flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    loam = os.environ.get("HPC_LOAM_HOME", "")
    here = os.path.dirname(os.path.abspath(__file__))
    python = site.get("python") or _DEFAULTS["python"]
    launcher = site.get("launcher")
    if not launcher:
        cand = os.path.join(loam, _DEFAULTS["launcher_rel"]) if loam else None
        if cand and os.path.isfile(cand):
            launcher = cand
        else:
            launcher = os.path.join(here, "rosetta_tool.py")  # committed beside runner
    env_extra = site.get("env", {}) or {}
    return python, launcher, env_extra


def _build_cmd(python, launcher, row, cwd):
    """Map the row's action onto a rosetta_tool.py invocation + the expected output."""
    pdb = row["pdb"]
    action = row.get("action", "score")
    sfxn = row.get("sfxn", "ref2015")
    if action == "score":
        out = os.path.join(cwd, "scores.json")
        cmd = [python, launcher, "score", pdb, "--sfxn", sfxn, "--json", out]
        return cmd, [out]
    if action == "relax":
        out = os.path.join(cwd, "relaxed.pdb")
        cmd = [python, launcher, "relax", pdb, "--sfxn", sfxn,
               "--out", out, "--rounds", str(row.get("rounds", 1))]
        if row.get("max_iter"):
            cmd += ["--max-iter", str(row["max_iter"])]
        if row.get("constrain"):
            cmd += ["--constrain"]
        return cmd, [out]
    if action == "interface":
        out = os.path.join(cwd, "interface.json")
        cmd = [python, launcher, "interface", pdb, "--sfxn", sfxn,
               "--partners", row.get("partners", "A_B"), "--json", out]
        if row.get("pack_input"):
            cmd += ["--pack-input"]
        if row.get("pack_separated"):
            cmd += ["--pack-separated"]
        return cmd, [out]
    sys.exit(f"[pyrosetta_run_row] unknown action {action!r}")


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")  # CPU job; logged only
    host = socket.gethostname()
    python, launcher, env_extra = _resolve()

    if not os.path.exists(python):
        sys.exit(f"[pyrosetta_run_row] skill python not found: {python!r}")
    if not launcher or not os.path.isfile(launcher):
        sys.exit(f"[pyrosetta_run_row] launcher not found: {launcher!r} "
                 f"(set $HPC_LOAM_HOME or site['pyrosetta']['launcher'])")
    pdb = row.get("pdb")
    if not pdb:
        sys.exit("[pyrosetta_run_row] row has no 'pdb'")
    if not os.path.isfile(pdb):
        sys.exit(f"[pyrosetta_run_row] input pdb not found: {pdb!r}")

    action = row.get("action", "score")
    cmd, expected = _build_cmd(python, launcher, row, cwd)

    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items()})

    print(f"[pyrosetta_run_row] START id={rid} tile={tile} host={host} "
          f"action={action} pdb={os.path.basename(pdb)}", flush=True)
    log = os.path.join(cwd, "pyrosetta.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode

    produced = [p for p in expected if os.path.isfile(p)]
    outputs = [os.path.basename(p) for p in produced]

    fail = (rc != 0) or (len(produced) != len(expected))
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "action": action,
        "outputs": outputs,
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[pyrosetta_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
