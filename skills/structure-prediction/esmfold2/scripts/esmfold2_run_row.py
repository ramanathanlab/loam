#!/usr/bin/env python3
"""Loam run_batch row runner for ESMFold2 (skill-MCP route).

`run_batch` executes `python esmfold2_run_row.py input.json` once per JSONL row, in
a per-row folder, with this row pinned to ONE XPU tile via ZE_AFFINITY_MASK. This is
"the function that takes inputs": it reads one row, resolves the ESMFold2 install
WORKER-SIDE (site manifest or built-in defaults), subprocesses the repo's
`fold_sequence.py` launcher under the esm venv (with the FORKED transformers first on
PYTHONPATH, HF cache + offline set), and writes the mmCIF into the per-row cwd.

It bakes in the ESMFold2 silent trap the SKILL warns about: `fold_sequence.py` folds
only the FIRST record of a FASTA. A row that hands it a multi-record FASTA (meaning to
fold a complex) would get a single-chain structure and a "successful" exit. Such a row
is turned into a FAILURE here so the silently-dropped chains surface as a nonzero exit
instead of a quietly wrong monomer. (The CLI is single-chain only; use the Python API
for real complexes.)

NOTE on cost: ESMFold2 amortizes its ~90 s model load + first-fold warm-up (253 s for
L=56 observed) by folding many sequences in ONE process. run_batch runs one FRESH
process per row, so every row pays that load+warm-up. That is inherent to the one-row-
per-process model, not something this runner can avoid; size timeout_each accordingly.

Input row (JSONL):
  id         : row identity
  seq        : amino-acid sequence (single chain)         [seq OR fasta required]
  fasta      : FASTA as inline text OR an absolute path (first record only is folded)
  loops      : --loops   trunk recycling loops            (default 20)
  steps      : --steps   diffusion sampling steps         (default 100)
  seed       : --seed    (note: NOT honored on XPU per the SKILL)   (default 0)
  chunk_size : --chunk-size  (optional; lower for long chains to cut O(L^2) memory)
  out        : output mmCIF name under cwd                 (default "pred.cif")

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B):
  python   : site["esmfold2"]["python"]   -> else built-in default (esm/.venv/bin/python)
  launcher : site["esmfold2"]["launcher"] -> else built-in default (repo fold_sequence.py)
             -> else $HPC_LOAM_HOME/skills/structure-prediction/esmfold2/scripts/<launcher_rel>
  repo     : site["esmfold2"]["repo"]     -> drives PYTHONPATH (fork first) + HF_HOME
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.
"""
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time

SKILL = "esmfold2"

# Built-in fallbacks so the runner works before a site manifest is deployed.
# These mirror the esmfold2 SKILL.md "Environment" block; the manifest overrides them.
_REPO = "/flare/FRAME-IDP/msinclair/software/esm"
_DEFAULTS = {
    "python": _REPO + "/.venv/bin/python",
    "launcher": _REPO + "/scripts/aurora/fold_sequence.py",
    "launcher_rel": "scripts/fold_sequence.py",
    "repo": _REPO,
}
_PROXY = "http://proxy.alcf.anl.gov:3128"


def _load_site_entry():
    """This skill's entry from the worker-side site manifest, or {} if none."""
    loam = os.environ.get("HPC_LOAM_HOME", "")
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("LOAM_SITE_MANIFEST"),
        os.path.join(here, "site.aurora.json"),          # staged beside the runner
        os.path.join(loam, "skills", "site.aurora.json") if loam else None,
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return (json.load(open(path)) or {}).get(SKILL, {}) or {}
            except Exception as exc:  # a bad manifest must not kill the row
                print(f"[esmfold2_run_row] WARN bad site manifest {path}: {exc}", flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    loam = os.environ.get("HPC_LOAM_HOME", "")
    python = site.get("python") or _DEFAULTS["python"]
    launcher = site.get("launcher") or _DEFAULTS["launcher"]
    if not launcher or not os.path.isfile(launcher):
        if loam:
            launcher = os.path.join(loam, "skills", "structure-prediction", SKILL,
                                    _DEFAULTS["launcher_rel"])
    repo = site.get("repo") or _DEFAULTS["repo"]
    env_extra = site.get("env", {}) or {}
    return python, launcher, repo, env_extra


def _count_fasta_records(text):
    return sum(1 for ln in text.splitlines() if ln.startswith(">"))


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    python, launcher, repo, env_extra = _resolve()

    if not launcher or not os.path.isfile(launcher):
        sys.exit(f"[esmfold2_run_row] launcher not found: {launcher!r} "
                 f"(set site['esmfold2']['launcher'] or $HPC_LOAM_HOME)")
    if not os.path.exists(python):
        sys.exit(f"[esmfold2_run_row] esm python not found: {python!r}")

    out_name = row.get("out", "pred.cif")
    out_path = os.path.join(cwd, out_name)
    cmd = [python, launcher,
           "--loops", str(row.get("loops", 20)),
           "--steps", str(row.get("steps", 100)),
           "--seed", str(row.get("seed", 0)),
           "--out", out_path]
    if row.get("chunk_size") is not None:
        cmd += ["--chunk-size", str(row.get("chunk_size"))]

    # --- resolve the input: seq (preferred) or fasta (first record only) ---
    multi_record = False
    seq = row.get("seq")
    fasta = row.get("fasta")
    if seq:
        cmd += ["--seq", str(seq)]
    elif fasta:
        if os.path.isfile(fasta):
            fasta_path = fasta
            ftext = open(fasta_path, errors="replace").read()
        else:
            fasta_path = os.path.join(cwd, "input.fasta")
            open(fasta_path, "w").write(fasta)
            ftext = fasta
        multi_record = _count_fasta_records(ftext) > 1
        cmd += ["--fasta", fasta_path]
    else:
        sys.exit("[esmfold2_run_row] row has neither 'seq' nor 'fasta'")

    env = os.environ.copy()
    # Fork transformers MUST come first on PYTHONPATH; then HF cache + offline.
    env["PYTHONPATH"] = f"{repo}/transformers/src:{repo}:" + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", repo + "/.hf_cache")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.update({k: str(v) for k, v in env_extra.items()})

    print(f"[esmfold2_run_row] START id={rid} tile={tile} host={host} "
          f"multi_record={multi_record}", flush=True)
    log = os.path.join(cwd, "esmfold2.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    logtext = open(log, errors="replace").read()

    cifs = sorted(glob.glob(os.path.join(cwd, "*.cif")))
    # Best-effort pLDDT scrape from the launcher's stdout (it prints the mean).
    plddt = None
    pm = re.search(r"pLDDT[^0-9]*([01]?\.\d+)", logtext)
    if pm:
        try:
            plddt = float(pm.group(1))
        except ValueError:
            plddt = None

    # The trap: a multi-record FASTA silently folds only the first chain.
    fail = (rc != 0) or (not cifs) or multi_record
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "n_cif": len(cifs),
        "outputs": [os.path.basename(c) for c in cifs],
        "plddt": plddt,
        "multi_record_dropped": multi_record,
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[esmfold2_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
