#!/usr/bin/env python3
"""Loam run_batch row runner for Chai-1 (skill-MCP route).

`run_batch` executes `python chai_run_row.py input.json` once per JSONL row, in a
per-row folder, with this row pinned to ONE XPU tile via ZE_AFFINITY_MASK. This
is "the function that takes inputs": it reads one row, resolves the chai install
WORKER-SIDE (site manifest or built-in defaults), subprocesses the chai_xpu.py
launcher under the chai venv, and writes outputs into the per-row cwd.

It bakes in the trap the chai SKILL warns about: RDKit 2026 can SILENTLY drop a
ligand, leaving only a `Failed to tokenize` line in the log and a "successful"
exit. A row that expected a ligand but hit that line is reported as a FAILURE, so
the mistake surfaces as a nonzero exit instead of a quietly wrong structure.

Input row (JSONL):
  id         : row identity (ignored by the batch distinctness check)
  fasta      : chai entity FASTA, as inline text OR an absolute path to a .fasta
  seed       : int; distinct seeds are distinct diffusion samples (real work)
  recycles   : --num-trunk-recycles   (default 3)
  timesteps  : --num-diffn-timesteps  (default 200)
  samples    : --num-diffn-samples    (default 5)
  use_msa    : bool; adds --use-msa-server (needs the ALCF proxy)   (default False)
  expect_ligand : bool override; else inferred from a ">ligand" entity in the FASTA

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B):
  python   : site["chai"]["python"]   -> else built-in default
  launcher : site["chai"]["launcher"] -> else $HPC_LOAM_HOME/<launcher_rel>
  weights  : site["chai"]["downloads"]-> else built-in default (CHAI_DOWNLOADS_DIR)
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.
"""
import glob
import json
import os
import socket
import subprocess
import sys
import time

SKILL = "chai"

# Built-in fallbacks so the runner works before a site manifest is deployed.
# These mirror the chai SKILL.md "Environment" block; the manifest overrides them.
_DEFAULTS = {
    "python": "/flare/FRAME-IDP/msinclair/software/chai-lab/.venv/bin/python",
    "launcher_rel": "skills/structure-prediction/chai/scripts/chai_xpu.py",
    "downloads": "/flare/FRAME-IDP/msinclair/software/chai-lab/downloads",
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
                print(f"[chai_run_row] WARN bad site manifest {path}: {exc}", flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    loam = os.environ.get("HPC_LOAM_HOME", "")
    python = site.get("python") or _DEFAULTS["python"]
    launcher = site.get("launcher")
    if not launcher:
        launcher = os.path.join(loam, _DEFAULTS["launcher_rel"]) if loam else None
    downloads = site.get("downloads") or _DEFAULTS["downloads"]
    env_extra = site.get("env", {}) or {}
    return python, launcher, downloads, env_extra


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    python, launcher, downloads, env_extra = _resolve()

    if not launcher or not os.path.isfile(launcher):
        sys.exit(f"[chai_run_row] launcher not found: {launcher!r} "
                 f"(set $HPC_LOAM_HOME or site['chai']['launcher'])")
    if not os.path.exists(python):
        sys.exit(f"[chai_run_row] chai python not found: {python!r}")

    # --- resolve the FASTA (inline text or an absolute path) ---
    fasta = row.get("fasta")
    if not fasta:
        sys.exit("[chai_run_row] row has no 'fasta'")
    if os.path.isfile(fasta):
        fasta_path = fasta
        ftext = open(fasta_path, errors="replace").read()
    else:
        fasta_path = os.path.join(cwd, "input.fasta")
        open(fasta_path, "w").write(fasta)
        ftext = fasta
    expect_ligand = bool(row.get(
        "expect_ligand",
        any(line.lower().startswith(">ligand") for line in ftext.splitlines()),
    ))

    out_dir = os.path.join(cwd, "out")  # chai requires OUT_DIR empty/nonexistent
    cmd = [
        python, launcher, "fold", fasta_path, out_dir,
        "--num-trunk-recycles", str(row.get("recycles", 3)),
        "--num-diffn-timesteps", str(row.get("timesteps", 200)),
        "--num-diffn-samples", str(row.get("samples", 5)),
        "--seed", str(row.get("seed", 42)),
        "--device", "xpu:0",
    ]
    if row.get("use_msa"):
        cmd.append("--use-msa-server")

    env = os.environ.copy()
    env["CHAI_DOWNLOADS_DIR"] = downloads
    env.update({k: str(v) for k, v in env_extra.items()})
    if row.get("use_msa"):
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.setdefault(key, _PROXY)

    print(f"[chai_run_row] START id={rid} tile={tile} host={host} seed={row.get('seed')} "
          f"expect_ligand={expect_ligand}", flush=True)
    log = os.path.join(cwd, "chai.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    logtext = open(log, errors="replace").read()

    cifs = sorted(glob.glob(os.path.join(out_dir, "*.cif")))
    npzs = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    dropped_ligand = "Failed to tokenize" in logtext

    # The trap: a ligand run that silently dropped the ligand is a FAILURE.
    fail = (rc != 0) or (not cifs) or (expect_ligand and dropped_ligand)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "n_cif": len(cifs), "n_npz": len(npzs),
        "cifs": [os.path.basename(c) for c in cifs],
        "expect_ligand": expect_ligand, "dropped_ligand": dropped_ligand,
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[chai_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
