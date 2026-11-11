#!/usr/bin/env python3
"""Loam run_batch row runner for Boltz-2 (skill-MCP route).

`run_batch` executes `python boltz_run_row.py input.json` once per JSONL row, in a
per-row folder, with this row pinned to ONE XPU tile via ZE_AFFINITY_MASK. This is
"the function that takes inputs": it reads one row, resolves the Boltz install
WORKER-SIDE (site manifest or built-in defaults), subprocesses the `boltz` console
script under the Boltz venv, and writes outputs into the per-row cwd.

It bakes in the Boltz silent-ish trap the SKILL warns about: Boltz prints
`Number of failed examples: N` and can still exit 0 with N>0 (a bad YAML, a SMILES
that would not parse, a missing CCD code). A row whose log reports N>0 failed
examples is turned into a FAILURE so the mistake surfaces as a nonzero exit instead
of a quietly empty prediction dir. An affinity run that produced no affinity JSON is
likewise a failure.

Input row (JSONL):
  id            : row identity
  input         : Boltz YAML or FASTA, as inline text OR an absolute path to a file
  input_format  : "yaml" | "fasta"  (only used to name an inline input; default "yaml")
  seed          : int  --seed                              (default 42)
  recycling_steps : --recycling_steps                      (default 3)
  sampling_steps  : --sampling_steps                       (default 200)
  diffusion_samples : --diffusion_samples                  (default 5)
  output_format : "pdb" | "mmcif"                          (default "pdb")
  use_msa       : bool; adds --use_msa_server (needs the ALCF proxy)   (default False)
  expect_affinity : bool override; else inferred from "affinity" in a YAML input
  sampling_steps_affinity   : int  (default = sampling_steps)   # TODO(verify) production value
  diffusion_samples_affinity: int  (default 1)                  # TODO(verify) production value

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B):
  boltz : site["boltz"]["boltz"] -> else built-in default (the .venv/bin/boltz console script)
  cache : site["boltz"]["cache"] -> else built-in default (~/.boltz)
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

SKILL = "boltz"

# Built-in fallbacks so the runner works before a site manifest is deployed.
# These mirror the boltz SKILL.md "Environment" block; the manifest overrides them.
_DEFAULTS = {
    "boltz": "/flare/FRAME-IDP/msinclair/software/boltz/.venv/bin/boltz",
    "cache": os.path.expanduser("~/.boltz"),
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
                print(f"[boltz_run_row] WARN bad site manifest {path}: {exc}", flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    boltz = site.get("boltz") or _DEFAULTS["boltz"]
    cache = site.get("cache") or _DEFAULTS["cache"]
    env_extra = site.get("env", {}) or {}
    return boltz, os.path.expanduser(cache), env_extra


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    boltz, cache, env_extra = _resolve()

    if not os.path.exists(boltz):
        sys.exit(f"[boltz_run_row] boltz executable not found: {boltz!r} "
                 f"(set site['boltz']['boltz'])")

    # --- resolve the input (inline text or an absolute path) ---
    inp = row.get("input")
    if not inp:
        sys.exit("[boltz_run_row] row has no 'input'")
    fmt = (row.get("input_format") or "yaml").lower()
    if os.path.isfile(inp):
        in_path = inp
        itext = open(in_path, errors="replace").read()
    else:
        ext = "fasta" if fmt == "fasta" else "yaml"
        in_path = os.path.join(cwd, f"input.{ext}")
        open(in_path, "w").write(inp)
        itext = inp
    expect_affinity = bool(row.get(
        "expect_affinity",
        ("affinity" in itext.lower()) and fmt != "fasta",
    ))

    out_dir = os.path.join(cwd, "out")
    sampling_steps = int(row.get("sampling_steps", 200))
    cmd = [
        boltz, "predict", in_path,
        "--out_dir", out_dir,
        "--cache", cache,
        "--accelerator", "xpu",
        "--devices", "1",
        "--output_format", "mmcif" if row.get("output_format") == "mmcif" else "pdb",
        "--num_workers", "0",
        "--override",
        "--recycling_steps", str(row.get("recycling_steps", 3)),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", str(row.get("diffusion_samples", 5)),
        "--seed", str(row.get("seed", 42)),
    ]
    if expect_affinity:
        # TODO(verify): production affinity sampling settings are untimed in the SKILL
        # (it used 50 / 1 for a smoke run). Defaults below mirror the structure settings.
        cmd += [
            "--sampling_steps_affinity",
            str(row.get("sampling_steps_affinity", sampling_steps)),
            "--diffusion_samples_affinity",
            str(row.get("diffusion_samples_affinity", 1)),
        ]
    if row.get("use_msa"):
        cmd.append("--use_msa_server")

    env = os.environ.copy()
    env["BOLTZ_CACHE"] = cache
    env.update({k: str(v) for k, v in env_extra.items()})
    if row.get("use_msa"):
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.setdefault(key, _PROXY)

    print(f"[boltz_run_row] START id={rid} tile={tile} host={host} seed={row.get('seed')} "
          f"expect_affinity={expect_affinity}", flush=True)
    log = os.path.join(cwd, "boltz.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    logtext = open(log, errors="replace").read()

    pdbs = sorted(glob.glob(os.path.join(out_dir, "boltz_results_*", "predictions", "*", "*_model_0.pdb")))
    cifs = sorted(glob.glob(os.path.join(out_dir, "boltz_results_*", "predictions", "*", "*_model_0.cif")))
    structs = pdbs + cifs
    confs = sorted(glob.glob(os.path.join(out_dir, "boltz_results_*", "predictions", "*", "confidence_*_model_0.json")))
    affs = sorted(glob.glob(os.path.join(out_dir, "boltz_results_*", "predictions", "*", "affinity_*.json")))

    # The trap: Boltz can report failed examples and still exit 0.
    m = re.search(r"Number of failed examples:\s*(\d+)", logtext)
    n_failed = int(m.group(1)) if m else 0

    fail = (rc != 0) or (not structs) or (n_failed > 0) or (expect_affinity and not affs)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "n_struct": len(structs), "n_conf": len(confs), "n_aff": len(affs),
        "n_failed_examples": n_failed,
        "outputs": [os.path.relpath(p, cwd) for p in structs],
        "expect_affinity": expect_affinity,
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[boltz_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
