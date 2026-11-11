#!/usr/bin/env python3
"""Loam run_batch row runner for Protenix (skill-MCP route).

`run_batch` executes `python protenix_run_row.py input.json` once per JSONL row, in a
per-row folder, with this row pinned to ONE XPU tile via ZE_AFFINITY_MASK. This is
"the function that takes inputs": it reads one row, resolves the Protenix install
WORKER-SIDE (site manifest or built-in defaults), subprocesses the `protenix` console
script under the Protenix venv (with the old-oneAPI LD_LIBRARY_PATH, PROTENIX_ROOT_DIR
and the MANDATORY `--trimul_kernel torch --triatt_kernel torch` XPU kernel swaps), and
writes outputs into the per-row cwd (protenix caches ./esm_embeddings into cwd, so a
per-row folder is exactly right).

Baked-in guards (the SKILL's traps, turned into clear failures):
  * Only three model names resolve offline. Any other `model` would silently try to
    DOWNLOAD weights and hang/fail; this runner rejects it up front with a message.
  * `--use_msa` defaults to TRUE in protenix (network MSA search); this runner passes
    an explicit value every time so a row never accidentally hits the network.
  * `--trimul_kernel`/`--triatt_kernel` default to CUDA cuequivariance; always forced
    to `torch` here.

Input row (JSONL):
  id        : row identity
  input     : Protenix input JSON (a list of entity dicts), inline text OR a .json path
  model     : -n/--model_name  (one of the 3 offline models)  (default protenix_base_default_v1.0.0)
  seeds     : -s/--seeds  comma-list string or int             (default "101")
  cycle     : -c/--cycle  Pairformer recycles                  (default 10)
  step      : -p/--step   diffusion steps                      (default 200)
  sample    : -e/--sample samples per seed                     (default 5)
  dtype     : -d/--dtype  "bf16" | "fp32"                      (default "bf16")
  use_msa   : bool -> --use_msa true/false (true needs proxy)  (default False)

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B):
  protenix : site["protenix"]["protenix"] -> else built-in default (.venv/bin/protenix)
  root_dir : site["protenix"]["root_dir"] -> else built-in default (PROTENIX_ROOT_DIR)
  env (LD_LIBRARY_PATH / PROTENIX_ROOT_DIR / PYTHONUNBUFFERED) overlaid from the manifest.
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.
"""
import glob
import json
import os
import socket
import subprocess
import sys
import time

SKILL = "protenix"

# Built-in fallbacks so the runner works before a site manifest is deployed.
# These mirror the protenix SKILL.md "Environment" block; the manifest overrides them.
_DEFAULTS = {
    "protenix": "/flare/FRAME-IDP/msinclair/software/Protenix/.venv/bin/protenix",
    "root_dir": "/flare/FRAME-IDP/msinclair/software/protenix_data",
    "ld_library_path": "/opt/aurora/26.26.0/oneapi/2025.3/lib",
}
_OFFLINE_MODELS = {
    "protenix_mini_esm_v0.5.0",
    "protenix_base_default_v1.0.0",
    "protenix-v2",
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
                print(f"[protenix_run_row] WARN bad site manifest {path}: {exc}", flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    protenix = site.get("protenix") or _DEFAULTS["protenix"]
    root_dir = site.get("root_dir") or _DEFAULTS["root_dir"]
    env_extra = site.get("env", {}) or {}
    return protenix, root_dir, env_extra


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    protenix, root_dir, env_extra = _resolve()

    if not os.path.exists(protenix):
        sys.exit(f"[protenix_run_row] protenix executable not found: {protenix!r} "
                 f"(set site['protenix']['protenix'])")

    model = row.get("model", "protenix_base_default_v1.0.0")
    if model not in _OFFLINE_MODELS:
        sys.exit(f"[protenix_run_row] model {model!r} is not one of the offline models "
                 f"{sorted(_OFFLINE_MODELS)} -- it would try to download weights.")

    # --- resolve the input JSON (inline text or an absolute path) ---
    inp = row.get("input")
    if not inp:
        sys.exit("[protenix_run_row] row has no 'input'")
    if isinstance(inp, (list, dict)):
        in_path = os.path.join(cwd, "input.json")
        json.dump(inp, open(in_path, "w"))
    elif os.path.isfile(inp):
        in_path = inp
    else:
        in_path = os.path.join(cwd, "input.json")
        open(in_path, "w").write(inp)

    out_dir = os.path.join(cwd, "out")
    use_msa = "true" if row.get("use_msa") else "false"
    cmd = [
        protenix, "pred",
        "-i", in_path,
        "-o", out_dir,
        "-s", str(row.get("seeds", "101")),
        "-n", model,
        "-c", str(row.get("cycle", 10)),
        "-p", str(row.get("step", 200)),
        "-e", str(row.get("sample", 5)),
        "-d", str(row.get("dtype", "bf16")),
        "--use_msa", use_msa,
        "--trimul_kernel", "torch",
        "--triatt_kernel", "torch",
    ]

    env = os.environ.copy()
    # Old oneAPI libs first (the venv was built on frameworks/2025.3.1 -> needs its MKL).
    env["LD_LIBRARY_PATH"] = _DEFAULTS["ld_library_path"] + ":" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("PROTENIX_ROOT_DIR", root_dir)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update({k: str(v) for k, v in env_extra.items()})
    if row.get("use_msa"):
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.setdefault(key, _PROXY)

    print(f"[protenix_run_row] START id={rid} tile={tile} host={host} model={model} "
          f"use_msa={use_msa}", flush=True)
    log = os.path.join(cwd, "protenix.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode

    cifs = sorted(glob.glob(os.path.join(out_dir, "*", "seed_*", "predictions", "*_sample_*.cif")))
    confs = sorted(glob.glob(os.path.join(out_dir, "*", "seed_*", "predictions",
                                          "*_summary_confidence_sample_*.json")))

    fail = (rc != 0) or (not cifs)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "n_cif": len(cifs), "n_conf": len(confs),
        "outputs": [os.path.relpath(c, cwd) for c in cifs],
        "model": model,
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[protenix_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
