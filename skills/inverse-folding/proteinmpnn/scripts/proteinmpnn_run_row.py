#!/usr/bin/env python3
"""Loam run_batch row runner for ProteinMPNN inverse folding (skill-MCP route).

`run_batch` executes `python proteinmpnn_run_row.py input.json` once per JSONL
row, in a per-row folder, with this row pinned to ONE XPU tile via
ZE_AFFINITY_MASK. This is "the function that takes inputs": it reads one row,
resolves the ProteinMPNN install WORKER-SIDE, subprocesses `protein_mpnn_run.py`
under the ProteinMPNN venv, and writes outputs into the per-row cwd.

It bakes in the trap the proteinmpnn SKILL warns about: the device is hard-coded
`xpu:0 if torch.xpu.is_available() else cpu`, and if the XPU is NOT visible the
run SILENTLY falls back to CPU and still "succeeds" (just wrong/slow). The log
line `model on device: cpu` is the only signal. A row that landed on CPU is
reported as a FAILURE so the mistake surfaces as a nonzero exit. The `--device`
flag is parsed but IGNORED (tile pinning is ZE_AFFINITY_MASK, set per-row by EL).

The IPEX shim (empty stand-in package) MUST be on PYTHONPATH: the Aurora port
does `import intel_extension_for_pytorch` but the 2026 frameworks stack has no
IPEX, so without the shim every run dies with ModuleNotFoundError.

Input row (JSONL):
  id                 : row identity
  pdb                : ABSOLUTE path to the input backbone PDB (required)
  designed_chains    : chains to DESIGN, space-separated e.g. "A" or "A B"
                       (unlisted chains stay as fixed context; "" => all)   (default "")
  num_seq_per_target : sequences per structure per temperature   (default 8)
  sampling_temp      : space-separated temps string, e.g. "0.1 0.2"   (default "0.1")
  seed               : int; 0 = random   (default 37)
  batch_size         : copies decoded in parallel   (default 1)
  model_name         : v_48_002|v_48_010|v_48_020|v_48_030   (default v_48_020)
  use_soluble_model  : bool; use the soluble weights   (default False)
  ca_only            : bool; CA-only model (002/010/020 only)   (default False)
  backbone_noise     : float; Gaussian noise added to the backbone   (optional)
  omit_AAs           : string of residues to globally forbid, e.g. "CX"   (optional)
  save_score         : bool; write per-sequence score .npz   (default False)
  save_probs         : bool; write per-position probability .npz   (default False)
  score_only         : bool; SCORE the native sequence instead of designing
                       (writes score_only/<name>.npz)   (default False)

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B/D):
  repo    : $PROTEINMPNN_PATH (EXPORTED on the worker, preferred) -> site["path"]
            -> built-in default. protein_mpnn_run.py + its weight folders are read
            from here.
  python  : site["python"] -> built-in default (the repo's .venv python).
  ipex_shim: site["ipex_shim"] -> $HPC_LOAM_HOME/<shim_rel> -> beside this runner.
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.
"""
import glob
import json
import os
import socket
import subprocess
import sys
import time

SKILL = "proteinmpnn"

# Built-in fallbacks so the runner works before a site manifest is deployed.
# These mirror the proteinmpnn SKILL.md "Environment" block; the manifest and the
# exported $PROTEINMPNN_PATH override them.
_DEFAULTS = {
    "repo": "/flare/FRAME-IDP/msinclair/software/ProteinMPNN",
    "python": "/flare/FRAME-IDP/msinclair/software/ProteinMPNN/.venv/bin/python",
    "shim_rel": "skills/inverse-folding/proteinmpnn/scripts/ipex_shim",
}


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
                print(f"[proteinmpnn_run_row] WARN bad site manifest {path}: {exc}",
                      flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    loam = os.environ.get("HPC_LOAM_HOME", "")
    here = os.path.dirname(os.path.abspath(__file__))

    # Repo root: PREFER the exported $PROTEINMPNN_PATH (verified present on the
    # worker), then the manifest, then the built-in default.
    repo = os.environ.get("PROTEINMPNN_PATH") or site.get("path") or _DEFAULTS["repo"]

    # Interpreter: manifest wins; else the built-in venv python. Kept independent
    # of the repo root because $PROTEINMPNN_PATH may point at a checkout without
    # its own .venv.  # TODO(verify): if $PROTEINMPNN_PATH's checkout has a working
    # .venv, set site["proteinmpnn"]["python"] to <repo>/.venv/bin/python.
    python = site.get("python") or _DEFAULTS["python"]

    # IPEX shim dir (REQUIRED on PYTHONPATH).
    shim = site.get("ipex_shim")
    if not shim:
        if loam:
            shim = os.path.join(loam, _DEFAULTS["shim_rel"])
        if not (shim and os.path.isdir(shim)):
            shim = os.path.join(here, "ipex_shim")  # beside this runner

    env_extra = site.get("env", {}) or {}
    return repo, python, shim, env_extra


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    repo, python, shim, env_extra = _resolve()

    run_py = os.path.join(repo, "protein_mpnn_run.py")
    if not os.path.isfile(run_py):
        sys.exit(f"[proteinmpnn_run_row] protein_mpnn_run.py not found: {run_py!r} "
                 f"(set $PROTEINMPNN_PATH or site['proteinmpnn']['path'])")
    if not os.path.exists(python):
        sys.exit(f"[proteinmpnn_run_row] proteinmpnn python not found: {python!r}")
    if not (shim and os.path.isdir(shim)):
        sys.exit(f"[proteinmpnn_run_row] ipex_shim dir not found: {shim!r} "
                 f"(set $HPC_LOAM_HOME or site['proteinmpnn']['ipex_shim'])")

    pdb = row.get("pdb")
    if not pdb:
        sys.exit("[proteinmpnn_run_row] row has no 'pdb'")
    if not os.path.isfile(pdb):
        sys.exit(f"[proteinmpnn_run_row] input pdb not found: {pdb!r}")

    score_only = bool(row.get("score_only", False))
    out_dir = os.path.join(cwd, "out")
    os.makedirs(out_dir, exist_ok=True)

    num_seq = int(row.get("num_seq_per_target", 8))
    batch_size = int(row.get("batch_size", 1))
    # Silent trap: num_seq_per_target < batch_size => NUM_BATCHES == 0 => zero outputs.
    if not score_only and num_seq < batch_size:
        print(f"[proteinmpnn_run_row] WARN num_seq_per_target ({num_seq}) < "
              f"batch_size ({batch_size}); would produce 0 sequences", flush=True)

    cmd = [
        python, run_py,
        "--pdb_path", pdb,
        "--out_folder", out_dir,
        "--num_seq_per_target", str(num_seq),
        "--sampling_temp", str(row.get("sampling_temp", "0.1")),
        "--seed", str(row.get("seed", 37)),
        "--batch_size", str(batch_size),
        "--model_name", str(row.get("model_name", "v_48_020")),
    ]
    designed = str(row.get("designed_chains", "")).strip()
    if designed:  # chains to DESIGN; unlisted chains kept as fixed context
        cmd += ["--pdb_path_chains", designed]
    if row.get("use_soluble_model"):
        cmd.append("--use_soluble_model")
    if row.get("ca_only"):
        cmd.append("--ca_only")
    if "backbone_noise" in row:
        cmd += ["--backbone_noise", str(row["backbone_noise"])]
    if "omit_AAs" in row:
        cmd += ["--omit_AAs", str(row["omit_AAs"])]
    if row.get("save_score"):
        cmd += ["--save_score", "1"]
    if row.get("save_probs"):
        cmd += ["--save_probs", "1"]
    if score_only:
        cmd += ["--score_only", "1"]

    env = os.environ.copy()
    # IPEX shim MUST be first on PYTHONPATH (SKILL.md Environment).
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = shim + (os.pathsep + prev_pp if prev_pp else "")
    env.update({k: str(v) for k, v in env_extra.items()})
    # ZE_AFFINITY_MASK is set per-row by EL; the device then appears as xpu:0.

    print(f"[proteinmpnn_run_row] START id={rid} tile={tile} host={host} "
          f"pdb={os.path.basename(pdb)} designed={designed!r} "
          f"score_only={score_only}", flush=True)
    log = os.path.join(cwd, "proteinmpnn.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    logtext = open(log, errors="replace").read()

    # The trap: silent CPU fallback (XPU not visible). A CPU run is a FAILURE.
    on_cpu = "model on device: cpu" in logtext

    if score_only:
        outs = sorted(glob.glob(os.path.join(out_dir, "score_only", "*.npz")))
        n_out = len(outs)
        n_key = "n_npz"
    else:
        fastas = sorted(glob.glob(os.path.join(out_dir, "seqs", "*.fa")))
        # Count DESIGNED sequence records (headers start with ">T="); the first
        # ">..." record in each file is the native sequence, not a design.
        n_out = 0
        for fa in fastas:
            for line in open(fa, errors="replace"):
                if line.startswith(">T="):
                    n_out += 1
        outs = [os.path.relpath(f, out_dir) for f in fastas]
        n_key = "n_seq"

    fail = (rc != 0) or on_cpu or (n_out == 0)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        n_key: n_out,
        "outputs": outs,
        "score_only": score_only,
        "on_cpu": on_cpu,
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[proteinmpnn_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
