#!/usr/bin/env python3
"""Loam run_batch row runner for UMA (fairchem-xpu) on Aurora XPU (skill-MCP route).

`run_batch` executes `python uma_run_row.py input.json` once per JSONL row, in a
per-row folder, with this row pinned to ONE XPU tile via ZE_AFFINITY_MASK (set by
the Ensemble Launcher, not by this runner). This is "the function that takes
inputs": it reads one row, resolves the fairchem-xpu install WORKER-SIDE (site
manifest or built-in defaults), subprocesses the fairchem venv python on a small
compute program, and writes outputs into the per-row cwd.

It bakes in the trap the uma SKILL warns about: `default` and `high-precision`
inference settings route to a compiled multigpu path on XPU that returns WRONG
forces (H2O max|F| 0.4254 vs the correct 0.4827 eV/A from `batch`/Nitro-XPU/CPU),
at the cost of 45-65 s of wasted compile. So `batch` is the default here, and a
row that explicitly asks for `default`/`high-precision` is REFUSED unless it also
sets `allow_wrong_xpu_forces: true` -- the mistake surfaces as a nonzero exit
instead of a quietly wrong force.

Input row (JSONL):
  id           : row identity
  task_name    : omol | omat | oc20 | odac | omc  (UMA head; default "omol")
  model        : registered UMA model name (default "uma-s-1p2p1")
  inference    : "batch" (default, general path) | "nitro-xpu" (single fixed
                 system fast path, uma-s-1p2p1 + omol only)
  # structure -- supply exactly one of:
  structure_path : absolute path to a file ASE can read (.xyz/.cif/.pdb/.traj)
  molecule       : an ase.build.molecule() name (e.g. "H2O", "CH3CH2OH")
  atoms          : {"symbols": [...], "positions": [[x,y,z],...], "cell": [...]? , "pbc": bool?}
  # omol electronic state (ignored by other heads):
  charge       : int  (default 0)
  spin         : int spin multiplicity (default 1)
  # action (manifest action "calc"):
  relax        : bool; if true run an LBFGS geometry optimization (default False)
  fmax         : relax force convergence, eV/A (default 0.02)
  steps        : relax max steps (default 200)
  allow_wrong_xpu_forces : escape hatch for default/high-precision (default False)

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md s4.B):
  python    : site["uma"]["python"]    -> else built-in default
  cache_dir : site["uma"]["cache_dir"] -> else built-in default (FAIRCHEM_CACHE_DIR)
  launcher  : site["uma"]["launcher"]  -> else $HPC_LOAM_HOME/skills/qm/uma/scripts/uma_compute.py
              -> else an embedded compute program written into the per-row cwd.
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.
"""
import json
import os
import socket
import subprocess
import sys
import time

SKILL = "uma"

# Built-in fallbacks so the runner works before a site manifest is deployed.
# These mirror the uma SKILL.md "Environment" block; the manifest overrides them.
_DEFAULTS = {
    "python": "/flare/FRAME-IDP/msinclair/software/fairchem-xpu/.venv/bin/python",
    "launcher_rel": "skills/qm/uma/scripts/uma_compute.py",
    "cache_dir": "/flare/FRAME-IDP/msinclair/hf_cache",
}
_BAD_XPU_MODES = {"default", "high-precision"}  # wrong forces on XPU (SKILL gotcha #1)

# Embedded compute program, run under the fairchem-xpu venv python when no separate
# launcher is deployed. Kept here so the runner is a single self-contained file
# (the deployment python that runs this runner is pure-stdlib and cannot import
# fairchem; the science must run under the skill venv as a subprocess -- C2 pattern).
_UMA_COMPUTE = r'''#!/usr/bin/env python3
"""UMA (fairchem-xpu) per-row compute: single point (energy+forces) +/- LBFGS relax.
Invoked as:  <fairchem venv python> uma_compute.py <row.json> <out.json>
Runs under whatever ZE_AFFINITY_MASK the Ensemble Launcher set (one visible tile).
"""
import json
import sys
import time

import torch
from ase import Atoms
from ase.build import molecule as ase_molecule
from ase.io import read as ase_read
from ase.optimize import LBFGS

from fairchem.core import FAIRChemCalculator


def build_atoms(row):
    if row.get("structure_path"):
        atoms = ase_read(row["structure_path"])
    elif row.get("molecule"):
        atoms = ase_molecule(row["molecule"])
    elif row.get("atoms"):
        a = row["atoms"]
        atoms = Atoms(symbols=a["symbols"], positions=a["positions"])
        if a.get("cell") is not None:
            atoms.set_cell(a["cell"])
        if a.get("pbc") is not None:
            atoms.set_pbc(a["pbc"])
    else:
        raise SystemExit("[uma_compute] row needs one of structure_path/molecule/atoms")
    return atoms


def make_calc(model, task_name, inference, cache_dir):
    if inference == "nitro-xpu":
        # XPU port of UMA Nitro: takes a checkpoint PATH; device="xpu" is hard-coded.
        # Merges MoLE experts for ONE composition/charge/spin -> one predictor per system.
        from huggingface_hub import hf_hub_download
        from fairchem.core.models.uma.uma_nitro_xpu import load_predict_unit
        ckpt = hf_hub_download("facebook/UMA", "checkpoints/%s.pt" % model,
                               cache_dir=cache_dir)
        predictor = load_predict_unit(ckpt)
    else:
        from fairchem.core import pretrained_mlip
        # per-row ZE_AFFINITY_MASK pins the single visible tile; device="xpu" (verified).
        predictor = pretrained_mlip.get_predict_unit(
            model, inference_settings=inference, device="xpu")
    return FAIRChemCalculator(predictor, task_name=task_name)


def main():
    row = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]
    import os as _os
    cache_dir = _os.environ.get("FAIRCHEM_CACHE_DIR")

    model = row.get("model", "uma-s-1p2p1")
    task_name = row.get("task_name", "omol")
    inference = row.get("inference", "batch")

    print("torch", torch.__version__, "xpu_available:", torch.xpu.is_available(),
          "device_count:", torch.xpu.device_count(), flush=True)
    if not torch.xpu.is_available():
        raise SystemExit("[uma_compute] torch.xpu not available (wrong python or no XPU alloc)")

    calc = make_calc(model, task_name, inference, cache_dir)
    atoms = build_atoms(row)
    if task_name == "omol":
        atoms.info.update({"charge": int(row.get("charge", 0)),
                           "spin": int(row.get("spin", 1))})
    atoms.calc = calc

    rec = {"model": model, "task_name": task_name, "inference": inference,
           "n_atoms": len(atoms), "relaxed": False}

    t0 = time.time()
    e = float(atoms.get_potential_energy())
    f = atoms.get_forces()
    rec.update({"energy_eV": e,
                "max_force_eV_per_A": float(abs(f).max()),
                "singlepoint_seconds": round(time.time() - t0, 2)})

    if row.get("relax"):
        fmax = float(row.get("fmax", 0.02))
        steps = int(row.get("steps", 200))
        if inference == "nitro-xpu":
            # Nitro merges for one system; a fresh predictor is needed if composition
            # changed, but an in-place relax keeps composition fixed, so reuse is fine.
            pass
        t1 = time.time()
        opt = LBFGS(atoms, logfile="-", trajectory="opt.traj")
        converged = bool(opt.run(fmax=fmax, steps=steps))
        atoms.write("relaxed.xyz")
        rec.update({"relaxed": True, "relax_converged": converged,
                    "relax_steps": int(opt.get_number_of_steps()),
                    "final_energy_eV": float(atoms.get_potential_energy()),
                    "final_max_force_eV_per_A": float(abs(atoms.get_forces()).max()),
                    "relax_seconds": round(time.time() - t1, 2)})

    json.dump(rec, open(out_path, "w"), indent=2)
    print("[uma_compute] OK " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
'''


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
                print(f"[uma_run_row] WARN bad site manifest {path}: {exc}", flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    loam = os.environ.get("HPC_LOAM_HOME", "")
    python = site.get("python") or _DEFAULTS["python"]
    cache_dir = site.get("cache_dir") or _DEFAULTS["cache_dir"]
    launcher = site.get("launcher")
    if not launcher and loam:
        launcher = os.path.join(loam, _DEFAULTS["launcher_rel"])
    env_extra = site.get("env", {}) or {}
    return python, launcher, cache_dir, env_extra


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    python, launcher, cache_dir, env_extra = _resolve()

    if not os.path.exists(python):
        sys.exit(f"[uma_run_row] fairchem python not found: {python!r}")

    # Bake in the XPU wrong-forces trap (SKILL gotcha #1).
    inference = row.get("inference", "batch")
    if inference in _BAD_XPU_MODES and not row.get("allow_wrong_xpu_forces"):
        sys.exit(f"[uma_run_row] inference={inference!r} gives WRONG forces on XPU "
                 f"(compiled multigpu fallback; SKILL gotcha #1). Use 'batch' or "
                 f"'nitro-xpu', or set allow_wrong_xpu_forces:true to override.")

    # Locate the compute program: a deployed launcher, else the embedded copy.
    if launcher and os.path.isfile(launcher):
        compute = launcher
    else:
        # TODO(verify): no separate uma_compute.py is deployed; use the embedded body.
        compute = os.path.join(cwd, "_uma_compute.py")
        with open(compute, "w") as cf:
            cf.write(_UMA_COMPUTE)

    out_json = os.path.join(cwd, "uma_out.json")
    cmd = [python, compute, sys.argv[1], out_json]

    env = os.environ.copy()
    env["FAIRCHEM_CACHE_DIR"] = cache_dir
    env.setdefault("HF_HUB_OFFLINE", "1")   # use cached weights; no gated download
    env.update({k: str(v) for k, v in env_extra.items()})

    print(f"[uma_run_row] START id={rid} tile={tile} host={host} "
          f"task={row.get('task_name', 'omol')} inference={inference} "
          f"relax={bool(row.get('relax'))}", flush=True)
    log = os.path.join(cwd, "uma.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode

    outputs = []
    result = {}
    if os.path.isfile(out_json):
        outputs.append("uma_out.json")
        try:
            result = json.load(open(out_json))
        except Exception:
            result = {}
    if row.get("relax") and os.path.isfile(os.path.join(cwd, "relaxed.xyz")):
        outputs.append("relaxed.xyz")

    # Fail if the subprocess errored or no energy was written.
    fail = (rc != 0) or ("energy_eV" not in result)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "outputs": outputs,
        "energy_eV": result.get("energy_eV"),
        "max_force_eV_per_A": result.get("max_force_eV_per_A"),
        "relaxed": result.get("relaxed", False),
        "relax_converged": result.get("relax_converged"),
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[uma_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
