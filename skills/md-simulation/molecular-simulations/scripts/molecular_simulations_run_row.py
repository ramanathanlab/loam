#!/usr/bin/env python3
"""Loam run_batch row runner for molecular-simulations (skill-MCP route).

`run_batch` executes `python molecular_simulations_run_row.py input.json` once per
JSONL row, in a per-row folder, with this row pinned to ONE XPU tile via
ZE_AFFINITY_MASK (set by the EL executor). This is "the function that takes
inputs": it reads one row, resolves the molecular-simulations install WORKER-SIDE
(site manifest or built-in defaults mirroring the SKILL.md Environment block),
subprocesses the AmberTools-build + OpenMM-simulate pipeline under the skill venv
on `platform='OpenCL'`, and writes outputs into the per-row cwd.

The runner stays pure-stdlib and runs under the deployment venv. The science
(importing `molecular_simulations`, which imports OpenMM/AmberTools) happens in a
subprocess under the skill venv, driven by a small driver emitted into the cwd
(the launcher is "stack-coupled" per docs/SKILL_MCP_DESIGN.md §4.A). A site
`launcher` path, if present worker-side, overrides the emitted driver.

Input row (JSONL):
  id          : row identity
  pdb         : ABSOLUTE path to the input protein PDB (worker-visible /flare)   [required]
  ligand      : optional ligand path (TODO(verify): ligand builders need rdkit/
                openbabel, which the SKILL.md says are NOT installed in the venv)
  pipeline    : "explicit" (verified) | "implicit" (TODO verify) |
                "complex"/"ligand" (TODO verify)                 (default "explicit")
  padding     : AmberTools solvent padding in Angstrom           (default 10.0)
  heat_steps, equil_steps, prod_steps, n_equil_cycles,
  temperature, prod_dt_in_ps, eq_reporter_frequency,
  prod_reporter_frequency, ff  : Simulator kwargs (omit -> package defaults)

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B):
  python    : site["molecular_simulations"]["python"]   -> else built-in default
  amberhome : site[...]["amberhome"]                     -> else built-in default
  launcher  : site[...]["launcher"]                      -> else emit driver into cwd
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.

NOT fully tested on Aurora (per SKILL.md) -> marked `# TODO(verify)` below:
  implicit-solvent pipeline, and any ligand/complex pipeline (rdkit/openbabel absent).
Only the ExplicitSolvent + Simulator path was verified (2026-10-29, 1L2Y smoke).
"""
import glob
import json
import os
import socket
import subprocess
import sys
import time

SKILL = "molecular_simulations"  # site-manifest key (identifier-safe)

# Built-in fallbacks so the runner works before a site manifest is deployed.
# These mirror the molecular-simulations SKILL.md "Environment" block.
_DEFAULTS = {
    "python": "/flare/FRAME-IDP/msinclair/software/molecular-simulations/.venv/bin/python",
    "amberhome": "/lus/flare/projects/FRAME-IDP/msinclair/envs/ambertools",
    # Optional committed launcher; if absent worker-side the runner emits a driver.
    "launcher_rel": "skills/md-simulation/molecular-simulations/scripts/molsim_run.py",
}

# Driver run UNDER THE SKILL VENV (imports molecular_simulations -> OpenMM/AmberTools).
# Plain string (NOT an f-string): it reads its params from argv[1] JSON at runtime.
_DRIVER = r'''
import json, sys
from pathlib import Path

p = json.load(open(sys.argv[1]))
pdb = Path(p["pdb"]).resolve()
out = Path(p["out"]).resolve()
out.mkdir(parents=True, exist_ok=True)
pipeline = p.get("pipeline", "explicit")

# Simulator kwargs: pass only what the row supplied, so the package defaults
# (heat 100k, equil 1.25M, prod 250M, 3 cycles, ...) apply otherwise.
_sim_keys = ("heat_steps", "equil_steps", "prod_steps", "n_equil_cycles",
             "temperature", "prod_dt_in_ps", "eq_reporter_frequency",
             "prod_reporter_frequency", "ff")
sim_kwargs = {k: p[k] for k in _sim_keys if k in p and p[k] is not None}
sim_kwargs["platform"] = "OpenCL"   # Aurora Intel GPU platform (NOT CUDA)

if pipeline == "explicit":
    from molecular_simulations.build import ExplicitSolvent
    from molecular_simulations.simulate import Simulator
    ExplicitSolvent(path=out, pdb=pdb, padding=float(p.get("padding", 10.0))).build()
    Simulator(path=out, **sim_kwargs).run()   # resumes from prod.rst.chk if present
elif pipeline == "implicit":
    # TODO(verify): implicit path documented in SKILL.md but NOT exercised on Aurora.
    from molecular_simulations.build import ImplicitSolvent
    from molecular_simulations.simulate import ImplicitSimulator
    ImplicitSolvent(path=out, pdb=pdb, protein=True).build()
    ImplicitSimulator(path=out, **sim_kwargs).run()
elif pipeline in ("complex", "ligand"):
    # TODO(verify): ligand/complex builders need rdkit + openbabel, which SKILL.md
    # says are NOT installed in this venv -> expect ImportError. Guarded so the row
    # fails loudly instead of silently producing an apo system.
    from molecular_simulations.build import ComplexBuilder  # TODO(verify) class/signature
    from molecular_simulations.simulate import Simulator
    lig = p.get("ligand")
    if not lig:
        sys.exit("[molsim driver] pipeline=%s requires a 'ligand' path" % pipeline)
    ComplexBuilder(path=out, pdb=pdb, ligand=Path(lig).resolve(),
                   padding=float(p.get("padding", 10.0))).build()  # TODO(verify) kwargs
    Simulator(path=out, **sim_kwargs).run()
else:
    sys.exit("[molsim driver] unknown pipeline %r" % pipeline)
print("[molsim driver] done pipeline=%s" % pipeline)
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
                print(f"[molsim_run_row] WARN bad site manifest {path}: {exc}", flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    loam = os.environ.get("HPC_LOAM_HOME", "")
    python = site.get("python") or _DEFAULTS["python"]
    amberhome = site.get("amberhome") or _DEFAULTS["amberhome"]
    launcher = site.get("launcher")
    if not launcher and loam:  # only if the committed launcher is actually present
        cand = os.path.join(loam, _DEFAULTS["launcher_rel"])
        launcher = cand if os.path.isfile(cand) else None
    env_extra = site.get("env", {}) or {}
    return python, amberhome, launcher, env_extra


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    python, amberhome, launcher, env_extra = _resolve()

    if not os.path.exists(python):
        sys.exit(f"[molsim_run_row] skill python not found: {python!r}")
    pdb = row.get("pdb")
    if not pdb:
        sys.exit("[molsim_run_row] row has no 'pdb'")
    if not os.path.isfile(pdb):
        sys.exit(f"[molsim_run_row] input pdb not found: {pdb!r}")

    # Driver params: outputs land in the per-row cwd.
    params = dict(row)
    params["out"] = cwd
    pinp = os.path.join(cwd, "_driver_input.json")
    json.dump(params, open(pinp, "w"), indent=2)

    # Launcher: a committed site launcher if present, else emit the driver into cwd.
    if launcher and os.path.isfile(launcher):
        cmd = [python, launcher, pinp]
    else:
        drv = os.path.join(cwd, "_molsim_driver.py")
        open(drv, "w").write(_DRIVER)
        cmd = [python, drv, pinp]

    env = os.environ.copy()
    env["AMBERHOME"] = amberhome            # required by the AmberTools builders
    env.update({k: str(v) for k, v in env_extra.items()})
    # ZE_AFFINITY_MASK is set per-row by the EL executor; OpenCL then uses that one
    # visible tile as device 0. Do NOT override it here.

    pipeline = row.get("pipeline", "explicit")
    print(f"[molsim_run_row] START id={rid} tile={tile} host={host} "
          f"pipeline={pipeline} pdb={os.path.basename(pdb)}", flush=True)
    log = os.path.join(cwd, "molsim.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    logtext = open(log, errors="replace").read()

    prmtop = sorted(glob.glob(os.path.join(cwd, "system.prmtop")))
    prod_dcd = sorted(glob.glob(os.path.join(cwd, "prod*.dcd")))
    eq_dcd = sorted(glob.glob(os.path.join(cwd, "eq*.dcd")))
    outputs = [os.path.basename(f) for f in (prmtop + eq_dcd + prod_dcd)]

    # Fail loudly: nonzero rc, no built system, or no production trajectory.
    # (AmberTools tleap failures surface only as CalledProcessError in the log;
    #  grep for the FATAL marker the SKILL.md calls out.)
    tleap_fatal = "FATAL" in logtext
    fail = (rc != 0) or (not prmtop) or (not prod_dcd)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "pipeline": pipeline,
        "outputs": outputs,
        "has_prmtop": bool(prmtop), "has_prod_dcd": bool(prod_dcd),
        "tleap_fatal": tleap_fatal,
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[molsim_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
