#!/usr/bin/env python3
"""Loam run_batch row runner for OpenMM on Aurora (skill-MCP route).

`run_batch` executes `python openmm_run_row.py input.json` once per JSONL row, in
a per-row folder, with this row pinned to ONE XPU tile via ZE_AFFINITY_MASK (set
by the EL executor). This is "the function that takes inputs": it reads one row,
resolves the openmm-sycl install WORKER-SIDE (site manifest or built-in defaults
mirroring the SKILL.md Environment block), subprocesses a minimal MD run under the
skill venv on `platform='OpenCL'` with `{'Precision':'mixed'}`, and writes outputs
into the per-row cwd. No ligand parameterization here.

The runner stays pure-stdlib and runs under the deployment venv. The science
(importing OpenMM) happens in a subprocess under the skill venv, driven by a small
driver emitted into the cwd that mirrors the verified `openmm_xpu_smoke.py`
settings (PME, HBonds, LangevinMiddle, mixed precision, OpenCL). A site `launcher`
path, if present worker-side, overrides the emitted driver.

Input row (JSONL):
  id          : row identity
  input_type  : "amber" (VERIFIED) | "gromacs" | "charmm" | "pdb"   (default "amber")
  prmtop, inpcrd                 : AMBER inputs (input_type="amber")
  gro, top                       : GROMACS inputs   (TODO verify)
  psf, crd, params               : CHARMM inputs    (TODO verify; params = list of files)
  pdb                            : PDB input        (TODO verify; amber14 FF)
  nsteps                 : production steps after warm-up            (default 5000)
  temperature            : kelvin                                    (default 300)
  timestep_ps            : integrator dt in ps                       (default 0.002)
  nonbonded_cutoff_nm    : nm                                        (default 1.0)
  report_interval        : DCD/StateData reporter stride             (default 1000)
  minimize               : minimizeEnergy(200) before MD             (default True)

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B):
  python   : site["openmm"]["python"]   -> else built-in default
  launcher : site["openmm"]["launcher"] -> else emit driver into cwd
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.

NOT fully tested on Aurora (per SKILL.md): only the AMBER prmtop/inpcrd path was
verified (2026-10-29). The GROMACS/CHARMM/PDB readers are documented standard
OpenMM app readers but were NOT live-tested -> marked `# TODO(verify)` below.
"""
import glob
import json
import os
import socket
import subprocess
import sys
import time

SKILL = "openmm"  # site-manifest key

_DEFAULTS = {
    "python": "/flare/FRAME-IDP/msinclair/software/openmm-sycl/.venv/bin/python",
    # Optional committed launcher; if absent worker-side the runner emits a driver.
    "launcher_rel": "skills/md-simulation/openmm/scripts/openmm_run.py",
}

# Driver run UNDER THE SKILL VENV (imports OpenMM). Plain string (NOT an f-string):
# it reads its params from argv[1] JSON at runtime. Mirrors openmm_xpu_smoke.py.
_DRIVER = r'''
import json, os, sys, time
from openmm.app import *
from openmm import *
from openmm.unit import *

p = json.load(open(sys.argv[1]))
outdir = p["out"]; os.makedirs(outdir, exist_ok=True)
itype = p.get("input_type", "amber")
nsteps = int(p.get("nsteps", 5000))
T = float(p.get("temperature", 300.0))
dt = float(p.get("timestep_ps", 0.002))
cutoff = float(p.get("nonbonded_cutoff_nm", 1.0))
report = int(p.get("report_interval", 1000))
minimize = bool(p.get("minimize", True))

if itype == "amber":
    inpcrd = AmberInpcrdFile(p["inpcrd"])
    top = AmberPrmtopFile(p["prmtop"], periodicBoxVectors=inpcrd.boxVectors)
    topology, positions = top.topology, inpcrd.positions
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=cutoff*nanometer,
                              constraints=HBonds)
elif itype == "gromacs":
    # TODO(verify): GROMACS reader documented in SKILL.md but NOT live-tested on Aurora.
    gro = GromacsGroFile(p["gro"])
    top = GromacsTopFile(p["top"], periodicBoxVectors=gro.getPeriodicBoxVectors())
    topology, positions = top.topology, gro.positions
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=cutoff*nanometer,
                              constraints=HBonds)
elif itype == "charmm":
    # TODO(verify): CHARMM reader documented but NOT live-tested on Aurora.
    psf = CharmmPsfFile(p["psf"])
    crd = CharmmCrdFile(p["crd"])
    params = CharmmParameterSet(*p["params"])
    topology, positions = psf.topology, crd.positions
    system = psf.createSystem(params, nonbondedMethod=PME,
                              nonbondedCutoff=cutoff*nanometer, constraints=HBonds)
elif itype == "pdb":
    # TODO(verify): PDB + amber14 FF documented but NOT live-tested on Aurora.
    pdb = PDBFile(p["pdb"])
    ff = ForceField("amber14-all.xml", "amber14/tip3pfx.xml")
    topology, positions = pdb.topology, pdb.positions
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME,
                             nonbondedCutoff=cutoff*nanometer, constraints=HBonds)
else:
    sys.exit("[openmm driver] unknown input_type %r" % itype)

integrator = LangevinMiddleIntegrator(T*kelvin, 1/picosecond, dt*picoseconds)
platform = Platform.getPlatformByName("OpenCL")   # Aurora Intel GPU platform
# ZE_AFFINITY_MASK already restricts visibility to one tile -> DeviceIndex stays 0.
sim = Simulation(topology, system, integrator, platform, {"Precision": "mixed"})
sim.context.setPositions(positions)
print("Platform:", sim.context.getPlatform().getName())
try:
    print("DeviceName =", platform.getPropertyValue(sim.context, "DeviceName"))
except Exception:
    pass
if minimize:
    sim.minimizeEnergy(maxIterations=200)
sim.reporters.append(DCDReporter(os.path.join(outdir, "traj.dcd"), report))
sim.reporters.append(StateDataReporter(os.path.join(outdir, "md.log"), report,
                     step=True, potentialEnergy=True, temperature=True, speed=True))
sim.step(100)   # warm-up: kernel JIT compile
t0 = time.time()
sim.step(nsteps)
dt_wall = time.time() - t0
sim.saveState(os.path.join(outdir, "final_state.xml"))
ns_day = nsteps * dt * 1e-3 / dt_wall * 86400 if dt_wall > 0 else 0.0
print("[openmm driver] %d steps in %.2f s -> %.1f ns/day (%d atoms)"
      % (nsteps, dt_wall, ns_day, system.getNumParticles()))
'''


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
                print(f"[openmm_run_row] WARN bad site manifest {path}: {exc}", flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    loam = os.environ.get("HPC_LOAM_HOME", "")
    python = site.get("python") or _DEFAULTS["python"]
    launcher = site.get("launcher")
    if not launcher and loam:
        cand = os.path.join(loam, _DEFAULTS["launcher_rel"])
        launcher = cand if os.path.isfile(cand) else None
    env_extra = site.get("env", {}) or {}
    return python, launcher, env_extra


# Which row keys must be present+on-disk for each input_type.
_NEEDED = {
    "amber": ["prmtop", "inpcrd"],
    "gromacs": ["gro", "top"],
    "charmm": ["psf", "crd", "params"],
    "pdb": ["pdb"],
}


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    python, launcher, env_extra = _resolve()

    if not os.path.exists(python):
        sys.exit(f"[openmm_run_row] skill python not found: {python!r}")

    itype = row.get("input_type", "amber")
    needed = _NEEDED.get(itype)
    if needed is None:
        sys.exit(f"[openmm_run_row] unknown input_type {itype!r}")
    for key in needed:
        val = row.get(key)
        if not val:
            sys.exit(f"[openmm_run_row] input_type={itype} needs row['{key}']")
        # `params` (CHARMM) is a list of files; validate each. Others are single paths.
        paths = val if isinstance(val, list) else [val]
        for pth in paths:
            if not os.path.isfile(pth):
                sys.exit(f"[openmm_run_row] input not found: {pth!r}")

    params = dict(row)
    params["out"] = cwd
    pinp = os.path.join(cwd, "_driver_input.json")
    json.dump(params, open(pinp, "w"), indent=2)

    if launcher and os.path.isfile(launcher):
        cmd = [python, launcher, pinp]
    else:
        drv = os.path.join(cwd, "_openmm_driver.py")
        open(drv, "w").write(_DRIVER)
        cmd = [python, drv, pinp]

    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items()})
    # ZE_AFFINITY_MASK is set per-row by EL; OpenCL uses that one visible tile as
    # device 0. Do NOT set DeviceIndex (SKILL.md gotcha).

    print(f"[openmm_run_row] START id={rid} tile={tile} host={host} "
          f"input_type={itype} nsteps={row.get('nsteps', 5000)}", flush=True)
    log = os.path.join(cwd, "openmm.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode

    dcd = sorted(glob.glob(os.path.join(cwd, "traj.dcd")))
    state = sorted(glob.glob(os.path.join(cwd, "final_state.xml")))
    outputs = [os.path.basename(f) for f in (dcd + state)]

    fail = (rc != 0) or (not dcd)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "input_type": itype,
        "outputs": outputs,
        "has_traj": bool(dcd),
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[openmm_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
