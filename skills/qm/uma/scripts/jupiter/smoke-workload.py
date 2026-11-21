# Jupiter smoke workload, ported from the moeen branch (ports/jupiter/smoke/case.py), UMA case only.
# ruff: noqa: E402 -- the tool is imported only after the environment provenance is saved
"""A neutral singlet water single point with UMA/OMol on one CUDA GPU, asserting finite energy and forces."""

import argparse
import faulthandler
import hashlib
import json
import os
import sys
import time
from importlib.metadata import distributions
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
os.chdir(a.out)
start = time.time()

faulthandler.dump_traceback_later(120, repeat=True)


def save(name, data):
    (a.out / name).write_text(json.dumps(data, indent=2, default=str) + "\n")


save("environment-packages.json", dict(sorted((d.metadata["Name"], d.version) for d in distributions())))
save(
    "test-provenance.json",
    {
        "case_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.executable,
        "python_version": sys.version,
        "started_at": start,
    },
)

import numpy as np
from ase import Atoms
from fairchem.core import FAIRChemCalculator, pretrained_mlip

pred = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda", inference_settings="batch")
atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
atoms.info.update(charge=0, spin=1)
atoms.calc = FAIRChemCalculator(pred, task_name="omol")
e = atoms.get_potential_energy()
forces = atoms.get_forces()
assert np.isfinite(e) and np.isfinite(forces).all()
save("energy-forces.json", {"energy_eV": e, "forces_eV_A": forces.tolist()})
result = {"energy_eV": e, "atoms": len(atoms)}
result.update(status="passed", case="uma", elapsed_seconds=time.time() - start, python=sys.executable)
save("result.json", result)
print(json.dumps(result, default=str))
