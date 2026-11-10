# Generated from ports/jupiter/smoke/case.py; SHA256 2880908c4c876f63b044212a86599284fcc3f13d8e5fcbdb5b2b62d7939d953a
"""Small real workloads for the isolated Jupiter skill environments (MCP only)."""

import argparse
import faulthandler
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from importlib.metadata import distributions
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--case", default='molecular-simulations')
p.add_argument("--out", type=Path, required=True)
p.add_argument("--software", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
os.chdir(a.out)
SW = a.software
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


def run(args, timeout=1500):
    with open("command.log", "a") as f:
        f.write(json.dumps([str(x) for x in args]) + "\n")
        f.flush()
        q = subprocess.run([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    assert q.returncode == 0, f"Command failed with {q.returncode}; see command.log"


def fixture():
    raw = urllib.request.urlopen("https://files.rcsb.org/download/1L2Y.pdb", timeout=60).read()
    (a.out / "1L2Y.pdb.raw").write_bytes(raw)
    text = raw.decode()
    mapping = dict(
        zip(
            "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
            "ARNDCQEGHILKMFPSTWYV",
        )
    )
    seq = "".join(
        mapping[x]
        for line in text.splitlines()
        if line.startswith("SEQRES") and line[11] == "A"
        for x in line[19:70].split()
    )
    assert len(seq) == 20
    lines = []
    active = False
    for line in text.splitlines():
        if line.startswith("MODEL"):
            active = True
            continue
        if line.startswith("ENDMDL") and active:
            break
        if active and line.startswith("ATOM") and line[76:78].strip() != "H":
            lines.append(line)
    (a.out / "input.pdb").write_text("\n".join(lines) + "\nTER\nEND\n")
    (a.out / "input.fasta").write_text(">protein|name=trpcage\n" + seq + "\n")
    save(
        "input-provenance.json",
        {
            "url": "https://files.rcsb.org/download/1L2Y.pdb",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sequence": seq,
            "model": 1,
        },
    )
    return seq


def validate_fold(path, expected_residues=20):
    try:
        import gemmi
    except ModuleNotFoundError:
        gemmi = None
    if gemmi is not None:
        structure = gemmi.read_structure(str(path))
        atoms = [atom for chain in structure[0] for residue in chain for atom in residue]
        count = sum(len(chain) for chain in structure[0])
        assert atoms and all(math.isfinite(x) for atom in atoms for x in (atom.pos.x, atom.pos.y, atom.pos.z))
        assert count == expected_residues, (count, expected_residues)
        return {"atoms": len(atoms), "residues": count, "coordinates_finite": True}

    import numpy as np
    from biotite.structure import get_residue_count
    from biotite.structure.io.pdbx import CIFFile, get_structure

    atoms = get_structure(CIFFile.read(str(path)), model=1)
    assert len(atoms) > 0 and np.isfinite(atoms.coord).all()
    count = get_residue_count(atoms)
    assert count == expected_residues, (count, expected_residues)
    return {"atoms": len(atoms), "residues": count, "coordinates_finite": True}


result = {}
from molecular_simulations.build import ExplicitSolvent
from molecular_simulations.simulate import Minimizer

fixture()
dest = a.out / "system"
ExplicitSolvent(path=str(dest), pdb=str(a.out / "input.pdb"), padding=8.0, glycans=False, debug=True).build()
assert (dest / "system.prmtop").is_file()
Minimizer(
    topology=dest / "system.prmtop", coordinates=dest / "system.inpcrd", out="min.pdb", platform="CUDA"
).minimize()
from openmm.app import PDBFile
from openmm import unit
import numpy as np

minimized = dest / "min.pdb"
assert minimized.is_file()
structure = PDBFile(str(minimized))
assert np.isfinite(structure.positions.value_in_unit(unit.nanometer)).all()
result = {
    "topology_bytes": (dest / "system.prmtop").stat().st_size,
    "minimized_structure": str(minimized),
    "atoms": structure.topology.getNumAtoms(),
    "coordinates_finite": True,
}
result.update(status="passed", case=a.case, elapsed_seconds=time.time() - start, python=sys.executable)
save("result.json", result)
print(json.dumps(result, default=str))
