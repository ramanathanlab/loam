# Jupiter smoke workload, ported from the moeen branch (ports/jupiter/smoke/case.py), ProteinMPNN case only.
# ruff: noqa: E402 -- the tool is imported only after the environment provenance is saved
"""One trained-model sequence design on a bundled monomer, through an mmCIF/PDB round trip."""

import argparse
import faulthandler
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import distributions
from pathlib import Path

p = argparse.ArgumentParser()
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


import gemmi

source = SW / "proteinmpnn/inputs/PDB_monomers/pdbs/5L33.pdb"
s = gemmi.read_structure(str(source))
s.make_mmcif_document().write_file("input.cif")
gemmi.read_structure("input.cif").write_pdb("input.pdb")
chain = s[0][0].name
run(
    [
        sys.executable,
        SW / "proteinmpnn/protein_mpnn_run.py",
        "--pdb_path",
        a.out / "input.pdb",
        "--pdb_path_chains",
        chain,
        "--out_folder",
        a.out / "design",
        "--num_seq_per_target",
        "1",
        "--batch_size",
        "1",
        "--sampling_temp",
        "0.1",
        "--seed",
        "42",
    ]
)
fasta = next((a.out / "design/seqs").glob("*.fa"))
text = fasta.read_text()
assert text.count(">") >= 2
sequences = ["".join(record.splitlines()[1:]) for record in text.split(">") if record.strip()]
assert len(sequences[-1]) == len(sequences[0])
assert set(sequences[-1]) <= set("ACDEFGHIKLMNPQRSTVWY")
result = {"generated_fasta": str(fasta), "records": text.count(">"), "conversion_gemmi": gemmi.__version__}
result.update(status="passed", case="proteinmpnn", elapsed_seconds=time.time() - start, python=sys.executable)
save("result.json", result)
print(json.dumps(result, default=str))
