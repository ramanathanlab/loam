# Jupiter smoke workload, ported from the moeen branch (ports/jupiter/smoke/case.py).
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
from importlib.metadata import distributions
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--case", default='rfd4-proteina')
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


repo = SW / "RFD4-Proteina"
run(
    [
        repo / ".venv/bin/rfd4-generate",
        "inputs=" + str(repo / "data/benchmarks/monomer/monomer-short.cif.zst"),
        "out_dir=" + str(a.out / "generation"),
        "num_replicates=1",
        "batch_size=1",
        "compile_lengths=null",
        "num_workers=0",
        "ncpus_=1",
        "seed=42",
        "metrics_tags=[]",
        "hydra.run.dir=" + str(a.out / "hydra"),
    ]
)
import gemmi  # noqa: E402 -- only needed once the generation has run

files = list((a.out / "generation/inference_samples").rglob("*.cif"))
assert files
s = gemmi.read_structure(str(files[0]))
assert sum(len(c) for c in s[0]) > 0
assert all(math.isfinite(x) for c in s[0] for r in c for at in r for x in (at.pos.x, at.pos.y, at.pos.z))
result = {"generated_cif": str(files[0]), "chains": len(s[0])}
result.update(status="passed", case=a.case, elapsed_seconds=time.time() - start, python=sys.executable)
save("result.json", result)
print(json.dumps(result, default=str))
