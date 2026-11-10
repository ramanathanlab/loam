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
p.add_argument("--case", required=True)
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
if a.case == "fetch-literature":
    import requests
    import xml.etree.ElementTree as ET

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    q = requests.get(
        base + "esearch.fcgi",
        params={
            "db": "pubmed",
            "term": "EGFR extracellular domain",
            "retmax": 2,
            "retmode": "json",
            "tool": "loam-jupiter-smoke",
        },
        timeout=45,
    )
    q.raise_for_status()
    ids = q.json()["esearchresult"]["idlist"]
    assert ids
    q = requests.get(
        base + "efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "loam-jupiter-smoke"},
        timeout=45,
    )
    q.raise_for_status()
    (a.out / "pubmed.xml").write_text(q.text)
    records = []
    for art in ET.fromstring(q.content).iter("PubmedArticle"):
        abstract = " ".join(" ".join(x.itertext()) for x in art.iter("AbstractText"))
        title = " ".join(art.find(".//ArticleTitle").itertext())
        if abstract:
            records.append({"text": abstract, "title": title, "pmid": art.findtext(".//PMID"), "source": "PubMed"})
    ep = requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": "EGFR extracellular", "format": "json", "pageSize": 1},
        timeout=45,
    )
    ep.raise_for_status()
    assert ep.json()["resultList"]["result"]
    assert records
    (a.out / "corpus.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    result = {"records": len(records), "europe_pmc": True}
elif a.case in ("rag-build-index", "rag-query"):
    import faiss
    import numpy as np
    from datasets import Dataset, load_from_disk
    from sentence_transformers import SentenceTransformer

    model = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = SentenceTransformer(model, device="cpu")
    corpus = [
        {"text": "This synthetic test record identifies the target as EGFR.", "id": "target"},
        {"text": "This synthetic test record identifies the folding tool as Chai.", "id": "folding"},
        {"text": "This synthetic test record identifies the scheduler as Slurm.", "id": "scheduler"},
    ]
    vec = encoder.encode([x["text"] for x in corpus], normalize_embeddings=True, convert_to_numpy=True)
    idx = faiss.IndexFlatIP(vec.shape[1])
    idx.add(vec)
    faiss.write_index(idx, "faiss.index")
    Dataset.from_list(corpus).save_to_disk("dataset")
    ds = load_from_disk("dataset")
    scores, ids = faiss.read_index("faiss.index").search(
        encoder.encode(["What is the target?"], normalize_embeddings=True, convert_to_numpy=True), 2
    )
    assert int(ids[0][0]) == 0
    assert np.isfinite(scores).all()
    save(
        "retrieval.json",
        {
            "query": "What is the target?",
            "passages": [ds[int(i)] for i in ids[0]],
            "scores": scores.tolist(),
            "model": model,
            "fixture": "synthetic test data; no scientific evidence",
        },
    )
    result = {
        "vectors": idx.ntotal,
        "dimension": vec.shape[1],
        "top_hit": ds[int(ids[0][0])]["id"],
        "retrieval_passed": True,
    }
elif a.case == "openmm":
    from openmm import Platform, LangevinMiddleIntegrator, unit
    from openmm.app import PDBFile, ForceField, Modeller, Simulation, NoCutoff, HBonds

    fixture()
    pdb = PDBFile("input.pdb")
    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    m = Modeller(pdb.topology, pdb.positions)
    m.addHydrogens(ff)
    system = ff.createSystem(m.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = Simulation(m.topology, system, integrator, Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
    sim.context.setPositions(m.positions)
    sim.minimizeEnergy(maxIterations=20)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 42)
    sim.step(10)
    state = sim.context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert math.isfinite(energy)
    import numpy as np

    assert np.isfinite(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).all()
    with open("final.pdb", "w") as f:
        PDBFile.writeFile(m.topology, state.getPositions(), f)
    result = {"energy_kj_mol": energy, "platform": sim.context.getPlatform().getName(), "steps": 10}
elif a.case == "molecular-simulations":
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
elif a.case == "proteinmpnn":
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
elif a.case == "rfd4-proteina":
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
    import gemmi

    files = list((a.out / "generation/inference_samples").rglob("*.cif"))
    assert files
    s = gemmi.read_structure(str(files[0]))
    assert sum(len(c) for c in s[0]) > 0
    assert all(math.isfinite(x) for c in s[0] for r in c for at in r for x in (at.pos.x, at.pos.y, at.pos.z))
    result = {"generated_cif": str(files[0]), "chains": len(s[0])}
elif a.case == "chai":
    fixture()
    run(
        [
            SW / "chai-lab/.venv/bin/chai-lab",
            "fold",
            a.out / "input.fasta",
            a.out / "fold",
            "--num-trunk-recycles",
            "1",
            "--num-diffn-timesteps",
            "50",
            "--num-diffn-samples",
            "1",
            "--seed",
            "42",
            "--no-use-msa-server",
        ]
    )
    import numpy as np

    files = list((a.out / "fold").glob("pred*.cif"))
    scores = list((a.out / "fold").glob("scores*.npz"))
    assert files and scores
    d = np.load(scores[0])
    assert math.isfinite(float(d["ptm"]))
    result = {"cif": str(files[0]), "ptm": float(d["ptm"]), **validate_fold(files[0])}
elif a.case == "boltz":
    seq = fixture()
    (a.out / "input.yaml").write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: " + seq + "\n      msa: empty\n"
    )
    run(
        [
            SW / "boltz/.venv/bin/boltz",
            "predict",
            a.out / "input.yaml",
            "--out_dir",
            a.out / "fold",
            "--cache",
            os.environ["BOLTZ_CACHE"],
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--recycling_steps",
            "1",
            "--sampling_steps",
            "50",
            "--diffusion_samples",
            "1",
            "--num_workers",
            "0",
            "--no_kernels",
        ]
    )
    files = list((a.out / "fold").rglob("*model*.cif"))
    conf = list((a.out / "fold").rglob("confidence*.json"))
    assert files and conf
    result = {"cif": str(files[0]), "confidence": json.loads(conf[0].read_text()), **validate_fold(files[0])}
elif a.case == "protenix":
    seq = fixture()
    save("input.json", [{"name": "trpcage", "sequences": [{"proteinChain": {"sequence": seq, "count": 1}}]}])
    run(
        [
            SW / "protenix/.venv/bin/protenix",
            "pred",
            "-i",
            a.out / "input.json",
            "-o",
            a.out / "fold",
            "-n",
            "protenix_mini_esm_v0.5.0",
            "-s",
            "42",
            "-c",
            "1",
            "-p",
            "20",
            "-e",
            "1",
            "--use_msa",
            "false",
            "--trimul_kernel",
            "torch",
            "--triatt_kernel",
            "torch",
        ]
    )
    files = list((a.out / "fold").rglob("*.cif"))
    assert files
    result = {"cif": str(files[0]), **validate_fold(files[0])}
elif a.case == "uma":
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
elif a.case == "esmfold2":
    import torch
    from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder, ProteinInput, StructurePredictionInput

    seq = fixture()
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
    with torch.inference_mode():
        r = ESMFold2InputBuilder().fold(
            model,
            StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)]),
            num_loops=1,
            num_sampling_steps=20,
            num_diffusion_samples=1,
            seed=42,
        )
    complex = r.complex if hasattr(r, "complex") else r
    (a.out / "prediction.cif").write_text(complex.to_mmcif())
    assert (a.out / "prediction.cif").stat().st_size > 100
    result = {"structure": "prediction.cif", **validate_fold(a.out / "prediction.cif")}
elif a.case == "genslm-esm":
    import torch
    from transformers import EsmTokenizer
    from genslm_esm.modeling import GenslmEsmcModel

    model_id = "genslm-test/genslm-esmc-300M-contrastive-2026-10-step-200000"
    model = GenslmEsmcModel.from_pretrained(model_id).cuda().eval()
    tokenizer = EsmTokenizer.from_pretrained(model_id)
    from genslm_esm.data import FastaDataset, GenslmEsmcDataCollator

    ds = FastaDataset(sequences=["ATGAAGGTACTACCACAAGAAACTGTAAGAATTGGA"], return_codon=True, return_aminoacid=True)
    collate = GenslmEsmcDataCollator(return_codon=True, return_aminoacid=True, tokenizer=tokenizer)
    batch = collate([ds[0]]).to("cuda")
    with torch.inference_mode():
        output = model(**batch)
    tensor = output.last_hidden_state
    assert torch.isfinite(tensor).all()
    torch.save(tensor.cpu(), "embeddings.pt")
    result = {"shape": list(tensor.shape), "model_id": model_id, "SAE_verified": False}
elif a.case == "pyrosetta":
    import pyrosetta

    fixture()
    pyrosetta.init(extra_options="-mute all")
    pose = pyrosetta.pose_from_pdb("input.pdb")
    score = pyrosetta.create_score_function("ref2015")(pose)
    assert math.isfinite(score)
    result = {"score_REU": score, "residues": pose.total_residue()}
elif a.case == "reactra":
    cli = SW / "reactra/.venv/bin/reactra"
    if not cli.is_file():
        raise RuntimeError(
            "BLOCKED: exact Reactra source, environment, mech-format and native providers are not installed"
        )
    raise RuntimeError(
        "BLOCKED: a reviewed native Reactra reaction/deployment fixture is required; CLI/import alone cannot pass this test"
    )
else:
    raise ValueError(a.case)
result.update(status="passed", case=a.case, elapsed_seconds=time.time() - start, python=sys.executable)
save("result.json", result)
print(json.dumps(result, default=str))
