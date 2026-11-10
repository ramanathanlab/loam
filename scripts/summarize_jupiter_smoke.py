#!/usr/bin/env python3
"""Summarize manual audit attempts without treating missing software as a pass."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--runs", type=Path, nargs="+", required=True)
a = p.parse_args()
repo = Path(__file__).resolve().parents[1]
workloads = {
    "fetch-literature": "Real PubMed abstracts and Europe PMC retrieval",
    "rag-build-index": "Pretrained embeddings, FAISS/Arrow persistence and reload",
    "rag-query": "FAISS retrieval followed by login-node Argo answer",
    "openmm": "Real protein CUDA minimization and 10 MD steps",
    "molecular-simulations": "AmberTools explicit-solvent build and CUDA minimization",
    "proteinmpnn": "mmCIF/PDB conversion and one trained-model sequence design",
    "rfd4-proteina": "Trained-model monomer backbone generation with zero loader workers",
    "chai": "Single-sequence trained-model folding and confidence output",
    "boltz": "Single-sequence CUDA folding with native PyTorch kernels",
    "protenix": "Mini ESM CUDA folding, real checkpoint/embedding path",
    "uma": "Neutral singlet water energy and forces using UMA/OMol",
    "esmfold2": "Native ESMFold2 trained-model folding",
    "genslm-esm": "Pretrained codon/amino-acid embedding; excludes unavailable SAE fork",
    "pyrosetta": "Real protein ref2015 scoring using licensed runtime",
    "reactra": "Excluded at user's request",
}
attempts = {name: [] for name in workloads}
for directory in a.runs:
    report = json.loads((directory / "summary.json").read_text())
    for name, case in report["cases"].items():
        if name in attempts:
            attempts[name].append({**case, "audit_directory": str(directory.resolve())})

cases = {}
skills = {main.parent.name.removesuffix("-jupiter"): main for main in (repo / "skills").rglob("*-jupiter/SKILL.md")}
for name, history in attempts.items():
    history.sort(key=lambda c: c.get("started_at", 0))
    current = history[-1] if history else {"status": "not_tested"}
    status = current["status"]
    note = ""
    logs = ""
    if current.get("audit_directory"):
        process = Path(current["audit_directory"]) / name / "process.log"
        if process.exists():
            logs = process.read_text()
    if name == "reactra":
        status, note = "excluded", "User excluded Reactra; no installation required."
    elif status == "failed":
        if "GatedRepoError" in logs:
            status, note = (
                "blocked",
                "Checkpoint access denied by Hugging Face; authorize and authenticate before retry.",
            )
        elif "RepositoryNotFoundError" in logs:
            status, note = "blocked", "Documented checkpoint is inaccessible to the current Hugging Face identity."
        elif name == "pyrosetta" and "No module named 'pyrosetta'" in logs:
            status, note = "blocked", "Only installer present; licensed ARM64 PyRosetta runtime is missing."
        else:
            note = "Workflow failed; inspect the latest attempt logs."
    cases[name] = {
        "status": status,
        "workload": workloads[name],
        "note": note,
        "latest_attempt": current,
        "attempts": history,
    }

report = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "scope": "Manual small workflows through MCP, isolated venvs; not campaign preflight or production qualification",
    "execution": "Initially sequential; repair reruns overlapped asset staging on a second GPU, leaving two GPUs free",
    "required_count": 14,
    "all_required_passed": all(c["status"] == "passed" for n, c in cases.items() if n != "reactra"),
    "cases": cases,
}
report["reference_reads"] = []
for directory in a.runs:
    for filename in ("skill-view.json", "skill-view-recipes.json"):
        evidence = directory / filename
        if evidence.exists():
            reads = json.loads(evidence.read_text())
            report["reference_reads"].append(
                {"evidence": str(evidence.resolve()), "successful_reads": sum(r.get("success", False) for r in reads)}
            )
    evidence = directory / "additional-output-checks.json"
    if evidence.exists():
        checks = json.loads(evidence.read_text())
        if str(checks.get("returncode")) == "0":
            for name, values in json.loads(checks["stdout"]).items():
                cases[name]["additional_output_validation"] = {"evidence": str(evidence.resolve()), "values": values}
target = repo / "ports/jupiter/software-smoke-status.json"
target.write_text(json.dumps(report, indent=2) + "\n")
lines = [
    "# Jupiter software workflow validation",
    "",
    report["scope"],
    "",
    report["execution"],
    "",
    "| Software | Latest status | Tested workload / blocker |",
    "|---|---|---|",
]
for name, case in cases.items():
    lines.append(f"| {name} | {case['status']} | {case['workload']}. {case['note']} |")
    skill = skills[name]
    reference = skill.parent / "references/smoke-test.md"
    marker = "\n## Latest recorded manual validation\n"
    text = reference.read_text().split(marker)[0]
    text += marker + f"\nStatus: **{case['status']}**. Workload: {case['workload']}. {case['note']}\n"
    current = case["latest_attempt"]
    if current.get("audit_directory"):
        text += f"\nEvidence directory: `{current['audit_directory']}/{name}`. Preserve failed attempts; the root `summary.json` records this run.\n"
    text += "\nThis status describes only the tested path and cache/auth state at audit time. Revalidate after environment or checkpoint changes.\n"
    reference.write_text(text)
lines += [
    "",
    "Full results and all attempts are recorded in [software-smoke-status.json](software-smoke-status.json).",
    "The older status.json is a package/import inventory, not this workflow audit.",
    "",
    "Repairs: matching OpenMM CUDA extra in both MD venvs; Boltz --no_kernels and zero loader workers.",
    "The first Boltz fallback prediction succeeded but its added checker assumed Biotite; the final clean rerun uses installed Gemmi.",
    "Checkpoint authorization and licensed-runtime blockers require user-provided access; do not substitute random models or fabricated outputs.",
    "",
    "The five scientific software skills selected by campaign-test11 (RFD4, ProteinMPNN, Chai, molecular-simulations and OpenMM) passed these small workflow tests. PyRosetta is optional and explicitly excluded by that campaign. Larger target/binder workflows require their actual campaign validation.",
]
(target.parent / "software-smoke-report.md").write_text("\n".join(lines) + "\n")
print({name: case["status"] for name, case in cases.items()})
