# Jupiter software workflow validation

Manual small workflows through MCP, isolated venvs; not campaign preflight or production qualification

Initially sequential; repair reruns overlapped asset staging on a second GPU, leaving two GPUs free

| Software | Latest status | Tested workload / blocker |
|---|---|---|
| fetch-literature | passed | Real PubMed abstracts and Europe PMC retrieval.  |
| rag-build-index | passed | Pretrained embeddings, FAISS/Arrow persistence and reload.  |
| rag-query | passed | FAISS retrieval followed by login-node Argo answer.  |
| openmm | passed | Real protein CUDA minimization and 10 MD steps.  |
| molecular-simulations | passed | AmberTools explicit-solvent build and CUDA minimization.  |
| proteinmpnn | passed | mmCIF/PDB conversion and one trained-model sequence design.  |
| rfd4-proteina | passed | Trained-model monomer backbone generation with zero loader workers.  |
| chai | passed | Single-sequence trained-model folding and confidence output.  |
| boltz | passed | Single-sequence CUDA folding with native PyTorch kernels.  |
| protenix | passed | Mini ESM CUDA folding, real checkpoint/embedding path.  |
| uma | blocked | Neutral singlet water energy and forces using UMA/OMol. Checkpoint access denied by Hugging Face; authorize and authenticate before retry. |
| esmfold2 | passed | Native ESMFold2 trained-model folding.  |
| genslm-esm | blocked | Pretrained codon/amino-acid embedding; excludes unavailable SAE fork. Documented checkpoint is inaccessible to the current Hugging Face identity. |
| pyrosetta | blocked | Real protein ref2015 scoring using licensed runtime. Only installer present; licensed ARM64 PyRosetta runtime is missing. |
| reactra | excluded | Excluded at user's request. User excluded Reactra; no installation required. |

Full results and all attempts are recorded in [software-smoke-status.json](software-smoke-status.json).
The older status.json is a package/import inventory, not this workflow audit.

Repairs: matching OpenMM CUDA extra in both MD venvs; Boltz --no_kernels and zero loader workers.
The first Boltz fallback prediction succeeded but its added checker assumed Biotite; the final clean rerun uses installed Gemmi.
Checkpoint authorization and licensed-runtime blockers require user-provided access; do not substitute random models or fabricated outputs.

The five scientific software skills selected by campaign-test11 (RFD4, ProteinMPNN, Chai, molecular-simulations and OpenMM) passed these small workflow tests. PyRosetta is optional and explicitly excluded by that campaign. Larger target/binder workflows require their actual campaign validation.

Reference verification: All 15 packaging tests passed, and all advertised paths and packaged Python recipes were checked. The earlier actual Hermes skill_view audit passed 107 reads. Fresh checks of the 15 newly added recipe files timed out at 90 and 180 seconds amid login-side I/O waits; those final interface reads remain unverified. No Hermes patches or security-boundary workarounds were applied.
