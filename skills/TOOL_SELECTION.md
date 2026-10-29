# Tool Selection Guide (Aurora)

Read this to pick a tool, then open that tool's `SKILL.md` for environment setup and commands.
All GPU tools run on one Intel Max 1550 tile: set `ZE_AFFINITY_MASK=<int>` (plain integer) before starting Python.
Never write into `/flare/FRAME-IDP/msinclair/software/*`; write outputs elsewhere.

## Decide by task

| I want to... | Use | Fallback / note |
|---|---|---|
| Fold a protein chain, no MSA | `esmfold2` | `boltz`/`protenix` (mini_esm) cold-start faster (~1 min vs ~6 min) for a one-off |
| Co-fold a complex (protein-protein, protein-DNA/RNA, protein-ligand) | `esmfold2` (Python API, gives ipTM) | `boltz`, `protenix` (AF3-style, PAE), `chai` |
| Co-fold protein + small molecule (SMILES/CCD) | `esmfold2` (Python API `LigandInput`) | `boltz`; `chai` (needs launcher; check logs for `Failed to tokenize`) |
| Estimate protein-ligand binding affinity | `boltz` (`properties: affinity`) | none |
| Fold with contact/pocket restraints or glycans | `chai` | |
| Generate new backbones / binders / motif or enzyme scaffolds | `rfdiffusion3` (JSON spec) | `rfd4-proteina` (Hydra, also DNA binders, faster ~1.5 min) |
| Design sequences for a given backbone | `proteinmpnn` | |
| Score a sequence against a backbone (log-probs) | `proteinmpnn` (`--score_only`) | |
| Score / relax a PDB, Rosetta energies, interface dG/SC/dSASA | `pyrosetta` (CPU only) | |
| Rank many binder designs by interface energy | `pyrosetta` (batch scorer + PBS) | |
| Build an AMBER system and run a full MD pipeline | `molecular-simulations` | |
| Build protein-ligand or ligand-in-solvent systems | `molecular-simulations` (`ComplexBuilder`, `LigandBuilder`, `LigandSolutionBuilder`) | imports OK (rdkit 2026.09.1, openbabel 3.2.1); no system built on Aurora yet |
| Constant-pH MD, EVB, coarse-grained (CALVADOS), MM/PBSA, free energy | `molecular-simulations` | in the package; not tested on Aurora |
| Analyze MD (SASA, interaction energy, ipSAE, clustering, PPI covariance) | `molecular-simulations` (`analysis/`) | not tested on Aurora |
| Raw/custom OpenMM MD (own forces, integrators, inputs) | `openmm` | platform name is `OpenCL` |
| Near-DFT energies, forces, geometry opt, MD (molecules, materials, catalysts, MOFs) | `uma` | use `batch` mode or Nitro, not `default` |
| Enzyme reaction mechanism / reaction-flooding workflow | `reactra` | only validate/plan verified; full runs ~1 h |
| Codon or protein LM embeddings, SAE features | `genslm-esm` | use `venvs/aurora` |

## Typical pipelines
- **De novo binder design:** `rfdiffusion3` or `rfd4-proteina` → `proteinmpnn` → `esmfold2` (refold many in one process, check pLDDT/ipTM; or `boltz`/`protenix`) → `pyrosetta` (interface metrics, ranking) → optional `molecular-simulations` (stability MD).
- **Redesign an existing protein:** `proteinmpnn` (fixed positions) → `esmfold2`/`boltz` (check fold).
- **Ligand complex:** `esmfold2` or `boltz` (pose; `boltz` for affinity) → `molecular-simulations` (`ComplexBuilder` MD) → `uma` (local energetics) or `reactra` (reaction).

## Tool cards

| Tool | Input → Output | Device | Measured time (small test) | Must-know |
|---|---|---|---|---|
| `boltz` | YAML/FASTA → PDB/mmCIF + confidence (+ affinity) | 1 XPU | 34-72 s | `--accelerator xpu`; MSA server needs ALCF proxy |
| `chai` | entity FASTA → mmCIF + scores | 1 XPU | ~50 s | Use `scripts/chai_xpu.py`, not `chai` CLI |
| `esmfold2` | sequence (CLI) or protein/DNA/ligand entities (Python API) → mmCIF (pLDDT/pTM/ipTM) | 1 XPU | ~6.5 min cold (90 s load + first-call JIT); later folds much faster | Forked transformers first on `PYTHONPATH`; `set_kernel_backend(None)`; CLI folds only the first FASTA record; batch many inputs in one process |
| `protenix` | JSON → mmCIF + pLDDT/pTM/ipTM/PAE JSON | 1 XPU | ~1 min | Prepend old oneAPI lib to `LD_LIBRARY_PATH`; torch kernels |
| `rfdiffusion3` | JSON/YAML spec → CIF(.gz) designs | 1 XPU | ~6 min | `module swap` to frameworks/2025.3.1 |
| `rfd4-proteina` | Hydra overrides / annotated CIF → CIF designs | 1 XPU | ~1.5 min (+~2 min imports) | `key=value` args only; sitecustomize shim; run from repo root |
| `proteinmpnn` | PDB (→ JSONL) → FASTA sequences + scores | 1 XPU | ~3-10 s per call | IPEX shim on `PYTHONPATH`; `--device` ignored |
| `pyrosetta` | PDB → scores JSON/CSV, relaxed PDB | CPU | <1 s score, 4-30 s relax | `init(extra_options=...)`; init once per process |
| `molecular-simulations` | PDB (+ ligand) → prmtop/inpcrd + DCD trajectories + analyses | 1 XPU | ~45 s smoke run | `platform='OpenCL'`; set `AMBERHOME` |
| `openmm` | prmtop/gro/psf/pdb → trajectories/state | 1 XPU | ~10 s smoke run | No CUDA; `DeviceIndex` stays 0 |
| `uma` | ASE Atoms → energy/forces/relaxed geometry | 1 XPU | seconds | `FAIRCHEM_CACHE_DIR` set; pick `task_name` (omol/omat/oc20/odac/omc) |
| `reactra` | YAML + .mech → staged reaction workflow | CPU (plan), XPU (run) | <25 s plan; ~54 min full | Use `scripts/reactra_cli.sh` (mech-format overlay) |
| `genslm-esm` | codon/AA sequences → embeddings, SAE features | 1 XPU | seconds | Codons space-separated, uppercase |

## Don't use for
- The `esmfold2` CLI (`fold_sequence.py`) for complexes: it takes one chain. Use the Python API.
- `esmfold2` for a single one-off fold when wall time matters: the cold start is ~6 min.
- `chai` without grepping logs: ligands can drop silently.
- `reactra` for plain QM single points → `uma`.
- `openmm` for ligand parameterization (no `openmmforcefields`). Use `molecular-simulations` builders (AmberTools antechamber/parmchk2) instead.
- `genslm-esm` `venvs/ci` (CUDA-only).
