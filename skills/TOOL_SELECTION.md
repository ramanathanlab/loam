# Tool Selection Guide (Aurora)

Pick a tool here, then open that tool's `SKILL.md` for the environment, example, arguments and
outputs. All GPU tools run on one Intel Max 1550 tile: set `ZE_AFFINITY_MASK=<int>` (a plain
integer) before starting Python. Never write into `/flare/FRAME-IDP/msinclair/software/*`.

## Decide by task

| I want to… | Use | Alternative |
|---|---|---|
| Fold a protein chain, no MSA | `esmfold2` | `boltz` / `protenix` start much faster for a one-off |
| Co-fold a complex (protein-protein, protein-DNA/RNA, protein-ligand) | `esmfold2` (Python API, gives ipTM) | `boltz`, `protenix`, `chai` |
| Co-fold protein + small molecule (SMILES/CCD) | `boltz` | `esmfold2` (`LigandInput`); `chai` (grep logs for `Failed to tokenize`) |
| Estimate protein-ligand binding affinity | `boltz` (`properties: affinity`) | none |
| Fold with contact/pocket restraints or glycans | `chai` | |
| Generate new backbones / binders / motif or enzyme scaffolds | `rfdiffusion3` (JSON spec, Hydra CLI) | `rfd4-proteina` (annotated CIF, resident worker per tile; also DNA binders) |
| Fan backbone generation across many tiles under Loam | `rfd4-proteina` (`run_batch`, one row per tile) | |
| Design sequences for a given backbone | `proteinmpnn` | |
| Score a sequence against a backbone (log-probs) | `proteinmpnn` (`--score_only`) | |
| Score / relax a PDB, Rosetta energies, interface dG/SC/dSASA | `pyrosetta` (CPU only) | |
| Rank many binder designs by interface energy | `pyrosetta` (batch scorer + PBS) | |
| Build an AMBER system and run a full MD pipeline | `molecular-simulations` | |
| Raw/custom OpenMM MD (own forces, integrators, inputs) | `openmm` | platform name is `OpenCL` |
| Near-DFT energies, forces, geometry opt, MD | `uma` | use `batch` mode or Nitro, not `default` |
| Enzyme reaction mechanism / reaction-flooding workflow | `reactra` | |
| Codon or protein LM embeddings, SAE features | `genslm-esm` | use `venvs/aurora` |

## Typical pipelines

- **De novo binder design:** `rfdiffusion3` or `rfd4-proteina` → `proteinmpnn` → `esmfold2` (refold
  many in one process, check pLDDT/ipTM; or `boltz`/`protenix`) → `pyrosetta` (interface metrics,
  ranking) → optional `molecular-simulations` (stability MD).
- **Redesign an existing protein:** `proteinmpnn` (fixed positions) → `esmfold2`/`boltz` (check fold).
- **Ligand complex:** `boltz` (pose + affinity) or `esmfold2` → `molecular-simulations`
  (`ComplexBuilder` MD) → `uma` (local energetics) or `reactra` (reaction).

## Tool cards

| Tool | Input → Output | Device |
|---|---|---|
| `boltz` | YAML/FASTA → PDB/mmCIF + confidence (+ affinity) | 1 XPU |
| `chai` | entity FASTA → mmCIF + scores | 1 XPU |
| `esmfold2` | sequence (CLI) or protein/DNA/ligand entities (Python API) → mmCIF + pLDDT/pTM/ipTM | 1 XPU |
| `protenix` | JSON → mmCIF + pLDDT/pTM/ipTM/PAE JSON | 1 XPU |
| `rfdiffusion3` | JSON/YAML spec → CIF(.gz) designs + metrics JSON | 1 XPU |
| `rfd4-proteina` | annotated CIF + plan JSON → CIF designs + receipts | 1 XPU per worker |
| `proteinmpnn` | PDB (→ JSONL) → FASTA sequences + scores | 1 XPU |
| `pyrosetta` | PDB → scores JSON/CSV, relaxed PDB | CPU |
| `molecular-simulations` | PDB (+ ligand) → prmtop/inpcrd + DCD trajectories | 1 XPU |
| `openmm` | prmtop/gro/psf/pdb → trajectories/state | 1 XPU |
| `uma` | ASE Atoms → energy/forces/relaxed geometry | 1 XPU |
| `reactra` | YAML + .mech → staged reaction workflow | CPU (plan), XPU (run) |
| `genslm-esm` | codon/AA sequences → embeddings, SAE features | 1 XPU |

## Don't use for

- The `esmfold2` CLI (`fold_sequence.py`) for complexes — it folds one chain. Use the Python API.
- `esmfold2` for a single one-off fold when wall time matters: the cold start dominates.
- `chai` without grepping logs — ligands can drop silently.
- `reactra` for plain QM single points → use `uma`.
- `openmm` for ligand parameterization (no `openmmforcefields`) → use `molecular-simulations`
  builders (AmberTools antechamber/parmchk2).
- `genslm-esm`'s `venvs/ci` (CUDA-only).
- `molecular-simulations` ligand builders — rdkit/openbabel are not installed.
