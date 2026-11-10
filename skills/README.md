# Jupiter port

For Julich Jupiter, select the `*-jupiter` variants and `hpc-deployment`.
Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh` for software and model-cache paths.
Read [JUPITER.md](../JUPITER.md) and the generated
[installation status](../ports/jupiter/status.json) before using a tool.
Jupiter has four GH200 GPUs per node; preserve CUDA_VISIBLE_DEVICES.
Use `rfd4-proteina-jupiter` for backbone generation. RFdiffusion3 is excluded from the skill library on this branch; historical recipes are archived under `ports/jupiter/disabled-skills/`.
Original Aurora skills below remain API references, with their original provenance.

# Aurora Tool Skills

Claude Code skills for the Aurora tool stack, written and tested on an Aurora
debug node (Intel Data Center GPU Max 1550, FLAT mode, 12 tiles) on 2026-10-29.
Software lives in `/flare/FRAME-IDP/msinclair/software/<dir>`.

Pick a GPU tile with `ZE_AFFINITY_MASK=<int>` (a plain integer; `4.0` hides the device).

| Category | Skill | Software dir | Status | Required workaround / key note |
|---|---|---|---|---|
| MD Simulation | [molecular-simulations](md-simulation/molecular-simulations/SKILL.md) | molecular-simulations | PASS | `platform='OpenCL'` (default is CUDA); `AMBERHOME=/lus/flare/projects/FRAME-IDP/msinclair/envs/ambertools`; no rdkit/openbabel → no ligand builders |
| MD Simulation | [openmm](md-simulation/openmm/SKILL.md) | openmm-sycl | PASS | GPU platform is named `OpenCL` |
| Structure Prediction | [boltz](structure-prediction/boltz/SKILL.md) | boltz | PASS | `--accelerator xpu`; single device only |
| Structure Prediction | [chai](structure-prediction/chai/SKILL.md) | chai-lab | PASS via launcher | Stock `chai` CLI broken (IPEX import, antipickle, pandas 3, RDKit 2026 silently drops ligands) → use `scripts/chai_xpu.py` |
| Structure Prediction | [esmfold2](structure-prediction/esmfold2/SKILL.md) | esm | PASS | Forked transformers first on PYTHONPATH; fold was slow (~4 min) |
| Structure Prediction | [protenix](structure-prediction/protenix/SKILL.md) | Protenix | PASS | Venv built on frameworks 2025.3.1 → prepend old oneAPI lib to `LD_LIBRARY_PATH`; `--trimul_kernel torch --triatt_kernel torch` |
| Diffusion Models | [rfd4-proteina](diffusion-models/rfd4-proteina/SKILL.md) | RFD4-Proteina | PASS via shim | Old lightning 2.3.0dev in venv → `scripts/sitecustomize.py` patch; weights from HF via proxy |
| Inverse Folding | [proteinmpnn](inverse-folding/proteinmpnn/SKILL.md) | ProteinMPNN | PASS | Empty `intel_extension_for_pytorch` shim on PYTHONPATH; `--device` ignored |
| QM | [reactra](qm/reactra/SKILL.md) | reactra | PARTIAL | Validate/plan only (no full run); `mech-format` installed in a `test_runs` overlay |
| QM | [uma](qm/uma/SKILL.md) | fairchem-xpu | PASS | `FAIRCHEM_CACHE_DIR=/flare/FRAME-IDP/msinclair/hf_cache`; `default`/`high-precision` modes give wrong forces on XPU → use `batch` or Nitro-XPU |
| Analysis | [pyrosetta](analysis/pyrosetta/SKILL.md) | pyrosetta_energy | PASS | CPU only; use `init(extra_options=...)` |
| Foundation LMs | [genslm-esm](foundation-language-models/genslm-esm/SKILL.md) | genslm-esm-sae | PASS | Use `venvs/aurora` (no `.venv`); `venvs/ci` is CUDA-only |

## Recommended permanent fixes (software dirs were left untouched)
- **chai-lab:** patch the source, or pin `pandas<3` and an older RDKit.
- **RFD4-Proteina:** upgrade `lightning` to `>=2.6`.
- **ProteinMPNN:** remove the unused `intel_extension_for_pytorch` imports.
- **reactra:** add `mech-format` to the venv.
- **Protenix, foundry (RFD3), pyrosetta_energy:** rebuild against frameworks 2026.1.0.

To use these as Claude Code skills, symlink or copy the tool folders into `~/.claude/skills/` or a project's `.claude/skills/`.

## Jupiter skill reference packaging

Each `*-jupiter` skill keeps its supporting files inside its own skill directory
and lists exact `skill_view(name="...", file_path="references/...")` calls.
Hermes does not resolve `../other-skill` or files from a software checkout as
paths in the selected skill. Do not use symlinks that resolve outside the skill.

All 15 Jupiter skills, including the three top-level literature/RAG skills,
package their own references. Literature/RAG skills include `references/api.md`
and a Jupiter workflow. RFD4 packages its binder/input guides directly. The other Jupiter wrappers
provide `references/workflow.md` for Jupiter execution, `references/aurora-notes.md`
for clearly labeled original provenance, and `references/resource-index.md` for
all bundled historical examples/helpers. Historical helpers are reference
material, not Jupiter launch scripts; original availability and timing claims
do not imply Jupiter readiness. `references/source-manifest.json` records
source paths and hashes. Update those snapshots deliberately when their sources
change, retaining the distinction between historical evidence and Jupiter use.

Validate packaging with `pytest tests/test_jupiter_skill_references.py`; also
verify changed references through the installed Hermes `skill_view` interface.
The campaign libraries link to these source directories, so files become
available immediately; agents must reread changed skills to receive new content.

Each Jupiter skill also packages `references/smoke-workload.py`, an executable
snapshot of its exact manual validation recipe. Refresh these snapshots after
changing the suite with `python3 scripts/package_jupiter_smoke_references.py`.
The snapshot header records the source SHA256. Missing licensed/private software
remains explicitly unavailable; packaging a recipe does not make it runnable.
