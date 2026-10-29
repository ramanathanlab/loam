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
| Diffusion Models | [rfdiffusion3](diffusion-models/rfdiffusion3/SKILL.md) | foundry | PASS | `module swap` to frameworks/2025.3.1 |
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
