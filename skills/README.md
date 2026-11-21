# Tool skills

Claude Code skills for loam's scientific tool stack. Each skill's install lives under the Site
profile's software root, `$LOAM_SOFTWARE_ROOT/<dir>` (the table's "Software dir"). Never write into
those directories; write outputs to the campaign workspace.

Read [TOOL_SELECTION.md](TOOL_SELECTION.md) to pick a tool, then that tool's `SKILL.md` for the
method, and `references/$HPC_SYSTEM.md` for the environment, device selection and launch commands on
the System you are on. On Aurora (Intel Data Center GPU Max 1550, FLAT mode, 12 tiles) that means
pinning one tile with `ZE_AFFINITY_MASK=<int>` — a plain integer (`4.0` hides the device).

| Category | Skill | Software dir | Key note |
|---|---|---|---|
| Structure Prediction | [boltz](structure-prediction/boltz/SKILL.md) | boltz | `--accelerator xpu`; single device only; also does affinity |
| Structure Prediction | [chai](structure-prediction/chai/SKILL.md) | chai-lab | Use `scripts/chai_xpu.py`; the stock `chai` CLI is broken |
| Structure Prediction | [esmfold2](structure-prediction/esmfold2/SKILL.md) | esm | Forked transformers first on `PYTHONPATH`; slow cold start |
| Structure Prediction | [protenix](structure-prediction/protenix/SKILL.md) | Protenix | Prepend old oneAPI lib to `LD_LIBRARY_PATH`; torch kernels |
| Diffusion Models | [rfdiffusion3](diffusion-models/rfdiffusion3/SKILL.md) | foundry | Aurora only, interactive: rows hang under `run_batch` |
| Diffusion Models | [rfd4-proteina](diffusion-models/rfd4-proteina/SKILL.md) | (`$RFD4_DEPLOYMENT`, or RFD4-Proteina) | Resident worker per accelerator; annotated-CIF inputs |
| Inverse Folding | [proteinmpnn](inverse-folding/proteinmpnn/SKILL.md) | ProteinMPNN (Aurora), proteinmpnn (Jupiter) | `--device` ignored; Aurora needs the IPEX shim on `PYTHONPATH` |
| MD Simulation | [molecular-simulations](md-simulation/molecular-simulations/SKILL.md) | molecular-simulations | `platform='OpenCL'`; set `AMBERHOME`; no ligand builders |
| MD Simulation | [openmm](md-simulation/openmm/SKILL.md) | openmm-sycl | GPU platform is named `OpenCL` |
| QM | [uma](qm/uma/SKILL.md) | fairchem-xpu | Use `batch` or Nitro-XPU; `default` gives wrong forces |
| QM | [reactra](qm/reactra/SKILL.md) | reactra | Needs the `mech-format` overlay; full runs ~1 h |
| Analysis | [pyrosetta](analysis/pyrosetta/SKILL.md) | pyrosetta_energy | Aurora only, CPU only; use `init(extra_options=...)` |
| Foundation LMs | [genslm-esm](foundation-language-models/genslm-esm/SKILL.md) | genslm-esm-sae | Aurora only; use `venvs/aurora`, `venvs/ci` is CUDA-only |
| Literature | [fetch-literature](fetch-literature/SKILL.md) | (none) | `systems: all`; `requests` against PubMed and Europe PMC |
| RAG | [rag-build-index](rag-build-index/SKILL.md) | (none) | `systems: all`; faiss + an Arrow dataset in one row order |
| RAG | [rag-query](rag-query/SKILL.md) | (none) | `systems: all`; needs an endpoint from `LOAM_INFERENCE_JSON` |

`rfdiffusion3` and `rfd4-proteina` are both working backbone generators — see TOOL_SELECTION.md for
which to pick.

## Skill layout

A skill is being converted, one family at a time, to a System-neutral layout (see
`docs/adr/0004-system-neutral-core-and-out-of-repo-site-profiles.md`):

```
<skill>/
  SKILL.md               # inputs, outputs and method; no System-specific or personal paths
                         # frontmatter: systems: [aurora, jupiter]
  references/<system>.md # one per claimed System: environment, device selection, launch commands
  scripts/<system>/      # that System's runners, launchers and smoke workloads
  examples/              # inputs valid on any System
```

The agent reads `references/$HPC_SYSTEM.md`; a skill with no reference file for a System is
unsupported there and its `SKILL.md` says so. Paths go only through Site profile variables such as
`$LOAM_SOFTWARE_ROOT`. `tests/test_skill_layout.py` enforces this for the converted skills (so far
the structure-prediction family: boltz, chai, esmfold2, protenix; the md-simulation family:
molecular-simulations, openmm; the qm family: uma, and reactra, which is Aurora-only; and the
diffusion-models family: rfd4-proteina, and rfdiffusion3, which is Aurora-only; inverse-folding:
proteinmpnn; analysis: pyrosetta, and foundation-language-models: genslm-esm, both Aurora-only; and
fetch-literature, rag-build-index and rag-query, which claim `systems: all`). It also fails when any
file a converted skill bundles holds a personal path.

A skill that runs the same way everywhere claims `systems: all` instead of naming Systems: it has no
reference file (and may not ship one), and takes its interpreter, caches, network and inference
endpoints from the Site profile rather than from a System.

Every tool skill in the table above is now converted. `hpc-deployment`, which is not a tool skill,
still has its own shape; #62 widens the check to it.

## Recommended permanent fixes (software dirs were left untouched)

- **chai-lab:** patch the source, or pin `pandas<3` and an older RDKit.
- **ProteinMPNN:** remove the unused `intel_extension_for_pytorch` imports.
- **reactra:** add `mech-format` to the venv.
- **Protenix, pyrosetta_energy:** rebuild against frameworks 2026.1.0.

To use these as Claude Code skills, symlink or copy the tool folders into `~/.claude/skills/` or a
project's `.claude/skills/`.
