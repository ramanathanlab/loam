---
name: aurora-binder-venvs
description: >-
  The authoritative map of the Aurora binder_design project: which venv runs
  which tool, how to activate it, and the per-venv gaps that waste a round if
  you discover them yourself. Read this FIRST on any Aurora protein-design task
  before writing code that activates an environment or invokes a model.
---

# Aurora binder_design: venvs and tools

Project root (the two spellings are the SAME directory; either works):

```
/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
/lus/flare/projects/FRAME-IDP/agentic_campaign_09_26/binder_design
```

Activate a venv with its script, which also loads `frameworks/2026.1.0` and sets
weight paths and GPU settings:

```bash
source <root>/envs/_scripts/activate_<venv>.sh
```

`chai-2026-venv` has **no** activate script. It needs:

```bash
module use /soft/modulefiles; module load frameworks/2026.1.0
source <root>/envs/chai-2026-venv/bin/activate
```

Run every tool through `subprocess.run(["bash","-lc", CMD])`. Do not import these
packages into a long-lived process: the activate scripts set module and library
state that only a login shell applies correctly.

## Which venv for what

| venv | tools | use for |
|---|---|---|
| `chai-2026-venv` | `chai fold` | complex co-folding, interface metrics |
| `foundry-2026-venv` | `rfd3`, `rfd3na`, `rf3`, `mpnn` | backbone generation, inverse folding |
| `rfd4-proteina-2026-venv` | `rfd4-generate`, `rfd4-evaluate` | all-atom flow-matching design |
| `proteina-complexa-2026-venv` | `complexa` (+ `rf3`, `rfd3`, `mpnn`, `pdbfixer`) | end-to-end binder pipeline |
| `boltz-2026-venv` | `boltz predict` | structure + affinity prediction |
| `protenix-2026-venv` | `protenix pred` | AlphaFold3-style prediction |
| `esm-2026-venv` | `$ESM_SCRIPTS/*.py` | ESMFold2, ESMC/ESM3 embeddings |
| `genie3-2026-venv` | `genie3 generate` | generation with built-in sharding |
| `rfdiffusion-idp-2026-venv` | `run_inference.py` | disordered-region backbones |
| `bindcraft2-2026-venv` | `bindcraft` | AF2+MPNN binder design (JAX) |
| `freebindcraft-2026-venv` | `bindcraft.py` | FreeBindCraft/ColabDesign (JAX, has pdbfixer + OpenMM 8.6.1) |
| `deepdrive-2026-venv` | `deepdrivewe`, OpenMM 8.6, MDAnalysis | MD, weighted ensemble |
| `rag-2026-venv` | `distllm` | retrieval over a literature corpus |
| `dgl-xpu-2026-venv` | `dgl` | graph learning — `GraphConv` is BROKEN on XPU |
| `gpu4mrh-2026-venv` | `pyscf`, `mrh` | quantum chemistry, not design |

## Gaps that cost a round

- **`envs/VENVS.txt` is the maintained inventory.** A copy named
  `venvs_aurora.txt` circulates in campaign inputs and is STALE: it omits
  `rfd4-proteina-2026-venv` entirely. Trust `envs/VENVS.txt`.
- **No protonation tooling.** In `deepdrive-2026-venv`, `propka`, `pdb2pqr` and
  `pdbfixer` are all ABSENT (verified: `openmm` true, `MDAnalysis` true,
  `deepdrivewe` true, `parmed` true, `mdtraj` true, `pdbfixer` **false**,
  `propka` **false**, `pdb2pqr` **false**). For pH-dependent protonation use
  `openmm.app.Modeller.addHydrogens(forcefield, pH=...)`, or `pdbfixer` from
  `freebindcraft-2026-venv` / `proteina-complexa-2026-venv`. See
  `aurora-openmm-ph`.
- **`mpnn` warns about `CCD_MIRROR_PATH` and `PDB_MIRROR_PATH` being unset.**
  Harmless for plain sequence design; functions needing those mirrors fail.
- **Activating `proteina-complexa-2026-venv` downgrades oneAPI**
  (`oneapi/release/2026.1.0 => oneapi/release/2025.3.1`). Do not mix it into a
  shell that already activated another venv; use one venv per `bash -lc`.
- **`bindcraft2` and `freebindcraft` cannot import torch** by design (JAX only).
- First activation of a venv can take 30–80 s (module load + imports). Budget
  for it; it is not a hang.

## Device pinning

One MPI-free process per tile. Chai and the torch tools take `--device xpu:0`
or honour `ZE_AFFINITY_MASK`; a batch row that uses one device needs `gpus=1`.
A node has 6 GPUs / 12 tiles.

Verification: every tool above was probed with `command -v` and `--help` on
Aurora (PBS 8982260, 2026-10-29); all returned rc=0. Entries for
`rfdiffusion-idp`, `bindcraft2`, `freebindcraft`, `dgl` and `gpu4mrh` come from
`envs/VENVS.txt` and were not individually probed.
