---
name: aurora-rfd4-proteina
description: >-
  Generate all-atom designs (proteins, nucleic acids, ligands, complexes) with
  RFD4-Proteina flow matching on an Aurora XPU, including which upstream
  features are unavailable in the XPU port. Use when a campaign asks for
  "RFdiffusion4" or all-atom/interface-conditioned generation.
---

# RFD4-Proteina on Aurora

This venv is NOT in the stale `venvs_aurora.txt` that circulates in campaign
inputs. It exists: `envs/rfd4-proteina-2026-venv`, built from the upstream
GitHub repo rather than the Apptainer recipes.

## Activate

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_rfd4-proteina-2026-venv.sh
```

Entry points: `rfd4-generate`, `rfd4-evaluate`. Python: `import rfproteina`,
`import rfd4_proteina`.

## Invoke

Hydra-style, from `envs/VENVS.txt`:

```bash
rfd4-generate hf_ckpt_path=nvidia/RFD4-Proteina \
  inference_experiment=quickstart \
  inputs=<spec>.yaml out_dir=<dir> compile_lengths=null
```

`rfd4-generate --help` prints the hydra config tree rather than flags. Verified
groups relevant to binder work (rc=0 on Aurora):

- `datasets/val/ppi`: `ppi_all_atom`, `ppi_all_atom_subset_12`, `ppi_bb_only`,
  `ppi_bb_only_subset_12` — protein-protein interface conditioning
- `datasets/val/antibody`: `nanobody_design`, `nanobody_design_subset`
- `datasets/val/motif_scaffolding`: indexed/unindexed motif scaffolding
- `datasets/val/small_molecule_binder`, `datasets/val/dna_binder`
- `datasets/val/adherence`: `interchain_interaction_conditioning`,
  `centroid_adherence`, `rasa_IAI`, `ss_conditioned`
- `debug`: `cpu`, `gpu`

Run `rfd4-generate --help` yourself and read the group list before composing a
command; do not invent config names.

## What is NOT available in this port

Upstream is CUDA-only; this is the base package plus four XPU patches in
`software/RFD4-Proteina-Aurora` (read its `aurora/README.md`). Unavailable:

- `inference_backend=bioir`
- `precision=fp8`
- the TMol CUDA metrics
- the AF2/AF3 refolding oracles

So generate with `inference_experiment=quickstart` or pass `'~metrics'` to drop
the metrics stage. `torch.compile` is validated only OFF: keep
`compile_lengths=null`.

## Cost

Verified upstream measurement: 2 designs in 1m29s on one Max 1550 tile
(PBS 8969397). Size a tier from that, not from CUDA benchmarks.
