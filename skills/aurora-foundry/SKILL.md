---
name: aurora-foundry
description: >-
  Generate protein backbones with RFdiffusion3 (rfd3) and design sequences onto
  a fixed backbone with ProteinMPNN/LigandMPNN/SolubleMPNN (mpnn) from the
  RosettaCommons Foundry venv on Aurora. Use for the generation and
  inverse-folding stages of a binder funnel.
---

# Foundry on Aurora: rfd3, rf3, mpnn

## Activate

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_foundry-2026-venv.sh
```

Binaries (verified present, rc=0):

```
$R/envs/foundry-2026-venv/bin/rfd3   RFdiffusion3
$R/envs/foundry-2026-venv/bin/rfd3na RFdiffusion3, nucleic acids
$R/envs/foundry-2026-venv/bin/rf3    RoseTTAFold3
$R/envs/foundry-2026-venv/bin/mpnn   ProteinMPNN / LigandMPNN / SolubleMPNN
```

## rfd3 — backbone generation

`rfd3 --help` reports: *"Run design using hydra config overrides and input
files."* It is a **hydra** app, so configuration is `key=value` overrides on the
command line, not conventional flags — `--help` lists only completion options.
Discover the available groups and their defaults before composing a run:

```bash
rfd3 --cfg job          # print the resolved config
rfd3 --help             # completion only; not the option list
```

Compose the run from config overrides (output dir, number of designs, contigs,
hotspots). Do not guess flag spellings from other RFdiffusion versions — v3's
config tree differs from v1/v2.

## mpnn — inverse folding

Real usage, from `mpnn --help`:

```
mpnn [-h] [--config_json CONFIG_JSON] [--model_type {protein_mpnn,ligand_mpnn}]
     [--checkpoint_path CHECKPOINT_PATH] [--is_legacy_weights {True,False}]
     [--out_directory OUT_DIRECTORY] [--write_fasta {True,False}]
     [--write_structures {True,False}] [--structure_path STRUCTURE_PATH]
     [--name NAME] [--seed SEED] ...
```

So: point `--structure_path` at the backbone, pick `--model_type`, set
`--out_directory`, and `--write_fasta True` to get sequences. `--seed` makes a
run reproducible; vary it (not the temperature alone) for independent samples.

Use `--model_type protein_mpnn` for a plain backbone and SolubleMPNN weights via
`--checkpoint_path` when solubility matters for developability.

## Pitfalls

- `mpnn` prints warnings that `CCD_MIRROR_PATH` and `PDB_MIRROR_PATH` are unset.
  Harmless for sequence design on a supplied structure; any atomworks function
  needing those mirrors will fail. Set them only if you actually need them.
- MPNN on this system runs on CPU. Uncapped, torch grabs a default thread count
  per process and 12 concurrent batch rows oversubscribe the node badly. Cap it
  (e.g. `OMP_NUM_THREADS=16`) in batch fan-out.
- MPNN can collapse to low-complexity sequences (poly-Q, poly-A) on extended or
  out-of-distribution backbones. Gate sequences on a complexity filter BEFORE
  spending folds on them; a shape-based interface score will not catch it.
- First activation takes ~40 s. Not a hang.
