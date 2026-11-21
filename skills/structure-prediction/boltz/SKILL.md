---
name: boltz
description: Predict biomolecular structures (proteins, complexes, protein-ligand, DNA/RNA) and protein-ligand binding affinity with Boltz-2. Use when asked to fold a sequence or complex, dock a SMILES/CCD ligand, or estimate protein-ligand binding affinity.
systems: [aurora, jupiter]
---

# Boltz-2

## Before you run it: read the reference file for this System

The environment, accelerator flags and launch command differ per System. Read
`references/<HPC_SYSTEM>.md` (for example `read_skill(name="boltz", file="references/aurora.md")`)
and follow it; it is authoritative over this file for anything System-specific. This skill has
reference files for **aurora** and **jupiter**. It is **unsupported** on any other System: report the
step as blocked rather than adapting another System's instructions.

Software is found only through Site profile variables: the install is under `$LOAM_SOFTWARE_ROOT`.
Never edit the install directory; write outputs to the campaign workspace.

## Input

YAML (recommended) or FASTA; `DATA` may be one file or a directory (batch). Full schema in
`<install>/docs/prediction.md`, more in `<install>/examples/`. Examples here: `examples/prot_no_msa.yaml`,
`examples/lig_aff.yaml`, `examples/ubq_msa.fasta`.

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      msa: empty              # single-sequence mode; omit it to use --use_msa_server
  - ligand:
      id: B
      smiles: 'CC(=O)Oc1ccccc1C(=O)O'    # or: ccd: SAH
properties:                   # optional, exactly ONE ligand binder
  - affinity:
      binder: B
```

Entity ids must be unique. Every protein needs an explicit MSA choice: `msa: empty`, an MSA path,
or `--use_msa_server`, matching the intended experiment. Multimers use several `protein` entries
(or `id: [A, B]` for homo-copies); nucleic acids use `dna:` / `rna:`. FASTA headers are
`>CHAIN_ID|ENTITY_TYPE|MSA_PATH` with type protein/dna/rna/smiles/ccd.

## Arguments (`predict DATA`)

| flag | default | meaning |
|---|---|---|
| `--accelerator`, `--devices` | auto | set per System (see the reference file); always one device |
| `--out_dir`, `--cache` | — | output dir; weight cache |
| `--recycling_steps` | 3 | trunk recycles |
| `--sampling_steps` | 200 | diffusion steps |
| `--diffusion_samples` | 1 | structures generated; 5 is typical for production |
| `--sampling_steps_affinity`, `--diffusion_samples_affinity` | 200 / 5 | affinity head equivalents |
| `--max_parallel_samples` | — | lower on OOM |
| `--step_scale` | 1.5 | diversity vs designability |
| `--output_format` | mmcif | `pdb` or `mmcif` |
| `--seed` | — | reproducibility |
| `--use_msa_server`, `--msa_server_url`, `--msa_pairing_strategy` | off | ColabFold MSAs (needs network access) |
| `--use_potentials` | off | physics steering |
| `--write_full_pae`, `--write_full_pde`, `--write_embeddings` | off | extra arrays |
| `--override` | off | overwrite existing predictions |
| `--num_workers` | — | use `0` |
| `--model` | boltz2 | `boltz1` or `boltz2` |
| `--no_kernels` | off | disables the CUDA-only cuequivariance kernels |

## Outputs

```
out/boltz_results_<name>/
  predictions/<name>/
    <name>_model_0.pdb|cif               # ranked by confidence; model_0 = best
    confidence_<name>_model_0.json       # confidence_score, ptm, iptm, ligand_iptm, protein_iptm,
                                         # complex_plddt, complex_iplddt, complex_pde/ipde,
                                         # chains_ptm, pair_chains_iptm
    plddt_/pae_/pde_<name>_model_0.npz
    affinity_<name>.json                 # affinity_pred_value (log10 IC50 in uM, lower = stronger),
                                         # affinity_probability_binary
  processed/, msa/                        # featurized inputs; MSAs when --use_msa_server
```

`confidence_score = 0.8*complex_plddt + 0.2*iptm` (ptm for monomers). **pLDDT is 0-1 here**, not 0-100.

## Must-know

- Multi-device inference is not implemented: `--devices >1` warns and uses one device. For many
  inputs, run one process per GPU.
- Use `--num_workers 0` to avoid dataloader fork issues.
- Without `--override`, existing predictions in `out_dir` are silently skipped. Prefer a fresh
  output directory over overriding prior predictions.
- `--use_msa_server` needs network access (the System's proxy), or it hangs.
- `Number of failed examples: N>0` means bad input — check unique chain ids, valid SMILES, and that
  any `ccd` code exists in the cache's `mols/`.
- Verify all requested chains and ligands are present in the actual output. Preserve input and
  checkpoint ids, seeds, MSA choice and failures.
- The affinity head is for protein-ligand pairs. It is not a protein-binder MM/GBSA calculation or a
  measured binding affinity; never report it as a protein-binder free energy.
