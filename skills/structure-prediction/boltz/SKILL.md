---
name: boltz
description: Predict biomolecular structures (proteins, complexes, protein-ligand, DNA/RNA) and protein-ligand binding affinity with Boltz-2 on Aurora Intel XPU. Use when asked to fold a sequence or complex, dock a SMILES/CCD ligand, or estimate binding affinity on Aurora.
---

# Boltz-2 on Aurora (Intel XPU)

Install: `/flare/FRAME-IDP/msinclair/software/boltz` — a fork with PyTorch Lightning removed
(vanilla inference loop) and `--accelerator xpu` added. Do not edit the install directory.

## Environment

```bash
BOLTZ=/flare/FRAME-IDP/msinclair/software/boltz/.venv/bin/boltz   # no module load or activate needed
export ZE_AFFINITY_MASK=2                   # ONE tile (0-11 in FLAT mode); appears as xpu:0
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy   # only for --use_msa_server
```

Weights and the CCD cache are already in `~/.boltz` (`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols/`);
pass `--cache ~/.boltz`. Copies live in `/flare/FRAME-IDP/msinclair/software/checkpoints/`.

## Example

```bash
# Minimal runner: scripts/run_boltz_xpu.sh <input> <out_dir> [tile] [extra args...]
bash scripts/run_boltz_xpu.sh prot_no_msa.yaml out 2

# Equivalent direct call (production settings)
$BOLTZ predict prot_no_msa.yaml --out_dir out --cache ~/.boltz --accelerator xpu --devices 1 \
  --recycling_steps 3 --sampling_steps 200 --diffusion_samples 5 \
  --output_format pdb --override --num_workers 0
```

Input is YAML (recommended) or FASTA; `DATA` may be one file or a directory (batch).
Examples: `scripts/prot_no_msa.yaml`, `scripts/lig_aff.yaml`, `scripts/ubq_msa.fasta`;
more in `<install>/examples/`, full schema in `<install>/docs/prediction.md`.

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

Multimers use several `protein` entries (or `id: [A, B]` for homo-copies); nucleic acids use
`dna:` / `rna:`. FASTA headers are `>CHAIN_ID|ENTITY_TYPE|MSA_PATH` with type
protein/dna/rna/smiles/ccd.

## Arguments (`predict DATA`)

| flag | default | meaning |
|---|---|---|
| `--accelerator xpu` | auto | required on Aurora (`auto`/`gpu` fall through when CUDA is absent) |
| `--out_dir`, `--cache` | — | output dir; weight cache (`~/.boltz`) |
| `--recycling_steps` | 3 | trunk recycles |
| `--sampling_steps` | 200 | diffusion steps |
| `--diffusion_samples` | 1 | structures generated; 5 is typical for production |
| `--sampling_steps_affinity`, `--diffusion_samples_affinity` | 200 / 5 | affinity head equivalents |
| `--max_parallel_samples` | — | lower on OOM |
| `--step_scale` | 1.5 | diversity vs designability |
| `--output_format` | mmcif | `pdb` or `mmcif` |
| `--seed` | — | reproducibility |
| `--use_msa_server`, `--msa_server_url`, `--msa_pairing_strategy` | off | ColabFold MSAs (needs the proxy) |
| `--use_potentials` | off | physics steering |
| `--write_full_pae`, `--write_full_pde`, `--write_embeddings` | off | extra arrays |
| `--override` | off | overwrite existing predictions |
| `--num_workers` | — | use `0` |
| `--model` | boltz2 | `boltz1` or `boltz2` |
| `--no_kernels` | off | disables CUDA-only cuequivariance kernels (not needed on XPU) |

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

- Always pin `ZE_AFFINITY_MASK` to one tile. Multi-device inference is not implemented: `--devices >1`
  warns and uses one device. For many inputs, run one process per tile with a different mask.
- Use `--num_workers 0` to avoid dataloader fork issues.
- Without `--override`, existing predictions in `out_dir` are silently skipped.
- `--use_msa_server` needs the proxy env vars, or it hangs.
- Harmless noise: `aten::topk` kernel-override warnings, `RequestsDependencyWarning`, dropped
  checkpoint `hyper_parameters`, and `Aten Op fallback from XPU to CPU` for `torch.linalg.svd`
  (slow but correct).
- `Number of failed examples: N>0` means bad input — check unique chain ids, valid SMILES, and that
  any `ccd` code exists in `~/.boltz/mols`.
