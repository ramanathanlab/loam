---
name: boltz
description: Predict biomolecular structures (protein, protein complexes, protein-ligand, DNA/RNA) and protein-ligand binding affinity with Boltz-2 on ALCF Aurora Intel XPUs. Use when the user wants to fold a sequence or complex, dock a small molecule (SMILES or CCD code) into a protein, or estimate binding affinity on Aurora with the local Boltz install at /flare/FRAME-IDP/msinclair/software/boltz.
---

# Boltz-2 on Aurora (Intel XPU)

Local install: `/flare/FRAME-IDP/msinclair/software/boltz`. This is a modified fork in which
PyTorch Lightning was removed (it uses a "vanilla" inference loop) and `--accelerator xpu` was added.
Do not edit files in the install directory.

## Environment

```bash
# Run on a compute node. The venv was built on the system frameworks python with
# include-system-site-packages = true, so no `module load` is needed and you do not need to activate it.
BOLTZ=/flare/FRAME-IDP/msinclair/software/boltz/.venv/bin/boltz
export ZE_AFFINITY_MASK=2          # pin to ONE tile (FLAT mode: 0-11). Inside the process it shows up as xpu:0
# Only needed for --use_msa_server (compute nodes reach the internet only through the proxy):
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
```
Python is aurora_frameworks-2026.1.0 (py3.12, torch 2.13 with native XPU). IPEX is not used.

## Weights / cache
Everything is already in `~/.boltz` (pass `--cache ~/.boltz` or set `BOLTZ_CACHE`):
`boltz2_conf.ckpt` (structure), `boltz2_aff.ckpt` (affinity), `mols/` (CCD), `mols.tar`.
Copies of the checkpoints are also in `/flare/FRAME-IDP/msinclair/software/checkpoints/`.
No download happens if these are present.

## Input formats
YAML (recommended):
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      msa: empty            # single-sequence mode; omit and use --use_msa_server, or give a path to an .a3m/.csv
  - ligand:
      id: B
      smiles: 'CC(=O)Oc1ccccc1C(=O)O'   # or: ccd: SAH
properties:                 # optional: affinity for ONE ligand binder
  - affinity:
      binder: B
```
Multimers use several `protein` entries (or `id: [A, B]` for homo-copies). DNA/RNA use `dna:` / `rna:`.
Examples in the install: `examples/{prot.yaml,prot_no_msa.yaml,multimer.yaml,ligand.yaml,affinity.yaml,pocket.yaml,cyclic_prot.yaml}`.
Full schema: `docs/prediction.md`.

FASTA: `>CHAIN_ID|ENTITY_TYPE|MSA_PATH` where the type is protein/dna/rna/smiles/ccd, for example
`>A|protein` (with --use_msa_server), `>A|protein|empty` (no MSA), `>B|smiles` followed by a SMILES string on the next line, `>C|ccd` followed by `SAH`.

`DATA` can be one file or a directory of files (batch).

## Verified commands (Aurora, tile 2, Oct 2026)

1. Protein only (76 aa ubiquitin), single sequence. Wall time 44 s, pLDDT 0.93:
```bash
export ZE_AFFINITY_MASK=2
$BOLTZ predict prot_no_msa.yaml --out_dir out --cache ~/.boltz --accelerator xpu --devices 1 \
  --recycling_steps 1 --sampling_steps 50 --diffusion_samples 1 --output_format pdb --override --num_workers 0
```
2. Protein + SMILES ligand + affinity. Wall time 72 s:
```bash
$BOLTZ predict lig_aff.yaml --out_dir out --cache ~/.boltz --accelerator xpu \
  --recycling_steps 1 --sampling_steps 50 --diffusion_samples 1 \
  --sampling_steps_affinity 50 --diffusion_samples_affinity 1 --output_format pdb --override --num_workers 0
```
3. FASTA with the ColabFold MSA server through the ALCF proxy. Wall time 34 s, pLDDT 0.94:
```bash
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
$BOLTZ predict ubq_msa.fasta --use_msa_server --out_dir out --cache ~/.boltz --accelerator xpu \
  --recycling_steps 1 --sampling_steps 50 --diffusion_samples 1 --output_format pdb --override --num_workers 0
```
The test inputs are in `scripts/` next to this file, together with the wrapper `scripts/run_boltz_xpu.sh <input> <out_dir> [tile] [extra args]`.
The test outputs are in `/lus/flare/projects/FRAME-IDP/msinclair/testing_grounds/env_tests/test_runs/boltz/`.
For production runs, use the defaults (`--recycling_steps 3 --sampling_steps 200`) and `--diffusion_samples 5`. Those settings were not timed here.

## Main options
`--accelerator xpu` (or `auto`/`gpu`, which fall through to XPU when CUDA is absent), `--recycling_steps` (3), `--sampling_steps` (200),
`--diffusion_samples` (1), `--max_parallel_samples`, `--step_scale` (1.5), `--output_format pdb|mmcif`, `--seed`,
`--use_msa_server`/`--msa_server_url`/`--msa_pairing_strategy`, `--use_potentials` (physics steering),
`--write_full_pae` / `--write_full_pde`, `--write_embeddings`, `--override`, `--model boltz1|boltz2`,
`--no_kernels` (disables the cuequivariance kernels, which are CUDA-only; the XPU runs above worked without this flag), `--max_msa_seqs`, `--num_subsampled_msa`.

## Output layout
```
out/boltz_results_<name>/
  processed/                      # featurized inputs, manifest
  msa/                            # only with --use_msa_server (<name>_0.csv)
  predictions/<name>/
    <name>_model_0.pdb|cif        # ranked by confidence; model_0 = best
    confidence_<name>_model_0.json   # confidence_score, ptm, iptm, ligand_iptm, protein_iptm,
                                     # complex_plddt, complex_iplddt, complex_pde, complex_ipde, chains_ptm, pair_chains_iptm
    plddt_/pae_/pde_<name>_model_0.npz
    affinity_<name>.json          # if affinity requested: affinity_pred_value (log10 IC50 in uM; lower = stronger),
                                  # affinity_probability_binary (P(binder)), plus ensemble members *1, *2
```
confidence_score = 0.8*complex_plddt + 0.2*iptm (ptm for monomers). pLDDT is on a 0-1 scale.

## Aurora/XPU gotchas
- Always set `ZE_AFFINITY_MASK` to one tile. Multi-device inference is not implemented in this fork: `--devices >1` prints a warning and uses one device. To process many inputs, run one process per tile, each with a different mask.
- Expected noise that you can ignore: the `Overriding a previously registered kernel ... aten::topk` warnings, the urllib3/chardet RequestsDependencyWarning, "Checkpoint contains hyper_parameters not in Boltz2.__init__, dropping ...", and "Aten Op fallback from XPU to CPU" for `torch.linalg.svd` (from the diffusion alignment; it is slow but correct).
- Use `--num_workers 0` (the tested setting) to avoid dataloader worker fork issues.
- Without `--override`, existing predictions in out_dir are silently skipped.
- The MSA server needs the proxy env vars. Without them it hangs or fails to connect.

## Troubleshooting
- `Number of failed examples: N>0`: check the YAML (unique chain ids, valid SMILES, ccd code exists in `~/.boltz/mols`).
- It tries to download weights: `--cache` is not pointing to `~/.boltz`, or `BOLTZ_CACHE` is set to some other path.
- OOM on large complexes: lower `--max_parallel_samples` and `--diffusion_samples`, or add `--no_kernels`. Use one full tile per job.
