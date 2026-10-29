---
name: protenix
description: Predict biomolecular structures (proteins, DNA/RNA, ligands, ions, complexes) with Protenix, ByteDance's AlphaFold3-style diffusion model, on Aurora Intel XPU, writing mmCIF plus pLDDT/pTM/ipTM/PAE confidence JSON. Use when you need AF3-like complex or single-chain prediction on an ALCF Aurora compute node, including fast no-MSA runs (mini_esm, v1.0.0, v2 checkpoints available locally).
---

# Protenix on Aurora (Intel Max 1550 / XPU)

The checkout at `/flare/FRAME-IDP/msinclair/software/Protenix` is XPU-patched
(`protenix/utils/device.py` auto-selects `xpu` -> `cuda` -> `cpu`). CUDA-only kernels
(cuequivariance, deepspeed, triattention) must be swapped for the `torch` kernels.

## Environment

```bash
REPO=/flare/FRAME-IDP/msinclair/software/Protenix
# The venv was built on frameworks/2025.3.1 (/opt/aurora/26.26.0). The default module is now
# frameworks/2026.1.0, which lacks that build's MKL -> "OSError: libmkl_intel_lp64.so.2".
# Fix (verified): prepend the old oneAPI lib dir. (Swapping modules via `module load
# frameworks/2025.3.1` FAILED: missing intel-oneapi-runtime/2026.1.0 dependency.)
export LD_LIBRARY_PATH=/opt/aurora/26.26.0/oneapi/2025.3/lib:${LD_LIBRARY_PATH:-}
export ZE_AFFINITY_MASK=4                   # one plain tile index (FLAT mode); NOT "4.0"
export PROTENIX_ROOT_DIR=/flare/FRAME-IDP/msinclair/software/protenix_data
export PYTHONUNBUFFERED=1                   # get logs live
PROTENIX=$REPO/.venv/bin/protenix           # CLI entry point (or $REPO/.venv/bin/python)
```
For MSA/template search (network): `export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128`.
Not needed for `--use_msa false` runs (verified without proxy).

## Data / weights (already downloaded)

`$PROTENIX_ROOT_DIR` = `/flare/FRAME-IDP/msinclair/software/protenix_data`
- `checkpoint/`: `protenix_mini_esm_v0.5.0.pt`, `protenix_base_default_v1.0.0.pt`,
  `protenix-v2.pt`, `esm2_t36_3B_UR50D.pt` (+ contact-regression; used by mini_esm)
- `common/`: `components.cif`, `components.cif.rdkit_mol.pkl` (CCD), cluster/release caches
- `mmcif/` (template structures)

Only these model names work offline: `protenix_mini_esm_v0.5.0`, `protenix_base_default_v1.0.0`,
`protenix-v2`. Others in `inference_demo.sh` would try to download.

## Input JSON

```json
[{
  "name": "gb1",
  "sequences": [
    {"proteinChain": {"sequence": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", "count": 1}}
  ]
}]
```
Other entity types: `dnaSequence`, `rnaSequence`, `ligand` (`"ligand": "CCD_ATP"` or a SMILES
string), `ion`; `count` = copies; optional per-chain `pairedMsaPath` / `unpairedMsaPath`
(absolute a3m paths), `templatesPath`, `modifications`. Full spec:
`$REPO/docs/infer_json_format.md`; examples in `$REPO/examples/`. `protenix json -i x.cif -o dir`
converts a PDB/mmCIF to input JSON (unverified here).

## Verified minimal command

```bash
cd /your/workdir        # NOTE: ESM embeddings are cached to ./esm_embeddings in the CWD
$PROTENIX pred -i gb1.json -o out_mini -s 101 \
  -n protenix_mini_esm_v0.5.0 -c 4 -p 20 -e 1 --use_msa false \
  --trimul_kernel torch --triatt_kernel torch
```
Tested script + input: `scripts/run_protenix_example.sh`, `scripts/example_gb1.json`
(run dir `/lus/flare/projects/FRAME-IDP/msinclair/testing_grounds/env_tests/test_runs/protenix/`).

Results 2026-10-29, GB1 (56 aa), 1 seed, 1 sample, 4 cycles, 20 steps, bf16:

| model (`-n`) | wall (incl. load) | forward | pLDDT | pTM |
|---|---|---|---|---|
| protenix_mini_esm_v0.5.0 | 56-64 s | ~4 s | 93.5 | 0.92 |
| protenix_base_default_v1.0.0 (no MSA) | 76 s | 5.7 s | 86.0 | 0.80 |
| protenix-v2 (no MSA) | 88 s | 6.4 s | 89.0 | 0.88 |

## Main options (`protenix pred`)
| flag | default | notes |
|---|---|---|
| `-i/--input` | — | JSON file or dir |
| `-o/--out_dir` | ./output | |
| `-s/--seeds` | 101 | comma list, e.g. `101,102` |
| `-c/--cycle` | 10 | Pairformer recycles |
| `-p/--step` | 200 | diffusion steps (use 200 for real runs) |
| `-e/--sample` | 5 | samples per seed |
| `-d/--dtype` | bf16 | `bf16` or `fp32` |
| `-n/--model_name` | protenix_base_default_v1.0.0 | see list above |
| `--use_msa` | **true** | true -> MSA server search (needs proxy; unverified on this node) |
| `--trimul_kernel` | **cuequivariance** | MUST set `torch` on XPU |
| `--triatt_kernel` | **cuequivariance** | MUST set `torch` on XPU |
| `--use_template`, `--use_rna_msa` | false | v1.0.0+; need kalign/hmmer + DBs (unverified) |
| `--use_default_params` | | load model's recommended cycles/steps |
| `--use_seeds_in_json` | | take seeds from input JSON |

Lower-level: `$REPO/.venv/bin/python $REPO/runner/inference.py --model_name ... --input_json_path ... --dump_dir ... --triangle_attention torch --triangle_multiplicative torch` (needs `PYTHONPATH=$REPO`; unverified this session).

## Output layout
```
<out_dir>/<name>/seed_<seed>/predictions/
    <name>_sample_<k>.cif                       # structure (one per sample)
    <name>_summary_confidence_sample_<k>.json   # plddt (0-100), ptm, iptm, ranking_score,
                                                # chain_* / chain_pair_* (iptm, pae_mean, plddt), has_clash, disorder
```
`--need_atom_confidence true` adds full per-atom pLDDT / PAE arrays. Rank samples by
`ranking_score`. iptm is 0 for single chains.

## Aurora / XPU gotchas
- `libmkl_intel_lp64.so.2` not found -> venv/module mismatch; set the `LD_LIBRARY_PATH` above.
- Always pass `--trimul_kernel torch --triatt_kernel torch` (defaults are CUDA cuequivariance).
- `--use_msa` defaults to true -> network MSA search; pass `--use_msa false` for offline/fast runs.
- Writes `./esm_embeddings/` (mini_esm) into the current dir — `cd` to a scratch/run dir first;
  never run from inside the software dir.
- Model load dominates small jobs (~45-80 s); batch many inputs in one JSON/dir per call.
- One unexplained incident: a first attempt sat idle ~3.5 min with no log output (no
  PYTHONUNBUFFERED, no proxy); identical reruns finished in ~1 min. If it stalls, kill and rerun
  with `PYTHONUNBUFFERED=1`.
- Older note in repo logs: `NameError: name 'self' is not defined` in
  `is_gpu_capability_between_7_and_8` and biotite `BondList._bonds` errors — both already fixed
  in the current checkout.

## Troubleshooting
- `torch.xpu.is_available()` False: check `ZE_AFFINITY_MASK` is an integer and you're on a compute node.
- Checkpoint download attempts: confirm `PROTENIX_ROOT_DIR` and use one of the three local models.
- OOM on large complexes: one tile = 64 GB; reduce `-e` samples, use bf16, or split jobs.
