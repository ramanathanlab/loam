---
name: protenix
description: Predict biomolecular structures (proteins, DNA/RNA, ligands, ions, complexes) with Protenix, an AlphaFold3-style diffusion model, on Aurora Intel XPU, writing mmCIF plus pLDDT/pTM/ipTM/PAE JSON. Use for AF3-like complex or single-chain prediction on Aurora, including fast no-MSA runs.
---

# Protenix on Aurora (Intel XPU)

Install: `/flare/FRAME-IDP/msinclair/software/Protenix` — XPU-patched (`protenix/utils/device.py`
auto-selects xpu → cuda → cpu). Do not edit the install directory. CUDA-only kernels
(cuequivariance, deepspeed, triattention) must be swapped for the `torch` kernels.

## Environment

```bash
REPO=/flare/FRAME-IDP/msinclair/software/Protenix
PROTENIX=$REPO/.venv/bin/protenix
# The venv was built on frameworks/2025.3.1; the default module is now 2026.1.0, which lacks that
# build's MKL. Prepend the old oneAPI lib dir (do NOT `module load frameworks/2025.3.1` — it fails
# on a missing intel-oneapi-runtime dependency).
export LD_LIBRARY_PATH=/opt/aurora/26.26.0/oneapi/2025.3/lib:${LD_LIBRARY_PATH:-}
export ZE_AFFINITY_MASK=4                   # ONE plain tile index; not "4.0"
export PROTENIX_ROOT_DIR=/flare/FRAME-IDP/msinclair/software/protenix_data
export PYTHONUNBUFFERED=1                   # get logs live
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy   # only for --use_msa true
```

`$PROTENIX_ROOT_DIR` already holds `checkpoint/` (`protenix_mini_esm_v0.5.0.pt`,
`protenix_base_default_v1.0.0.pt`, `protenix-v2.pt`, `esm2_t36_3B_UR50D.pt`), `common/` (CCD) and
`mmcif/` templates. **Only those three model names work offline**; others try to download.

## Example

```bash
# Minimal runner: scripts/run_protenix_example.sh <work_dir> [tile]
bash scripts/run_protenix_example.sh ./work 4

# Equivalent direct call (production settings)
cd /your/workdir        # ESM embeddings are cached to ./esm_embeddings in the CWD
$PROTENIX pred -i gb1.json -o out -s 101 \
  -n protenix_mini_esm_v0.5.0 -c 10 -p 200 -e 5 --use_msa false \
  --trimul_kernel torch --triatt_kernel torch
```

Input JSON (example: `scripts/example_gb1.json`):

```json
[{
  "name": "gb1",
  "sequences": [
    {"proteinChain": {"sequence": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", "count": 1}}
  ]
}]
```

Other entity types: `dnaSequence`, `rnaSequence`, `ligand` (`"CCD_ATP"` or a SMILES string), `ion`.
`count` sets copies; optional per-chain `pairedMsaPath` / `unpairedMsaPath` (absolute a3m paths),
`templatesPath`, `modifications`. Full spec: `$REPO/docs/infer_json_format.md`; examples in
`$REPO/examples/`. `protenix json -i x.cif -o dir` converts a PDB/mmCIF into input JSON.

## Arguments (`protenix pred`)

| flag | default | meaning |
|---|---|---|
| `-i/--input` | — | JSON file or directory |
| `-o/--out_dir` | ./output | output directory |
| `-s/--seeds` | 101 | comma list, e.g. `101,102` |
| `-c/--cycle` | 10 | Pairformer recycles |
| `-p/--step` | 200 | diffusion steps |
| `-e/--sample` | 5 | samples per seed |
| `-d/--dtype` | bf16 | `bf16` or `fp32` |
| `-n/--model_name` | protenix_base_default_v1.0.0 | one of the three local models above |
| `--use_msa` | **true** | `false` for offline/fast runs; `true` needs the proxy |
| `--trimul_kernel` | cuequivariance | **must be `torch`** on XPU |
| `--triatt_kernel` | cuequivariance | **must be `torch`** on XPU |
| `--use_template`, `--use_rna_msa` | false | need kalign/hmmer + databases |
| `--use_default_params` | — | load the model's recommended cycles/steps |
| `--use_seeds_in_json` | — | take seeds from the input JSON |
| `--need_atom_confidence` | false | adds full per-atom pLDDT / PAE arrays |

## Outputs

```
<out_dir>/<name>/seed_<seed>/predictions/
  <name>_sample_<k>.cif                       # one structure per sample
  <name>_summary_confidence_sample_<k>.json   # plddt (0-100), ptm, iptm, ranking_score,
                                              # chain_* / chain_pair_* (iptm, pae_mean, plddt),
                                              # has_clash, disorder
```

Rank samples by `ranking_score`. `iptm` is 0 for single chains.

## Must-know

- `OSError: libmkl_intel_lp64.so.2` means the `LD_LIBRARY_PATH` line above is missing.
- **Always pass `--trimul_kernel torch --triatt_kernel torch`** — the defaults are CUDA-only.
- `--use_msa` defaults to **true**, which triggers a network MSA search. Pass `--use_msa false`
  for offline runs.
- It writes `./esm_embeddings/` into the current directory, so `cd` to a scratch dir first and never
  run from inside the software dir.
- Model load dominates small jobs. Batch many inputs into one JSON or directory per call.
- If a run sits silent with no log output, kill it and rerun with `PYTHONUNBUFFERED=1`.
- Lower-level entry point (needs `PYTHONPATH=$REPO`):
  `$REPO/.venv/bin/python $REPO/runner/inference.py --model_name … --input_json_path … --dump_dir …
  --triangle_attention torch --triangle_multiplicative torch`.
