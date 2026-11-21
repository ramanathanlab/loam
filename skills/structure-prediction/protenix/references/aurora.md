# Protenix on Aurora (Intel XPU)

Install: `$LOAM_SOFTWARE_ROOT/Protenix` — XPU-patched (`protenix/utils/device.py` auto-selects
xpu → cuda → cpu). Do not edit the install directory. CUDA-only kernels (cuequivariance, deepspeed,
triattention) must be swapped for the `torch` kernels.

## Environment

```bash
REPO=$LOAM_SOFTWARE_ROOT/Protenix
PROTENIX=$REPO/.venv/bin/protenix
# The venv was built on frameworks/2025.3.1; the default module is now 2026.1.0, which lacks that
# build's MKL. Prepend the old oneAPI lib dir (do NOT `module load frameworks/2025.3.1` — it fails
# on a missing intel-oneapi-runtime dependency).
export LD_LIBRARY_PATH=/opt/aurora/26.26.0/oneapi/2025.3/lib:${LD_LIBRARY_PATH:-}
export ZE_AFFINITY_MASK=4                   # ONE plain tile index; not "4.0"
export PROTENIX_ROOT_DIR=$LOAM_SOFTWARE_ROOT/protenix_data
export PYTHONUNBUFFERED=1                   # get logs live
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy   # only for --use_msa true
```

`$PROTENIX_ROOT_DIR` already holds `checkpoint/` (`protenix_mini_esm_v0.5.0.pt`,
`protenix_base_default_v1.0.0.pt`, `protenix-v2.pt`, `esm2_t36_3B_UR50D.pt`), `common/` (CCD) and
`mmcif/` templates. **Only those three model names work offline**; others try to download.

## Example

```bash
# Minimal runner: scripts/aurora/run_protenix_example.sh <work_dir> [tile]
bash scripts/aurora/run_protenix_example.sh ./work 4

# Equivalent direct call (production settings)
cd /your/workdir        # ESM embeddings are cached to ./esm_embeddings in the CWD
$PROTENIX pred -i gb1.json -o out -s 101 \
  -n protenix_mini_esm_v0.5.0 -c 10 -p 200 -e 5 --use_msa false \
  --trimul_kernel torch --triatt_kernel torch
```

## Aurora-specific arguments

| flag | value | meaning |
|---|---|---|
| `-n/--model_name` | one of the three local models above | others try to download |
| `--use_msa true` | — | needs the proxy env vars above |
| `--trimul_kernel` | **must be `torch`** on XPU | default is CUDA-only |
| `--triatt_kernel` | **must be `torch`** on XPU | default is CUDA-only |

## Must-know

- `OSError: libmkl_intel_lp64.so.2` means the `LD_LIBRARY_PATH` line above is missing.
- **Always pass `--trimul_kernel torch --triatt_kernel torch`** — the defaults are CUDA-only.
- If a run sits silent with no log output, kill it and rerun with `PYTHONUNBUFFERED=1`.
- Lower-level entry point (needs `PYTHONPATH=$REPO`):
  `$REPO/.venv/bin/python $REPO/runner/inference.py --model_name … --input_json_path … --dump_dir …
  --triangle_attention torch --triangle_multiplicative torch`.

## Resource index

- `read_skill(name="protenix", file="scripts/aurora/run_protenix_example.sh")` — minimal one-tile runner
- `read_skill(name="protenix", file="examples/example_gb1.json")` — example input (any System)
