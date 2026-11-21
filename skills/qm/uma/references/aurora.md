# UMA on Aurora (fairchem-xpu, Intel XPU)

Install: `$LOAM_SOFTWARE_ROOT/fairchem-xpu` (fairchem-core fork, commit `e47bd5fe`, "Merge qualified
Intel XPU and UMA Nitro acceleration"). Do not edit it. XPU Nitro docs:
`src/fairchem/core/models/uma/uma_nitro_xpu/README.md` in that checkout. The venv runs on the Aurora
frameworks stack (torch with XPU, Python 3.12); `torch.xpu.device_count()` is 1 when the mask is
set.

## Environment

```bash
PY=$LOAM_SOFTWARE_ROOT/fairchem-xpu/.venv/bin/python
export ZE_AFFINITY_MASK=9               # ONE tile; use only tiles you were assigned
export FAIRCHEM_CACHE_DIR=$LOAM_CACHE_ROOT   # where the UMA checkpoints live
export HF_HUB_OFFLINE=1                 # use cached weights; the facebook/UMA repo is gated
```

The device is `"xpu"`. Checkpoints (local, nothing to download) under
`$FAIRCHEM_CACHE_DIR/models--facebook--UMA/snapshots/ce3b2744ff57c0f49a010b763efcbfe9ba82e2e1/checkpoints/`:
`uma-s-1p2p1.pt` (recommended), `uma-s-1p2.pt`, `uma-s-1p1.pt`, `uma-m-1p1.pt`,
`uma-s-1p2p1-ft-r2scan.pt`, `uma-s-1p2p1-ft-d4rev.pt`; `references/` beside it holds
`iso_atom_elem_refs.yaml` and `form_elem_refs.yaml`. The repo sits at the **root** of the cache dir,
not under `hub/`: fairchem passes `cache_dir=FAIRCHEM_CACHE_DIR` to `hf_hub_download`. Leaving
`FAIRCHEM_CACHE_DIR` unset sends fairchem to `~/.cache/fairchem` and a gated download.

## Example

```bash
# Minimal runner: scripts/aurora/uma_xpu_test.py <model> <inference_settings>
# (H2O single point + ethanol LBFGS)
cd <a run dir>
$PY $HPC_LOAM_HOME/skills/qm/uma/scripts/aurora/uma_xpu_test.py uma-s-1p2p1 batch   # or: nitro-xpu
```

General path: the `SKILL.md` method with `device="xpu"` and `inference_settings="batch"`.

Nitro fast path (`uma-s-1p2p1` + `omol` only; best for long MD/opt of ONE fixed system):

```python
import os
from huggingface_hub import hf_hub_download
from fairchem.core import FAIRChemCalculator
from fairchem.core.models.uma.uma_nitro_xpu import load_predict_unit

ckpt = hf_hub_download("facebook/UMA", "checkpoints/uma-s-1p2p1.pt",
                       cache_dir=os.environ["FAIRCHEM_CACHE_DIR"])   # local, offline OK
predictor = load_predict_unit(ckpt)        # device="xpu" is hard-coded; FP32 forward
calc = FAIRChemCalculator(predictor, task_name="omol")
```

`load_predict_unit(path)` loads without reference energies (irrelevant for multi-atom `omol`).

## Recorded results (tiles 9/10, 2026-10-29)

| inference_settings | H2O E (eV) | H2O max abs F (eV/Å) | first call | ethanol LBFGS (28 steps, fmax 0.011) | wall |
|---|---|---|---|---|---|
| `batch` | -2079.861972 | **0.4827** | 1.6 s | 1.97 s, Ef = -4218.478884 | 13 s |
| `nitro-xpu` (uma_nitro_xpu) | -2079.861984 | **0.4827** | 26–48 s (Triton JIT) | 0.34 s, Ef = -4218.478931 | 61 s |
| `default` | -2079.861972 | 0.4254 (wrong) | 66 s (compile) | 1.78 s (after fallback) | 89 s |
| `high-precision` | -2079.861972 | 0.4254 (wrong) | 47 s (compile) | 1.50 s (after fallback) | 63 s |
| CPU `batch` reference | -2079.861972 | 0.4827 | – | – | – |

## Must-know

- **Avoid `inference_settings="default"` and `"high-precision"` on XPU.** They log "UMA Nitro cannot
  serve this request: nitro mode requires a CUDA device", fall back to a compiled `multigpu` path,
  and that compiled single point returns **forces that differ from CPU, `batch` and Nitro-XPU**
  (0.4254 vs 0.4827 eV/Å max abs F on H₂O; energies match). It also costs 45–65 s of compile time.
  Use `"batch"` or the `uma_nitro_xpu` loader, and compare against CPU for any new setting.
- **Nitro merges MoLE experts for ONE composition/charge/spin.** A second, different molecule on the
  same predictor raises `MergeMoleConsistencyError` → `UMANitroNotSupportedError: merge_mole=False`.
  Build a fresh predictor per system (~1.6 s warm), or use `inference_settings="batch"` for many
  different molecules.
- The first Nitro-XPU call JIT-compiles Intel Triton kernels (~25–50 s); later calls are ms-scale.
- Compiled/merged paths auto-fall back when the composition changes ("Compositions differ from
  merged model … Falling back"). Results after the fallback are fine, but the compile time is wasted.
- On Aurora, checkpoint errors (see `SKILL.md`) mean `FAIRCHEM_CACHE_DIR` is wrong or
  `HF_HUB_OFFLINE` is unset on a node with no proxy. For real downloads you also need an HF token
  with `facebook/UMA` access plus `export http_proxy=http://proxy.alcf.anl.gov:3128
  https_proxy=$http_proxy`.
- `torch.xpu.is_available()` False means the wrong python (use the venv above) or no GPU allocation.
- If you use UMA together with OpenMM OpenCL in one process, initialize Triton/XPU (one UMA call)
  **before** creating the OpenCL Context.
- A harmless `UserWarning: Overriding a previously registered kernel … aten::topk … XPU` prints at
  import.
- Unverified on XPU: periodic systems (omat/oc20), MD speed, multi-tile, `uma-m-1p1`.

## Resource index

- `read_skill(name="uma", file="SKILL.md")` — the System-neutral method
- `read_skill(name="uma", file="references/aurora.md")` — this file
- `read_skill(name="uma", file="scripts/aurora/uma_xpu_test.py")` — the minimal runner
