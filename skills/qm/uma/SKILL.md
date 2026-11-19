---
name: uma
description: Run Meta FAIR's UMA universal machine-learning interatomic potential (fairchem-core, Intel XPU port) on Aurora GPU tiles for near-DFT energies, forces, geometry optimizations and MD of molecules, materials, catalysts, MOFs or molecular crystals through the ASE FAIRChemCalculator. Use for fast near-DFT energies/forces or relaxations on Aurora.
---

# UMA (fairchem-xpu) on Aurora (Intel XPU)

Install: `/flare/FRAME-IDP/msinclair/software/fairchem-xpu` (fairchem-core fork, commit `e47bd5fe`,
"Merge qualified Intel XPU and UMA Nitro acceleration"). Do not edit it. XPU Nitro docs:
`src/fairchem/core/models/uma/uma_nitro_xpu/README.md` in that checkout.

## Environment

```bash
PY=/flare/FRAME-IDP/msinclair/software/fairchem-xpu/.venv/bin/python
export ZE_AFFINITY_MASK=9            # ONE tile; use only tiles you were assigned
export FAIRCHEM_CACHE_DIR=/flare/FRAME-IDP/msinclair/hf_cache   # where the UMA checkpoints live
export HF_HUB_OFFLINE=1              # use cached weights; the facebook/UMA repo is gated
```

Checkpoints (local, nothing to download) under
`$FAIRCHEM_CACHE_DIR/models--facebook--UMA/snapshots/ce3b2744…/checkpoints/`:
`uma-s-1p2p1.pt` (recommended), `uma-s-1p2.pt`, `uma-s-1p1.pt`, `uma-m-1p1.pt`,
`uma-s-1p2p1-ft-r2scan.pt`, `uma-s-1p2p1-ft-d4rev.pt`. Note the repo sits at the **root** of the
cache dir, not under `hub/`. Leaving `FAIRCHEM_CACHE_DIR` unset sends fairchem to
`~/.cache/fairchem` and a gated download.

`task_name` picks the UMA head: `omol` (molecules, ωB97M-V; needs `atoms.info["charge"]` and
`atoms.info["spin"]` = multiplicity), `omat` (inorganic bulk), `oc20` (catalyst surfaces + adsorbates),
`odac` (MOFs), `omc` (molecular crystals).

## Example

```bash
# Minimal runner: scripts/uma_xpu_test.py <model> <inference_settings>
cd <a run dir>
$PY <this skill>/scripts/uma_xpu_test.py uma-s-1p2p1 batch      # or: nitro-xpu
```

General path (recommended — handles many different molecules):

```python
from ase.build import molecule
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2p1", inference_settings="batch", device="xpu")   # "batch", NOT "default"
calc = FAIRChemCalculator(predictor, task_name="omol")

atoms = molecule("CH3CH2OH")
atoms.info.update({"charge": 0, "spin": 1})                    # set these explicitly
atoms.calc = calc
print(atoms.get_potential_energy(), atoms.get_forces())        # eV, eV/Å
LBFGS(atoms, trajectory="opt.traj").run(fmax=0.02, steps=200)
```

Nitro fast path (`uma-s-1p2p1` + `omol` only; best for long MD/opt of ONE fixed system):

```python
from huggingface_hub import hf_hub_download
from fairchem.core import FAIRChemCalculator
from fairchem.core.models.uma.uma_nitro_xpu import load_predict_unit

ckpt = hf_hub_download("facebook/UMA", "checkpoints/uma-s-1p2p1.pt",
                       cache_dir="/flare/FRAME-IDP/msinclair/hf_cache")   # local, offline OK
predictor = load_predict_unit(ckpt)        # device="xpu" is hard-coded; FP32 forward
calc = FAIRChemCalculator(predictor, task_name="omol")
```

## Arguments

| argument | value | meaning |
|---|---|---|
| model name | `uma-s-1p2p1` | also uma-s-1p2, uma-s-1p1, uma-m-1p1, and the ft-d4rev / ft-r2scan variants |
| `inference_settings` | **`batch`** | use this on XPU; see Must-know for why not `default`/`high-precision` |
| `device` | `xpu` | |
| `task_name` | `omol` | or omat / oc20 / odac / omc |
| `atoms.info["charge"]`, `["spin"]` | 0, 1 | required for `omol`; defaults are silently applied |

## Outputs

Standard ASE results: `atoms.get_potential_energy()` in **eV**, `atoms.get_forces()` in **eV/Å**,
and an optimizer trajectory file (`opt.traj`) when one is requested. Relaxed geometry is in `atoms`
after `.run()`.

## Must-know

- **Avoid `inference_settings="default"` and `"high-precision"` on XPU.** They log "UMA Nitro cannot
  serve this request: nitro mode requires a CUDA device", fall back to a compiled `multigpu` path,
  and that compiled single point returns **forces that differ from CPU, `batch` and Nitro-XPU**
  (0.4254 vs 0.4827 eV/Å max abs F on H₂O; energies match). It also costs 45-65 s of compile time.
  Use `"batch"` or the `uma_nitro_xpu` loader, and compare against CPU for any new setting.
- **Nitro merges MoLE experts for ONE composition/charge/spin.** A second, different molecule on the
  same predictor raises `MergeMoleConsistencyError` → `UMANitroNotSupportedError: merge_mole=False`.
  Build a fresh predictor per system (~1.6 s warm), or use `inference_settings="batch"` for many
  different molecules.
- The first Nitro-XPU call JIT-compiles Intel Triton kernels (~25-50 s); later calls are ms-scale.
- Compiled/merged paths auto-fall back when the composition changes ("Compositions differ from
  merged model … Falling back"). Results after the fallback are fine, but the compile time is wasted.
- Without `charge`/`spin` in `atoms.info`, `omol` silently uses (0, 1). Set them explicitly.
- `UnknownCheckpointError` / HF 401 / gated errors mean `FAIRCHEM_CACHE_DIR` is wrong or
  `HF_HUB_OFFLINE` is unset on a node with no proxy. For real downloads you also need an HF token
  with `facebook/UMA` access plus `export http_proxy=http://proxy.alcf.anl.gov:3128
  https_proxy=$http_proxy`.
- If you use UMA together with OpenMM OpenCL in one process, initialize Triton/XPU (one UMA call)
  **before** creating the OpenCL Context.
- A harmless `UserWarning: Overriding a previously registered kernel … aten::topk … XPU` prints at
  import.
- Unverified on XPU: periodic systems (omat/oc20), MD speed, multi-tile, `uma-m-1p1`.
