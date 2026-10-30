---
name: uma
description: Run Meta FAIR's UMA universal machine-learning interatomic potential (fairchem-core, Intel XPU port) on ALCF Aurora GPU tiles for energies, forces, geometry optimizations and MD of molecules (omol), materials (omat), catalysts (oc20), MOFs (odac) or molecular crystals (omc) through the ASE FAIRChemCalculator. Use when a task needs fast near-DFT energies/forces or relaxations on Aurora (Intel Data Center GPU Max 1550 / XPU), including the XPU-native UMA Nitro fast path for uma-s-1p2p1 + omol.
---

# UMA (fairchem-xpu) on Aurora XPU

Source checkout: `/flare/FRAME-IDP/msinclair/software/fairchem-xpu` (fairchem-core fork, commit
`e47bd5fe` "Merge qualified Intel XPU and UMA Nitro acceleration"). Do not edit it.
XPU Nitro docs: `src/fairchem/core/models/uma/uma_nitro_xpu/README.md` in that checkout.

## Environment (verified)

```bash
export ZE_AFFINITY_MASK=9            # pin ONE tile; use only tiles you were assigned
export FAIRCHEM_CACHE_DIR=/flare/FRAME-IDP/msinclair/hf_cache   # where the UMA checkpoints are
export HF_HUB_OFFLINE=1              # use cached weights; no gated download needed
PY=/flare/FRAME-IDP/msinclair/software/fairchem-xpu/.venv/bin/python
# or: source /flare/FRAME-IDP/msinclair/software/fairchem-xpu/.venv/bin/activate
```

The venv runs on the Aurora frameworks stack: torch `2.13.0a0+gitcf30153` with XPU,
Python 3.12. `torch.xpu.device_count()` == 1 when the mask is set.

## Checkpoints (local, no download needed)

`$FAIRCHEM_CACHE_DIR/models--facebook--UMA/snapshots/ce3b2744ff57c0f49a010b763efcbfe9ba82e2e1/`
- `checkpoints/`: `uma-s-1p2p1.pt` (recommended), `uma-s-1p2.pt`, `uma-s-1p1.pt`, `uma-m-1p1.pt`,
  `uma-s-1p2p1-ft-r2scan.pt`, `uma-s-1p2p1-ft-d4rev.pt`
- `references/`: `iso_atom_elem_refs.yaml`, `form_elem_refs.yaml`

Note the layout: the repo sits at the root of the cache dir, NOT under `hf_cache/hub/`
(the `HF_HOME` env var points at the same dir, but fairchem passes
`cache_dir=FAIRCHEM_CACHE_DIR` to `hf_hub_download`). If you leave `FAIRCHEM_CACHE_DIR` unset,
fairchem looks in `~/.cache/fairchem` and tries to download from the gated `facebook/UMA` repo.
An identical copy of `uma-s-1p2p1.pt` (sha256 `b2673b85…5ce`) is at
`/lus/flare/projects/FRAME-IDP/dkastner/Projects/Reactra/assets/uma-s-1p2p1.pt`.

Registered names (`pretrained_mlip.available_models`): uma-s-1p2p1, uma-s-1p2, uma-s-1p1,
uma-m-1p1, uma-s-1p2p1-ft-d4rev, uma-s-1p2p1-ft-r2scan, plus eSEN/AllScAIP OMol/OC25/ODAC models
(the non-UMA ones are not cached here).

## Tasks

`task_name` picks the UMA head: `omol` (molecules, ωB97M-V; needs `atoms.info["charge"]` and
`atoms.info["spin"]` = spin multiplicity), `omat` (inorganic bulk, PBE/PBE+U), `oc20` (catalyst
surfaces + adsorbates), `odac` (MOFs), `omc` (molecular crystals). Only `omol` was tested here.

## Verified minimal code (general path, recommended)

```python
from ase.build import molecule
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2p1", inference_settings="batch", device="xpu")
calc = FAIRChemCalculator(predictor, task_name="omol")

atoms = molecule("CH3CH2OH")
atoms.info.update({"charge": 0, "spin": 1})
atoms.calc = calc
print(atoms.get_potential_energy(), atoms.get_forces())   # eV, eV/Å
LBFGS(atoms, trajectory="opt.traj").run(fmax=0.02, steps=200)
```

## Verified XPU UMA Nitro fast path (uma-s-1p2p1 + omol only)

```python
from huggingface_hub import hf_hub_download
from fairchem.core import FAIRChemCalculator
from fairchem.core.models.uma.uma_nitro_xpu import load_predict_unit

ckpt = hf_hub_download("facebook/UMA", "checkpoints/uma-s-1p2p1.pt",
                       cache_dir="/flare/FRAME-IDP/msinclair/hf_cache")   # local path, offline OK
predictor = load_predict_unit(ckpt)          # device="xpu" is hard-coded; FP32 forward
calc = FAIRChemCalculator(predictor, task_name="omol")
```
Nitro merges MoLE experts for ONE composition/charge/spin. A second molecule on the same
predictor raises `MergeMoleConsistencyError` → `UMANitroNotSupportedError: merge_mole=False`.
Build a fresh predictor per system (≈1.6 s once the process is warm). Good for long MD/opt of
a single fixed system; use `inference_settings="batch"` for many different molecules.
`load_predict_unit(path)` loads without reference energies (irrelevant for multi-atom omol).

Tested script: `scripts/uma_xpu_test.py` (H2O single point + ethanol LBFGS):
```bash
cd <run dir>; $PY /path/to/skills/qm/uma/scripts/uma_xpu_test.py uma-s-1p2p1 batch      # or nitro-xpu
```

## Test results (tile 9/10, 2026-10-29)

| inference_settings | H2O E (eV) | H2O max abs F (eV/Å) | first-call | ethanol LBFGS (28 steps, fmax 0.011) | wall |
|---|---|---|---|---|---|
| `batch` | -2079.861972 | **0.4827** | 1.6 s | 1.97 s, Ef = -4218.478884 | 13 s |
| `nitro-xpu` (uma_nitro_xpu) | -2079.861984 | **0.4827** | 26–48 s (Triton JIT) | 0.34 s, Ef = -4218.478931 | 61 s |
| `default` | -2079.861972 | 0.4254 (wrong) | 66 s (compile) | 1.78 s (after fallback) | 89 s |
| `high-precision` | -2079.861972 | 0.4254 (wrong) | 47 s (compile) | 1.50 s (after fallback) | 63 s |
| CPU `batch` reference | -2079.861972 | 0.4827 | – | – | – |

## Aurora/XPU gotchas

1. **Avoid `"default"`/`"high-precision"` (merge_mole + torch.compile) on XPU.** `default` logs
   "UMA Nitro cannot serve this request: nitro mode requires a CUDA device" and falls back to
   `multigpu` (umas_fast_gpu + compile). The compiled single point gave H2O forces that
   differ from CPU, `batch` and Nitro-XPU (max abs F 0.4254 vs 0.4827 eV/Å; energy matches). It
   also costs 45–65 s of compile time. Use `"batch"` (general path) or the `uma_nitro_xpu`
   loader. Compare against CPU for any new setting.
2. Compiled/merged paths auto-fall back when composition changes ("Compositions differ from merged
   model … Falling back"). The results after the fallback are fine but the compile time is wasted.
3. First Nitro-XPU call JIT-compiles Intel Triton kernels (~25–50 s). Later calls are ms-scale.
4. A harmless `UserWarning: Overriding a previously registered kernel … aten::topk … XPU`
   is printed at import.
5. If you use UMA together with OpenMM OpenCL in one process, initialize Triton/XPU (one UMA call)
   before creating the OpenCL Context (Nitro README). Not tested here.
6. Without `charge`/`spin` in `atoms.info`, omol uses defaults (0, 1). Set them explicitly.

## Troubleshooting

- `UnknownCheckpointError`/HF 401/gated errors → `FAIRCHEM_CACHE_DIR` is not set to the hf_cache
  dir, or `HF_HUB_OFFLINE` is not set while the node has no proxy. For real downloads:
  `export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy` plus an HF token with
  facebook/UMA access.
- `UMANitroNotSupportedError: merge_mole=False` → you reused a Nitro predictor on a new system.
- `torch.xpu.is_available()` False → wrong python (use the venv above) or no GPU allocation.
- Unverified: periodic systems/omat/oc20 on XPU, MD speed, multi-tile, `uma-m-1p1` on XPU.
