"""UMA (fairchem-xpu) minimal test: energy/forces + short geometry optimization on XPU.

Usage:
  export ZE_AFFINITY_MASK=9
  export FAIRCHEM_CACHE_DIR=/flare/FRAME-IDP/msinclair/hf_cache HF_HUB_OFFLINE=1
  /flare/FRAME-IDP/msinclair/software/fairchem-xpu/.venv/bin/python uma_xpu_test.py [model] [default|batch|high-precision|nitro-xpu]
"""
import sys
import time

import torch
from ase.build import molecule
from ase.optimize import LBFGS

from fairchem.core import FAIRChemCalculator, pretrained_mlip

model = sys.argv[1] if len(sys.argv) > 1 else "uma-s-1p2p1"
settings = sys.argv[2] if len(sys.argv) > 2 else "default"

print("torch", torch.__version__, "xpu available:", torch.xpu.is_available(),
      "device_count:", torch.xpu.device_count())
print("device:", torch.xpu.get_device_name(0))

def make_calc():
  t0 = time.time()
  if settings == "nitro-xpu":
    # XPU port of UMA Nitro: takes a checkpoint *path* (tested with uma-s-1p2p1 + omol).
    # It merges MoLE experts for ONE composition/charge/spin: build one predictor per system.
    from huggingface_hub import hf_hub_download
    from fairchem.core.models.uma.uma_nitro_xpu import load_predict_unit
    ckpt = hf_hub_download("facebook/UMA", f"checkpoints/{model}.pt",
                           cache_dir=__import__("os").environ.get("FAIRCHEM_CACHE_DIR"))
    predictor = load_predict_unit(ckpt)
  else:
    predictor = pretrained_mlip.get_predict_unit(model, inference_settings=settings, device="xpu")
  print(f"loaded {model} ({settings}) in {time.time() - t0:.1f}s; predictor={type(predictor).__name__}")
  return FAIRChemCalculator(predictor, task_name="omol")


calc = make_calc()

# --- single point: water, neutral singlet ---
atoms = molecule("H2O")
atoms.info.update({"charge": 0, "spin": 1})  # omol needs charge + spin multiplicity
atoms.calc = calc
t0 = time.time()
e = atoms.get_potential_energy()
f = atoms.get_forces()
print(f"H2O E = {e:.6f} eV  max|F| = {abs(f).max():.4f} eV/A  ({time.time() - t0:.2f}s incl. warmup)")

# --- geometry optimization: ethanol, slightly distorted ---
mol = molecule("CH3CH2OH")
mol.info.update({"charge": 0, "spin": 1})
mol.rattle(stdev=0.05, seed=0)
mol.calc = make_calc() if settings == "nitro-xpu" else calc
e0 = mol.get_potential_energy()
t0 = time.time()
opt = LBFGS(mol, logfile="-", trajectory="ethanol_opt.traj")
conv = opt.run(fmax=0.02, steps=100)
fmax = abs(mol.get_forces()).max()
print(f"ethanol opt converged={conv} steps={opt.get_number_of_steps()} "
      f"E0={e0:.6f} Ef={mol.get_potential_energy():.6f} eV fmax={fmax:.4f} "
      f"({time.time() - t0:.2f}s)")
mol.write("ethanol_opt.xyz")
print("PASS")
