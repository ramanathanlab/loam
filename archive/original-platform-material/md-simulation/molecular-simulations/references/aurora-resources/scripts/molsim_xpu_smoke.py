#!/usr/bin/env python
"""Build (tleap, explicit OPC water) + run a tiny Simulator pipeline on Intel PVC via OpenCL.
Usage: AMBERHOME=... ZE_AFFINITY_MASK=0 python molsim_xpu_smoke.py <protein.pdb> <outdir>
"""
import sys, time
from pathlib import Path
from molecular_simulations.build import ExplicitSolvent
from molecular_simulations.simulate import Simulator

pdb, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
t0 = time.time()
ExplicitSolvent(path=out, pdb=pdb, padding=8.0).build()
print(f'build done in {time.time()-t0:.1f}s:', sorted(p.name for p in out.glob('system.*')))

t1 = time.time()
sim = Simulator(
    path=out,
    heat_steps=1000,          # tiny values for a smoke test only
    equil_steps=1000,
    prod_steps=2000,
    n_equil_cycles=1,
    eq_reporter_frequency=500,
    prod_reporter_frequency=500,
    platform='OpenCL',        # Intel GPU platform name in this OpenMM build (no 'SYCL'/'CUDA')
)
sim.run()
print(f'simulate done in {time.time()-t1:.1f}s')
