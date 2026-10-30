#!/usr/bin/env python
"""Minimal OpenMM MD on Intel PVC (Aurora) via the OpenCL platform.
Usage: ZE_AFFINITY_MASK=0 python openmm_xpu_smoke.py <prmtop> <inpcrd> [nsteps] [outdir]
"""
import sys, time, os
from openmm.app import *
from openmm import *
from openmm.unit import *

prmtop_f, inpcrd_f = sys.argv[1], sys.argv[2]
nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
outdir = sys.argv[4] if len(sys.argv) > 4 else '.'
os.makedirs(outdir, exist_ok=True)

print('ZE_AFFINITY_MASK =', os.environ.get('ZE_AFFINITY_MASK'))
inpcrd = AmberInpcrdFile(inpcrd_f)
prmtop = AmberPrmtopFile(prmtop_f, periodicBoxVectors=inpcrd.boxVectors)
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
                             constraints=HBonds)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
platform = Platform.getPlatformByName('OpenCL')
props = {'Precision': 'mixed'}   # with ZE_AFFINITY_MASK set, the one visible tile is device 0
sim = Simulation(prmtop.topology, system, integrator, platform, props)
sim.context.setPositions(inpcrd.positions)
print('Platform:', sim.context.getPlatform().getName())
for k in ('DeviceName', 'OpenCLPlatformName', 'DeviceIndex', 'Precision'):
    print(f'  {k} = {platform.getPropertyValue(sim.context, k)}')
sim.minimizeEnergy(maxIterations=200)
sim.reporters.append(DCDReporter(os.path.join(outdir, 'traj.dcd'), 1000))
sim.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True,
                                       temperature=True, speed=True))
sim.step(100)  # warm-up (kernel JIT compile)
t0 = time.time()
sim.step(nsteps)
dt = time.time() - t0
ns_day = nsteps * 0.002e-3 / dt * 86400
print(f'{nsteps} steps in {dt:.2f} s -> {ns_day:.1f} ns/day ({system.getNumParticles()} atoms)')
