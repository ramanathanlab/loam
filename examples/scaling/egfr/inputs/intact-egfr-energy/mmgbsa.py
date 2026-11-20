"""Minimized single-frame MM/GBSA diagnostic, with explicit chemistry records."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, GBn2, HBonds, NoCutoff, PDBFile, PDBxFile, Simulation

row = json.loads(Path(sys.argv[1]).read_text())
assert not row.get('binder_disulfides'), 'This reference handles target disulfides; adapt and verify chemistry for an engineered disulfide binder'
root = Path(row['root'])
amber = Path(row['software']) / 'ambertools'
out = root / row.get('mmgbsa_directory', 'mmgbsa')
out.mkdir(exist_ok=False)
os.environ['AMBERHOME'] = str(amber)
os.environ['PATH'] = str(amber / 'bin') + os.pathsep + os.environ.get('PATH', '')
os.environ['OMP_NUM_THREADS'] = '4'
fold = json.loads((root / 'validation-report.json').read_text())
best = max(fold['predictions'], key=lambda p: p['metrics']['aggregate_score'][0])
sequence = json.loads((root / 'sequence-report.json').read_text())
pdb = PDBxFile(best['cif'])
chains = list(pdb.topology.chains())
assert [len(list(c.residues())) for c in chains] == [sequence['target_residues'], sequence['binder_residues']]
annotation_path = Path(row['target_annotation'])
annotation = json.loads(annotation_path.read_text())
assert annotation['primaryAccession'] == 'P00533'
start, end = sequence['canonical_interval']
assert annotation['sequence']['value'][start-1:end] == ''.join(
    record.splitlines()[1] for record in (root / 'cofold.fasta').read_text().split('>')
    if record.startswith('protein|name=human_EGFR'))
residues = list(chains[0].residues())
disulfides = []
initial_disulfide_distances = []
positions = np.array(pdb.positions.value_in_unit(unit.angstrom))
for feature in annotation.get('features', []):
    if feature['type'] != 'Disulfide bond':
        continue
    first, second = feature['location']['start']['value'], feature['location']['end']['value']
    assert (start <= first <= end) == (start <= second <= end), 'Crop cuts an annotated target disulfide'
    if not start <= first <= end:
        continue
    a, b = residues[first-start], residues[second-start]
    assert a.name == b.name == 'CYS'
    sg_a = next(atom for atom in a.atoms() if atom.name == 'SG')
    sg_b = next(atom for atom in b.atoms() if atom.name == 'SG')
    distance = float(np.linalg.norm(positions[sg_a.index] - positions[sg_b.index]))
    initial_disulfide_distances.append(distance)
    assert distance < 4.0, f'Predicted geometry is incompatible with annotated disulfide {first}-{second}: {distance:.2f} Angstrom'
    pdb.topology.addBond(sg_a, sg_b)
    disulfides.append([a.index + 1, b.index + 1])
    a.name = b.name = 'CYX'
assert disulfides, 'An oxidized EGFR ectodomain needs its annotated disulfide bonds'
with (out / 'input.pdb').open('w') as stream:
    PDBFile.writeFile(pdb.topology, pdb.positions, stream, keepIds=True)
leap = ['source leaprc.protein.ff14SB', 'set default PBRadii mbondi3',
        f'complex = loadpdb {out / "input.pdb"}']
# PDBFile writes these added bonds as CONECT records, which this tleap
# version reads. Adding explicit bond commands as well duplicates them.
leap += [f'saveamberparm complex {out / "complex.prmtop"} {out / "complex.inpcrd"}', 'quit']
(out / 'leap.in').write_text('\n'.join(leap) + '\n')


def run(name, command, timeout=120):
    log = out / (name + '.log')
    log.touch(mode=0o600)
    with log.open('w') as stream:
        result = subprocess.run(list(map(str, command)), cwd=out, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    assert result.returncode == 0, (name, result.returncode, str(log))


run('tleap', [amber / 'bin/tleap', '-f', out / 'leap.in'])
assert (out / 'complex.prmtop').stat().st_size > 0
top = AmberPrmtopFile(str(out / 'complex.prmtop'))
# Amber can discard chain labels. Verify chemical separation using residue
# indices before decomposing the receptor/binder energy.
receptor_count = sequence['target_residues']
cross_boundary_bonds = [(a.index, b.index) for a, b in top.topology.bonds()
                        if (a.residue.index < receptor_count) != (b.residue.index < receptor_count)]
assert not cross_boundary_bonds, 'Target and binder are covalently joined in the Amber topology'
prepared_disulfides = sorted(sorted([a.residue.index+1, b.residue.index+1])
                              for a, b in top.topology.bonds() if a.name == b.name == 'SG')
assert prepared_disulfides == sorted(sorted(pair) for pair in disulfides), 'Amber topology did not preserve the declared disulfide graph'
coords = AmberInpcrdFile(str(out / 'complex.inpcrd'))
system = top.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds, implicitSolvent=GBn2,
                          implicitSolventSaltConc=0.15 * unit.molar)
integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
sim = Simulation(top.topology, system, integrator, Platform.getPlatformByName('CUDA'), {'Precision': 'mixed'})
sim.context.setPositions(coords.positions)
before = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
sim.minimizeEnergy(tolerance=10 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=500)
state = sim.context.getState(getEnergy=True, getPositions=True)
after = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
assert math.isfinite(before) and math.isfinite(after)
assert np.isfinite(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).all()
with (out / 'minimized.pdb').open('w') as stream:
    PDBFile.writeFile(top.topology, state.getPositions(), stream)
report = {'purpose': row.get('purpose', 'hypothesis-scoped EGFR experiment'),
          'method': 'minimized single-frame MM/GBSA; entropy omitted; not an ensemble or measured affinity',
          'source_complex': best['cif'], 'force_field': 'ff14SB', 'solvent_model': 'GBn2/igb=8',
          'salt_molar': 0.15, 'radii': 'mbondi3', 'disulfides_from_uniprot': disulfides,
          'target_annotation': str(annotation_path), 'initial_disulfide_distances_angstrom': initial_disulfide_distances,
          'target_chemistry': 'oxidized, unglycosylated canonical EGFR ectodomain',
          'minimization_max_iterations': 500, 'energy_before_kcal_mol': before,
          'energy_after_kcal_mol': after, 'platform': sim.context.getPlatform().getName(),
          'receptor_residues': sequence['target_residues'], 'binder_residues': sequence['binder_residues'],
          'target_binder_covalent_bonds': len(cross_boundary_bonds),
          'partition_by_residue_ranges': {'receptor': [1, receptor_count],
                                         'binder': [receptor_count+1, receptor_count+sequence['binder_residues']]},
          'minimization_completed': True, 'mmgbsa_completed': False}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
del sim, integrator
run('ante-mmpbsa', [amber / 'bin/ante-MMPBSA.py', '-p', out / 'complex.prmtop',
                   '-r', out / 'receptor.prmtop', '-l', out / 'binder.prmtop',
                   '-n', f':1-{sequence["target_residues"]}'])
(out / 'cpptraj.in').write_text(f'parm {out / "complex.prmtop"}\ntrajin {out / "minimized.pdb"}\n'
                               f'trajout {out / "minimized.nc"} netcdf\nrun\n')
run('cpptraj', [amber / 'bin/cpptraj', '-i', out / 'cpptraj.in'])
(out / 'mmpbsa.in').write_text('&general\n startframe=1, endframe=1, interval=1, verbose=1,\n/\n'
                              '&gb\n igb=8, saltcon=0.15,\n/\n')
run('mmpbsa', [amber / 'bin/MMPBSA.py', '-O', '-i', out / 'mmpbsa.in',
              '-cp', out / 'complex.prmtop', '-rp', out / 'receptor.prmtop', '-lp', out / 'binder.prmtop',
              '-y', out / 'minimized.nc', '-o', out / 'FINAL_RESULTS_MMPBSA.dat'], timeout=180)
text = (out / 'FINAL_RESULTS_MMPBSA.dat').read_text()
match = re.search(r'^DELTA TOTAL\s+([-+\d.eE]+)', text, re.M)
assert match, 'MM/GBSA total is absent from the actual results file'
estimate = float(match[1])
assert math.isfinite(estimate)
report.update(mmgbsa_completed=True, static_delta_total_kcal_mol=estimate, frames=1,
              result_file=str(out / 'FINAL_RESULTS_MMPBSA.dat'))
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
