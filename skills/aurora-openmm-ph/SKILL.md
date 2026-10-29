---
name: aurora-openmm-ph
description: >-
  Set protonation states and run OpenMM on Aurora, and what is actually
  installed for pH work. Read this BEFORE promising a pH-dependent result:
  propka, pdb2pqr and pdbfixer are NOT in the MD venv, so a naive pKa workflow
  cannot run as written.
---

# Protonation and OpenMM on Aurora

## What is installed, verified

In `deepdrive-2026-venv`, probed directly (rc=0, 2026-10-29):

```json
{"openmm": true, "MDAnalysis": true, "deepdrivewe": true,
 "parmed": true, "mdtraj": true,
 "pdbfixer": false, "propka": false, "pdb2pqr": false}
```

So **there is no propka, no pdb2pqr and no pdbfixer in the MD venv.** A plan that
assigns pKa values with propka or prepares a structure with pdb2pqr cannot run
there. Say so rather than substituting a guess.

Where the missing pieces do exist:

- `pdbfixer` — `freebindcraft-2026-venv` (with OpenMM 8.6.1) and
  `proteina-complexa-2026-venv`
- nothing in the tree provides `propka` or `pdb2pqr`

## Setting protonation by pH with what you have

OpenMM itself can place hydrogens for a target pH:

```python
from openmm.app import PDBFile, Modeller, ForceField
pdb = PDBFile("complex.pdb")
ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")
mod = Modeller(pdb.topology, pdb.positions)
mod.addHydrogens(ff, pH=6.5)     # and a second system at pH=7.4
```

`Modeller.addHydrogens(forcefield, pH=7.0, variants=None, ...)` selects
protonation variants from the pH; `variants` lets you force a specific state per
residue (e.g. doubly-protonated HIS) when you want to test a mechanism rather
than a titration estimate. Read `inspect.getdoc(Modeller.addHydrogens)` on the
system — do not trust a remembered signature.

## Making a pH claim honestly

A pH-selective claim needs a DIFFERENTIAL on the SAME complex: build the system
twice, at 6.5 and at 7.4, with everything else identical, and report both plus
the difference. Fixed-state controls (a histidine held neutral, or mutated out)
separate "protonation did it" from "the interface is just weak". Note in the
result that `addHydrogens(pH=...)` is a heuristic variant selection, not a pKa
calculation — with propka absent, that is the strongest statement available.

## Running MD

`deepdrive-2026-venv` has OpenMM 8.6, MDAnalysis, parmed, mdtraj and
`deepdrivewe` for weighted-ensemble work. One process per tile; `gpus=1` per
batch row for a single-device simulation.
