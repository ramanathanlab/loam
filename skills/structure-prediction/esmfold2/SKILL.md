---
name: esmfold2
description: Predict protein structure from a single sequence (no MSA) with ESMFold2 (ESMC-6B + folding trunk + diffusion module), writing mmCIF with pLDDT/pTM. Use for fast single-sequence folds, or protein/DNA/ligand complexes via the Python API.
systems: [aurora, jupiter]
---

# ESMFold2

## Before you run it: read the reference file for this System

The install, the Python API's import path, device selection and launch command differ per System.
Read `references/<HPC_SYSTEM>.md` (for example
`read_skill(name="esmfold2", file="references/aurora.md")`) and follow it; it is authoritative over
this file for anything System-specific. This skill has reference files for **aurora** and
**jupiter**. It is **unsupported** on any other System: report the step as blocked rather than
adapting another System's instructions.

Software is found only through Site profile variables: the install is under `$LOAM_SOFTWARE_ROOT`.
Never edit the install directory; write outputs to the campaign workspace. Do not substitute
ESMFold v1.

## Method

ESMFold2 folds from a single sequence: no MSA, no templates. It needs the `biohub/ESMFold2` and
`biohub/ESMC-6B` (~25 GB) weights in the Hugging Face cache. Through the Python API it folds
complexes (protein, DNA, ligand entities) and draws several diffusion samples:

```python
spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
res = ESMFold2InputBuilder().fold(model, spi, num_loops=20, num_sampling_steps=100,
                                 num_diffusion_samples=1)
open("out.cif", "w").write(res.complex.to_mmcif())
print(float(res.plddt.mean()), res.ptm, res.iptm)
```

Production settings are 20 trunk loops and 100 diffusion steps. How to load `model` differs per
System; see the reference file.

## Outputs

- mmCIF via `result.complex.to_mmcif()`, all-atom coordinates.
- `result.plddt` per-residue on a **0-1** scale (not 0-100); >0.7 is confident.
- `result.ptm` global; `result.iptm` for interfaces (None/meaningless for a single chain).
- A random/synthetic sequence yields pLDDT ≈ 0.2 — a useful sanity contrast.

## Must-know

- **Cold start is expensive** (model load plus first-call kernel compilation). Fold many sequences
  in ONE process to amortize it; for a single one-off fold, boltz or protenix start faster.
- Memory: weights ~14 GB; long chains (L ≳ 1400) may OOM — reduce the trunk chunk size.
- Supply real campaign sequences and report their source; keep exact chain/domain definitions.
  Record model/checkpoint versions, sequence, sample settings, seed and failures.
- Inspect the actual output coordinates, chain/residue identities and confidence fields. A
  prediction is not a binder design, a measured affinity or experimental validation.
