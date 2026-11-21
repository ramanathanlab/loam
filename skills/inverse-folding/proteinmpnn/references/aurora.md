# ProteinMPNN on Aurora (Intel XPU)

Install: `$LOAM_SOFTWARE_ROOT/ProteinMPNN` (do not edit). Entry point `protein_mpnn_run.py`; JSONL
helpers in `helper_scripts/`; example inputs in `inputs/` (`PDB_complexes/pdbs/{3HTN,4YOW}.pdb`,
`PDB_monomers/pdbs/{5L33,6MRR}.pdb`, `PDB_homooligomers/pdbs/{6EHB,4GYT}.pdb`).

## Environment

```bash
MPNN=$LOAM_SOFTWARE_ROOT/ProteinMPNN
PY=$MPNN/.venv/bin/python                   # uv venv on frameworks/2026.1.0 (torch 2.13, system site-packages)
SKILL=$HPC_LOAM_HOME/skills/inverse-folding/proteinmpnn
export ZE_AFFINITY_MASK=8                   # ONE tile; it then appears as xpu:0
export PYTHONPATH=$SKILL/scripts/aurora/ipex_shim${PYTHONPATH:+:$PYTHONPATH}   # REQUIRED, see below
```

## Example

```bash
# Minimal runner: scripts/aurora/run_proteinmpnn.sh <pdb> <out_dir> [tile] [extra args...]
bash $SKILL/scripts/aurora/run_proteinmpnn.sh $MPNN/inputs/PDB_monomers/pdbs/5L33.pdb out 8
```

Everything else is the SKILL.md example set, with `$MPNN` and `$PY` as above.

## Must-know

- **The IPEX shim is required.** `protein_mpnn_run.py` and `protein_mpnn_utils.py` do
  `import intel_extension_for_pytorch as ipex` but never use it, and IPEX is absent from the 2026
  frameworks stack. `scripts/aurora/ipex_shim/` is an empty stand-in package; without it on
  `PYTHONPATH` the run fails with `ModuleNotFoundError`. Native `torch.xpu` does the real work.
- **`--device` is ignored.** The device is hard-coded to `xpu:0` if available, else CPU — so it
  **silently falls back to CPU**. Check the log line `model on device: xpu:0`. Pick the tile with
  `ZE_AFFINITY_MASK` (a plain integer; `8.0` hides the device). With the mask set, the tile is
  device 0 — don't pass `DeviceIndex`-style selection.
- `--batch_size` can go well above 1: the 64 GB tiles have room.
- mmCIF conversion needs gemmi **in this venv**: `"$PY" -c "import gemmi; print(gemmi.__version__)"`.
  During a campaign, report a missing dependency rather than installing into the shared software.
- Harmless noise: the `aten::topk` kernel-override warning and a `torch.cross` deprecation warning
  (CA model).
- Under Loam `run_batch`, reference the runner by its worker-visible copy (as `$SKILL` above) — the
  Aurora worker cannot see a local skill directory.

## Resource index

- `read_skill(name="proteinmpnn", file="scripts/aurora/run_proteinmpnn.sh")` — minimal design runner
- `read_skill(name="proteinmpnn", file="scripts/aurora/run_tests.sh")` — the verified suite:
  multi-chain, fixed positions, homooligomer, soluble, CA-only and score-only, with wall times
- `read_skill(name="proteinmpnn", file="scripts/aurora/ipex_shim/intel_extension_for_pytorch/__init__.py")`
  — the empty IPEX stand-in
