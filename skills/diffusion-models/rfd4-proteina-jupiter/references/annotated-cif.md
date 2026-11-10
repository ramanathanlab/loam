# Target-conditioned annotated CIF input

This packaged Jupiter reference adapts the installed checkout's `skills/rfd4-proteina-annotated-cif/SKILL.md` and the Loam input helper. It is accessible as `skill_view(name="rfd4-proteina-jupiter", file_path="references/annotated-cif.md")`. The [protein-binder workflow](tasks/protein-binder.md) describes generation after validation.

## Validate identity before constructing inputs

1. Retrieve the authoritative target sequence and experimental or predicted coordinates at runtime; record source URL/accession, file checksum, assembly/model, chain and numbering convention.
2. Verify the selected protein chain maps to the requested target sequence and retained domain. Inspect actual residue coverage, unresolved residues, ligands/glycans, alternate conformers and mutations. Do not infer target identity from a search hit or a remembered PDB identifier.
3. Choose a justified crop and hotspots that exist in the selected target chain and use the numbering expected by the parser. Do not confuse author numbering, label numbering, and UniProt positions. Include the mapping in provenance.
4. Preserve true coordinates and native target identity; the generated binder is not a copied target segment. Do not hand-author fake atoms, placeholder coordinates, or binder sequences.

Inputs must have recognized `.cif`/`.bcif` suffixes (with supported compression when applicable). Naming downloaded CIF bytes `rcsb_ID_cif` without a `.cif` extension makes the upstream loader reject them. Do not merely rename PDB or compressed bytes as plain CIF; convert/decompress using supported IO and audit chemistry if needed.

## Existing Loam input helper

The helper uses RFD4's `write_condition_spec_cif` builder, not a hand-written CIF. Define the following from the verified campaign input: `TARGET`, `TARGET_CHAIN`, `TARGET_RANGE`, `HOTSPOTS`, `BINDER_LEN`, and `INPUT`. Use absolute paths for `TARGET` and `INPUT`:

```bash
source "$HPC_LOAM_HOME/scripts/jupiter_env.sh"
REPO="$LOAM_SOFTWARE_ROOT/RFD4-Proteina"
source "$REPO/.venv/bin/activate"
export CLUSTER=local WANDB_MODE=disabled OMP_NUM_THREADS=4
cd "$REPO"
"$REPO/.venv/bin/python" \
  "$HPC_LOAM_HOME/skills/diffusion-models/rfd4-proteina/scripts/make_rfd4_input.py" \
  --target "$TARGET" \
  --target-chain "$TARGET_CHAIN" \
  --target-range "$TARGET_RANGE" \
  --hotspots "$HOTSPOTS" \
  --binder-len "$BINDER_LEN" \
  --out "$INPUT" \
  --example-id campaign_binder
```

The helper constructs the binder contig plus target selection, uses one rigid internal-coordinate group for resolved target backbone atoms (`C_CRDGRP`), retains target sequence (`C_SEQ`), and marks selected target hotspot atoms (`C_HOT`; CA by default). It imports `rfproteina.conditions` to register annotations and validates before writing. Do not use `--no-validate` to bypass a failed input gate. A segment matching no atoms requires correcting the chain/range mapping, not fabricating coordinates.

## Inspect the saved input

Read it back and verify chain/residue selections, binder-length specification, target sequence and coordinates, hotspots, and conditioning counts against the intended experiment. An annotated input is not a generated binder or a scientific result.

For deeper input authoring, read these exact paths under `$LOAM_SOFTWARE_ROOT/RFD4-Proteina/` through MCP:

- `skills/rfd4-proteina-annotated-cif/references/python-authoring.md`
- `skills/rfd4-proteina-annotated-cif/references/annotators-and-selections.md`
- `skills/rfd4-proteina-annotated-cif/references/validation.md`
- `docs/reference/annotations.md`
- `docs/concepts/conditioning.md`

Those additional upstream files are not packaged as paths within this Jupiter skill. Use their full checkout paths through `inspect_filesystem` or the campaign's permitted MCP reader, rather than inventing a `skill_view(file_path=...)` path.
