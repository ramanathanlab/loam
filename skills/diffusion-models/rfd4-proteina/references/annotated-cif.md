# Target-conditioned annotated CIF input

This packaged Jupiter reference adapts the installed checkout's `skills/rfd4-proteina-annotated-cif/SKILL.md` and the Loam input helper. It is accessible as `read_skill(name="rfd4-proteina", file="references/annotated-cif.md")`. The [protein-binder workflow](tasks/protein-binder.md) describes generation after validation.

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

Those additional upstream files are not packaged as paths within this Jupiter skill. Use their full checkout paths through `inspect_filesystem` or the campaign's permitted MCP reader, rather than inventing a `read_skill(file=...)` path.

The portable helper is bundled at `scripts/make_rfd4_input.py` and can also be read with `read_skill(name="rfd4-proteina", file="scripts/make_rfd4_input.py")`. Read this exact path before searching installed software. Never recursively scan `.venv`, caches, the entire software root, or campaign outputs to locate a documented helper.

## Native serialized representation and numbering

The builder **renumbers** preserved target residues in the annotated CIF. For source chain A author residues 286–490, the native input uses chain A residues 1–205. This does not change target coordinates or identity. Do not reject the native file because its author numbering differs from the source contig. The helper writes `<input>.report.json`, mapping source author residues to native input residues by preserved CA coordinates. The report also contains the actual source contig and hotspot mapping.

Configuration aliases are not serialized literally: `C_SEQ` is `_atom_site.condition_sequence_1_residue`, `C_HOT` is `_atom_site.condition_hotspot_1_atom`, and `C_CRDGRP` is `_atom_site.condition_CoordinateGroup_1_atom`. Hotspot values are True/False strings, sequence conditions are residue names, and coordinate-group value 1 marks constrained backbone atoms. Inspect these native columns and the helper report, rather than searching CIF text for alias names or a literal contig. The expandable binder sentinel encodes the requested length/range; it is not a generated backbone.
# Source coordinates and annotated coordinates are different inputs

For the verified EGFR example, `3NJP.cif` is the full downloaded source: receptor chain A has 614 observed protein residues. It is not a 205-residue cropped file. To rebuild an annotation from that source, use `--target-range 286-490` and source-author hotspot IDs within that range. Do not assert that the full source has only 205 residues, or change its range to 1-205 to satisfy that assertion: that selects a different receptor region.

The helper-produced `3NJP_two_loop_annotated.cif` is the normalized input: target chain A1-205 corresponds to source A286-490. A hotspot measured at normalized position k maps to source author k+285 for this verified crop. When changing hotspot conditioning, map back through the report and invoke the helper on the full source with source-author IDs. Inspect the actual input's chain and condition columns rather than guessing its coordinate system from its filename.
