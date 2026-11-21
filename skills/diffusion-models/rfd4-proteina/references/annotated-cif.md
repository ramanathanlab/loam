# Target-conditioned annotated CIF input

RFD4 inputs are **annotated CIFs**: the target structure plus AtomWorks `StandardAnnotation`
columns and the required `pdbx_PDB_model_num` field. A generic PDB encodes no fixed atoms, sequence
constraints or ligand conditioning and will fail. The commonest failure is a hand-written file with
no `pdbx_PDB_model_num` column and/or placeholder zero coordinates, which makes the loader die with
`biotite.DeserializationError: Failed to deserialize category 'atom_site'` and produce zero
backbones. So do not hand-author the CIF: use this skill's input helper for the current System
(`scripts/<HPC_SYSTEM>/make_rfd4_input.py`), which drives RFD4's own `write_condition_spec_cif`
builder. Take the interpreter and environment for it from `references/<HPC_SYSTEM>.md`.

Read this exact path before searching installed software. Never recursively scan `.venv`, caches,
the whole software root or campaign outputs to locate a documented helper.

## Validate identity before constructing inputs

1. Retrieve the authoritative target sequence and experimental or predicted coordinates at runtime;
   record source URL/accession, file checksum, assembly/model, chain and numbering convention.
2. Verify the selected protein chain maps to the requested target sequence and retained domain.
   Inspect actual residue coverage, unresolved residues, ligands/glycans, alternate conformers and
   mutations. Do not infer target identity from a search hit or a remembered PDB identifier.
3. Choose a justified crop and hotspots that exist in the selected target chain, in the numbering
   the parser expects. Do not confuse author numbering, label numbering and UniProt positions.
   Include the mapping in provenance.
4. Preserve true coordinates and native target identity; the generated binder is not a copied
   target segment. Never hand-author fake atoms, placeholder coordinates or binder sequences.

Inputs must have recognized `.cif`/`.bcif` suffixes (with supported compression where applicable).
Naming downloaded CIF bytes `rcsb_ID_cif` without a `.cif` extension makes the loader reject them.
Do not merely rename PDB or compressed bytes as plain CIF; convert or decompress with supported IO
and audit the chemistry if needed.

## What the helper builds

Define `TARGET`, `TARGET_CHAIN`, `TARGET_RANGE`, `HOTSPOTS`, `BINDER_LEN` and `INPUT` from the
verified campaign input, with absolute paths for `TARGET` and `INPUT`. Hotspots accept
comma-separated author residue IDs or inclusive ranges such as `360-386`.

The helper constructs the binder contig plus the target selection, uses one rigid
internal-coordinate group for resolved target backbone atoms (`C_CRDGRP`), retains the target
sequence (`C_SEQ`) and marks selected target hotspot atoms (`C_HOT`; CA by default). It imports
`rfproteina.conditions` to register the annotations and validates before writing. Validation is on
by default; do not use `--no-validate` to bypass a failed input gate. A segment matching no atoms
means the chain/range mapping is wrong, not that coordinates should be fabricated.

Then read the saved input back and verify chain/residue selections, the binder-length
specification, target sequence and coordinates, hotspots and conditioning counts against the
intended experiment. An annotated input is not a generated binder and not a scientific result.

For deeper input authoring, read these paths inside the installed checkout
(`$LOAM_SOFTWARE_ROOT/RFD4-Proteina/`, or the engine directory the System's reference file names)
through the campaign's filesystem reader — they are not `read_skill` paths:

- `skills/rfd4-proteina-annotated-cif/references/python-authoring.md`
- `skills/rfd4-proteina-annotated-cif/references/annotators-and-selections.md`
- `skills/rfd4-proteina-annotated-cif/references/validation.md`
- `docs/reference/annotations.md`
- `docs/concepts/conditioning.md`

## Native serialized representation and numbering

The builder **renumbers** preserved target residues in the annotated CIF: for source chain A author
residues 286–490, the native input uses chain A residues 1–205. This changes neither target
coordinates nor identity, so do not reject the native file because its author numbering differs
from the source contig. The helper writes `<input>.report.json`, mapping source author residues to
native input residues by preserved CA coordinates, and recording the actual source contig and
hotspot mapping.

Configuration aliases are not serialized literally: `C_SEQ` is
`_atom_site.condition_sequence_1_residue`, `C_HOT` is `_atom_site.condition_hotspot_1_atom`, and
`C_CRDGRP` is `_atom_site.condition_CoordinateGroup_1_atom`. Hotspot values are True/False strings,
sequence conditions are residue names, and coordinate-group value 1 marks constrained backbone
atoms. Inspect these native columns and the helper report rather than searching CIF text for alias
names or a literal contig. The expandable binder sentinel encodes the requested length or range; it
is not a generated backbone.

## Source coordinates and annotated coordinates are different inputs

For the verified EGFR example, `3NJP.cif` is the full downloaded source: receptor chain A has 614
observed protein residues. It is not a 205-residue cropped file. To rebuild an annotation from that
source, use `--target-range 286-490` and source-author hotspot IDs within that range. Do not assert
that the full source has only 205 residues, and do not change its range to 1-205 to satisfy that
assertion: that selects a different receptor region.

The helper-produced `3NJP_two_loop_annotated.cif` is the normalized input: target chain A1-205
corresponds to source A286-490, so a hotspot measured at normalized position k maps to source
author k+285 for this verified crop. When changing hotspot conditioning, map back through the report
and invoke the helper on the full source with source-author IDs. Inspect the actual input's chain
and condition columns rather than guessing its coordinate system from its filename.
