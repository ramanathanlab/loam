# Packaged reference and example files

All paths below belong to `boltz-jupiter` and can be read through `skill_view`.

For example: `skill_view(name="boltz-jupiter", file_path="references/workflow.md")`.

## Jupiter references

- `references/workflow.md` — authoritative Jupiter workflow.
- `references/aurora-notes.md` — historical original notes, with explicit platform caveats.

## Historical supporting resources

These are read-only historical examples and launcher sources. Read them for API context; do not execute them unchanged on Jupiter. The resource path is the full path below, not an inferred `scripts/...` path.

- `skill_view(name="boltz-jupiter", file_path="references/aurora-resources/scripts/lig_aff.yaml")`
- `skill_view(name="boltz-jupiter", file_path="references/aurora-resources/scripts/prot_no_msa.yaml")`
- `skill_view(name="boltz-jupiter", file_path="references/aurora-resources/scripts/run_boltz_xpu.sh")`
- `skill_view(name="boltz-jupiter", file_path="references/aurora-resources/scripts/ubq_msa.fasta")`

## Provenance

`references/source-manifest.json` records original source paths and SHA-256 checksums; it is also readable with `skill_view`.
