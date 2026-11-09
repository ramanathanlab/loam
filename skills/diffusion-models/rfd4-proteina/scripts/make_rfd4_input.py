#!/usr/bin/env python
"""Build a VALID RFD4-Proteina annotated-CIF input from a real folded target.

Why this exists
---------------
RFD4-Proteina conditions binder generation on an *annotated* mmCIF: the target
structure plus AtomWorks `StandardAnnotation` columns (C_CRDGRP / C_SEQ / C_HOT /
C_IDX ...) and the required `pdbx_PDB_model_num` field. Hand-writing that CIF is
error-prone -- the common failure is emitting a file with no `pdbx_PDB_model_num`
column and/or placeholder zero coordinates, which makes RFD4's loader die with
`biotite.DeserializationError: Failed to deserialize category 'atom_site'` and
produce zero backbones.

Instead of hand-authoring, this wrapper drives RFD4's OWN builder,
`write_condition_spec_cif`, from a parameterized PPI condition spec. The builder
parses the real target coordinates, emits the correct columns, inserts the de-novo
binder chain via the contig, and runs RFD4's pre-save validation. The output is an
annotated CIF that `rfd4-generate inputs=<file>` accepts.

Run it on the Aurora worker in RFD4's venv (it imports `rfproteina`):

    /flare/FRAME-IDP/msinclair/software/RFD4-Proteina/.venv/bin/python make_rfd4_input.py \
        --target /path/to/folded_target.cif \
        --target-chain A \
        --hotspots 55,100,150,500,550,600 \
        --binder-len 55 \
        --out /path/to/input_annotated.cif

Then generate:

    rfd4-generate inputs=/path/to/input_annotated.cif out_dir=... num_designs=...

Validate separately with the shipped checker:

    bash .../skills/rfd4-proteina-annotated-cif/scripts/check_annotated_cif.sh <out.cif>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_spec(
    *,
    target_path: str,
    target_chain: str,
    hotspots: list[int],
    binder_len: str,
    target_range: str | None,
    hotspot_atoms: str,
) -> dict:
    """A PPI condition spec: de-novo binder chain + fixed-backbone target + hotspots.

    Mirrors RFD4's own examples/.../condition_spec_ppi.yaml, parameterized.
    contig  = "<binder_len>, /0, <target>"  -> new binder chain, chain break,
              then the target chain (whole chain, or <chain><start>-<end>).
    C_CRDGRP = one rigid group over the target backbone (N,CA,C,O), skipping NaN
               coords -> fixes the target's internal backbone geometry.
    C_SEQ    = the whole target chain sequence is known (only the binder is designed).
    C_HOT    = atom-level hotspots the binder should engage.
    """
    tgt = f"{target_chain}{target_range}" if target_range else target_chain
    spec: dict = {
        "input": target_path,
        "contig": f"{binder_len}, /0, {tgt}",
        "conditions": {
            # Rigid backbone group over the target: fixes geometry without pinning
            # absolute global pose. `has_nan_coord()` guard skips unresolved atoms.
            "C_CRDGRP": {
                1: [{"select": f"sel('{target_chain}/*/*/[N,CA,C,O]') and not has_nan_coord()"}]
            },
            # The target sequence is fully given; the binder is generated.
            "C_SEQ": {
                "true": [{"select": f"sel('{target_chain}')"}]
            },
        },
    }
    if hotspots:
        spec["conditions"]["C_HOT"] = {
            "true": [
                {"select": f"sel('{target_chain}/*/{r}/[{hotspot_atoms}]')"}
                for r in hotspots
            ]
        }
    return spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True,
                    help="Folded target structure (CIF/PDB) with REAL coordinates, e.g. the Chai fold of the target ectodomain.")
    ap.add_argument("--target-chain", default="A", help="Target chain id in the structure (default: A).")
    ap.add_argument("--target-range", default=None,
                    help="Optional residue range on the target chain, e.g. 25-645. Default: whole chain.")
    ap.add_argument("--hotspots", default="",
                    help="Comma-separated target residue ids the binder should engage, e.g. 55,100,150.")
    ap.add_argument("--hotspot-atoms", default="CA",
                    help="Atom selection per hotspot residue (default: CA). Use e.g. 'CA,CB' or specific sidechain atoms.")
    ap.add_argument("--binder-len", default="55",
                    help="Binder length, fixed ('55') or a sampled range ('50-120'). Length is set HERE, not by a CLI flag.")
    ap.add_argument("--out", required=True, help="Output annotated CIF path.")
    ap.add_argument("--example-id", default="denovo_binder", help="Label used in the builder's summary.")
    ap.add_argument("--no-validate", action="store_true", help="Skip RFD4 pre-save validation (NOT recommended).")
    args = ap.parse_args(argv)

    target_path = Path(args.target).resolve()
    if not target_path.exists():
        print(f"ERROR: target not found: {target_path}", file=sys.stderr)
        return 2
    out_path = Path(args.out).resolve()

    hotspots = [int(x) for x in args.hotspots.split(",") if x.strip()]

    # Import RFD4's builder. `rfproteina.conditions` must be imported to register
    # the RFProteina condition classes before building, or columns get dropped.
    try:
        import rfproteina.conditions  # noqa: F401  -- registration side effect
        from rfproteina.inference_utils.condition_spec import write_condition_spec_cif
    except Exception as exc:  # pragma: no cover
        print(
            "ERROR: could not import rfproteina. Run this with RFD4's venv, e.g.\n"
            "  /flare/FRAME-IDP/msinclair/software/RFD4-Proteina/.venv/bin/python "
            f"{Path(__file__).name} ...\n"
            f"(import error: {exc})",
            file=sys.stderr,
        )
        return 3

    spec = build_spec(
        target_path=str(target_path),
        target_chain=args.target_chain,
        hotspots=hotspots,
        binder_len=args.binder_len,
        target_range=(f"{args.target_range}" if args.target_range else None),
        hotspot_atoms=args.hotspot_atoms,
    )

    # spec_dir resolves any relative `input` paths in the spec; we pass an absolute
    # input, so the target's own directory is a safe choice.
    written = write_condition_spec_cif(
        args.example_id,
        spec,
        out_path,
        spec_dir=target_path.parent,
        validate=not args.no_validate,
    )
    print(f"OK: wrote annotated CIF -> {written}")
    print(f"    binder contig: {spec['contig']}  hotspots: {hotspots or 'none'}")
    print(f"    feed to: rfd4-generate inputs={written} out_dir=<dir> num_designs=<n>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
