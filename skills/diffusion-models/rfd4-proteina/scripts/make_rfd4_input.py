#!/usr/bin/env python
"""Build a validated RFD4 annotated CIF from real Jupiter target coordinates.

Run with $LOAM_SOFTWARE_ROOT/RFD4-Proteina/.venv/bin/python after sourcing
$HPC_LOAM_HOME/scripts/jupiter_env.sh. Author residue numbers are used for
--target-range and --hotspots; hotspots accept comma-separated ids or ranges.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _chain_span(target_path: str, chain: str) -> str:
    """Read observed protein author residue ids with the installed CIF parser."""
    import gemmi
    structure = gemmi.read_structure(target_path)
    ids = [res.seqid.num for ch in structure[0] if ch.name == chain
           for res in ch if res.het_flag == 'A']
    if not ids:
        raise SystemExit(f"Target chain {chain!r} has no protein residues in {target_path}")
    return f"{min(ids)}-{max(ids)}"


def parse_hotspots(value: str) -> list[int]:
    """Expand author-residue ranges without inventing or silently dropping ids."""
    result = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part[1:]:
            lo, hi = part.split('-', 1)
            if int(lo) > int(hi):
                raise ValueError(f'Reversed hotspot range: {part}')
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return list(dict.fromkeys(result))


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
    # Explicit ranges must match the observed author numbering.
    actual = _chain_span(target_path, target_chain)
    lo, hi = (int(x) for x in actual.split("-"))
    if target_range:
        want = target_range.split("-")
        try:
            w_lo = int(want[0])
            w_hi = int(want[1]) if len(want) > 1 else w_lo
        except (ValueError, IndexError):
            raise SystemExit(
                f"make_rfd4_input: --target-range {target_range!r} is not "
                f"<start>-<end>; chain {target_chain} spans {actual}."
            )
        c_lo, c_hi = max(w_lo, lo), min(w_hi, hi)
        if c_lo > c_hi:
            raise SystemExit(
                f"make_rfd4_input: --target-range {target_range} does not "
                f"overlap chain {target_chain}, which spans {actual}."
            )
        if (c_lo, c_hi) != (w_lo, w_hi):
            raise SystemExit(f"Target range {target_range} exceeds observed chain {target_chain} span {actual}; verify the residue mapping")
        target_range = f"{c_lo}-{c_hi}"
    else:
        target_range = actual
    tgt = f"{target_chain}{target_range}"
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


def input_report(target_path: Path, out_path: Path, chain_id: str, spec: dict, hotspots: list[int]) -> dict:
    """Describe native annotations and map renumbered residues by preserved CAs."""
    import hashlib
    import json
    import math
    import gemmi
    source = gemmi.read_structure(str(target_path))[0]
    built = gemmi.read_structure(str(out_path))[0]
    source_by_ca = {}
    for chain in source:
        if chain.name != chain_id:
            continue
        for residue in chain:
            ca = residue.find_atom('CA', '*')
            if ca:
                key = tuple(round(x, 3) for x in (ca.pos.x, ca.pos.y, ca.pos.z))
                source_by_ca[key] = residue.seqid.num
    mapping = []
    for chain in built:
        if chain.name != chain_id:
            continue
        for residue in chain:
            ca = residue.find_atom('CA', '*')
            if ca:
                key = tuple(round(x, 3) for x in (ca.pos.x, ca.pos.y, ca.pos.z))
                mapping.append({'source_author_residue': source_by_ca.get(key),
                                'input_residue': residue.seqid.num})
    atoms = gemmi.cif.read(str(out_path)).sole_block().get_mmcif_category('_atom_site.')
    fields = {'C_SEQ': 'condition_sequence_1_residue',
              'C_HOT': 'condition_hotspot_1_atom',
              'C_CRDGRP': 'condition_CoordinateGroup_1_atom'}
    actual_hotspots = sorted({int(atoms['auth_seq_id'][i])
        for i, value in enumerate(atoms.get(fields['C_HOT'], []))
        if str(value).lower() in {'true', '1'} and atoms['auth_asym_id'][i] == chain_id})
    source_to_input = {item['source_author_residue']: item['input_residue'] for item in mapping}
    expected = sorted({source_to_input[h] for h in hotspots if h in source_to_input})
    if hotspots and (len(expected) != len(set(hotspots)) or expected != actual_hotspots):
        raise ValueError(f'Hotspot annotation mismatch: expected {hotspots} mapped to {expected}, observed {actual_hotspots}')
    report = {'input': str(out_path), 'source': str(target_path),
              'source_sha256': hashlib.sha256(target_path.read_bytes()).hexdigest(),
              'input_sha256': hashlib.sha256(out_path.read_bytes()).hexdigest(),
              'contig': spec['contig'], 'conditions': fields,
              'native_condition_columns_present': {name: field in atoms for name, field in fields.items()},
              'residue_counts_per_chain': {chain.name: len(chain) for chain in built},
              'target_residue_mapping': mapping, 'source_hotspot_residues': hotspots,
              'input_hotspot_residues': actual_hotspots,
              'target_coordinates_finite': all(math.isfinite(v) for chain in built if chain.name == chain_id
                  for residue in chain for atom in residue for v in (atom.pos.x, atom.pos.y, atom.pos.z)),
              'note': 'RFD4 renumbers target residues; the expandable binder sentinel is not a generated binder.'}
    sidecar = out_path.with_suffix('.report.json')
    sidecar.write_text(json.dumps(report, indent=2) + '\n')
    return {**report, 'report_path': str(sidecar)}


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
    ap.add_argument("--validate", action="store_true", help="Explicitly request validation (enabled by default).")
    ap.add_argument("--no-validate", action="store_true", help="Skip RFD4 pre-save validation (NOT recommended).")
    args = ap.parse_args(argv)

    target_path = Path(args.target).resolve()
    if not target_path.exists():
        print(f"ERROR: target not found: {target_path}", file=sys.stderr)
        return 2
    out_path = Path(args.out).resolve()

    hotspots = parse_hotspots(args.hotspots)

    # Import RFD4's builder. `rfproteina.conditions` must be imported to register
    # the RFProteina condition classes before building, or columns get dropped.
    try:
        import rfproteina.conditions  # noqa: F401  -- registration side effect
        from rfproteina.inference_utils.condition_spec import write_condition_spec_cif
    except Exception as exc:  # pragma: no cover
        print(
            "ERROR: could not import rfproteina. Run this with RFD4's venv, e.g.\n"
            "  $LOAM_SOFTWARE_ROOT/RFD4-Proteina/.venv/bin/python "
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
    report = input_report(target_path, Path(written), args.target_chain, spec, hotspots)
    import json
    print("INPUT_REPORT=" + json.dumps(report))
    print(f"OK: wrote annotated CIF -> {written}")
    print(f"    binder contig: {spec['contig']}  hotspots: {hotspots or 'none'}")
    print(f"    feed to: rfd4-generate inputs={written} out_dir=<dir> num_replicates=<n> recycle_strategy=medium/front_loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
