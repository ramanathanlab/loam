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

Run it on an Aurora worker with the RFD4 interpreter `references/aurora.md` names -- the pinned
deployment's `$RFD4_DEPLOYMENT/envs/rfd4-xpu-2026.1.0/bin/python`, which is what the verified route
generates with (it imports `rfproteina`):

    "$RFD4_PYTHON" make_rfd4_input.py \
        --target /path/to/folded_target.cif \
        --target-chain A \
        --hotspots 55,100,150,500,550,600 \
        --binder-len 55 \
        --out /path/to/input_annotated.cif

Then generate as `references/aurora.md` describes, and read the saved input back before fanning
out: `references/annotated-cif.md` says what to check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _chain_span(target_path: str, chain: str) -> str:
    """Return "<start>-<end>" for `chain`'s observed residues in a CIF or PDB.

    Reads author-assigned residue ids, which is what a contig token refers to.
    Falls back to raising rather than guessing: a wrong span silently designs
    against the wrong part of the target.
    """
    import gzip

    path = Path(target_path)
    opener = gzip.open if path.suffix == ".gz" else open
    ids: list[int] = []
    with opener(path, "rt", errors="replace") as handle:
        lines = handle.read().splitlines()

    if any(line.startswith(("_atom_site.", "data_")) for line in lines[:400]):
        # mmCIF: resolve columns from the loop header rather than assuming order.
        cols: dict[str, int] = {}
        for line in lines:
            if line.startswith("_atom_site."):
                cols[line.strip().split(".", 1)[1]] = len(cols)
            elif cols and line.startswith("ATOM"):
                f = line.split()
                ch = cols.get("auth_asym_id", cols.get("label_asym_id"))
                rs = cols.get("auth_seq_id", cols.get("label_seq_id"))
                if ch is None or rs is None or len(f) <= max(ch, rs):
                    continue
                if f[ch] == chain and f[rs].lstrip("-").isdigit():
                    ids.append(int(f[rs]))
            elif cols and line.startswith("#") and ids:
                break
    else:
        # PDB: fixed columns 22 (chain) and 23-26 (resSeq).
        for line in lines:
            if line.startswith("ATOM") and len(line) >= 26:
                if line[21] == chain and line[22:26].strip().lstrip("-").isdigit():
                    ids.append(int(line[22:26]))

    if not ids:
        raise SystemExit(
            f"make_rfd4_input: chain {chain!r} has no parsable residues in "
            f"{target_path}; pass --target-range explicitly."
        )
    return f"{min(ids)}-{max(ids)}"


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
    # A contig's preserved-segment token is chain letters FOLLOWED BY digits
    # ("A1-645", "A300"); rfproteina's _parse_preserved_segment rejects a bare
    # chain id. So --target-range's documented "Default: whole chain" could not
    # work: it emitted "A" and rfd4 died 87s in with
    #   ValueError: Malformed preserved-segment token: 'A'
    # on all 48 rows of run 16. Derive the chain's real span instead, so the
    # default means what it says.
    # Always reconcile the range against the structure. Agents supply ranges
    # from UniProt (e.g. the 1-645 ectodomain) that overrun the deposited
    # model, and rfd4 then refuses the whole run:
    #   ValueError: Preserved segment 'A1-645' runs outside chain A, whose
    #               residues are 1-622.
    # That killed all 48 rows of a run 16 batch. Clamping keeps the experiment
    # alive with a correct contig instead of failing 20s in; the adjustment is
    # printed so the narrowing is visible in the row's log.
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
            print(f"    note: --target-range {target_range} clamped to "
                  f"{c_lo}-{c_hi} (chain {target_chain} spans {actual})")
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
            "ERROR: could not import rfproteina. Run this with the RFD4 interpreter this "
            "System's reference file names, e.g.\n"
            "  \"$RFD4_PYTHON\" "
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
