#!/usr/bin/env python3
"""Loam run_batch row runner for RFD4-Proteina (resident-worker route).

run_batch executes `python rfd4_run_row.py input.json` once per JSONL row, in a
per-row folder, with this row pinned to ONE XPU tile via ZE_AFFINITY_MASK. The
row either supplies a ready-made annotated CIF (`input`), OR a folded `target`
plus a design spec, in which case this runner ANNOTATES it first (annotate_rfd4.sh
-> RFD4's write_condition_spec_cif) and then generates. Either way it writes a
1-sample plan and invokes run_rfd4_resident.sh (the pinned Kastner stack), which
inherits the tile, loads the model, and generates the backbone. Output (CIF +
receipt) lands under <row>/rfd4_run/predictions/.

Input row (JSONL) -- one of two modes:
  GENERATE from a ready annotated CIF:
    {"id","input"(abs annotated .cif),"seed"[,"seconds"][,"wrapper"]}
  ANNOTATE-THEN-GENERATE from a folded target (the MCP builds the input itself):
    {"id","target"(abs folded .cif/.pdb),"target_chain"(default A),
     "target_range"("start-end"; auto-derived from the target if omitted),
     "hotspots"(csv residue indices, optional),"binder_len"(default 55),
     "hotspot_atoms"(default CA),"seed"[,"seconds"][,"wrapper"][,"annotator"]}

Wrapper resolution order: row["wrapper"] -> $RFD4_WRAPPER ->
  $HPC_LOAM_HOME/skills/diffusion-models/rfd4-proteina/scripts/run_rfd4_resident.sh
Annotator resolution: row["annotator"] -> $RFD4_ANNOTATOR ->
  $HPC_LOAM_HOME/skills/diffusion-models/rfd4-proteina/scripts/annotate_rfd4.sh
"""
import json, os, subprocess, sys, time, glob, socket


def _resolve(row, key, env, rel):
    w = row.get(key) or os.environ.get(env)
    if not w and os.environ.get("HPC_LOAM_HOME"):
        w = os.path.join(os.environ["HPC_LOAM_HOME"], rel)
    if not w or not os.path.isfile(w):
        sys.exit(f"[rfd4_run_row] {key} script not found "
                 f"(row.{key} / ${env} / $HPC_LOAM_HOME): got {w!r}")
    return w


def resolve_wrapper(row):
    return _resolve(row, "wrapper", "RFD4_WRAPPER",
                    "skills/diffusion-models/rfd4-proteina/scripts/run_rfd4_resident.sh")


def resolve_annotator(row):
    return _resolve(row, "annotator", "RFD4_ANNOTATOR",
                    "skills/diffusion-models/rfd4-proteina/scripts/annotate_rfd4.sh")


def _derive_range(cif_path, chain):
    """Min-max residue number for `chain` from an mmCIF/PDB atom_site, or None.

    The engine contig needs an explicit <start>-<end>; a bare chain id is
    rejected. Best-effort so the MCP user need not know the target's numbering.
    """
    try:
        if cif_path.endswith((".pdb", ".ent")):
            nums = [int(l[22:26]) for l in open(cif_path, errors="replace")
                    if l.startswith(("ATOM", "HETATM")) and l[21:22].strip() == chain]
            return f"{min(nums)}-{max(nums)}" if nums else None
        # mmCIF: locate the atom_site loop header, then read columns.
        lines = open(cif_path, errors="replace").read().splitlines()
        cols, i = [], 0
        while i < len(lines):
            if lines[i].strip() == "loop_":
                hdr = []
                j = i + 1
                while j < len(lines) and lines[j].lstrip().startswith("_atom_site."):
                    hdr.append(lines[j].strip())
                    j += 1
                if hdr:
                    cols, start = hdr, j
                    break
            i += 1
        if not cols:
            return None
        def idx(name):
            for k, c in enumerate(cols):
                if c.endswith(name):
                    return k
            return None
        asym = idx(".auth_asym_id") or idx(".label_asym_id")
        seq = idx(".auth_seq_id") or idx(".label_seq_id")
        if asym is None or seq is None:
            return None
        nums = []
        for ln in lines[start:]:
            if ln.startswith(("ATOM", "HETATM")):
                parts = ln.split()
                if len(parts) > max(asym, seq) and parts[asym] == chain:
                    try: nums.append(int(parts[seq]))
                    except ValueError: pass
            elif ln.strip() in ("#", "loop_") or ln.startswith("_"):
                if nums: break
        return f"{min(nums)}-{max(nums)}" if nums else None
    except Exception as exc:
        print(f"[rfd4_run_row] WARN range derive failed: {exc}", flush=True)
        return None


def annotate(row, cwd):
    """Build an annotated CIF from a folded target; return its path."""
    target = row["target"]
    if not os.path.isfile(target):
        sys.exit(f"[rfd4_run_row] target not found: {target!r}")
    chain = str(row.get("target_chain", "A"))
    rng = row.get("target_range") or _derive_range(target, chain)
    if not rng:
        sys.exit("[rfd4_run_row] target_range not given and could not be derived "
                 f"from {target!r} chain {chain!r}; pass 'target_range' as 'start-end'")
    out_cif = os.path.join(cwd, "input_annotated.cif")
    annotator = resolve_annotator(row)
    cmd = ["bash", "-lc", 'exec bash "$0" "$1" "$2" "$3" "$4" "$5" "$6" "$7"',
           annotator, target, chain, str(rng), str(row.get("hotspots", "")),
           str(row.get("binder_len", 55)), out_cif, str(row.get("hotspot_atoms", "CA"))]
    print(f"[rfd4_run_row] ANNOTATE target={target} chain={chain} range={rng} "
          f"hotspots={row.get('hotspots','')} binder_len={row.get('binder_len',55)}", flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0 or not os.path.isfile(out_cif):
        sys.exit(f"[rfd4_run_row] annotation failed (rc={p.returncode}); no {out_cif}")
    return out_cif


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    run_dir = os.path.join(cwd, "rfd4_run")          # wrapper mkdir rejects existing
    plan_path = os.path.join(cwd, "plan.json")
    seconds = str(row.get("seconds", 700))
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()

    # Mode select: ready annotated CIF, or annotate a folded target here.
    if row.get("input"):
        input_cif = row["input"]
    elif row.get("target"):
        input_cif = annotate(row, cwd)
    else:
        sys.exit("[rfd4_run_row] row needs 'input' (annotated CIF) or 'target' (folded structure)")

    wrapper = resolve_wrapper(row)
    json.dump({"samples": [{"input": input_cif, "seed": row.get("seed", 101)}]},
              open(plan_path, "w"), indent=2)
    print(f"[rfd4_run_row] START id={row.get('id')} tile={tile} host={host} "
          f"input={input_cif} seed={row.get('seed')}", flush=True)
    json.dump({"id": row.get("id"), "host": host, "tile": tile,
               "status": "started", "t_start": time.time()},
              open(os.path.join(cwd, "result.json"), "w"), indent=2)
    t0 = time.time()
    # bash -lc so `module` is available; stream stdout/stderr -> run_batch captures it.
    p = subprocess.run(["bash", "-lc",
                        f'exec bash "{wrapper}" "{plan_path}" "{run_dir}" "{seconds}"'])
    recs = glob.glob(run_dir + "/predictions/**/receipt.json", recursive=True)
    cifs = glob.glob(run_dir + "/predictions/**/*_generated.cif", recursive=True)
    rec = {"id": row.get("id"), "host": host, "tile": tile,
           "seconds": round(time.time() - t0, 1), "rc": p.returncode,
           "input": input_cif, "annotated": bool(row.get("target") and not row.get("input")),
           "n_receipts": len(recs), "n_cifs": len(cifs), "t_end": time.time()}
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[rfd4_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(p.returncode if (p.returncode or cifs) else 1)


if __name__ == "__main__":
    main()
