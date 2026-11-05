#!/usr/bin/env python3
"""Loam run_batch row runner for RFD4-Proteina (resident-worker route).

run_batch executes `python rfd4_run_row.py input.json` once per JSONL row, in a
per-row folder, with this row pinned to ONE XPU tile via ZE_AFFINITY_MASK. We
write a 1-sample plan and invoke run_rfd4_resident.sh (the pinned Kastner stack),
which inherits the tile, loads the model, and generates the backbone. Output
(CIF + receipt) lands under <row>/rfd4_run/predictions/. This runs under the
exact MPI/PALS launcher that deadlocked the old rfd4-generate path; the resident
adapter is single-device by construction, so it does not hang.

Input row (JSONL): {"id","input","seed"[,"seconds"][,"wrapper"]}
  input   : absolute path to an annotated CIF (see make_rfd4_input.py / benchmarks)
  seed    : integer; distinct seeds are distinct designs
  seconds : bounded worker budget (default 700)
  wrapper : optional override for the resident wrapper path

Wrapper resolution order: row["wrapper"] -> $RFD4_WRAPPER ->
  $HPC_LOAM_HOME/skills/diffusion-models/rfd4-proteina/scripts/run_rfd4_resident.sh
"""
import json, os, subprocess, sys, time, glob, socket


def resolve_wrapper(row):
    w = row.get("wrapper") or os.environ.get("RFD4_WRAPPER")
    if not w and os.environ.get("HPC_LOAM_HOME"):
        w = os.path.join(os.environ["HPC_LOAM_HOME"],
                         "skills/diffusion-models/rfd4-proteina/scripts/run_rfd4_resident.sh")
    if not w or not os.path.isfile(w):
        sys.exit(f"[rfd4_run_row] resident wrapper not found (row.wrapper / $RFD4_WRAPPER / "
                 f"$HPC_LOAM_HOME): got {w!r}")
    return w


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    run_dir = os.path.join(cwd, "rfd4_run")          # wrapper mkdir rejects existing
    plan_path = os.path.join(cwd, "plan.json")
    seconds = str(row.get("seconds", 700))
    wrapper = resolve_wrapper(row)
    json.dump({"samples": [{"input": row["input"], "seed": row.get("seed", 101)}]},
              open(plan_path, "w"), indent=2)
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    print(f"[rfd4_run_row] START id={row.get('id')} tile={tile} host={socket.gethostname()} "
          f"input={row['input']} seed={row.get('seed')}", flush=True)
    json.dump({"id": row.get("id"), "host": socket.gethostname(), "tile": tile,
               "status": "started", "t_start": time.time()},
              open(os.path.join(cwd, "result.json"), "w"), indent=2)
    t0 = time.time()
    # bash -lc so `module` is available; stream stdout/stderr -> run_batch captures it.
    p = subprocess.run(["bash", "-lc",
                        f'exec bash "{wrapper}" "{plan_path}" "{run_dir}" "{seconds}"'])
    recs = glob.glob(run_dir + "/predictions/**/receipt.json", recursive=True)
    cifs = glob.glob(run_dir + "/predictions/**/*_generated.cif", recursive=True)
    rec = {"id": row.get("id"), "host": socket.gethostname(), "tile": tile,
           "seconds": round(time.time() - t0, 1), "rc": p.returncode,
           "n_receipts": len(recs), "n_cifs": len(cifs), "t_end": time.time()}
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[rfd4_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
