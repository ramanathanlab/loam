#!/usr/bin/env python3
"""Loam run_batch row runner for RFdiffusion3 (RFD3, skill-MCP route).

`run_batch` executes `python rfdiffusion3_run_row.py input.json` once per JSONL
row, in a per-row folder, with this row already pinned to ONE XPU tile via
ZE_AFFINITY_MASK (set by the Ensemble Launcher BEFORE this process starts). This
is "the function that takes inputs": it reads one row, resolves the Foundry
install WORKER-SIDE (site manifest or built-in defaults), writes a small per-row
shell launcher, and subprocesses `rfd3 design` under the correct frameworks
module, writing outputs into the per-row cwd.

Two traps from the SKILL are baked in:
  * RFD3's venv is pinned to `frameworks/2025.3.1`; the node default is now
    `frameworks/2026.1.0`, so `import torch` fails unless we `module swap` first.
    We therefore run generation via `bash -lc` so `module` is available, and we
    `module use /opt/aurora/26.26.0/modulefiles` before the swap (the 2025.3.1
    modulefile is not on the default MODULEPATH).  [SKILL.md §1, §7.1]
  * `set -u` + Lmod deadlock on Aurora, so the generated launcher uses
    `set -eo pipefail` (NOT `-euo`).  [SKILL.md §7.2]
  * We do NOT set ZE_AFFINITY_MASK here: the EL assigns the tile per row before
    this process launches; with one tile visible RFD3 auto-selects XPU.  [§7.3]

Input row (JSONL):
  id          : row identity (ignored by the batch distinctness check)
  spec        : design spec -- inline JSON text OR an absolute path to a
                JSON/YAML inputs file (spec-name -> design config, SKILL.md §4).
                Omit for a length-only unconditional design (needs `length`).
  length      : int or "min-max" string; used ONLY in the unconditional path
                (inputs=null +specification.length=<length>).  [SKILL.md §3]
  seed        : int; distinct seeds are distinct designs            (default 42)
  timesteps   : inference_sampler.num_timesteps                     (default 200)
  batches     : n_batches                                           (default 1)
  batch_size  : diffusion_batch_size (total designs = batches*batch_size) (default 1)
  keys_subset : optional list -> json_keys_subset=[k1,k2]           (spec path only)
  extra       : optional list of extra raw Hydra overrides (strings)

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B):
  foundry     : site["rfdiffusion3"]["foundry"]     -> else built-in default
  rfd3        : site["rfdiffusion3"]["rfd3"]         -> else <foundry>/.venv/bin/rfd3
  checkpoints : site["rfdiffusion3"]["checkpoints"]  -> else built-in default
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.
  (RFD3 is invoked directly -- there is no separate loam-tree plan launcher the
  way chai has chai_xpu.py; the per-row launcher below is generated in cwd. A
  site-provided "launcher" wrapper, or $HPC_LOAM_HOME/skills/diffusion-models/
  rfdiffusion3/scripts/<script>, is honored if present.)
"""
import glob
import json
import os
import shlex
import socket
import subprocess
import sys
import time

SKILL = "rfdiffusion3"

# Built-in fallbacks so the runner works before a site manifest is deployed.
# These mirror the rfdiffusion3 SKILL.md "Environment" block; the manifest
# overrides them.  [SKILL.md §1, §2]
_DEFAULTS = {
    "foundry": "/flare/FRAME-IDP/msinclair/software/foundry",
    "rfd3_rel": ".venv/bin/rfd3",  # under <foundry>
    "checkpoints": "/flare/FRAME-IDP/msinclair/software/checkpoints",
}
# The frameworks module swap RFD3 needs; `module use` first (2025.3.1 is off the
# default MODULEPATH), `|| module load` so a clean env still works.  [SKILL.md §1]
_MODULE_SETUP = (
    "module use /opt/aurora/26.26.0/modulefiles\n"
    "module swap frameworks/2026.1.0 frameworks/2025.3.1 2>/dev/null "
    "|| module load frameworks/2025.3.1\n"
)


def _load_site_entry():
    """This skill's entry from the worker-side site manifest, or {} if none."""
    loam = os.environ.get("HPC_LOAM_HOME", "")
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("LOAM_SITE_MANIFEST"),
        os.path.join(here, "site.aurora.json"),          # staged beside the runner
        os.path.join(loam, "skills", "site.aurora.json") if loam else None,
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return (json.load(open(path)) or {}).get(SKILL, {}) or {}
            except Exception as exc:  # a bad manifest must not kill the row
                print(f"[rfdiffusion3_run_row] WARN bad site manifest {path}: {exc}",
                      flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    foundry = site.get("foundry") or _DEFAULTS["foundry"]
    rfd3 = site.get("rfd3") or os.path.join(foundry, _DEFAULTS["rfd3_rel"])
    checkpoints = site.get("checkpoints") or _DEFAULTS["checkpoints"]
    module_setup = site.get("module_setup") or _MODULE_SETUP
    env_extra = site.get("env", {}) or {}
    # Optional site-provided wrapper, else a loam-tree script if one is ever added.
    launcher = site.get("launcher")
    if not launcher:
        loam = os.environ.get("HPC_LOAM_HOME", "")
        cand = (os.path.join(loam, "skills/diffusion-models/rfdiffusion3/scripts",
                             "run_rfd3_row.sh") if loam else None)
        if cand and os.path.isfile(cand):
            launcher = cand
    return foundry, rfd3, checkpoints, module_setup, env_extra, launcher


def _build_overrides(row, out_dir):
    """Hydra key=value overrides for `rfd3 design`, as a token list."""
    spec = row.get("spec")
    tokens = []
    if spec:
        # inline JSON text -> write to cwd; else treat as an absolute path
        if os.path.isfile(spec):
            spec_path = spec
        else:
            spec_path = os.path.join(os.getcwd(), "input.json")
            open(spec_path, "w").write(spec)
        tokens.append(f"inputs={spec_path}")
        if row.get("keys_subset"):
            subset = ",".join(str(k) for k in row["keys_subset"])
            tokens.append(f"json_keys_subset=[{subset}]")
    else:
        # length-only unconditional design; `+` is REQUIRED for specification.*
        length = row.get("length")
        if length is None:
            sys.exit("[rfdiffusion3_run_row] row has neither 'spec' nor 'length'")
        tokens.append("inputs=null")
        tokens.append(f"+specification.length={length}")
    tokens += [
        f"out_dir={out_dir}",
        f"diffusion_batch_size={row.get('batch_size', 1)}",
        f"n_batches={row.get('batches', 1)}",
        f"inference_sampler.num_timesteps={row.get('timesteps', 200)}",
        f"seed={row.get('seed', 42)}",
        "skip_existing=False",
        "prevalidate_inputs=True",
    ]
    for raw in (row.get("extra") or []):
        tokens.append(str(raw))
    return tokens


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    foundry, rfd3, checkpoints, module_setup, env_extra, launcher = _resolve()

    if not os.path.isdir(foundry):
        sys.exit(f"[rfdiffusion3_run_row] foundry install not found: {foundry!r} "
                 f"(set $HPC_LOAM_HOME or site['rfdiffusion3']['foundry'])")
    if not os.path.exists(rfd3):
        sys.exit(f"[rfdiffusion3_run_row] rfd3 entry point not found: {rfd3!r}")

    out_dir = os.path.join(cwd, "out")
    overrides = _build_overrides(row, out_dir)

    # Extra env exports for the generated launcher (quoted).
    exports = [f"export FOUNDRY_CHECKPOINT_DIRS={shlex.quote(checkpoints)}"]
    for k, v in env_extra.items():
        exports.append(f"export {k}={shlex.quote(str(v))}")

    # Per-row launcher: module swap (bash -lc gives us `module`), cd into the
    # install (config discovery), then rfd3 with absolute out_dir/inputs.
    override_str = " ".join(shlex.quote(t) for t in overrides)
    if launcher:
        # A site/loam wrapper: <launcher> <foundry> <checkpoints> <overrides...>
        run_line = (f'exec bash {shlex.quote(launcher)} {shlex.quote(foundry)} '
                    f'{shlex.quote(checkpoints)} {override_str}')
    else:
        run_line = (f'cd {shlex.quote(foundry)}\n'
                    f'exec {shlex.quote(rfd3)} design {override_str}')
    script = (
        "#!/usr/bin/env bash\n"
        "# generated per-row by rfdiffusion3_run_row.py -- do not edit\n"
        "set -eo pipefail\n"               # NOT -u: set -u + Lmod deadlock [§7.2]
        + module_setup
        + "\n".join(exports) + "\n"
        + run_line + "\n"
    )
    script_path = os.path.join(cwd, "run_rfd3.sh")
    open(script_path, "w").write(script)

    print(f"[rfdiffusion3_run_row] START id={rid} tile={tile} host={host} "
          f"seed={row.get('seed')} spec={'yes' if row.get('spec') else 'no'} "
          f"length={row.get('length')}", flush=True)
    log = os.path.join(cwd, "rfd3.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(["bash", "-lc", f'exec bash {shlex.quote(script_path)}'],
                            cwd=cwd, stdout=lf, stderr=subprocess.STDOUT).returncode

    # Output: <out_dir>/<...>_model_<n>.cif.gz (+ .json). [SKILL.md §6]
    cifs = sorted(glob.glob(os.path.join(out_dir, "*.cif.gz"))
                  + glob.glob(os.path.join(out_dir, "*.cif")))

    fail = (rc != 0) or (not cifs)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "n_cif": len(cifs),
        "outputs": [os.path.basename(c) for c in cifs],
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[rfdiffusion3_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
