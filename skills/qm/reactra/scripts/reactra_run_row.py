#!/usr/bin/env python3
"""Loam run_batch row runner for Reactra enzyme-reaction simulations (skill-MCP route).

`run_batch` executes `python reactra_run_row.py input.json` once per JSONL row, in
a per-row folder, with this row pinned to ONE XPU tile via ZE_AFFINITY_MASK (set by
the Ensemble Launcher; note Reactra's native workers normally take their tile from
`native_device_slots[].environment.ZE_AFFINITY_MASK` in the deployment profile --
see the reactra SKILL). This is "the function that takes inputs": it reads one row,
resolves the reactra control-CLI wrapper WORKER-SIDE (site manifest or built-in
defaults), and invokes `scripts/reactra_cli.sh` through `bash -lc`, capturing a log.

Reactra drives a whole staged pipeline from one YAML + a MECH mechanism file. Only
the cheap authoring path (validate/plan, verified < 25 s, no GPU) is validated on
Aurora; a full `run` is long (the CM reference took ~54 min) and is NOT validated
here -- it is gated behind stage="run" and marked TODO(verify).

The wrapper REQUIRES $REACTRA_MECH_OVERLAY (the one-time mech-format overlay built
in a directory you own; there is no shared copy). The runner fails loudly if it is
unset, rather than letting every MECH-dependent command refuse opaquely.

Input row (JSONL):
  id          : row identity / default request-id
  stage       : "plan" (default; validate + plan, cheap) | "run" (full, ~1h, UNVERIFIED)
  run_yaml    : absolute path to the reactra.run/1 YAML (references its .mech + assets)
  store       : stage="run" only -- store dir (default <cwd>/store)
  request_id  : stage="run" only -- request id (default == id)
  mech_overlay: optional override for $REACTRA_MECH_OVERLAY (else site env / process env)

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md s4.B):
  wrapper      : site["reactra"]["wrapper"] -> else $HPC_LOAM_HOME/skills/qm/reactra/scripts/reactra_cli.sh
  mech_overlay : row -> site["reactra"]["env"]["REACTRA_MECH_OVERLAY"] -> $REACTRA_MECH_OVERLAY
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.

Reactra CLI exit codes (preserved in result.json "rc"; see SKILL):
  0 completed | 2 refusal/error | 3 run not terminal | 4 terminal but outputs unmet.
  Exit 4 is a terminal SCIENTIFIC outcome -- do NOT auto-resubmit it as transient.
"""
import json
import os
import shlex
import socket
import subprocess
import sys
import time

SKILL = "reactra"

# Built-in fallbacks so the runner works before a site manifest is deployed.
# These mirror the reactra SKILL.md "Environments" block; the manifest overrides them.
_DEFAULTS = {
    "wrapper_rel": "skills/qm/reactra/scripts/reactra_cli.sh",
}


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
                print(f"[reactra_run_row] WARN bad site manifest {path}: {exc}", flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    loam = os.environ.get("HPC_LOAM_HOME", "")
    wrapper = site.get("wrapper")
    if not wrapper and loam:
        wrapper = os.path.join(loam, _DEFAULTS["wrapper_rel"])
    env_extra = site.get("env", {}) or {}
    return wrapper, env_extra


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    stage = row.get("stage", "plan")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    wrapper, env_extra = _resolve()

    if not wrapper or not os.path.isfile(wrapper):
        sys.exit(f"[reactra_run_row] reactra_cli.sh wrapper not found: {wrapper!r} "
                 f"(set $HPC_LOAM_HOME or site['reactra']['wrapper'])")

    run_yaml = row.get("run_yaml")
    if not run_yaml:
        sys.exit("[reactra_run_row] row has no 'run_yaml'")
    if not os.path.isfile(run_yaml):
        sys.exit(f"[reactra_run_row] run_yaml not found: {run_yaml!r}")

    # Resolve the mech-format overlay (the wrapper refuses without it).
    overlay = (row.get("mech_overlay")
               or env_extra.get("REACTRA_MECH_OVERLAY")
               or os.environ.get("REACTRA_MECH_OVERLAY"))
    if not overlay:
        sys.exit("[reactra_run_row] REACTRA_MECH_OVERLAY unset: build the one-time "
                 "mech-format overlay (SKILL 'One-time setup') in a dir you own and "
                 "set it via row['mech_overlay'], site['reactra']['env'], or the env.")
    if not os.path.isdir(overlay):
        sys.exit(f"[reactra_run_row] REACTRA_MECH_OVERLAY does not exist: {overlay!r}")

    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items()})
    env["REACTRA_MECH_OVERLAY"] = overlay

    rx = shlex.quote(wrapper)
    yaml_q = shlex.quote(run_yaml)
    ov_q = shlex.quote(overlay)

    if stage == "plan":
        # Cheap authoring path (verified < 25 s, no GPU): validate then plan.
        bash_cmd = (f"export REACTRA_MECH_OVERLAY={ov_q}; "
                    f"{rx} validate {yaml_q} > validate.json; v=$?; "
                    f"{rx} plan {yaml_q} --format text > plan.txt; p=$?; "
                    f"exit $(( v>p ? v : p ))")
        expected_outputs = ["validate.json", "plan.txt"]
    elif stage == "run":
        # TODO(verify): full run not validated on Aurora. Long (~1h); exit 4 is a
        # terminal scientific outcome, not a transient failure -- do not auto-retry.
        store = row.get("store") or os.path.join(cwd, "store")
        req = row.get("request_id") or rid
        bash_cmd = (f"export REACTRA_MECH_OVERLAY={ov_q}; "
                    f"{rx} run {yaml_q} --store {shlex.quote(store)} "
                    f"--request-id {shlex.quote(str(req))}")
        expected_outputs = [os.path.relpath(store, cwd) if store.startswith(cwd) else store]
    else:
        sys.exit(f"[reactra_run_row] unknown stage {stage!r} (expected 'plan' or 'run')")

    print(f"[reactra_run_row] START id={rid} stage={stage} tile={tile} host={host} "
          f"run_yaml={run_yaml}", flush=True)
    log = os.path.join(cwd, "reactra.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(["bash", "-lc", bash_cmd], cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode

    outputs = [o for o in expected_outputs
               if os.path.exists(os.path.join(cwd, o)) or os.path.exists(o)]

    # Success = exit 0 AND the expected artifact(s) present. Non-zero reactra exits
    # (2/3/4) are preserved in rc for the caller to interpret per the SKILL table.
    fail = (rc != 0) or (not outputs)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "stage": stage,
        "outputs": outputs,
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[reactra_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
