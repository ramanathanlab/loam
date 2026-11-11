#!/usr/bin/env python3
"""Loam run_batch row runner for GenSLM-ESM embeddings + SAE features (skill-MCP).

`run_batch` executes `python genslm_esm_run_row.py input.json` once per JSONL
row, in a per-row folder, with this row pinned to ONE XPU tile via
ZE_AFFINITY_MASK. This is "the function that takes inputs": it reads one row,
resolves the genslm-esm-sae install WORKER-SIDE, and runs the verified embed
recipe (SKILL.md "Verified recipe 1") under the Aurora frameworks module + the
`venvs/aurora` venv, writing outputs into the per-row cwd.

This runner is PURE STDLIB. The heavy torch/genslm work runs in a small driver
(embedded below as _DRIVER_SRC, lifted from the verified scripts/embed_and_sae.py
API) that the runner writes into the per-row cwd and subprocesses under the
genslm venv. The runner never imports torch itself.

Two traps from the SKILL are baked in as FAILURES:
  1. XPU device gotcha / wrong venv. The backbone device is
     `torch.device('xpu' if torch.xpu.is_available() else 'cpu')`; if the XPU is
     not visible (e.g. the CUDA-only `venvs/ci` was used instead of
     `venvs/aurora`, or the frameworks module was not loaded) the run SILENTLY
     runs on CPU. The driver exits nonzero when it lands on CPU, and the runner
     also fails the row if results.json reports device != xpu.
  2. Non-finite activations. The driver asserts the activations are finite and
     records it; a non-finite run is reported as a failure.

The SKILL also warns about load/IO contention: cold load is ~160 s solo and
contends on Lustre under ~12 rows/node, so the manifest sets a generous
timeout_each (>= 1800). PMIX/PMI rank env is scrubbed before Python so a stray
HF/Lightning distributed init cannot fire with no peers (harmless here since the
embed path uses no HF Trainer, but kept for safety).

Input row (JSONL):
  id         : row identity
  sequence   : the sequence to embed (required). For modality=codon it is
               space-separated UPPERCASE codons ('ATG GCT ... TAA'); a bare DNA
               string with no spaces is split into triplets and upper-cased. For
               modality=aminoacid it is space-separated one-letter residues.
  modality   : "codon" | "aminoacid"   (default "codon")
  layer      : int, 1-based, 1..36; the hidden layer to embed/pool   (default 32)
  pool       : "mean" | "none"; mean-pool per sequence over real tokens (default "mean")
  max_length : int   (default 2048)
  sae        : bool; also run the layer's trained SAE   (default False)
  sae_dir    : path to the SAE checkpoint (a .../layerNN dir). Required when
               sae=True and layer != 32; else defaults to best/layer32. The SAE's
               trained layer+modality MUST match `layer`/`modality`.

Path resolution (WORKER-SIDE, see docs/SKILL_MCP_DESIGN.md §4.B):
  python    : site["python"]    -> built-in default (venvs/aurora python)
  model_id  : site["model_id"]  -> built-in default
  sae_default: site["sae_default"] -> built-in default (best/layer32)
  env       : site["env"] merged over the built-in Aurora env block
  The site manifest is $LOAM_SITE_MANIFEST or $HPC_LOAM_HOME/skills/site.aurora.json.
"""
import json
import os
import socket
import subprocess
import sys
import time

SKILL = "genslm_esm"

# Built-in fallbacks mirroring the genslm-esm SKILL.md "Environment"/"Weights"
# blocks; the manifest overrides them.
_SOFT = "/flare/FRAME-IDP/msinclair/software/genslm-esm-sae"
_DEFAULTS = {
    "python": f"{_SOFT}/venvs/aurora/bin/python",   # NOT .venv, NOT venvs/ci (CUDA-only)
    "model_id": "genslm-test/genslm-esmc-600M-contrastive-2026-10-step-200000",
    "sae_default": f"{_SOFT}/campaigns/600m-codon/runs/l18-36-lr2e-4/best/layer32",
}
# Env the SKILL requires for an offline XPU run. Overridable via site["env"].
_ENV = {
    "ZE_FLAT_DEVICE_HIERARCHY": "FLAT",
    "HF_HOME": "/flare/FRAME-IDP/msinclair/hf_cache",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TMPDIR": "/tmp",
}

# The embed driver, parametrized by the row JSON (argv[1]) + out dir (argv[2]).
# Runs under the genslm venv; uses the verified Backbone/SaeModel API.
_DRIVER_SRC = r'''
import json, sys, time
from pathlib import Path
import torch
from genslm_esm_sae.backbone import Backbone
from genslm_esm_sae.sae import SaeModel

row = json.load(open(sys.argv[1]))
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
model_id = sys.argv[3]
sae_default = sys.argv[4]

modality = row.get("modality", "codon")
layer = int(row.get("layer", 32))
pool = row.get("pool", "mean")
max_length = int(row.get("max_length", 2048))
seq = row.get("sequence")
if not seq:
    sys.exit("[genslm driver] row has no 'sequence'")

# Codon normalization: space-separated UPPERCASE codons. A bare DNA string with
# no spaces is split into triplets (SKILL: codon = space-separated uppercase).
if modality == "codon":
    seq = seq.upper()
    if " " not in seq.strip():
        s = seq.strip()
        if len(s) % 3 != 0:
            sys.exit(f"[genslm driver] bare codon string length {len(s)} not a "
                     f"multiple of 3; pass space-separated codons")
        seq = " ".join(s[i:i+3] for i in range(0, len(s), 3))

dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
res = {"device": str(dev), "modality": modality, "layer": layer}

t = time.time()
# layer=[1, layer] so acts[-1] is the requested layer (matches embed_and_sae.py).
bb = Backbone(model_id, layer=[1, layer], modality=modality,
              max_length=max_length, dtype=torch.bfloat16)
bb.model.to(dev)   # Backbone loads on CPU; must move it ourselves
res["load_s"] = round(time.time() - t, 1)

t = time.time()
batch = {k: v.to(dev) for k, v in bb.collator([{modality: seq}]).items()}
acts = bb(batch)                      # (n_real_tokens, d) for one seq, or (n_layers, n_tok, d)
a = acts if acts.dim() == 2 else acts[-1]
finite = bool(torch.isfinite(a).all())
real = batch[f"{modality}_labels"] != -100
counts = real.sum(1).tolist()
res["acts_shape"] = list(acts.shape)
res["tokens_per_seq"] = counts
res["finite"] = finite

if pool == "mean":
    pooled = torch.stack([c.mean(0) for c in torch.split(a, counts)])
    res["pooled_shape"] = list(pooled.shape)
    torch.save(pooled.cpu(), out / ("%s_layer%02d_meanpool.pt" % (modality, layer)))
else:
    torch.save(a.cpu(), out / ("%s_layer%02d_tokens.pt" % (modality, layer)))
res["embed_s"] = round(time.time() - t, 1)

if row.get("sae"):
    sae_dir = row.get("sae_dir") or sae_default
    sae = SaeModel.from_pretrained(sae_dir).eval().to(dev)
    # Guard the documented "garbage features" trap: layer/modality must match.
    if int(sae.config.layer) != layer:
        sys.exit(f"[genslm driver] SAE layer {sae.config.layer} != requested layer {layer}")
    with torch.inference_mode():
        sout = sae(a)
    fm = sout.feature_magnitudes
    res["sae"] = {
        "sae_dir": sae_dir,
        "layer": int(sae.config.layer),
        "modality": getattr(sae.config, "modality", None),
        "features_shape": list(fm.shape),
        "active_per_token_max": int((fm != 0).sum(-1).max()),
        "mean_recon_loss": float(sout.reconstruction_loss.mean()),
    }
    torch.save({"feature_magnitudes": fm.to_sparse().cpu(), "tokens_per_seq": counts},
               out / ("sae_layer%02d_features.pt" % layer))

if dev.type == "xpu":
    res["xpu_max_mem_GB"] = round(torch.xpu.max_memory_allocated() / 1e9, 2)
(out / "results.json").write_text(json.dumps(res, indent=2))
print("[genslm driver] " + json.dumps(res))

# Fail hard on the silent traps so the runner sees a nonzero exit.
if dev.type != "xpu":
    sys.exit("[genslm driver] FATAL: ran on CPU (XPU not visible; wrong venv or "
             "frameworks module not loaded)")
if not finite:
    sys.exit("[genslm driver] FATAL: non-finite activations")
'''


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
                print(f"[genslm_esm_run_row] WARN bad site manifest {path}: {exc}",
                      flush=True)
    return {}


def _resolve():
    site = _load_site_entry()
    python = site.get("python") or _DEFAULTS["python"]
    model_id = site.get("model_id") or _DEFAULTS["model_id"]
    sae_default = site.get("sae_default") or _DEFAULTS["sae_default"]
    env_extra = dict(_ENV)
    env_extra.update(site.get("env", {}) or {})
    return python, model_id, sae_default, env_extra


def main():
    row = json.load(open(sys.argv[1]))
    cwd = os.getcwd()
    rid = row.get("id", "row")
    tile = os.environ.get("ZE_AFFINITY_MASK", "unset")
    host = socket.gethostname()
    python, model_id, sae_default, env_extra = _resolve()

    if not os.path.exists(python):
        sys.exit(f"[genslm_esm_run_row] genslm venv python not found: {python!r} "
                 f"(set site['genslm_esm']['python']; MUST be venvs/aurora, not "
                 f"venvs/ci which is CUDA-only)")
    if not row.get("sequence"):
        sys.exit("[genslm_esm_run_row] row has no 'sequence'")

    out_dir = os.path.join(cwd, "out")
    driver = os.path.join(cwd, "_genslm_embed_driver.py")
    open(driver, "w").write(_DRIVER_SRC)
    row_path = os.path.join(cwd, "_row.json")
    json.dump(row, open(row_path, "w"))

    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items()})
    # Scrub stray MPI/PMIx rank env so no distributed init fires with no peers.
    for k in ("PMIX_RANK", "PMIX_NAMESPACE", "PMI_RANK", "PMI_SIZE", "RANK",
              "LOCAL_RANK", "LOCAL_WORLD_SIZE", "PALS_RANKID", "PALS_NODEID",
              "PARSL_WORKER_RANK"):
        env.pop(k, None)
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    # ZE_AFFINITY_MASK is set per-row by EL; keep it so torch.device('xpu') picks
    # the pinned tile.

    # The venvs/aurora venv needs the Aurora frameworks module loaded for XPU
    # torch (it is built on the module with include-system-site-packages=true).
    # Run through a login shell so `module` is available, load frameworks, then
    # exec the venv python on the driver.
    # TODO(verify): if worker_init already `module load frameworks`, this reload
    # is a harmless no-op; keep it so the runner is self-sufficient.
    inner = (
        "set -e; "
        "module use /soft/modulefiles >/dev/null 2>&1 || true; "
        "module load frameworks >/dev/null 2>&1 || true; "
        f"exec {python} {driver} {row_path} {out_dir} {model_id} {sae_default}"
    )
    cmd = ["bash", "-lc", inner]

    print(f"[genslm_esm_run_row] START id={rid} tile={tile} host={host} "
          f"modality={row.get('modality', 'codon')} layer={row.get('layer', 32)} "
          f"sae={bool(row.get('sae'))}", flush=True)
    log = os.path.join(cwd, "genslm_esm.log")
    t0 = time.time()
    with open(log, "wb") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode

    results_path = os.path.join(out_dir, "results.json")
    device = None
    finite = None
    if os.path.isfile(results_path):
        try:
            r = json.load(open(results_path))
            device = r.get("device")
            finite = r.get("finite")
        except Exception:
            pass

    import glob
    outs = sorted(os.path.relpath(p, out_dir)
                  for p in glob.glob(os.path.join(out_dir, "*.pt")))
    n_out = len(outs)

    # Traps: ran on CPU (silent XPU-missing / wrong venv), or non-finite acts,
    # or no results/output produced.
    on_cpu = device is not None and not str(device).startswith("xpu")
    fail = (rc != 0) or (not os.path.isfile(results_path)) or on_cpu \
        or (finite is False) or (n_out == 0)
    rec = {
        "id": rid, "host": host, "tile": tile, "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "device": device,
        "finite": finite,
        "n_pt": n_out,
        "outputs": outs,
        "status": "fail" if fail else "ok",
    }
    json.dump(rec, open(os.path.join(cwd, "result.json"), "w"), indent=2)
    print("[genslm_esm_run_row] RESULT " + json.dumps(rec), flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
