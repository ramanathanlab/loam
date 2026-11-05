#!/usr/bin/env python3
"""Fill the live allocation shape into the hpc-deployment skill's LOAM_LIVE_ALLOC
sentinel so code-generators size run_batch batches to the REAL pool, not the
campaign goal's headline node count. Idempotent; never raises into the launch.

Usage: fill_alloc_skill.py <SKILL.md>
Reads top_k from $LOAM_CONFIG_PATH, node count from the live allocation
(hpc_control) or $HPC_NNODES. Exits 0 if it rewrote the sentinel, 1 otherwise.
"""
import os
import re
import sys

def top_k():
    try:
        import yaml
        d = yaml.safe_load(open(os.environ["LOAM_CONFIG_PATH"]))
        p = d["profiles"][d["active_profile"]]
        return int(p.get("hypothesis_tournament", {}).get("top_k", 3)) or 3
    except Exception:
        return 3

def nodes():
    nn = os.environ.get("HPC_NNODES", "")
    if nn.isdigit() and int(nn) > 0:
        return int(nn)
    try:
        sys.path.insert(0, os.path.join(os.environ.get("LOAM_REPO", "."),
                                        "skills/hpc-deployment/scripts"))
        import hpc_control
        st = hpc_control.Controller().status()
        n = int((st.get("settings") or {}).get("nodes") or 0)
        return n or None
    except Exception:
        return None

def main():
    if len(sys.argv) < 2:
        return 1
    f = sys.argv[1]
    n = nodes()
    if not n:
        return 1
    k = top_k()
    per_node = {"aurora": 12, "polaris": 4, "alps": 4, "cscs": 4, "jupiter": 4}.get(
        os.environ.get("HPC_SYSTEM", "aurora"), 12)
    tiles = n * per_node
    share = max(tiles // max(k, 1), 1)
    line = (f"**Live pool for this run: {n} nodes x {per_node} = {tiles} tiles.** Your fair "
            f"share is ~{share} tiles (the pool divided {k} ways by top_k; all "
            f"code-generators draw on the same tiles). A run_batch row = 1 tile "
            f"(gpus=1): size each batch to ~{share} rows per wave, poll batch_status "
            f"to unfinished==0, then submit the next wave. More rows just queue and "
            f"will not finish in the hour.")
    try:
        s = open(f).read()
        s2 = re.sub(r"(<!-- LOAM_LIVE_ALLOC -->).*", r"\1 " + line.replace("\\", "\\\\"),
                    s, count=1)
        if s2 != s:
            open(f, "w").write(s2)
            return 0
    except Exception:
        pass
    return 1

sys.exit(main())
