#!/bin/bash
# Reproduce the endpoint's worker_init exactly, then prove a worker can import
# everything mcp_server.py's agents need. Run on Aurora (login node is fine for
# the import check; GPU presence is not asserted).
export TMPDIR=/tmp
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
export NO_PROXY=localhost,127.0.0.1,.alcf.anl.gov
export no_proxy=$NO_PROXY
module use /soft/modulefiles >/dev/null 2>&1
module load frameworks >/dev/null 2>&1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
source /flare/FRAME-IDP/msinclair/envs/loam/bin/activate
cd /flare/FRAME-IDP/msinclair/loam-hpc
export PYTHONPATH=/flare/FRAME-IDP/msinclair/loam-hpc:$PYTHONPATH

python - <<'PY'
import sys
CHECKS = [
    ("hpc_agent",          ["HpcAgent", "HpcClientAgent", "HpcClusterAgent"]),
    ("pool_manager",       ["CampaignPool", "ClusterPool"]),
    ("ensemble_launcher",  []),
    ("academy.manager",    ["Manager"]),
    ("academy.exchange",   ["HttpExchangeFactory"]),
    ("globus_compute_sdk", ["Executor"]),
]
ok = True
print("  frameworks python:", sys.version.split()[0])
print("  prefix           :", sys.prefix)
for mod, names in CHECKS:
    try:
        m = __import__(mod, fromlist=names or ["__name__"])
        missing = [n for n in names if not hasattr(m, n)]
        print("  {:22s} OK{}".format(mod, "  MISSING {}".format(missing) if missing else ""))
        if missing:
            ok = False
    except Exception as exc:
        print("  {:22s} FAIL {}: {}".format(mod, type(exc).__name__, str(exc)[:60]))
        ok = False

# Version parity with the manager matters: dill/cloudpickle deserialize the
# payloads, academy speaks a wire protocol, and a mismatch shows up as an
# unpickling error or a bare EOFError rather than a version complaint.
import importlib.metadata as md
print("  -- versions (must match the local manager venv) --")
for p in ("academy-py", "globus-compute-sdk", "globus-compute-endpoint",
          "dill", "cloudpickle", "ensemble-launcher"):
    try:
        print("  {:26s} {}".format(p, md.version(p)))
    except Exception:
        print("  {:26s} MISSING".format(p))
print("  AGENT CHAIN OK" if ok else "  AGENT CHAIN FAILED")
sys.exit(0 if ok else 1)
PY
