#!/bin/bash
# Archit's copy of set_env.sh for the dev_akv tool library on Aurora.
# Differs from set_env.sh in: HPC_BASE_DIR, HPC_CAMPAIGN_ID, and worker_init
# (own venv; PYTHONPATH -> /flare/FRAME-IDP/avasan/loam-hpc, which holds the
# dev_akv hpc_agent.py with register_tool / list_tools / run_batch(tool=)).

# loam-aurora-flex on Aurora (~/.globus_compute/loam-aurora-flex, aurora-uan-0009).
export HPC_GLOBUS_ENDPOINT_ID="e9a292e7-a158-4418-8897-aef6a113657e"
# Client agents in the pool = how many children can submit AT THE SAME TIME.
# Raised 1 -> 4 on 2026-10-21. At 1, the orchestration's three concurrent
# validators (each allowed up to three code-generator children) all queued on a
# single submitter slot; whoever waited timed out, and three consecutive
# timeouts open Hermes' circuit breaker, which is a PROCESS-GLOBAL dict keyed by
# server name (tools/mcp_tool.py:_server_breaker_opened_at). So one child's
# queueing convinced every other child the server was dead: 46 "endpoint
# unreachable" child reports in one evening against an endpoint with zero
# ManagerLost events and a verified working GPU.
# These slots are cheap: in cluster mode the client agent only submits and holds
# no accelerator (pool_manager.py), and max_workers_per_node below follows this
# value automatically.
export HPC_POOL_SIZE=4
# batches/ and tools/ are created here; code-gen scripts and the profile's
# workdir_root must live under it.
export HPC_BASE_DIR="/flare/FRAME-IDP/avasan/hpc-run"
export HPC_CAMPAIGN_ID="akv-aurora-1"
export HPC_SYSTEM="aurora"
# Read-only paths inspect_filesystem may look at (colon-separated). Unset, it is
# confined to HPC_BASE_DIR, so the agent could not stat or read the very
# artifacts it had just written: folds land under the binder_design project tree,
# and every inspect_filesystem call on them failed with "path is outside allowed
# inspection roots" (observed 2026-10-21, after the round1_h1 Chai-1 fold). Both
# spellings of each tree are listed because /flare and /lus/flare/projects are
# the same filesystem and the agent uses them interchangeably.
export HPC_INSPECT_ROOTS="/flare/FRAME-IDP/avasan/hpc-run:/lus/flare/projects/FRAME-IDP/avasan/hpc-run:/flare/FRAME-IDP/agentic_campaign_09_26/binder_design:/lus/flare/projects/FRAME-IDP/agentic_campaign_09_26/binder_design"
# Nodes in the allocation. Both the PBS request (`nodes` below, rendered into
# nodes_per_block) and the EL cluster's leaf count must agree.
export HPC_NNODES=2
export ACADEMY_EXCHANGE_URL="https://exchange.academy-agents.org/v1"
export HPC_MCP_TRANSPORT="http"
# One worker slot per client agent, plus one for the long-running
# HpcClusterAgent, which occupies its slot for the life of the allocation.
# No available_accelerators: EL owns tile placement, not Parsl.
# venv (on frameworks, py3.12, --system-site-packages): academy-py 1.0.0,
# globus-compute-endpoint/sdk 4.19.0, dill 0.3.9, ensemble_launcher@c36dc43.
# Keys match globus_compute_endpoint_configs/alcf_aurora/user_config_template.yaml.j2.
# worker_init must be one line: the template renders it as one quoted YAML string.
export HPC_ENDPOINT_CONFIG_JSON='{"account": "FRAME-IDP", "queue": "debug", "walltime": "00:30:00", "nodes_per_block": '"$HPC_NNODES"', "max_workers_per_node": '"$((HPC_POOL_SIZE + 1))"', "worker_init": "export TMPDIR=/tmp; export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128 NO_PROXY=localhost,127.0.0.1,.alcf.anl.gov no_proxy=localhost,127.0.0.1,.alcf.anl.gov; module reset; module load frameworks; source /flare/FRAME-IDP/avasan/loam-hpc/venv/bin/activate; which python; export PYTHONPATH=/flare/FRAME-IDP/avasan/loam-hpc:$PYTHONPATH"}'
# Optional: HPC_TOOL_SMOKE_TIMEOUT=120, HPC_LOCAL_STAGE=/tmp/loam-tools
