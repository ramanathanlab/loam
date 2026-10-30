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
# ---- Allocation knobs (edit these) -----------------------------------------
# These render into HPC_ENDPOINT_CONFIG_JSON below, so change the allocation's
# queue/walltime/account here rather than editing the JSON string by hand. The
# queue constrains walltime and node count -- validate with `qstat -Qf <queue>`
# on Aurora before a long run. (HPC_NNODES and HPC_POOL_SIZE are set above;
# max_workers_per_node follows HPC_POOL_SIZE automatically.)
#
# Defaults are a 30-minute, 2-node DEBUG block -- the smoke-test footprint. For
# a real campaign raise these, e.g. HPC_QUEUE=capacity and HPC_WALLTIME=12:00:00
# (and remember the debug queue reclaims at its short walltime -> ManagerLost).
export HPC_ACCOUNT="FRAME-IDP"
export HPC_QUEUE="debug"
export HPC_WALLTIME="00:30:00"
# Block maintenance -- QUEUE-AGNOSTIC. These make Loam keep one block of whatever
# queue is selected above alive and request a REPLACEMENT by itself when a block
# ends or its node is lost, so a 30-min debug block self-replaces during a longer
# campaign, and prod/capacity behave the same with no code change. min_blocks=1
# is what makes the endpoint request a fresh node on its own; init_blocks=1 grabs
# one at startup; max_idletime is how long an idle block is held before scale-down
# (kept high so a lull between folds does not drop the allocation). Raise
# HPC_MAX_BLOCKS to 2 if you want the replacement to overlap the expiring block
# (no coverage gap) at the cost of briefly holding two blocks.
export HPC_INIT_BLOCKS="${HPC_INIT_BLOCKS:-1}"
export HPC_MIN_BLOCKS="${HPC_MIN_BLOCKS:-1}"
export HPC_MAX_BLOCKS="${HPC_MAX_BLOCKS:-1}"
export HPC_MAX_IDLETIME="${HPC_MAX_IDLETIME:-3600}"
# ----------------------------------------------------------------------------
# One worker slot per client agent, plus one for the long-running
# HpcClusterAgent, which occupies its slot for the life of the allocation.
# No available_accelerators: EL owns tile placement, not Parsl.
# venv (on frameworks, py3.12, --system-site-packages): academy-py 1.0.0,
# globus-compute-endpoint/sdk 4.19.0, dill 0.3.9, ensemble_launcher@c36dc43.
# Keys match globus_compute_endpoint_configs/alcf_aurora/user_config_template.yaml.j2.
# worker_init must be one line: the template renders it as one quoted YAML string.
export HPC_ENDPOINT_CONFIG_JSON='{"account": "'"$HPC_ACCOUNT"'", "queue": "'"$HPC_QUEUE"'", "walltime": "'"$HPC_WALLTIME"'", "nodes_per_block": '"$HPC_NNODES"', "max_workers_per_node": '"$((HPC_POOL_SIZE + 1))"', "init_blocks": '"$HPC_INIT_BLOCKS"', "min_blocks": '"$HPC_MIN_BLOCKS"', "max_blocks": '"$HPC_MAX_BLOCKS"', "max_idletime": '"$HPC_MAX_IDLETIME"', "worker_init": "export TMPDIR=/tmp; export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128 NO_PROXY=localhost,127.0.0.1,.alcf.anl.gov no_proxy=localhost,127.0.0.1,.alcf.anl.gov; module reset; module load frameworks; module load copper; launch_copper_aurora.sh; source /flare/FRAME-IDP/avasan/loam-hpc/venv/bin/activate; which python; export PYTHONPATH=/flare/FRAME-IDP/avasan/loam-hpc:$PYTHONPATH"}'
# Optional: HPC_TOOL_SMOKE_TIMEOUT=120, HPC_LOCAL_STAGE=/tmp/loam-tools
