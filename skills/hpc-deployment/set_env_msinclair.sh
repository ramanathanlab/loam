#!/bin/bash
# msinclair's copy of set_env.sh for Aurora.
#
# LIVE for campaign1, which runs execution.mode: remote_mcp against the EGFR
# binder-design goal. HPC_ENV_FILE in .hermes/.loam-home/config.yaml points here.
#
# Allocation: 10 nodes for one hour on the debug-scaling queue, charged to
# FRAME-IDP. Endpoint, account, queue and paths are all set.
#
# Fully populated, and the worker-side import chain is verified: hpc_agent,
# pool_manager, ensemble_launcher, academy.manager and globus_compute_sdk all
# import under the on-node venv. Nothing is left to fill in.
#
# The endpoint ignores a client-supplied worker_init -- see the long note above
# HPC_ENDPOINT_CONFIG_JSON. Venv, modules, TMPDIR and the ALCF proxy live in
# ~/.globus_compute/flex_scaling/user_config_template.yaml.j2 instead.

# The Globus Compute MANAGER UUID of your endpoint -- the value `globus-compute-
# endpoint list` shows for it, not the endpoint name. Create one first:
#   globus-compute-endpoint configure <name> && globus-compute-endpoint start <name>
export HPC_GLOBUS_ENDPOINT_ID="d83212d9-5751-4892-a78a-bd60d7232fc2"

# Client agents in the pool = how many children can submit AT THE SAME TIME.
# Keep this >= the number of concurrent validators, each of which may spawn up
# to three code-generator children. At 1, children queue on a single submitter
# slot; whoever waits times out, and three consecutive timeouts open Hermes'
# circuit breaker -- which is a PROCESS-GLOBAL dict keyed by server name
# (tools/mcp_tool.py:_server_breaker_opened_at). So one child's queueing
# convinces every other child the server is dead, producing a flood of
# "endpoint unreachable" reports against a perfectly healthy endpoint.
# In cluster mode a client agent only SUBMITS into the EL cluster and holds no
# accelerator. EL owns GPU placement and fanout across all HPC_NNODES x 12
# XPUs, so throughput does NOT scale with this number.
#
# A pool agent is NOT a concurrency slot. HpcClientAgent.run_code does
#   future = await asyncio.to_thread(self._client.submit, ...)
#   return await asyncio.wrap_future(future)
# with no lock or semaphore anywhere in the agent, so ONE agent multiplexes
# many in-flight run_code calls: the submit hops to a thread and the await
# costs nothing. ClusterPool.next_agent() is round-robin "with no work per
# call" -- it hands agents out, it does not lease them exclusively.
#
# The earlier comment here claimed one client agent per concurrent child was
# required, and sized this at 110 to match top_k. That described a serialized
# submitter the code does not implement. The real per-agent limits are
# asyncio.to_thread's default pool (~32 threads) and EL's internal submit lock,
# so 5 agents carry ~160 concurrent submits -- well past any child count this
# campaign produces.
#
# Small is actively better here. On 2026-10-31 a 110-agent pool took ~620 s to
# assemble and missed the preflight's 900 s gate by 15 s, and its 111 exchange
# mailboxes drew 19 HTTP 500s on /v1/mailbox/listen during startup. Six agents
# cut both, shrink the blast radius of the non-refreshing exchange token, and
# stay far inside the HPC_NNODES x 12 slot budget.
export HPC_POOL_SIZE=5

# batches/ and tools/ are created here; code-gen scripts and the profile's
# execution.workdir_root must live under it.
export HPC_BASE_DIR="/flare/FRAME-IDP/msinclair/hpc-run"
export HPC_CAMPAIGN_ID="msinclair-egfr-1"
export HPC_SYSTEM="aurora"

# Read-only paths inspect_filesystem may look at (colon-separated). Left unset,
# it is confined to HPC_BASE_DIR -- so the agent cannot stat or read artifacts it
# wrote outside that tree, and every inspect_filesystem call on them fails with
# "path is outside allowed inspection roots". List BOTH spellings of each tree:
# /flare and /lus/flare/projects are the same filesystem and the agent uses them
# interchangeably.
export HPC_INSPECT_ROOTS="/flare/FRAME-IDP/msinclair/hpc-run:/lus/flare/projects/FRAME-IDP/msinclair/hpc-run"

# Nodes in the allocation. Both the PBS request (rendered into nodes_per_block)
# and the EL cluster's leaf count must agree.
#
# This also fixes the agent slot budget at HPC_NNODES x 12 -- see the CEILING
# note above HPC_POOL_SIZE before changing it. The two must move together.
export HPC_NNODES=10
export ACADEMY_EXCHANGE_URL="https://exchange.academy-agents.org/v1"
export HPC_MCP_TRANSPORT="http"

# ---- Allocation knobs (edit these) -----------------------------------------
# These render into HPC_ENDPOINT_CONFIG_JSON below, so change the allocation's
# queue/walltime/account here rather than editing the JSON string by hand. The
# queue constrains walltime and node count -- validate with `qstat -Qf <queue>`
# on Aurora before a long run.
#
# Defaults are a 30-minute, 2-node DEBUG block -- the smoke-test footprint. For
# a real campaign raise these, e.g. HPC_QUEUE=capacity and HPC_WALLTIME=12:00:00
# (and remember the debug queue reclaims at its short walltime -> ManagerLost).
export HPC_ACCOUNT="FRAME-IDP"
export HPC_QUEUE="debug-scaling"
export HPC_WALLTIME="01:00:00"

# Guardrail caps hpc_control.validate() checks every allocation against. Its
# defaults (debug,debug-scaling,capacity; 4 nodes) reject the reservation above.
# They must be here, not only in the Hermes config: the launch preflight sources
# this file and starts the allocation before Hermes exists.
export HPC_ALLOWED_QUEUES="debug,debug-scaling,capacity,R8984162"
export HPC_MAX_NODES=256
export HPC_MAX_WALLTIME="12:00:00"

# Block maintenance -- QUEUE-AGNOSTIC. These make Loam keep one block of
# whatever queue is selected above alive and request a REPLACEMENT by itself
# when a block ends or its node is lost, so a 30-min debug block self-replaces
# during a longer campaign. min_blocks=1 is what makes the endpoint request a
# fresh node on its own; init_blocks=1 grabs one at startup; max_idletime is how
# long an idle block is held before scale-down (kept high so a lull between jobs
# does not drop the allocation). Raise HPC_MAX_BLOCKS to 2 if you want the
# replacement to overlap the expiring block at the cost of holding two.
export HPC_INIT_BLOCKS="${HPC_INIT_BLOCKS:-1}"
export HPC_MIN_BLOCKS="${HPC_MIN_BLOCKS:-1}"
export HPC_MAX_BLOCKS="${HPC_MAX_BLOCKS:-1}"
export HPC_MAX_IDLETIME="${HPC_MAX_IDLETIME:-3600}"
# ----------------------------------------------------------------------------

# This endpoint (d83212d9 = `flex_scaling`) accepts ONLY these Jinja variables:
#   account available_accelerators base_dir block_error_handler display_name
#   filesystems ifname init_blocks max_blocks max_workers_per_node min_blocks
#   nodes_per_block place prefetch_capacity queue walltime
#
# `worker_init` is NOT among them. The template hardcodes its own worker_init
# and never interpolates a client-supplied one, so a worker_init key here is
# silently DISCARDED -- it does not fail, it just has no effect. The venv
# activation, module loads, TMPDIR fix and ALCF proxy exports all live in the
# template at ~/.globus_compute/flex_scaling/user_config_template.yaml.j2.
#
# `base_dir` is the seam that replaces it: the template does
#   cd <base_dir>; export PYTHONPATH=<base_dir>/skills/globus-compute-mcp:<base_dir>
# and workers must import hpc_agent and pool_manager by reference (cloudpickle
# serializes module-level functions by reference, so `cd` alone is not enough).
# The directory below holds copies of those two modules taken from this repo's
# skills/hpc-deployment/scripts/ -- keep them in sync, or the local server and
# the remote workers run different code.
#
# max_workers_per_node is 12, NOT HPC_POOL_SIZE + 1: an Aurora node has 6 Intel
# Data Center GPU Max x 2 tiles = 12 addressable XPUs, and this template pins
# available_accelerators to match. The POOL_SIZE + 1 formula was written for a
# 2-node block where the pool's agents had to fit on few nodes.
#
# Because available_accelerators is pinned, 12 is a CEILING and not a request:
# Parsl derives the per-node worker count from the accelerator list, so raising
# this number buys nothing. At HPC_NNODES=10 the block therefore holds 10 x 12
# = 120 agent slots, which is what caps HPC_POOL_SIZE at 119.
#
# Do NOT quote walltime here: the manager's Jinja env already quotes
# client-supplied strings, and a second pair renders ""01:00:00"" -> ParserError.
# Unquoted is also wrong (YAML 1.1 reads 01:00:00 as sexagesimal 3600), so a
# plain JSON string is exactly right.
export HPC_ENDPOINT_CONFIG_JSON='{"account": "'"$HPC_ACCOUNT"'", "queue": "'"$HPC_QUEUE"'", "walltime": "'"$HPC_WALLTIME"'", "nodes_per_block": '"$HPC_NNODES"', "max_workers_per_node": 12, "available_accelerators": 12, "init_blocks": '"$HPC_INIT_BLOCKS"', "min_blocks": '"$HPC_MIN_BLOCKS"', "max_blocks": '"$HPC_MAX_BLOCKS"', "max_idletime": '"$HPC_MAX_IDLETIME"', "base_dir": "/flare/FRAME-IDP/msinclair/loam-hpc"}'
# Optional: HPC_TOOL_SMOKE_TIMEOUT=120, HPC_LOCAL_STAGE=/tmp/loam-tools
