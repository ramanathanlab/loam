#!/bin/bash

export HPC_GLOBUS_ENDPOINT_ID="3724d6df-cdaa-47c7-8f1d-61b86847627e"
export HPC_POOL_SIZE=1
export HPC_BASE_DIR="/lus/flare/projects/datascience/hari/loam/skills/hpc-deployment/run"
export HPC_CAMPAIGN_ID="test2"
export HPC_SYSTEM="aurora"
# Nodes in the allocation. Both the PBS request (`nodes` below, rendered into
# nodes_per_block) and the EL cluster's leaf count must agree.
export HPC_NNODES=1
export ACADEMY_EXCHANGE_URL="https://exchange.academy-agents.org/v1"
export HPC_MCP_TRANSPORT="http"
# One worker slot per client agent, plus one for the long-running
# HpcClusterAgent, which occupies its slot for the life of the allocation.
# A cluster rebuild shuts the stale agents down before launching replacements,
# so +1 is sufficient; +2 gives that handover headroom if the endpoint can
# spare it and a slow slot release is making restarts drag.
# gpus=0 leaves accelerator placement to EL rather than Parsl.
export HPC_ENDPOINT_CONFIG_JSON='{"target": "pbs", "queue": "debug", "walltime": "00:30:00", "nodes": '"$HPC_NNODES"', "workers":'"$((HPC_POOL_SIZE + 1))"', "gpus": 0, "worker_init": "module reset; module load frameworks; source /lus/flare/projects/datascience/hari/loam/venv/bin/activate; which python; export PYTHONPATH=/lus/flare/projects/datascience/hari/loam/skills/hpc-deployment/scripts:$PYTHONPATH; export TMPDIR=/tmp"}'