#!/bin/bash

export HPC_GLOBUS_ENDPOINT_ID="36bd2d76-4591-43c5-b3ae-9cd2e3fafd82"
export HPC_POOL_SIZE=1
export HPC_BASE_DIR="/lus/eagle/projects/datascience/hari/loam/skills/hpc-deployment/scripts"
export HPC_CAMPAIGN_ID="test2"
export ACADEMY_EXCHANGE_URL="https://exchange.academy-agents.org/v1"
# One worker slot per client agent, plus one for the long-running
# HpcClusterAgent, which occupies its slot for the life of the allocation.
# gpus=0 leaves accelerator placement to EL rather than Parsl.
export HPC_ENDPOINT_CONFIG_JSON='{"target": "pbs", "queue": "debug", "walltime": "00:10:00", "workers":'"$((HPC_POOL_SIZE + 1))"', "gpus": 0, "worker_init": "module reset; module use /soft/modulefiles; module load conda/2026-10-01; conda activate base; source /lus/eagle/projects/datascience/hari/loam/venv/bin/activate; which python; export PYTHONPATH=/lus/eagle/projects/datascience/hari/loam/skills/hpc-deployment/scripts:$PYTHONPATH"}'