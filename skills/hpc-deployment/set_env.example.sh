#!/bin/bash
# Template for the HPC MCP server environment. Copy it to set_env.local.sh
# (gitignored) and fill in every <PLACEHOLDER>:
#
#     cp skills/hpc-deployment/set_env.example.sh skills/hpc-deployment/set_env.local.sh
#
# run-campaign, campaign_preflight.py and hpc_control.py use set_env.local.sh
# unless HPC_ENV_FILE names another file. Values below are for Aurora; see
# set_env_akv.sh for a filled-in example.

# Your Globus Compute endpoint on the login node (`globus-compute-endpoint list`).
# It must be one your Globus identity may submit to; someone else's endpoint
# returns 403 ENDPOINT_ACCESS_FORBIDDEN. Register your own with
# globus_compute_endpoint_configs/alcf_aurora/user_config_template.yaml.j2.
export HPC_GLOBUS_ENDPOINT_ID="<ENDPOINT_UUID>"
# Client agents in the pool = how many children can submit at the same time.
# At 1, concurrent validators queue on one submitter slot and time out; see
# set_env_akv.sh for the full history.
export HPC_POOL_SIZE=4
# batches/ and tools/ are created here; code-gen scripts and the profile's
# workdir_root must live under it. Must be writable by you.
export HPC_BASE_DIR="/flare/<PROJECT>/<USER>/hpc-run"
export HPC_CAMPAIGN_ID="<CAMPAIGN_ID>"
export HPC_SYSTEM="aurora"
# Read-only paths inspect_filesystem may look at (colon-separated). Unset, it is
# confined to HPC_BASE_DIR. List both /flare and /lus/flare/projects spellings.
export HPC_INSPECT_ROOTS="/flare/<PROJECT>/<USER>/hpc-run:/lus/flare/projects/<PROJECT>/<USER>/hpc-run"
# Nodes in the allocation. Both the PBS request (`nodes_per_block` below) and the
# EL cluster's leaf count must agree.
export HPC_NNODES=1
export ACADEMY_EXCHANGE_URL="https://exchange.academy-agents.org/v1"
export HPC_MCP_TRANSPORT="http"
# Worker venv: py3.12 on `module load frameworks`, --system-site-packages, with
# globus-compute-endpoint matching the endpoint (4.19.0), academy-py 1.0.0,
# dill 0.3.9 and ensemble_launcher. WORKER_PYTHONPATH must hold this repo's
# skills/hpc-deployment/scripts (hpc_agent.py, pool_manager.py) on the cluster.
HPC_WORKER_VENV="/flare/<PROJECT>/<USER>/loam-hpc/venv"
HPC_WORKER_PYTHONPATH="/flare/<PROJECT>/<USER>/loam-hpc"
# One worker slot per client agent, plus one for the long-running
# HpcClusterAgent. No available_accelerators: EL owns tile placement, not Parsl.
# Keys match globus_compute_endpoint_configs/alcf_aurora/user_config_template.yaml.j2.
# worker_init must be one line: the template renders it as one quoted YAML string.
export HPC_ENDPOINT_CONFIG_JSON='{"account": "<PROJECT>", "queue": "debug", "walltime": "00:30:00", "nodes_per_block": '"$HPC_NNODES"', "max_workers_per_node": '"$((HPC_POOL_SIZE + 1))"', "worker_init": "export TMPDIR=/tmp; export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128 NO_PROXY=localhost,127.0.0.1,.alcf.anl.gov no_proxy=localhost,127.0.0.1,.alcf.anl.gov; module reset; module load frameworks; source '"$HPC_WORKER_VENV"'/bin/activate; which python; export PYTHONPATH='"$HPC_WORKER_PYTHONPATH"':$PYTHONPATH"}'
# Optional: HPC_TOOL_SMOKE_TIMEOUT=120, HPC_LOCAL_STAGE=/tmp/loam-tools
