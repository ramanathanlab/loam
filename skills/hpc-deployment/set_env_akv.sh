#!/bin/bash
# Archit's copy of set_env.sh for the dev_akv tool library on Aurora.
# Differs from set_env.sh in: HPC_BASE_DIR, HPC_CAMPAIGN_ID, and worker_init
# (own venv; PYTHONPATH -> /flare/FRAME-IDP/avasan/loam-hpc, which holds the
# dev_akv hpc_agent.py with register_tool / list_tools / run_batch(tool=)).

# TODO: set to the loam-aurora-flex endpoint UUID once it is configured.
export HPC_GLOBUS_ENDPOINT_ID="REPLACE_WITH_LOAM_AURORA_FLEX_UUID"
export HPC_POOL_SIZE=1
# batches/ and tools/ are created here; code-gen scripts and the profile's
# workdir_root must live under it.
export HPC_BASE_DIR="/flare/FRAME-IDP/avasan/hpc-run"
export HPC_CAMPAIGN_ID="akv-aurora-1"
export HPC_SYSTEM="aurora"
# Nodes in the allocation. Both the PBS request (`nodes` below, rendered into
# nodes_per_block) and the EL cluster's leaf count must agree.
export HPC_NNODES=1
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
