#!/bin/bash
# CSCS Alps / Daint (GH200) operator env for the HPC MCP server, pointing at the
# SLURM-templatable Globus Compute endpoint `loam-cscs-alps`. SLURM port of
# set_env_akv.sh (which targets Aurora/PBS). Sourced by hpc_control.py and
# mcp_server.py; keys in HPC_ENDPOINT_CONFIG_JSON match
# globus_compute_endpoint_configs/cscs_alps/user_config_template.yaml.j2.

# loam-cscs-alps on Alps (~/.globus_compute/loam-cscs-alps, daint-ln004).
export HPC_GLOBUS_ENDPOINT_ID="6bdbfb6e-42e4-4590-b891-9c153d81c33d"
# Client agents in the pool = how many children can submit at the same time.
# max_workers_per_node below follows this value (POOL_SIZE + 1 for the
# long-running HpcClusterAgent slot).
export HPC_POOL_SIZE=4
# batches/ and tools/ are created here; code-gen scripts and the profile's
# workdir_root must live under it. On /ritom scratch (mounted on compute nodes,
# as the endpoint smoke test confirmed).
export HPC_BASE_DIR="/ritom/scratch/cscs/avasan/hpc-run"
export HPC_CAMPAIGN_ID="akv-cscs-1"
# 'alps' selects Hari's (EL lead) canonical Alps profile: CUDA_VISIBLE_DEVICES
# pinning (4 GH200s/node), a multiprocessing leaf-child launch for single-node
# clusters (Alps has no mpiexec/mpirun -- it is an srun system), and srun-flavored
# MPI (--nodelist/--nodefile, --overlap --gpus-per-node=4 --gpu-bind=none,
# cpu_bind none) for multi-node. 'cscs' is accepted as an alias.
export HPC_SYSTEM="alps"
# Read-only paths inspect_filesystem may look at (colon-separated). Unset, it is
# confined to HPC_BASE_DIR. Add your campaign input/output trees here.
export HPC_INSPECT_ROOTS="/ritom/scratch/cscs/avasan/hpc-run"
# Nodes in the allocation. Must agree with nodes_per_block in the JSON below.
export HPC_NNODES="${HPC_NNODES:-1}"
export ACADEMY_EXCHANGE_URL="https://exchange.academy-agents.org/v1"
export HPC_MCP_TRANSPORT="http"

# ---- Allocation knobs (edit these) -----------------------------------------
# These render into HPC_ENDPOINT_CONFIG_JSON below. HPC_QUEUE is the SLURM
# PARTITION (the template maps the `queue` key -> SlurmProvider.partition).
# Alps partitions (sinfo): debug (<=30m, 24 nodes), low / normal (<=24h, 922
# nodes). Account g245, QOS normal. All four knobs are ${VAR:-default}, so you
# can override per launch without editing this file, e.g.:
#
#   HPC_QUEUE=normal HPC_WALLTIME=04:00:00 HPC_NNODES=2 ./run-campaign
#
# The hpcctl caps in .hermes/.loam-home/config.yaml (HPC_ALLOWED_QUEUES,
# HPC_MAX_NODES, HPC_MAX_WALLTIME) gate this independently -- raise them there
# too before asking for more.
export HPC_ACCOUNT="${HPC_ACCOUNT:-g245}"
export HPC_QUEUE="${HPC_QUEUE:-debug}"
export HPC_WALLTIME="${HPC_WALLTIME:-00:30:00}"
# Block maintenance. min_blocks=1 makes the endpoint request a fresh node on its
# own when a block ends/is lost; init_blocks=1 grabs one at startup; max_idletime
# is how long an idle block is held before scale-down (high so a lull between
# runs does not drop the allocation). Raise HPC_MAX_BLOCKS to 2 to overlap an
# expiring block with its replacement (no coverage gap, briefly two blocks).
export HPC_INIT_BLOCKS="${HPC_INIT_BLOCKS:-1}"
export HPC_MIN_BLOCKS="${HPC_MIN_BLOCKS:-1}"
export HPC_MAX_BLOCKS="${HPC_MAX_BLOCKS:-1}"
export HPC_MAX_IDLETIME="${HPC_MAX_IDLETIME:-3600}"
# ----------------------------------------------------------------------------
# No `available_accelerators`: as in the Aurora config, Loam runs POOL_SIZE+1
# non-accelerator worker slots and manages GPUs inside run_code, so Parsl leaves
# all 4 GH200s visible to the worker rather than pinning one per slot.
# worker_init is ONE line (rendered as a single quoted YAML string): /tmp TMPDIR
# first (socket-path limit), activate the loam venv (has globus-compute-endpoint,
# academy, parsl), and put the hpc-deployment scripts on PYTHONPATH so cluster
# workers import hpc_agent.py. No ALCF proxy / `module load frameworks` -- those
# are Aurora-only.
export HPC_ENDPOINT_CONFIG_JSON='{"account": "'"$HPC_ACCOUNT"'", "queue": "'"$HPC_QUEUE"'", "partition": "'"$HPC_QUEUE"'", "walltime": "'"$HPC_WALLTIME"'", "nodes_per_block": '"$HPC_NNODES"', "max_workers_per_node": '"$((HPC_POOL_SIZE + 1))"', "init_blocks": '"$HPC_INIT_BLOCKS"', "min_blocks": '"$HPC_MIN_BLOCKS"', "max_blocks": '"$HPC_MAX_BLOCKS"', "max_idletime": '"$HPC_MAX_IDLETIME"', "worker_init": "export TMPDIR=/tmp; source /ritom/scratch/cscs/avasan/loam_venv/bin/activate; which python; export PYTHONPATH=/ritom/scratch/cscs/avasan/loam/skills/hpc-deployment/scripts:$PYTHONPATH"}'
# Optional: HPC_TOOL_SMOKE_TIMEOUT=120, HPC_LOCAL_STAGE=/tmp/loam-tools
