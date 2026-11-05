#!/bin/bash
# TEMPLATE / RECIPE for a new campaign's HPC environment. It is user- and
# campaign-neutral: copy it, fill the blanks, and nothing in it names a person.
#
# Recipe:
#   1. cp set_env.sh set_env_<campaign-or-you>.sh
#   2. Fill every FILL_IN below (grep '^export .*FILL_IN' to list them):
#        HPC_GLOBUS_ENDPOINT_ID  your Globus Compute endpoint/manager UUID
#        HPC_BASE_DIR            a writable campaign dir on the parallel FS
#        HPC_CAMPAIGN_ID         a unique id for this campaign
#        HPC_LOAM_HOME           the loam clone on the shared FS the workers run
#                                from (skills resolve "$HPC_LOAM_HOME/skills/...")
#        HPC_VENV                the on-node venv the workers activate
#      Review HPC_SYSTEM (aurora/polaris), HPC_NNODES, HPC_POOL_SIZE, and the
#      queue/walltime in HPC_ENDPOINT_CONFIG_JSON for your allocation.
#   3. export HPC_ENV_FILE=$PWD/skills/hpc-deployment/set_env_<...>.sh  (or pass
#      it to run-campaign). run-campaign sources THAT file, not this template.
#   4. source your copy in the shell you run every later step from.
#
# This file is run-campaign's DEFAULT when HPC_ENV_FILE is unset, so it
# deliberately does not run: launch.sh greps for FILL_IN and refuses to start.
# It used to hold one person's real endpoint, base directory and campaign id,
# which meant a launch that forgot HPC_ENV_FILE silently started an allocation
# on a stranger's endpoint and wrote into their tree. Those values now live in
# set_env_hari.sh alongside the other per-user copies.
#
# The Copper block below is general and worth keeping; only the identity fields
# are blanked.

export HPC_GLOBUS_ENDPOINT_ID="FILL_IN_endpoint_manager_uuid"
export HPC_POOL_SIZE=1
export HPC_BASE_DIR="FILL_IN_absolute_path_on_the_parallel_filesystem"
export HPC_CAMPAIGN_ID="FILL_IN_campaign_id"
# The loam checkout on the shared filesystem that the compute-node workers run
# from. Campaign skills reference their worker-visible scripts/launchers as
# "$HPC_LOAM_HOME/skills/..." -- a worker cannot see your local checkout, so set
# the clone location ONCE here and every skill resolves against it; no skill
# hardcodes a path. It is also exported into worker_init below (a worker sees
# only the environment that block sets).
export HPC_LOAM_HOME="FILL_IN_absolute_path_to_the_loam_clone_on_the_shared_fs"
# Read-only paths inspect_filesystem / run_batch may look at (colon-separated),
# beyond HPC_BASE_DIR (always allowed). Include "$HPC_LOAM_HOME/skills" so
# run_batch can validate/copy a skill runner script submitted straight from the
# deployed loam tree (e.g. the rfd4-proteina route's rfd4_run_row.py); without it
# a script_path under $HPC_LOAM_HOME is rejected as "outside allowed inspection
# roots" and the agent must first copy the runner into HPC_BASE_DIR. Add the
# project output tree(s) your tools write to. If /flare and /lus/flare/projects
# (or similar) name the same filesystem on your site, list BOTH spellings since
# agents use them interchangeably.
export HPC_INSPECT_ROOTS="${HPC_LOAM_HOME}/skills:FILL_IN_absolute_path_to_tool_output_tree"
export HPC_SYSTEM="aurora"
# Nodes in the allocation. Both the PBS request (`nodes_per_block` below)
# and the EL cluster's leaf count must agree.
export HPC_NNODES=1
export ACADEMY_EXCHANGE_URL="https://exchange.academy-agents.org/v1"
export HPC_MCP_TRANSPORT="http"
# One worker slot per client agent, plus one for the long-running
# HpcClusterAgent, which occupies its slot for the life of the allocation.
# A cluster rebuild shuts the stale agents down before launching replacements,
# so +1 is sufficient; +2 gives that handover headroom if the endpoint can
# spare it and a slow slot release is making restarts drag.
# No available_accelerators: EL owns tile placement, not Parsl.
# Keys match globus_compute_endpoint_configs/alcf_aurora/user_config_template.yaml.j2.
#
# The shared venv and code the compute-node workers import from. The agent
# modules live in the loam clone, so the worker PYTHONPATH derives from
# HPC_LOAM_HOME above (override only if you keep the worker code elsewhere).
HPC_VENV="FILL_IN_absolute_path_to_the_on_node_venv"
HPC_WORKER_PYTHONPATH="${HPC_LOAM_HOME}/skills/hpc-deployment/scripts"

# Copper (ALCF cooperative caching) accelerates importing the shared venv from
# Lustre when many nodes start at once. Turn it on with `export HPC_USE_COPPER=1`
# before sourcing this file; default off reproduces the plain launch exactly.
# When on, a per-node Copper daemon starts and the venv + PYTHONPATH are read
# through Copper's READ-ONLY mount at /tmp/$USER/copper/<original absolute path>;
# TMPDIR and every write stay off it. The venv path then points INTO that mount,
# so the mount point below must match where launch_copper_aurora.sh actually
# mounts (its default is /tmp/$USER/copper) or the workers will not find the
# venv. $USER is left literal so it resolves on the compute node, not here.
export HPC_USE_COPPER="${HPC_USE_COPPER:-0}"
if [ "$HPC_USE_COPPER" = "1" ]; then
  _cop='/tmp/$USER/copper'
  # Export the mount so the agent routes user-supplied run_code/run_batch
  # venv_paths through Copper too (hpc_agent._copper_venv reads HPC_COPPER_MOUNT).
  _copper_init="module load copper; launch_copper_aurora.sh; export HPC_COPPER_MOUNT=${_cop}; "
  _venv="${_cop}${HPC_VENV}"
  _pypath="${_cop}${HPC_WORKER_PYTHONPATH}"
else
  _copper_init=""
  _venv="$HPC_VENV"
  _pypath="$HPC_WORKER_PYTHONPATH"
fi
# worker_init must stay one line: the template renders it as one quoted YAML string.
# HPC_LOAM_HOME is exported here too: a worker's environment is ONLY what this
# block sets, and the campaign skills reference "$HPC_LOAM_HOME/skills/...".
_worker_init="module reset; module load frameworks; ${_copper_init}source ${_venv}/bin/activate; which python; export HPC_LOAM_HOME=${HPC_LOAM_HOME}; export PYTHONPATH=${_pypath}:\$PYTHONPATH; export TMPDIR=/tmp"
export HPC_ENDPOINT_CONFIG_JSON='{"queue": "debug", "walltime": "00:30:00", "nodes_per_block": '"$HPC_NNODES"', "max_workers_per_node": '"$((HPC_POOL_SIZE + 1))"', "worker_init": "'"$_worker_init"'"}'
unset _cop _copper_init _venv _pypath _worker_init
