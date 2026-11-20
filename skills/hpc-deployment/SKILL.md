---
name: hpc-deployment
description: Run loam scientific sweeps on Jupiter Slurm GH200 allocations through Academy and Ensemble Launcher, in Globus-managed or existing Slurm allocations.
---

# Jupiter deployment

Launch the integrated campaign with `./launch.sh`; `loam.yaml` is the single source of campaign and allocation settings. The endpoint uses SlurmProvider, account deepwing, partition booster, four GH200 GPUs and 288 Grace CPU cores per node. The default launcher obtains allocation through Globus Compute. Campaign agents never issue scheduler commands.

The operator may instead provide one combined inference/science job. In that mode `HPC_LAUNCH_MODE=existing` requires an active parent `SLURM_JOB_ID` and explicit `HPC_SCIENCE_NODELIST`. Academy control agents run in the batch process; Ensemble Launcher uses only the science subset, validated against the parent allocation. The MCP tools and scientific recipes are unchanged. Nemotron nodes are a separate, disjoint subset of the same job. `scripts/stage_single_allocation.py` stages this operator-managed route; campaign agents must use the supplied live MCP service and must not launch additional allocations.

The cluster agent uses CUDA_VISIBLE_DEVICES pinning. One-node clusters use async_processpool; multi-node clusters use srun with GPU binding disabled so Ensemble Launcher can assign GPUs. Preserve assigned visibility; use cuda:0 inside each one-GPU row.

Read the relevant scientific skill before writing code. Stage scripts and input JSONL with write_files, register reusable stages with register_tool, then submit run_batch(gpus=1). Check pool_status and list_batches to size real distinct sweeps. Poll batch_status until unfinished is zero, check failures, and inspect produced artifacts before reporting metrics. run_code is for bounded harvesting and diagnosis. Never submit duplicate filler candidates.

The default batch floor is 32 rows and two distinct jobs. The Jupiter launcher explicitly sets HPC_MIN_BATCH_ROWS=1 and HPC_MIN_BATCH_DISTINCT=1 for a four-GPU development allocation; larger campaigns can override these. Respect the live server's error messages and campaign setup.

Each tool uses its dedicated software environment, activated in a bash subprocess. Source scripts/jupiter_env.sh by absolute path first. Outputs belong under execution.workdir, never inside installed software. A path that does not exist is an error, not an empty result.

The local deployment server and Globus workers must share compatible Academy and Ensemble Launcher versions. Jupiter supports the existing internal Redis exchange (sourced from the software redis/env.sh) as well as the HTTP exchange. Authentication remains in the user's Globus store. The launcher checks configuration and required credentials before starting the campaign; it does not start or reconfigure an endpoint.
