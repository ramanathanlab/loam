#!/usr/bin/env bash
# Source before the user-initiated endpoint start; never starts services itself.
export GLOBUS_COMPUTE_USER_DIR=/e/home/jusers/meigooni1/jupiter/.globus_compute
export GLOBUS_COMPUTE_ENDPOINT_NAME=loam
export TMPDIR=/tmp
source /e/project1/deepwing/meigooni1/agent/software/globus-compute/.venv/bin/activate
