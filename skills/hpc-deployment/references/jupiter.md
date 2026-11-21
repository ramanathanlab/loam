# Jupiter

Jupiter-specific notes for this skill. The System itself (288 Grace cores and
four GH200 GPUs per node, `CUDA_VISIBLE_DEVICES` pinning, Slurm node discovery,
the `srun --overlap` launcher, how the endpoint sizes its workers, and which
overrides it allows) is defined in `loam.systems`; this file only covers what
an operator does by hand.

## Endpoint

The endpoint's Slurm template takes no `available_accelerators` or `base_dir`,
so loam leaves both out of `HPC_ENDPOINT_CONFIG_JSON`. Its workers host only
loam's agents: `max_workers_per_node` is the client pool plus the cluster
agent, divided over the allocation's nodes. Check the live template before a
first run, as for any endpoint.

## Cluster mode

The cluster agent finds its nodes with `scontrol show hostnames
$SLURM_JOB_NODELIST` and refuses an allocation whose host count differs from
the campaign's `allocation.nodes`. Every EL step runs through
`scripts/srun_overlap.sh`, so it shares the endpoint's allocation instead of
waiting for a new one. A one-node cluster runs without a launcher.

## Overrides and shared GPUs

`execution.hpc.gpus_per_node` and `cpus_per_node` override the node shape, for
a partition whose nodes differ from the default. `science_workers_per_gpu: 2`
runs two science workers on each GPU: EL schedules logical slots and each
`run_batch` row maps its slot back to one GPU. Under sharing, rows take
`gpus=0` or `gpus=1`, `run_code` takes no GPU, and a batch resumes only under
the sharing it started with.
