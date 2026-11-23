# Polaris

The System definition in `loam.systems` supplies node shape, device pinning,
node discovery and the launcher. Site profile values supply the allocation
account, queue, paths and worker initialization.

## Endpoint and scheduler checks

Use the [PBS checks](pbs.md) before deployment. For endpoint discovery and
per-submission allocation sizing, follow the
[Globus Compute flex-scaling procedure](polaris-globus-compute-flex-scaling.md).

## Direct mode

Request four addressable GPUs per node and pin each worker with a distinct
`CUDA_VISIBLE_DEVICES`. The direct-mode pool size is requested nodes × four
unless the endpoint exposes another count.

## Cluster mode

EL owns GPU placement; client agents report an empty device pin. Node discovery
uses `PBS_NODEFILE` and the launcher comes from the System definition.
