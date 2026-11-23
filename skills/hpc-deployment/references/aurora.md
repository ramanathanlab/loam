# Aurora

Aurora-specific notes for this skill. The System itself (cores, the 12 tiles
per node, `ZE_AFFINITY_MASK` pinning, PBS node discovery, the MPI launcher) is
defined in `loam.systems`; this file only covers what an operator does by hand.

Before deployment, follow the [PBS endpoint and scheduler checks](pbs.md).

## Direct mode

Request 12 addressable tiles per node and pin each worker with
`ZE_FLAT_DEVICE_HIERARCHY=FLAT` plus a distinct `ZE_AFFINITY_MASK`.
`pool_size` is requested nodes × 12 unless the endpoint
exposes another count.

## Cluster mode

EL owns tile placement and sets `ZE_AFFINITY_MASK` per task, so client agents
report an empty device pin by design.
