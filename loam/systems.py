"""Systems: every System loam can deploy onto, as every user sees it.

A System is defined once, here, and resolved into explicit deployment
environment variables. Nothing downstream (the MCP server, the cluster agent)
looks a System up by name, so adding one is a new entry in SYSTEMS rather than
another branch in every consumer. See docs/adr/0004.

The independent truth for a System's shape is ensemble_launcher itself:
`ensemble_launcher.config.get_system_config` (cores, accelerators) and
`ensemble_launcher.helper_functions.get_nodes` (PBS node discovery).
The deployment environment written here is read back, without importing
loam, by `SystemSpec` in skills/hpc-deployment/scripts/hpc_agent.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class System:
    name: str
    scheduler: Literal['pbs', 'slurm']
    # Core ids the launcher may place work on; cores held back for the OS are omitted.
    cpu_ids: tuple[int, ...]
    # Accelerator ids, in the form the pinning variable takes (Aurora tiles are strings).
    gpu_ids: tuple[int | str, ...]
    # The variable that pins a process to one accelerator.
    gpu_pin_var: str
    # How the cluster agent finds the allocation's nodes.
    node_discovery: Literal['pbs-nodefile']
    # ensemble_launcher MPI flavour used to launch onto those nodes.
    mpi_flavor: str

    @property
    def gpus_per_node(self) -> int:
        return len(self.gpu_ids)

    def environment(self) -> dict[str, str]:
        return {
            'HPC_CPU_IDS': json.dumps(list(self.cpu_ids), separators=(',', ':')),
            'HPC_GPU_IDS': json.dumps(list(self.gpu_ids), separators=(',', ':')),
            'HPC_GPU_PIN_VAR': self.gpu_pin_var,
            'HPC_NODE_DISCOVERY': self.node_discovery,
            'HPC_MPI_FLAVOR': self.mpi_flavor,
        }


# Shapes match ensemble_launcher's own aurora_config / polaris_config.
AURORA = System(
    name='aurora',
    scheduler='pbs',
    cpu_ids=(*range(1, 52), *range(53, 104)),
    gpu_ids=tuple(str(i) for i in range(12)),
    gpu_pin_var='ZE_AFFINITY_MASK',
    node_discovery='pbs-nodefile',
    mpi_flavor='mpich',
)

POLARIS = System(
    name='polaris',
    scheduler='pbs',
    cpu_ids=tuple(range(32)),
    gpu_ids=tuple(range(4)),
    gpu_pin_var='CUDA_VISIBLE_DEVICES',
    node_discovery='pbs-nodefile',
    mpi_flavor='mpich',
)

SYSTEMS: dict[str, System] = {system.name: system for system in (AURORA, POLARIS)}

SystemName = Literal[tuple(SYSTEMS)]  # type: ignore[valid-type]
