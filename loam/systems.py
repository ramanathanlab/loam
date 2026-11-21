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

A System may also name how to read back what a job cost and did: its
scheduler's accounting of allocated time, and the format of the per-host GPU
telemetry samples. Saturation accounting (`loam.saturation`) asks the System
for both, and a System without them cannot be audited.

Single-allocation staging (`loam.staging`) and its coordinator (`loam.coordinator`)
ask the System how to submit a job, find the hosts of the allocation they run in,
and run one step on chosen hosts; nothing else in loam names a scheduler command.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ScalingMeasurement:
    """A dense scaling recipe measured on a System: the node shape and node counts it ran at (loam.scaling)."""
    gpus_per_node: int
    cpus_per_node: int
    node_counts: tuple[int, ...]
    science_workers_per_gpu: int
    # Where the measurements are written up.
    source: str


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
    node_discovery: Literal['pbs-nodefile', 'slurm-nodelist']
    # How the cluster agent launches onto those nodes: an ensemble_launcher MPI flavour, or
    # 'srun-overlap', srun steps that share the allocation the endpoint already holds.
    mpi_flavor: Literal['mpich', 'srun-overlap']
    # How the Globus endpoint config sizes its workers. 'per-accelerator': one worker per
    # accelerator, with available_accelerators and base_dir. 'pool-across-allocation': the
    # workers only host loam's agents (the pool plus the cluster agent), spread over the
    # allocation's nodes, and the endpoint template takes no accelerator list or base_dir.
    endpoint_workers: Literal['per-accelerator', 'pool-across-allocation'] = 'per-accelerator'
    # Whether a campaign config may set the node's GPU and CPU counts.
    node_shape_overrides: bool = False
    # How many science workers may share one GPU.
    max_science_workers_per_gpu: int = 1
    # How the scheduler reports a job's allocated time, if loam can read it.
    allocation_accounting: Literal['sacct'] | None = None
    # The format of one host's GPU telemetry sample, if loam can read it.
    gpu_telemetry: Literal['nvidia-smi-csv'] | None = None
    # The scaling recipe measured on this System, if any; loam.scaling refuses to size other Systems.
    scaling_measurement: ScalingMeasurement | None = None

    @property
    def gpus_per_node(self) -> int:
        return len(self.gpu_ids)

    @property
    def job_id_variable(self) -> str:
        """The variable the scheduler sets, inside a job, to that job's id."""
        return {'pbs': 'PBS_JOBID', 'slurm': 'SLURM_JOB_ID'}[self.scheduler]

    def with_node_shape(self, *, gpus: int | None = None, cpus: int | None = None) -> System:
        """This System with a campaign config's GPU and CPU counts, numbered from zero."""
        if gpus is None and cpus is None:
            return self
        if not self.node_shape_overrides:
            raise ValueError(f'{self.name} does not allow GPU/CPU count overrides')
        if (gpus is not None and gpus < 1) or (cpus is not None and cpus < 1):
            raise ValueError('GPU and CPU counts per node must be positive')
        return replace(self, gpu_ids=self.gpu_ids if gpus is None else tuple(range(gpus)),
                       cpu_ids=self.cpu_ids if cpus is None else tuple(range(cpus)))

    def check_science_workers_per_gpu(self, workers: int) -> None:
        if workers > self.max_science_workers_per_gpu:
            most = self.max_science_workers_per_gpu
            raise ValueError(f'{self.name} allows at most {most} science worker{"s" * (most != 1)} per GPU')

    def endpoint_workers_config(self, *, pool_size: int, nodes: int, base_dir: str) -> dict:
        """The endpoint config keys that size its workers."""
        if self.endpoint_workers == 'pool-across-allocation':
            # One worker per pool agent plus one for the cluster agent, divided over the nodes.
            return {'max_workers_per_node': (pool_size + nodes) // nodes}
        return {'available_accelerators': self.gpus_per_node, 'max_workers_per_node': self.gpus_per_node,
                'base_dir': base_dir}

    def environment(self) -> dict[str, str]:
        return {
            'HPC_CPU_IDS': json.dumps(list(self.cpu_ids), separators=(',', ':')),
            'HPC_GPU_IDS': json.dumps(list(self.gpu_ids), separators=(',', ':')),
            'HPC_GPU_PIN_VAR': self.gpu_pin_var,
            'HPC_NODE_DISCOVERY': self.node_discovery,
            'HPC_MPI_FLAVOR': self.mpi_flavor,
        }

    def allocation_attempts(self, job_id: str, *, gpus_per_node: int) -> dict:
        """Every physical allocation of one parent job, with its allocated GPU-seconds."""
        if self.allocation_accounting != 'sacct':
            raise NotImplementedError(f'loam cannot read allocation accounting on {self.name}')
        if not job_id.isdigit():
            raise ValueError('Expected exact numeric parent job ID')
        output = sacct(['--duplicates', '-X', '-n', '-P', '-j', job_id, '--format=' + ','.join(SACCT_FIELDS)])
        return parse_sacct_attempts(output, job_id, gpus_per_node=gpus_per_node)

    def parent_jobs(self, ids: list[str]) -> dict[str, dict]:
        """The latest record of each parent job: state, epoch start/end/submit and queue wait."""
        if self.allocation_accounting != 'sacct':
            raise NotImplementedError(f'loam cannot read allocation accounting on {self.name}')
        output = sacct(['-n', '-X', '-j', ','.join(ids), '--parsable2', '--format=' + ','.join(SACCT_FIELDS)])
        return parse_sacct_parent_jobs(output, ids)

    def batch_submission(self, *, script: str, name: str, workdir: str, account: str, queue: str, nodes: int,
                         walltime: str, reservation: str, dependencies: tuple[tuple[str, str], ...]) -> list[str]:
        """The command that submits `script` as one exclusive job of `nodes` whole nodes.

        `dependencies` are (kind, job id) pairs such as ('afterany', '41'). On Slurm the queue is the partition.
        The job starts with no environment from the submitting shell; its script sets what it needs.
        """
        if self.scheduler != 'slurm':
            raise NotImplementedError(f'loam cannot submit a single-allocation job on {self.name}')
        args = ['sbatch', '--parsable', '--export=NIL', f'--account={account}', f'--partition={queue}',
                f'--nodes={nodes}', '--ntasks-per-node=1', f'--cpus-per-task={len(self.cpu_ids)}',
                f'--gres=gpu:{self.gpus_per_node}', '--exclusive', f'--time={walltime}', f'--job-name={name}',
                f'--chdir={workdir}', f'--output={workdir}/slurm-%j.log']
        if reservation:
            args.append(f'--reservation={reservation}')
        args += [f'--dependency={kind}:{job_id}' for kind, job_id in dependencies]
        return [*args, script]

    def submitted_job_id(self, stdout: str) -> str:
        """The job id in the scheduler's acknowledgement of `batch_submission`."""
        job_id = stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError(f'Ambiguous {self.scheduler} acknowledgement {stdout!r}; '
                               'inspect the submission intent before retrying')
        return job_id

    def allocation_hosts(self, environ: Mapping[str, str]) -> list[str]:
        """The short hostnames of the allocation this process runs inside, in the scheduler's order."""
        variable = {'pbs-nodefile': 'PBS_NODEFILE', 'slurm-nodelist': 'SLURM_JOB_NODELIST'}[self.node_discovery]
        if not environ.get(variable):
            raise RuntimeError(f'This process is not inside an allocation: {variable} is not set')
        if self.node_discovery == 'pbs-nodefile':
            listed = Path(environ[variable]).read_text().split()
        else:
            listed = subprocess.run(['scontrol', 'show', 'hostnames', environ[variable]], capture_output=True,
                                    text=True, check=True, timeout=30).stdout.split()
        return list(dict.fromkeys(host.split('.')[0] for host in listed))

    def host_step(self, hosts: list[str], command: list[str], *, cpus: int = 1) -> list[str]:
        """`command` run once on each of `hosts` with `cpus` cores and every GPU, inside the allocation."""
        if self.mpi_flavor == 'srun-overlap':
            return ['srun', '--overlap', '--exact', '--mpi=none', f'--nodes={len(hosts)}', f'--ntasks={len(hosts)}',
                    '--ntasks-per-node=1', f'--cpus-per-task={cpus}', f'--nodelist={",".join(hosts)}',
                    f'--gpus-per-node={self.gpus_per_node}', '--gpu-bind=none', *command]
        return ['mpiexec', '-n', str(len(hosts)), '--ppn', '1', '--hosts', ','.join(hosts), *command]

    def step_environment(self) -> dict[str, str]:
        """Variables a `host_step` needs so the step sees the job's own environment.

        A Slurm job submitted with --export=NIL would otherwise hand its steps none of it.
        """
        return {'SLURM_EXPORT_ENV': 'ALL'} if self.scheduler == 'slurm' else {}

    def check_gpu_telemetry(self) -> None:
        if self.gpu_telemetry != 'nvidia-smi-csv':
            raise NotImplementedError(f'loam cannot read GPU telemetry on {self.name}')

    def gpu_sample(self) -> dict:
        """One host's GPU part of a telemetry sample, in the form `gpu_busy` and `every_gpu_reported` read."""
        try:
            query = subprocess.run(self.gpu_telemetry_query(), capture_output=True, text=True, timeout=8)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {'nvidia_error': type(error).__name__}
        return {'gpu_csv': query.stdout.strip(), 'nvidia_returncode': query.returncode}

    def gpu_telemetry_query(self) -> list[str]:
        """The command whose output is one host's `gpu_csv` telemetry sample."""
        self.check_gpu_telemetry()
        return ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,'
                'memory.total,power.draw,temperature.gpu', '--format=csv,noheader,nounits']

    def gpu_busy(self, sample: dict, *, gpus_per_node: int) -> list[float] | None:
        """Busy percent of each GPU, by index, in one telemetry sample; None if it does not cover every GPU."""
        self.check_gpu_telemetry()
        return parse_nvidia_smi_busy(sample.get('gpu_csv', ''), gpus_per_node=gpus_per_node)

    def every_gpu_reported(self, sample: dict) -> bool:
        """Whether one host's telemetry sample is a successful query that covers each of this System's GPUs."""
        self.check_gpu_telemetry()
        return sample.get('nvidia_returncode') == 0 \
            and self.gpu_busy(sample, gpus_per_node=self.gpus_per_node) is not None


# --- Slurm accounting (`sacct`) -------------------------------------------
#
# Every physical allocation attempt counts, including attempts Slurm requeued
# during boot: they cost GPU-hours though the campaign never ran on them.
# Queue-only records and job steps cost nothing. `sacct` formats times in the
# zone of the process that runs it, which is this one, so they parse as local.

SACCT_FIELDS = ['JobIDRaw', 'JobName', 'State', 'NNodes', 'AllocTRES', 'Submit', 'Start', 'End', 'ElapsedRaw',
                'NodeList', 'ExitCode', 'Reason', 'FailedNode']
STILL_RUNNING = ('RUNNING', 'COMPLETING')


def sacct(arguments: list[str]) -> str:
    return subprocess.run(['sacct', *arguments], capture_output=True, text=True, check=True, timeout=60).stdout


def sacct_time(value: str | None) -> float | None:
    """Epoch seconds for a `sacct` time, or None for 'Unknown', 'None' and the like."""
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def sacct_end(row: dict, start: float, end: float | None) -> float | None:
    """A record's end, taking a still-running job's end as start plus elapsed."""
    if end is None and row['State'].startswith(STILL_RUNNING):
        return start + float(row.get('ElapsedRaw') or 0)
    return end


def sacct_rows(output: str):
    for line in output.splitlines():
        yield dict(zip(SACCT_FIELDS, line.strip().split('|')))


def allocated_gpus(tres: str) -> int:
    parts = dict(part.split('=', 1) for part in tres.split(',') if '=' in part)
    if 'gres/gpu' in parts:
        return int(parts['gres/gpu'])
    return sum(int(value) for key, value in parts.items() if key.startswith('gres/gpu:'))


def parse_sacct_attempts(output: str, job_id: str, *, gpus_per_node: int) -> dict:
    """Each physical allocation of `job_id` in `sacct --duplicates` output, counted once."""
    allocations = {}
    unallocated = []
    for row in sacct_rows(output):
        if row.get('JobIDRaw') != job_id:
            continue
        start = sacct_time(row.get('Start'))
        if start is None:
            unallocated.append(row)
            continue
        end = sacct_end(row, start, sacct_time(row.get('End')))
        if end is None:
            raise ValueError('Allocated attempt has no authoritative end: ' + str(row['State']))
        elapsed = max(float(row.get('ElapsedRaw') or 0), end - start)
        if elapsed <= 0:
            continue
        nodes = int(row['NNodes'])
        gpus = allocated_gpus(row['AllocTRES'])
        if not nodes or gpus != gpus_per_node * nodes:
            raise ValueError(f'Job {job_id} has {gpus} GPUs on {nodes} nodes; '
                             f'expected {gpus_per_node} GPUs per node')
        record = {**row, 'start_epoch': start, 'end_epoch': end, 'allocated_seconds': elapsed,
                  'allocated_GPUs': gpus, 'allocated_GPU_seconds': gpus * elapsed}
        # sacct may repeat one controller transition. Those records describe one
        # physical allocation, not additional billed node-time.
        allocations[(start, end, nodes, row['NodeList'], gpus)] = record
    values = sorted(allocations.values(), key=lambda r: r['start_epoch'])
    return {'job_id': job_id, 'allocations': values, 'unallocated_records': unallocated,
            'allocation_attempts': len(values),
            'total_allocated_seconds': sum(r['allocated_seconds'] for r in values),
            'total_allocated_GPU_seconds': sum(r['allocated_GPU_seconds'] for r in values),
            'scope': 'Each physical parent allocation counted once. '
                     'Queue-only records and job steps contribute no GPU time.'}


def parse_sacct_parent_jobs(output: str, ids: list[str]) -> dict[str, dict]:
    jobs = {}
    for row in sacct_rows(output):
        if row.get('JobIDRaw') not in ids:
            continue
        start = row['start_epoch'] = sacct_time(row['Start'])
        submit = row['submit_epoch'] = sacct_time(row['Submit'])
        row['end_epoch'] = sacct_end(row, start, sacct_time(row['End'])) if start else None
        row['queue_seconds'] = max(0, start - submit) if start and submit else None
        jobs[row['JobIDRaw']] = row
    return jobs


# --- NVIDIA GPU telemetry -------------------------------------------------


def parse_nvidia_smi_busy(rows: str, *, gpus_per_node: int) -> list[float] | None:
    """Busy percents from `nvidia-smi --format=csv,noheader` rows: index first, utilization third."""
    gpu = list(csv.reader(io.StringIO(rows)))
    if len(gpu) != gpus_per_node or not all(len(g) >= 8 for g in gpu):
        return None
    if {int(g[0]) for g in gpu} != set(range(gpus_per_node)):
        raise ValueError(f'GPU telemetry has indices other than 0..{gpus_per_node - 1}')
    values = [float(g[2]) for g in sorted(gpu, key=lambda g: int(g[0]))]
    if not all(0 <= value <= 100 for value in values):
        raise ValueError('GPU telemetry has a busy percent outside 0..100')
    return values


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
    gpu_telemetry='nvidia-smi-csv',  # A100s; allocation accounting is still unread on PBS
)

# A GH200 node: 288 Grace cores and four CUDA GPUs.
JUPITER = System(
    name='jupiter',
    scheduler='slurm',
    cpu_ids=tuple(range(288)),
    gpu_ids=tuple(range(4)),
    gpu_pin_var='CUDA_VISIBLE_DEVICES',
    node_discovery='slurm-nodelist',
    mpi_flavor='srun-overlap',
    endpoint_workers='pool-across-allocation',
    node_shape_overrides=True,
    max_science_workers_per_gpu=2,
    allocation_accounting='sacct',
    gpu_telemetry='nvidia-smi-csv',
    # Jobs 2979745 (20 nodes, "20i") and 2999784 (256 nodes, "256c"), November 2026.
    scaling_measurement=ScalingMeasurement(gpus_per_node=4, cpus_per_node=288, node_counts=(20, 256),
                                           science_workers_per_gpu=2, source='docs/scaling-recipe-book.md'),
)

SYSTEMS: dict[str, System] = {system.name: system for system in (AURORA, POLARIS, JUPITER)}

SystemName = Literal[tuple(SYSTEMS)]  # type: ignore[valid-type]
