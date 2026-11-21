"""The cluster agent builds its cluster setup from the deployment environment alone."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from loam.config import Config
from test_round import hpc_config_values

pytest.importorskip('academy')
el_config = pytest.importorskip('ensemble_launcher.config')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills' / 'hpc-deployment' / 'scripts'))
import hpc_agent  # noqa: E402


def deployment_environment(tmp_path, system):
    values, _ = hpc_config_values(tmp_path)
    values['execution']['hpc']['system'] = system
    return Config.model_validate(values).execution.hpc.environment()


@pytest.mark.parametrize(('system', 'el_system_config', 'pin_var'), [
    ('aurora', el_config.aurora_config, 'ZE_AFFINITY_MASK'),
    ('polaris', el_config.polaris_config, 'CUDA_VISIBLE_DEVICES'),
])
def test_cluster_agent_reproduces_the_system_from_the_environment(tmp_path, system, el_system_config, pin_var):
    environment = deployment_environment(tmp_path, system)

    agent = hpc_agent.HpcClusterAgent('/campaign', 'ckpt', 2, hpc_agent.SystemSpec.from_environment(environment))

    # ensemble_launcher's own aurora_config / polaris_config are the independent truth.
    assert agent.system_config.model_dump(exclude={'name'}) == el_system_config.model_dump(exclude={'name'})
    assert agent.launcher_config.gpu_selector == pin_var
    assert agent.launcher_config.mpi_config.launcher == 'mpiexec'
    assert agent.launcher_config.policy_config.leaf_nodes == 2


def test_cluster_agent_rejects_an_environment_missing_the_system(tmp_path):
    environment = deployment_environment(tmp_path, 'aurora')
    del environment['HPC_GPU_PIN_VAR']
    with pytest.raises(RuntimeError, match='HPC_GPU_PIN_VAR'):
        hpc_agent.SystemSpec.from_environment(environment)


def test_cluster_agent_starts_its_cluster_on_the_pbs_allocation_nodes(tmp_path, monkeypatch):
    nodefile = tmp_path / 'nodefile'
    nodefile.write_text('x4102c5s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov\nx4102c5s1b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov\n')
    monkeypatch.setenv('PBS_NODEFILE', str(nodefile))
    started = {}

    class Launcher:  # stands in for EL, which would fork the cluster
        def __init__(self, **kwargs):
            started.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(hpc_agent, 'EnsembleLauncher', Launcher)
    spec = hpc_agent.SystemSpec.from_environment(deployment_environment(tmp_path, 'aurora'))
    agent = hpc_agent.HpcClusterAgent('/campaign', 'ckpt', 2, spec)

    asyncio.run(agent.agent_on_startup())

    assert started['Nodes'] == ['x4102c5s0b0n0', 'x4102c5s1b0n0']
    assert started['system_config'] is agent.system_config


def test_pbs_node_discovery_outside_an_allocation_uses_this_host(monkeypatch):
    monkeypatch.setattr(hpc_agent.socket, 'gethostname', lambda: 'login01')
    assert hpc_agent.discover_nodes('pbs-nodefile', {}) == ['login01']


def site_profile_environment(tmp_path, monkeypatch):
    """The deployment environment of a config whose Site profile sets the software root."""
    import yaml

    from loam.config import load

    profile = tmp_path / 'profile.yaml'
    profile.write_text(yaml.safe_dump({'software_root': '/soft/loam', 'model_weights': {'chai': '/w/chai'}}))
    monkeypatch.setenv('LOAM_SITE_PROFILE', str(profile))
    values, _ = hpc_config_values(tmp_path)
    config = tmp_path / 'loam.yaml'
    config.write_text(yaml.safe_dump(values))
    return load(config).execution.hpc.environment()


def test_run_code_on_a_worker_sees_the_site_profile(tmp_path, monkeypatch):
    environment = site_profile_environment(tmp_path, monkeypatch)
    for name in ('LOAM_SOFTWARE_ROOT', 'LOAM_MODEL_WEIGHTS_JSON'):
        monkeypatch.delenv(name, raising=False)  # the worker's own environment does not have them
    agent = hpc_agent.HpcAgent('campaign', str(tmp_path), worker_env=hpc_agent.worker_environment(environment))

    result = asyncio.run(agent.run_code("import os; print(os.environ['LOAM_SOFTWARE_ROOT'], os.environ['LOAM_MODEL_WEIGHTS_JSON'])"))

    assert result['stdout'].split() == ['/soft/loam', '{"chai":"/w/chai"}'], result['stderr']


def test_registered_tools_run_on_a_worker_with_the_site_profile(tmp_path, monkeypatch):
    environment = site_profile_environment(tmp_path, monkeypatch)
    monkeypatch.delenv('LOAM_SOFTWARE_ROOT', raising=False)
    agent = hpc_agent.HpcAgent('campaign', str(tmp_path), worker_env=hpc_agent.worker_environment(environment))
    script = tmp_path / 'tool.py'
    script.write_text("import os, pathlib; pathlib.Path('root.txt').write_text(os.environ['LOAM_SOFTWARE_ROOT'])\n")

    # The smoke run is the same row runner run_batch uses on every node.
    registered = asyncio.run(agent.register_tool('root', str(script), {'name': 'str'}, 'writes the software root',
                                                 {'name': 'x'}, expects=['root.txt']))

    assert (Path(registered['dir']) / 'smoke' / 'root.txt').read_text() == '/soft/loam'


def test_without_a_site_profile_workers_get_no_extra_environment(tmp_path):
    assert hpc_agent.worker_environment(deployment_environment(tmp_path, 'aurora')) == {}


class InlineCluster:
    """Stands in for EL's ClusterClient: runs each task here, the way an EL worker would, from its args alone."""

    def submit(self, task):
        return self.submit_batch([task])[0]

    def submit_batch(self, tasks):
        from concurrent.futures import Future

        futures = []
        for task in tasks:
            future = Future()
            future.set_result(task.executable(*task.args))
            futures.append(future)
        return futures


def test_batch_rows_on_cluster_workers_see_the_site_profile(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(hpc_agent, '_MIN_BATCH_ROWS', 1)  # one row is enough to see the environment
    monkeypatch.setattr(hpc_agent, '_MIN_BATCH_DISTINCT', 1)
    environment = site_profile_environment(tmp_path, monkeypatch)
    monkeypatch.delenv('LOAM_SOFTWARE_ROOT', raising=False)
    agent = hpc_agent.HpcClientAgent('campaign', str(tmp_path), 'ckpt',
                                     worker_env=hpc_agent.worker_environment(environment))
    agent._client, agent._el_pool = InlineCluster(), ThreadPoolExecutor(max_workers=1)
    script = tmp_path / 'row.py'
    script.write_text("import os, pathlib; pathlib.Path('root.txt').write_text(os.environ['LOAM_SOFTWARE_ROOT'])\n")
    inputs = tmp_path / 'rows.jsonl'
    inputs.write_text('{"id": "r1"}\n')

    code = asyncio.run(agent.run_code("import os; print(os.environ['LOAM_SOFTWARE_ROOT'])"))
    batch = asyncio.run(agent.run_batch(str(script), str(inputs)))

    assert code['stdout'].strip() == '/soft/loam', code['stderr']
    assert [p.read_text() for p in tmp_path.rglob('root.txt')] == ['/soft/loam'], batch


# --- Slurm, srun and shared GPUs (Jupiter) ---------------------------------


def jupiter_spec(tmp_path, **hpc):
    values, _ = hpc_config_values(tmp_path)
    values['execution']['hpc'].update(system='jupiter', **hpc)
    return hpc_agent.SystemSpec.from_environment(Config.model_validate(values).execution.hpc.environment())


def fake_command(bin_dir, name, script):
    """An executable `name` on a directory that the caller puts first on PATH."""
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text('#!/bin/sh\n' + script)
    path.chmod(0o755)


@pytest.fixture
def scontrol(tmp_path, monkeypatch):
    """`scontrol show hostnames` that expands the one node list these tests use, as Slurm would."""
    bin_dir = tmp_path / 'bin'
    fake_command(bin_dir, 'scontrol', '[ "$3" = "jpbo-[012-013]" ] && printf "jpbo-012\\njpbo-013\\n"\n')
    monkeypatch.setenv('PATH', f'{bin_dir}:{os.environ["PATH"]}')


def start_cluster(monkeypatch, agent):
    started = {}

    class Launcher:  # stands in for EL, which would fork the cluster
        def __init__(self, **kwargs):
            started.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(hpc_agent, 'EnsembleLauncher', Launcher)
    asyncio.run(agent.agent_on_startup())
    return started


def test_cluster_agent_starts_its_cluster_on_the_slurm_allocation_nodes(tmp_path, monkeypatch, scontrol):
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'jpbo-[012-013]')
    agent = hpc_agent.HpcClusterAgent('/campaign', 'ckpt', 2, jupiter_spec(tmp_path))

    started = start_cluster(monkeypatch, agent)

    # Full hostnames: EL's one-shot child launch would split them at '-'.
    assert started['Nodes'] == ['jpbo-012', 'jpbo-013']
    assert agent.launcher_config.sequential_child_launch is True


def test_cluster_agent_refuses_a_slurm_allocation_of_the_wrong_size(tmp_path, monkeypatch, scontrol):
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'jpbo-[012-013]')
    agent = hpc_agent.HpcClusterAgent('/campaign', 'ckpt', 3, jupiter_spec(tmp_path))
    with pytest.raises(RuntimeError, match='2 hosts; requested 3'):
        start_cluster(monkeypatch, agent)


def test_cluster_agent_needs_a_slurm_node_list_for_more_than_one_node(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_NODELIST', raising=False)
    monkeypatch.delenv('SLURM_NODELIST', raising=False)
    monkeypatch.setattr(hpc_agent.socket, 'gethostname', lambda: 'jpbo-login')

    assert hpc_agent.discover_nodes('slurm-nodelist', {}, expected=1) == ['jpbo-login']
    with pytest.raises(RuntimeError, match='SLURM_JOB_NODELIST'):
        hpc_agent.discover_nodes('slurm-nodelist', {}, expected=2)


def test_cluster_agent_builds_srun_launch_commands_that_overlap_the_allocation(tmp_path, monkeypatch):
    from ensemble_launcher.executors.async_mpi_executor import AsyncMPIExecutor
    from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

    agent = hpc_agent.HpcClusterAgent('/campaign', 'ckpt', 2, jupiter_spec(tmp_path))
    mpi = agent.launcher_config.mpi_config
    assert agent.system_config.cpus == list(range(288)) and agent.system_config.gpus == [0, 1, 2, 3]
    assert agent.launcher_config.gpu_selector == 'CUDA_VISIBLE_DEVICES'
    assert agent.launcher_config.child_executor_name == 'async_mpi'

    # A one-rank-per-node step on both nodes, built the way EL builds every launch.
    executor = AsyncMPIExecutor(mpi_config=mpi, tmp_dir=str(tmp_path / 'el'))
    resources = JobResource(resources=[NodeResourceList(cpus=(0,))] * 2, nodes=['jpbo-012', 'jpbo-013'])
    flags, _, _ = executor._build_resource_cmd('t1', resources)
    command = [mpi.launcher, *mpi.extra_launcher_flags, *flags, 'python', 'child.py']
    assert command[1:] == ['--gpus-per-node=4', '--gpu-bind=none', '-n', '2', '--ntasks-per-node', '1',
                           '--nodelist', 'jpbo-012,jpbo-013', 'python', 'child.py']

    # The launcher runs every step, EL's node-local copy steps included, as an overlapping srun.
    bin_dir = tmp_path / 'bin'
    fake_command(bin_dir, 'srun', 'echo srun "$@"\n')
    run = subprocess.run(command, capture_output=True, text=True, check=True,
                         env={**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}'})
    assert run.stdout.split() == ['srun', '--overlap', '--cpu-bind=none', *command[1:]]


def test_a_single_node_cluster_runs_without_a_launcher(tmp_path):
    agent = hpc_agent.HpcClusterAgent('/campaign', 'ckpt', 1, jupiter_spec(tmp_path))
    assert agent.launcher_config.child_executor_name == 'async_processpool'


def test_node_shape_overrides_reach_the_cluster(tmp_path):
    agent = hpc_agent.HpcClusterAgent('/campaign', 'ckpt', 2, jupiter_spec(tmp_path, gpus_per_node=2, cpus_per_node=72))
    assert agent.system_config.cpus == list(range(72)) and agent.system_config.gpus == [0, 1]
    assert agent.launcher_config.mpi_config.extra_launcher_flags[0] == '--gpus-per-node=2'


def test_shared_gpus_give_the_cluster_a_slot_per_science_worker(tmp_path):
    agent = hpc_agent.HpcClusterAgent('/campaign', 'ckpt', 2, jupiter_spec(tmp_path, science_workers_per_gpu=2))
    # EL schedules eight slots on four GPUs; each row maps its slot back to a GPU.
    assert agent.system_config.gpus == list(range(8))
    assert agent.launcher_config.gpu_selector == 'LOAM_EL_GPU_SLOTS'


class SlotCluster(InlineCluster):
    """An EL worker that hands each task the next science slot, through the cluster's GPU selector."""

    def __init__(self, slots):
        self.slots = iter(slots)

    def submit_batch(self, tasks):
        futures = []
        for task in tasks:
            os.environ['LOAM_EL_GPU_SLOTS'] = '' if task.ngpus_per_process == 0 else str(next(self.slots))
            os.environ.update(task.env)
            futures.extend(super().submit_batch([task]))
        return futures


def shared_gpu_client(tmp_path, monkeypatch, cluster):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(hpc_agent, '_MIN_BATCH_ROWS', 1)
    monkeypatch.setattr(hpc_agent, '_MIN_BATCH_DISTINCT', 1)
    monkeypatch.setenv('LOAM_EL_GPU_SLOTS', '')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    spec = jupiter_spec(tmp_path, science_workers_per_gpu=2)
    agent = hpc_agent.HpcClientAgent('campaign', str(tmp_path), 'ckpt', gpu_sharing=spec.gpu_sharing())
    agent._client, agent._el_pool = cluster, ThreadPoolExecutor(max_workers=1)
    script = tmp_path / 'row.py'
    script.write_text("import os, pathlib; pathlib.Path('gpu.txt').write_text(os.environ['CUDA_VISIBLE_DEVICES'])\n")
    inputs = tmp_path / 'rows.jsonl'
    inputs.write_text('{"id": "r1"}\n{"id": "r2"}\n')
    return agent, script, inputs


def test_shared_gpu_rows_run_on_the_gpu_their_slot_maps_to(tmp_path, monkeypatch):
    agent, script, inputs = shared_gpu_client(tmp_path, monkeypatch, SlotCluster([5, 2]))

    asyncio.run(agent.run_batch(str(script), str(inputs)))

    assert sorted(p.read_text() for p in tmp_path.rglob('gpu.txt')) == ['1', '2']


def test_zero_gpu_rows_under_shared_gpus_see_no_gpu(tmp_path, monkeypatch):
    agent, script, inputs = shared_gpu_client(tmp_path, monkeypatch, SlotCluster([]))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')  # left over from an earlier task on this worker

    asyncio.run(agent.run_batch(str(script), str(inputs), gpus=0))

    assert [p.read_text() for p in tmp_path.rglob('gpu.txt')] == ['', '']


def test_shared_gpus_refuse_rows_that_need_more_than_one_gpu(tmp_path, monkeypatch):
    agent, script, inputs = shared_gpu_client(tmp_path, monkeypatch, SlotCluster([]))
    with pytest.raises(ValueError, match='one-device'):
        asyncio.run(agent.run_batch(str(script), str(inputs), gpus=2))


def test_a_batch_resumes_only_under_the_gpu_sharing_it_started_with(tmp_path, monkeypatch):
    agent, script, inputs = shared_gpu_client(tmp_path, monkeypatch, SlotCluster([0, 1]))
    batch = asyncio.run(agent.run_batch(str(script), str(inputs)))
    unshared = hpc_agent.HpcClientAgent('campaign', str(tmp_path), 'ckpt')
    unshared._client, unshared._el_pool = agent._client, agent._el_pool

    with pytest.raises(ValueError, match='different GPU sharing'):
        asyncio.run(unshared.run_batch(batch_id=batch['batch_id']))
