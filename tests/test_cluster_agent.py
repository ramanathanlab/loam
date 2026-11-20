"""The cluster agent builds its cluster setup from the deployment environment alone."""

import asyncio
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
