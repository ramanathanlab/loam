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
