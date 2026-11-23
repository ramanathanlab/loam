"""The MCP server, built from a loam-generated deployment environment.

Only the services the server talks to are faked: the Globus Compute executor and
endpoint, the Academy exchange and manager, and the agent pools that would launch
onto the allocation. The server itself, its configuration and its tools are real.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from loam.config import Config
from test_round import hpc_config_values, hpc_environment

pytest.importorskip('academy')
pytest.importorskip('globus_compute_sdk')
fastmcp = pytest.importorskip('fastmcp')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills' / 'hpc-deployment' / 'scripts'))
import mcp_server  # noqa: E402
import smoke_test_client  # noqa: E402


class Services:
    """Stands in for everything outside the server; records whatever it was asked to submit."""

    def __init__(self, monkeypatch):
        self.submitted = []
        services = self

        class Executor:
            def __init__(self, **kwargs):
                services.submitted.append(('executor', kwargs))

            def submit(self, *args, **kwargs):
                services.submitted.append(('task', args))

            def shutdown(self, **kwargs):
                pass

        class Manager:
            exchange_client = None

            @classmethod
            async def from_exchange_factory(cls, **kwargs):
                services.submitted.append(('manager', kwargs))
                return cls()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def launch(self, agent, **kwargs):
                services.submitted.append(('agent', agent))
                return SimpleNamespace(agent_id='agent-id')

        class Pool:
            def __init__(self, *args, **kwargs):
                pass

            async def start(self, launch_cluster, launch_client):
                await launch_cluster('ckpt')

            async def stop(self):
                pass

            async def next_agent(self, launch_one):
                async def action(name, *args, **kwargs):
                    services.submitted.append(('action', name))
                    if name == 'run_code':
                        return {'returncode': '0', 'stdout': 'hpc smoke ok', 'stderr': ''}
                    if name == 'inspect_filesystem':
                        return {'entries': []}
                    raise AssertionError(f'Unexpected smoke-test action: {name}')

                return SimpleNamespace(handle=SimpleNamespace(action=action))

        async def endpoint_ready(*args, **kwargs):
            services.submitted.append(('probe', args))

        monkeypatch.setattr(mcp_server, 'Executor', Executor)
        monkeypatch.setattr(mcp_server, 'Manager', Manager)
        monkeypatch.setattr(mcp_server, 'HttpExchangeFactory', lambda *args, **kwargs: None)
        monkeypatch.setattr(mcp_server, 'await_endpoint_ready', endpoint_ready)
        monkeypatch.setattr(mcp_server, 'ClusterPool', Pool)
        monkeypatch.setattr(mcp_server, 'CampaignPool', Pool)


@pytest.fixture
def services(monkeypatch):
    return Services(monkeypatch)


def pool_status(server):
    async def call():
        async with fastmcp.Client(server) as client:
            return (await client.call_tool('pool_status', {})).structured_content

    return asyncio.run(call())


# Accelerators per node are ensemble_launcher's aurora_config / polaris_config;
# the config gives 2 nodes and a pool of 5 agents.
@pytest.mark.parametrize(('system', 'cluster', 'nodes', 'per_node', 'total'), [
    ('aurora', True, 2, 12, 24),
    ('aurora', False, None, 12, 5),
    ('polaris', True, 2, 4, 8),
    ('polaris', False, None, 4, 5),
])
def test_pool_status_reports_the_capacity_of_the_system(tmp_path, services, system, cluster, nodes, per_node, total):
    environment = hpc_environment(tmp_path, system)
    nnodes = int(environment['HPC_NNODES']) if cluster else None

    server = mcp_server.create_server(cluster=cluster, nnodes=nnodes, environ=environment)
    status = pool_status(server)

    assert status['system'] == system
    assert status['mode'] == ('cluster' if cluster else 'direct')
    assert status['nodes'] == nodes
    assert status['tiles_per_node'] == per_node
    assert status['total_tiles'] == total
    assert status['pool_size'] == 5
    assert (status['queue'], status['walltime'], status['account']) == ('debug', '01:02:03', 'project')
    assert status['campaign_id'] == 'campaign-17'


def test_pool_status_counts_science_rows_on_shared_gpus(tmp_path, services):
    values, _ = hpc_config_values(tmp_path)
    values['execution']['hpc'].update(system='jupiter', science_workers_per_gpu=2)
    environment = Config.model_validate(values).execution.hpc.environment()

    status = pool_status(mcp_server.create_server(cluster=True, nnodes=2, environ=environment))

    # Two nodes of four GPUs, two science workers on each.
    assert (status['total_tiles'], status['science_workers_per_gpu'], status['concurrent_scientific_rows']) == (8, 2, 16)


@pytest.mark.parametrize('cluster', [True, False])
@pytest.mark.parametrize('variable', ['HPC_CPU_IDS', 'HPC_GPU_IDS', 'HPC_GPU_PIN_VAR', 'HPC_NODE_DISCOVERY',
                                      'HPC_MPI_FLAVOR', 'HPC_SCIENCE_WORKERS_PER_GPU', 'HPC_GLOBUS_ENDPOINT_ID'])
def test_a_missing_variable_stops_the_server_before_anything_is_submitted(tmp_path, services, cluster, variable):
    environment = hpc_environment(tmp_path, 'aurora')
    del environment[variable]

    with pytest.raises(Exception, match=variable):
        pool_status(mcp_server.create_server(cluster=cluster, nnodes=2 if cluster else None, environ=environment))

    assert services.submitted == []


def existing_allocation(tmp_path, system='polaris', **changes):
    environment = {**hpc_environment(tmp_path, system), 'HPC_LAUNCH_MODE': 'existing'}
    del environment['HPC_GLOBUS_ENDPOINT_ID']
    nodefile = tmp_path / 'nodefile'
    nodefile.write_text('node-a.hsn\nnode-b.hsn\n')
    return {**environment, 'PBS_NODEFILE': str(nodefile), **changes}


def test_an_existing_allocation_is_attached_to_without_globus(tmp_path, services):
    pool_status(mcp_server.create_server(cluster=True, nnodes=2, environ=existing_allocation(tmp_path)))

    kinds = [kind for kind, _ in services.submitted]
    assert 'executor' not in kinds and 'probe' not in kinds and 'task' not in kinds
    # Academy hosts the agents in this process, which runs inside the allocation.
    manager = next(kwargs for kind, kwargs in services.submitted if kind == 'manager')
    assert manager['executors'] == {'allocation': None}
    # The cluster agent places science on the allocation's nodes, found once, up front.
    cluster = next(agent for kind, agent in services.submitted if kind == 'agent')
    assert cluster.nodes == ['node-a', 'node-b']


def test_attaching_outside_an_allocation_fails_before_anything_starts(tmp_path, services):
    environment = existing_allocation(tmp_path)
    del environment['PBS_NODEFILE']

    with pytest.raises(RuntimeError, match='not inside an allocation'):
        mcp_server.create_server(cluster=True, nnodes=2, environ=environment)

    assert services.submitted == []


def test_globus_submission_still_launches_through_the_endpoint(tmp_path, services):
    pool_status(mcp_server.create_server(cluster=True, nnodes=2, environ=hpc_environment(tmp_path, 'polaris')))

    kinds = [kind for kind, _ in services.submitted]
    assert 'executor' in kinds and 'probe' in kinds
    cluster = next(agent for kind, agent in services.submitted if kind == 'agent')
    assert cluster.nodes is None  # found by the worker the endpoint starts, inside its allocation


def test_deployment_smoke_client_accepts_the_current_server(tmp_path, services, monkeypatch, capsys):
    environment = hpc_environment(tmp_path, 'polaris')
    server = mcp_server.create_server(cluster=True, nnodes=2, environ=environment)
    monkeypatch.setenv('HPC_MCP_URL', 'http://unused.invalid/mcp')
    monkeypatch.setenv('HPC_BASE_DIR', environment['HPC_BASE_DIR'])
    monkeypatch.setattr(smoke_test_client, 'Client', lambda url, **kwargs: fastmcp.Client(server, **kwargs))

    asyncio.run(smoke_test_client.main())

    assert [name for kind, name in services.submitted if kind == 'action'] == ['run_code', 'inspect_filesystem']
    output = capsys.readouterr().out
    assert 'hpc smoke ok' in output and 'INSPECT' in output and 'POOL' in output
