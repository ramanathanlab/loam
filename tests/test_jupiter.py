"""Offline tests of Jupiter deployment contracts and packaged CUDA recipes."""
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

from loam.config import load
from loam.skills import discover, read

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills/hpc-deployment/scripts'


def test_jupiter_endpoint_schema():
    import jsonschema

    cfg = load(ROOT / 'loam.yaml')
    hpc = cfg.execution.hpc
    options = json.loads(hpc.environment()['HPC_ENDPOINT_CONFIG_JSON'])
    schema_path = ROOT / 'globus_compute_endpoint_configs/jupiter/user_config_schema.json'
    if schema_path.exists():
        jsonschema.validate(options, json.loads(schema_path.read_text()))
    assert options['max_workers_per_node'] == hpc.pool_size + 1
    assert 'available_accelerators' not in options
    assert 'base_dir' not in options
    assert 'scripts/jupiter_env.sh' in options['worker_init']
    assert Path(cfg.goal).is_file()
    assert all(path.is_file() for path in cfg.rules)


def test_jupiter_worker_capacity_is_allocation_wide():
    cfg = load(ROOT / 'loam.yaml')
    hpc = cfg.execution.hpc.model_copy(deep=True)
    hpc.pool_size = 24
    hpc.allocation.nodes = 19
    options = json.loads(hpc.environment()['HPC_ENDPOINT_CONFIG_JSON'])
    workers = options['max_workers_per_node']
    assert workers * 19 >= 25  # cluster owner plus all client agents can start
    assert (workers - 1) * 19 < 25  # no unnecessary full worker per node


@pytest.mark.parametrize('nodes,executor', [(1, 'async_processpool'), (2, 'async_mpi')])
def test_jupiter_cluster_configuration(nodes, executor, tmp_path, monkeypatch):
    pytest.importorskip('academy')
    pytest.importorskip('ensemble_launcher')
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = importlib.import_module('hpc_agent')
    agent = module.HpcClusterAgent(str(tmp_path), 'checkpoint', nodes, 'jupiter')
    assert agent.system_config.ngpus == 4
    assert agent.system_config.ncpus == 288
    assert agent.launcher_config.gpu_selector == 'CUDA_VISIBLE_DEVICES'
    assert agent.launcher_config.child_executor_name == executor
    assert agent.launcher_config.mpi_config.flavor == 'srun'
    assert agent.launcher_config.mpi_config.launcher == str(SCRIPTS / 'jupiter_srun.sh')
    assert '--gpu-bind=none' in agent.launcher_config.mpi_config.extra_launcher_flags
    assert agent.launcher_config.sequential_child_launch is True


def test_packaged_jupiter_recipes():
    skills = discover([ROOT / 'skills'])
    assert len(skills) == 17
    for name, directory in skills.items():
        body = read(skills, name)
        assert '/flare/' not in body and '/Users/' not in body
        assert 'skill_view(' not in body
        for match in re.finditer(r'file="([^"]+)"', body):
            assert not read(skills, name, match[1]).startswith('No file'), (name, match[1])
    assert '--no_kernels' in read(skills, 'boltz')
    assert 'recycle_strategy=medium/front_loaded' in read(skills, 'rfd4-proteina')
    assert 'num_workers=0' in read(skills, 'rfd4-proteina')
    assert 'licensed ARM64' in read(skills, 'pyrosetta')


def test_jupiter_discovers_all_allocated_slurm_hosts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(SCRIPTS))
    agent = importlib.import_module('hpc_agent')
    monkeypatch.delenv('SLURM_JOB_NODELIST', raising=False)
    monkeypatch.delenv('SLURM_NODELIST', raising=False)
    with pytest.raises(RuntimeError, match='requires SLURM_JOB_NODELIST'):
        agent._jupiter_nodes(5)
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'jpbo-018-[10-14]')
    hosts = [f'jpbo-018-{i}' for i in range(10, 15)]
    monkeypatch.setattr(agent.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='\n'.join(hosts) + '\n'))
    assert agent._jupiter_nodes(5) == hosts
    with pytest.raises(RuntimeError, match='contains 5 hosts; requested 4'):
        agent._jupiter_nodes(4)


def test_no_original_platform_install_paths_in_readable_skills():
    # read_skill can expose every supporting file, not only SKILL.md.
    forbidden = ('/flare/', '/lus/flare/', '/opt/aurora/', '/Users/', 'proxy.alcf.anl.gov')
    for path in (ROOT / 'skills').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            text = path.read_text(errors='replace')
            assert not any(value in text for value in forbidden), path
    assert not list((ROOT / 'skills').rglob('aurora-resources'))


def test_globus_readiness_retries_only_known_startup_races(monkeypatch):
    pytest.importorskip('fastmcp')
    cfg = load(ROOT / 'loam.yaml')
    for key, value in cfg.execution.hpc.environment().items():
        monkeypatch.setenv(key, value)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    server = importlib.import_module('mcp_server')
    for code in ('ENDPOINT_NOT_ONLINE', 'RESOURCE_CONFLICT'):
        error = RuntimeError('startup race')
        error.code = code
        assert server._is_cold_endpoint(error)
    assert not server._is_cold_endpoint(RuntimeError('UNAUTHORIZED'))
    assert not server._is_cold_endpoint(RuntimeError('invalid endpoint configuration'))


def test_queue_allowance_reaches_cluster_without_changing_allocation():
    cfg = load(ROOT / 'loam.yaml')
    deployment = cfg.execution.hpc
    original = deployment.environment()
    deployment.startup_timeout = 172800.0
    extended = deployment.environment()
    assert extended['HPC_CLUSTER_STARTUP_TIMEOUT'] == '172800.0'
    assert extended['HPC_ENDPOINT_CONFIG_JSON'] == original['HPC_ENDPOINT_CONFIG_JSON']


def test_token_is_refreshed_after_queue_wait(monkeypatch):
    import asyncio
    from loam import launcher
    cfg = load(ROOT / 'loam.yaml')
    monkeypatch.setenv('ALCF_TOKENS_BIN', '/fake/alcf-tokens')
    monkeypatch.setenv('ALCF_INFERENCE_TOKEN', 'old-test-token')
    class TokenProcess:
        returncode = 0
        async def communicate(self):
            return b'refreshed-test-token\n', b''
    async def spawn(*args, **kwargs):
        assert args == ('/fake/alcf-tokens', 'get-token', 'inference')
        return TokenProcess()
    monkeypatch.setattr(launcher.asyncio, 'create_subprocess_exec', spawn)
    asyncio.run(launcher._refresh_inference_token(cfg))
    import os
    assert os.environ['ALCF_INFERENCE_TOKEN'] == 'refreshed-test-token'


def test_only_code_generator_can_read_declared_external_software_docs(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from pydantic_ai.exceptions import ToolFailed
    from loam.execution import Deps, OwnedMCP
    from loam.schemas import Hypothesis
    from loam.workspace import Workspace
    cfg = load(ROOT / 'loam.yaml')
    cfg.state_dir = tmp_path / 'state'
    workspace = Workspace(cfg, Hypothesis(id='read-test', summary='test', content='test', elo=1200))
    _, run_dir = workspace.open_run(1, 'read docs')
    class Reader:
        async def call_tool(self, name, args, ctx, tool):
            return args['path']
    wrapped = OwnedMCP(Reader())
    doc = cfg.execution.hpc.inspect_roots[-1] + '/RFD4-Proteina/docs/reference/annotations.md'
    async def inspect(path, full):
        deps = Deps(cfg, {}, workspace, 1, 'test', run_dir=run_dir if full else '')
        return await wrapped.call_tool('inspect_filesystem', {'operation': 'cat', 'path': path},
                                       SimpleNamespace(deps=deps), None)
    assert asyncio.run(inspect(doc, True)) == doc
    with pytest.raises(ToolFailed):
        asyncio.run(inspect(doc, False))
    with pytest.raises(ToolFailed):
        asyncio.run(inspect(cfg.execution.hpc.base_dir + '/state/jnana.json', True))
    with pytest.raises(ToolFailed):
        asyncio.run(inspect('/e/project1/deepwing/meigooni1/agent/software/../other/file', True))


def test_shared_hpc_inputs_are_copied_once_per_hypothesis(tmp_path):
    from loam.schemas import Hypothesis
    from loam.workspace import Workspace
    cfg = load(ROOT / 'loam.yaml')
    cfg.state_dir = tmp_path / 'state'
    cfg.execution.hpc.base_dir = str(tmp_path / 'campaign')
    cfg.execution.workdir = str(tmp_path / 'campaign/work')
    inputs = tmp_path / 'verified-inputs'
    inputs.mkdir()
    (inputs / 'target.cif').write_text('verified source fixture')
    cfg.workspace_inputs = inputs
    cfg = type(cfg).model_validate(cfg.model_dump())
    h = Hypothesis(id='shared-inputs', summary='test', content='test', elo=1200)
    ws = Workspace(cfg, h)
    assert (Path(ws.root) / 'data/target.cif').read_text() == 'verified source fixture'
    (inputs / 'target.cif').write_text('later source version')
    Workspace(cfg, h)
    assert (Path(ws.root) / 'data/target.cif').read_text() == 'verified source fixture'


def test_read_roles_can_inspect_only_their_recorded_batch_outputs(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from pydantic_ai.exceptions import ToolFailed
    from loam.execution import Deps, OwnedMCP
    from loam.schemas import Hypothesis
    from loam.workspace import Workspace
    cfg = load(ROOT / 'loam.yaml')
    cfg.state_dir = tmp_path / 'state'
    ws = Workspace(cfg, Hypothesis(id='batch-read', summary='test', content='test', elo=1200))
    run_id, run_dir = ws.open_run(1, 'generate and inspect')
    ws.add_batch(run_id, 'b-owned')
    class Reader:
        async def call_tool(self, name, args, ctx, tool):
            return args['path']
    wrapped = OwnedMCP(Reader())
    root = cfg.execution.hpc.base_dir + '/batches'
    async def inspect(path, full):
        deps = Deps(cfg, {}, ws, 1, 'test', run_dir=run_dir if full else '')
        return await wrapped.call_tool('inspect_filesystem', {'operation': 'cat', 'path': path},
                                       SimpleNamespace(deps=deps), None)
    for full in (False, True):
        own = root + '/b-owned/run_101/result.json'
        assert asyncio.run(inspect(own, full)) == own
        for path in (root, root + '/b-other/run_101/result.json',
                     root + '/b-owned/../b-other/result.json'):
            with pytest.raises(ToolFailed):
                asyncio.run(inspect(path, full))


def test_registered_tool_environment_is_inherited_and_frozen_on_resume(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    agent = importlib.import_module('hpc_agent')
    monkeypatch.setattr(agent, '_MIN_BATCH_ROWS', 1)
    monkeypatch.setattr(agent, '_MIN_BATCH_DISTINCT', 1)
    source = tmp_path / 'stage.py'
    source.write_text('print("scientific stage")\n')
    inputs = tmp_path / 'inputs.jsonl'
    inputs.write_text('{"seed": 1}\n')
    environment = tmp_path / 'scientific/.venv'
    (environment / 'bin').mkdir(parents=True)
    (environment / 'bin/python').symlink_to(sys.executable)
    agent._register_tool('environment_test', str(source), {'seed': 'int'}, 'test', {'seed': 1},
                         None, 'json', str(environment), True, None, ['result.json'],
                         str(tmp_path), [str(tmp_path)])
    entry, _, _ = agent._prepare_batch(None, str(inputs), None, {'hypothesis_id': 'test'},
        900, 1, None, str(tmp_path), [str(tmp_path)], 'environment_test', None)
    assert entry['venv_path'] == str(environment)
    resumed, _, _ = agent._prepare_batch(None, None, entry['batch_id'], {'hypothesis_id': 'test'},
        900, 1, '/different/env', str(tmp_path), [str(tmp_path)], None, None)
    assert resumed['venv_path'] == str(environment)


def test_promoted_outputs_preserve_native_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    agent = importlib.import_module('hpc_agent')
    folder = tmp_path / 'run_candidate'
    attempt = folder / 'attempt_1'
    product = attempt / 'generation' / 'binder.cif'
    product.parent.mkdir(parents=True)
    product.write_text('real output placeholder for path test\n')
    (attempt / 'result.json').write_text(json.dumps({'structure': str(product)}))
    agent._promote_outputs(attempt, folder)
    stored = json.loads((folder / 'result.json').read_text())
    assert Path(stored['structure']).read_text() == (folder / 'generation/binder.cif').read_text()
    agent._promote_outputs(attempt, folder)
    assert product.exists()  # Repeated promotion must not create self-referential links.


def test_missing_interpreter_finishes_failed_instead_of_waiting_forever(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    agent = importlib.import_module('hpc_agent')
    script = tmp_path / 'stage.py'
    script.write_text('print("never runs")\n')
    folder = tmp_path / 'row'
    status = agent._run_one(str(script), {'seed': 1}, str(folder), 5, str(tmp_path / 'absent'))
    assert status['returncode'] == 127
    assert (folder / 'done').exists()
    assert 'FileNotFoundError' in (folder / 'stderr.txt').read_text()
    assert not (folder / 'result.json').exists()
