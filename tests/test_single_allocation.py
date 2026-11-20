"""The combined allocation must never send science to inference/foreign hosts."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from loam.allocation import read_allocation, validate_role

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills/hpc-deployment/scripts'


def manifest(tmp_path, **changes):
    data = dict(mode='single-allocation', job_id='123', all_nodes=['a', 'b', 'c'],
                science_nodes=['a', 'b'], inference_nodes=['c'])
    data.update(changes)
    path = tmp_path / 'allocation.json'
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize('changes', [
    {'science_nodes': ['a', 'b', 'c']},  # overlapping roles
    {'inference_nodes': ['foreign']},
    {'all_nodes': ['a', 'b', 'c', 'orphan']},
    {'science_nodes': ['a', 'a', 'b']},
    {'inference_nodes': []},
    {'job_id': ''},
])
def test_rejects_invalid_role_partition(tmp_path, changes):
    with pytest.raises(RuntimeError):
        read_allocation(manifest(tmp_path, **changes))


def test_enforces_parent_job_and_host_role(tmp_path):
    path = manifest(tmp_path)
    assert validate_role(path, 'science', hostname='a.local', job_id='123')['job_id'] == '123'
    assert validate_role(path, 'inference', hostname='c', job_id='123')['job_id'] == '123'
    for role, host, job in [('science', 'c', '123'), ('inference', 'a', '123'),
                            ('science', 'a', '456'), ('science', 'foreign', '123')]:
        with pytest.raises(RuntimeError, match='assigned Slurm'):
            validate_role(path, role, hostname=host, job_id=job)


def test_el_sees_only_science_subset(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    agent = importlib.import_module('hpc_agent')
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'all')
    monkeypatch.setenv('HPC_SCIENCE_NODELIST', 'science')
    replies = {'all': 'a\nb\nc\n', 'science': 'a\nb\n'}
    monkeypatch.setattr(agent.subprocess, 'run', lambda args, **kw: SimpleNamespace(stdout=replies[args[-1]]))
    assert agent._jupiter_nodes(2) == ['a', 'b']
    for invalid in ('a\nc\nforeign\n', 'a\na\n', 'a\n'):
        replies['science'] = invalid
        with pytest.raises(RuntimeError, match='Assigned science nodes'):
            agent._jupiter_nodes(2)


def test_subprocess_timeout_kills_descendants(tmp_path, monkeypatch):
    import signal
    import subprocess
    import sys
    import time

    monkeypatch.syspath_prepend(str(SCRIPTS))
    agent = importlib.import_module('hpc_agent')
    payload = (
        'import subprocess,sys,time; '
        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"]); '
        'print(p.pid,flush=True); time.sleep(10)'
    )
    start = time.monotonic()
    code, stdout, _, timed_out = agent._run_process(
        [sys.executable, '-c', payload], tmp_path, 0.5,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert timed_out and code == -signal.SIGKILL
    assert time.monotonic() - start < 4  # An inherited pipe must not wait for the child.
    child = Path('/proc') / stdout.decode().strip() / 'status'
    if child.exists():
        assert '\nState:\tZ' in child.read_text()  # Killed, awaiting its new parent's reap.


def test_existing_mode_does_not_construct_globus_executor(monkeypatch):
    from loam.config import load
    config = load(SCRIPTS.parents[2] / 'loam.yaml')
    for key, value in config.execution.hpc.environment().items():
        monkeypatch.setenv(key, value)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    server = importlib.import_module('mcp_server')
    monkeypatch.setattr(server, 'LAUNCH_MODE', 'existing')
    def forbidden(**kwargs):
        raise AssertionError('Existing-allocation mode attempted Globus allocation')
    monkeypatch.setattr(server, 'Executor', forbidden)
    assert server.make_executor() is None


def test_frozen_campaign_rejects_placeholder_scientific_tools(tmp_path):
    import asyncio
    from pydantic_ai.exceptions import ToolFailed
    from loam.config import load
    from loam.execution import Deps, OwnedMCP
    from loam.schemas import Hypothesis
    from loam.workspace import Workspace
    cfg=load(SCRIPTS.parents[2]/'loam.yaml')
    cfg.state_dir=tmp_path/'state';cfg.execution.hpc.base_dir=str(tmp_path/'campaign')
    cfg.execution.hpc.frozen_tool_library=True
    ws=Workspace(cfg,Hypothesis(id='frozen',summary='test',content='test',elo=1200))
    _,run_dir=ws.open_run(1,'test')
    base=Path(cfg.execution.hpc.base_dir);base.mkdir(exist_ok=True,parents=True)
    (base/'campaign-tool-registrations.json').write_text(json.dumps([{'tool':'frozen_science','version':1}]))
    class Recorder:
        async def call_tool(self,name,args,ctx,tool):
            assert name=='run_batch' and args['tool']=='frozen_science' and args['tool_version']==1
            return {'ok':True}
    wrapped=OwnedMCP(Recorder());ctx=SimpleNamespace(deps=Deps(cfg,{},ws,1,'test',run_dir=run_dir))
    async def run():
        for name,args in [('register_tool',{'name':'placeholder'}),('run_batch',{'tool':'placeholder'}),('run_batch',{'script_path':'script.py'})]:
            with pytest.raises(ToolFailed,match='operator-frozen scientific tool library'):
                await wrapped.call_tool(name,args,ctx,None)
        assert await wrapped.call_tool('run_batch',{'tool':'frozen_science','tool_version':99},ctx,None)=={'ok':True}
    asyncio.run(run())
