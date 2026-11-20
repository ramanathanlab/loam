"""A full round offline: scripted models, a stub tournament, real local execution."""

import asyncio
import contextlib
import json
import math
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from loam.agents import build
from loam.config import Config
from loam.orchestrator import Campaign
from loam.schemas import Hypothesis

SKILLS = Path(__file__).resolve().parents[1] / 'skills'


def scripted(*steps, seen: list | None = None):
    """A model that makes step i's tool call on its i-th turn; tool None means 'final output'.
    seen, if given, collects the text of every tool result the model was shown."""
    def respond(messages, info: AgentInfo):
        if seen is not None and isinstance(messages[-1], ModelRequest):
            seen.extend(str(getattr(p, 'content', '')) for p in messages[-1].parts if hasattr(p, 'tool_name'))
        tool, args = steps[sum(isinstance(m, ModelResponse) for m in messages)]
        return ModelResponse(parts=[ToolCallPart(tool or info.output_tools[0].name, args)])
    return FunctionModel(respond)


H = Hypothesis(id='h-abc123', summary='n=1 maximizes divisors of n^2+1', content='...', elo=1250)


class StubTournament:
    def run(self, goal, *, initial, brief=''):
        return [H]

    def pool(self):
        return {H.id: H, 'h-other': Hypothesis(id='h-other', summary='n=7', content='...', elo=1150)}

    def top(self):
        return [H]


class BrokenTournament(StubTournament):
    def run(self, goal, *, initial, brief=''):
        raise ConnectionError('Jnana endpoint down')


def run_round(cfg, agents, tournament=None, **models):
    """One round with the given scripted models (validator=..., reviewer=..., ...)."""
    models.setdefault('reasoner', scripted((None, {'report': '# synthesis', 'tournament_brief': 'next'})))
    with contextlib.ExitStack() as stack:
        for role, model in models.items():
            stack.enter_context(getattr(agents, role).override(model=model))
        return asyncio.run(Campaign(cfg, agents=agents, tournament=tournament or StubTournament()).round())


def report(claims=(), outcome='inconclusive'):
    return (None, {'outcome': outcome, 'evidence_score': 0, 'ledger': '', 'claims': list(claims), 'next_step': ''})


def make_cfg(tmp_path) -> Config:
    test = {'model': 'test'}
    return Config(goal='Find n in [1,10] maximizing d(n^2+1).', state_dir=tmp_path / 'state',
                  skills_dirs=[SKILLS],
                  models=dict.fromkeys(['validator', 'code_generator', 'reviewer', 'reasoner'], test),
                  execution={'mode': 'local', 'workdir': str(tmp_path / 'work')},
                  tournament={'jnana_root': tmp_path, 'generation': test, 'ranking': test})


def test_full_round(tmp_path):
    cfg = make_cfg(tmp_path)
    agents = build(cfg)
    script = 'import json; json.dump({"best_n": max(range(1, 11), key=lambda n: sum((n*n+1) % d == 0 for d in range(1, n*n+2)))}, open("result.json", "w"))'
    claim = {'claim': 'best_n is 7, not 1; gate: exhaustive search', 'source': 'outputs/r001-01/result.json', 'stance': 'contradicts'}

    validator = scripted(
        ('code_generator', {'instruction': 'Exhaustively compute best n; write result.json.'}),
        (None, {'outcome': 'contradicted', 'evidence_score': -1, 'ledger': '| 1 | search | n=10 |',
                'claims': [claim], 'next_step': 'none'}))
    code_generator = scripted(
        ('write_files', {'files': {'search.py': script}}),
        ('run_shell', {'command': 'python3 search.py && cat result.json'}),
        (None, {'status': 'executed', 'code': script, 'result': '{"best_n": 7}', 'artifacts': ['result.json']}))
    reviewer = scripted(
        ('read_file', {'path': 'outputs/r001-01/result.json'}),
        (None, {'verdicts': [{'index': 0, 'verdict': 'accepted', 'verification': 'artifact_attested',
                              'source_ref': 'outputs/r001-01/result.json#best_n', 'checked': 'read result.json: best_n 7'}]}))
    reasoner = scripted((None, {'report': '# synthesis', 'tournament_brief': 'retire n=1 family'}))

    with agents.validator.override(model=validator), agents.code_generator.override(model=code_generator), \
            agents.reviewer.override(model=reviewer), agents.reasoner.override(model=reasoner):
        record = asyncio.run(Campaign(cfg, agents=agents, tournament=StubTournament()).round())

    thread = record['threads'][0]
    assert thread['error'] is None, thread['error']
    assert thread['code_results'][0]['status'] == 'executed'

    # workspace: the run wrote into its own outputs dir and was recorded
    out = Path(cfg.execution.workdir) / 'h-abc123' / 'outputs' / 'r001-01' / 'result.json'
    assert json.loads(out.read_text()) == {'best_n': 7}
    records = cfg.state_dir / 'hypotheses' / 'h-abc123'
    assert json.loads((records / 'runs' / 'r001-01.json').read_text())['state'] == 'executed'
    experiment = json.loads((records / 'experiment.json').read_text())
    assert experiment['rounds'][0]['outcome'] == 'contradicted' and experiment['rounds'][0]['runs'] == ['r001-01']

    # reconcile: an artifact claim gets attested weight and its run as independence group
    [accepted] = thread['accepted']
    assert (accepted['policy'], accepted['group'], accepted['stance']) == ('artifact_attested', 'run:r001-01', 'contradicts')
    assert accepted['delta'] < 0

    # steering: the contradicted favourite drops below the untested hypothesis
    table = {row['hypothesis_id']: row for row in record['steering']}
    assert table['h-abc123']['evidence_rating'] < 1200 < table['h-abc123']['elo']
    assert table['h-abc123']['recommended_action'] == 'resolve_rank_evidence_disagreement'
    assert table['h-other']['uncertainty'] == 1.0

    assert record['brief']['tournament_brief'] == 'retire n=1 family'
    assert (cfg.state_dir / 'rounds' / 'round-001.json').is_file()


def test_reconciliation_requires_a_recorded_owned_run(tmp_path):
    from loam.reconcile import accepted
    from loam.schemas import Review, ValidationReport
    from loam.workspace import Workspace

    cfg = make_cfg(tmp_path)
    ws = Workspace(cfg, H)
    run_id, _ = ws.open_run(1, 'produce y')
    report = ValidationReport(outcome='supported', evidence_score=0.5, ledger='', next_step='', claims=[
        {'claim': 'x', 'source': 'https://example.test/record', 'stance': 'supports'},
        {'claim': 'y', 'source': f'outputs/{run_id}/y.json', 'stance': 'supports'}])
    review = Review(verdicts=[
        {'index': 0, 'verdict': 'accepted', 'verification': 'artifact_attested',
         'source_ref': 'result.json', 'checked': 'read result.json'},
        {'index': 1, 'verdict': 'accepted', 'verification': 'artifact_attested',
         'source_ref': f'outputs/{run_id}/y.json#y', 'checked': 'read y.json'}])

    assert accepted(report, review, unit='h:r1') == []
    [row] = accepted(report, review, unit='h:r1', workspace=ws)
    assert (row['claim'], row['group']) == ('y', f'run:{run_id}')


def test_agent_outputs_reject_unknown_fields_and_require_complete_review():
    from loam.reconcile import accepted
    from loam.schemas import CodeResult, Review, ValidationReport

    report = ValidationReport(outcome='supported', evidence_score=0.5, ledger='', next_step='', claims=[
        {'claim': 'x', 'source': 'https://example.test/x', 'stance': 'supports'},
        {'claim': 'y', 'source': 'https://example.test/y', 'stance': 'supports'}])
    with pytest.raises(ValueError, match='Extra inputs'):
        Review(verdicts=[{'index': 0, 'verdict': 'rejected', 'checked': 'absent', 'unexpected': True}])
    with pytest.raises(ValueError, match='Extra inputs'):
        CodeResult(status='blocked', code='', result='no run', invented='field')
    with pytest.raises(ValueError, match='ordered, zero-based'):
        Review(verdicts=[{'index': 1, 'verdict': 'rejected', 'checked': 'absent'}])
    with pytest.raises(ValueError, match='one ordered verdict for every claim'):
        accepted(report, Review(verdicts=[
            {'index': 0, 'verdict': 'rejected', 'checked': 'absent'}]), unit='h:r1')


def test_jnana_list_matches_are_normalized(tmp_path):
    from loam.tournament import Tournament

    cfg = make_cfg(tmp_path)
    tournament = Tournament(cfg.tournament, cfg.state_dir)
    tournament.state_file.parent.mkdir(parents=True)
    tournament.state_file.write_text(json.dumps({'hypotheses': [{
        'hypothesis_id': 'h-list', 'summary': 'candidate', 'content': 'body',
        'elo_rating': 1234, 'tournament_matches': [{'winner': 'h-list'}, {'winner': 'h-other'}],
    }]}))
    assert tournament.pool()['h-list'].matches == 2

def test_jnana_pool_skips_malformed_rows(tmp_path):
    from loam.tournament import Tournament

    tournament = Tournament(make_cfg(tmp_path).tournament, tmp_path / 'state')
    tournament.state_file.parent.mkdir(parents=True)
    tournament.state_file.write_text(json.dumps({'hypotheses': [
        None,
        {'hypothesis_id': '', 'summary': 'empty id', 'content': 'body', 'elo_rating': 1200},
        {'hypothesis_id': 'no-summary', 'content': 'body', 'elo_rating': 1200},
        {'hypothesis_id': 'nan', 'summary': 'bad rating', 'content': 'body', 'elo_rating': 'nan'},
        {'hypothesis_id': 'good', 'summary': '  valid   candidate ', 'content': 'body',
         'elo_rating': '1230', 'tournament_matches': 'bad'},
    ]}))

    pool = tournament.pool()
    assert list(pool) == ['good']
    assert (pool['good'].summary, pool['good'].elo, pool['good'].matches) == ('valid candidate', 1230, 0)


def test_round_generation_control_reuses_the_previous_brief(tmp_path):
    class ObservingTournament(StubTournament):
        def __init__(self):
            self.calls = []

        def run(self, goal, *, initial, brief=''):
            self.calls.append((goal, initial, brief))
            return [H]

    cfg = make_cfg(tmp_path)
    tournament = ObservingTournament()
    agents = build(cfg)
    run_round(cfg, agents, tournament=tournament, validator=scripted(report()),
              reasoner=scripted((None, {'report': 'first', 'tournament_brief': 'probe neighbors'})))
    run_round(cfg, agents, tournament=tournament, validator=scripted(report()))

    assert tournament.calls == [(cfg.goal_text(), True, ''),
                                (cfg.goal_text(), False, 'probe neighbors')]


def test_sparse_round_numbers_advance_from_the_maximum(tmp_path):
    cfg = make_cfg(tmp_path)
    rounds = cfg.state_dir / 'rounds'
    rounds.mkdir(parents=True)
    (rounds / 'round-002.json').write_text(json.dumps({'brief': {'tournament_brief': 'stale'}}))
    (rounds / 'round-009.json').write_text(json.dumps({'brief': {'tournament_brief': 'latest'}}))

    class Observer(StubTournament):
        def run(self, goal, *, initial, brief=''):
            self.control = initial, brief
            return [H]

    tournament = Observer()
    record = run_round(cfg, build(cfg), tournament=tournament, validator=scripted(report()))
    assert record['round'] == 10
    assert tournament.control == (False, 'latest')
    assert (rounds / 'round-010.json').is_file()


def test_unsafe_and_colliding_hypothesis_ids_are_rejected(tmp_path):
    from loam.workspace import Workspace

    cfg = make_cfg(tmp_path)
    unsafe = Hypothesis(id='../escape', summary='bad', content='bad', elo=1200)
    with pytest.raises(ValueError, match='unsafe hypothesis id'):
        Workspace(cfg, unsafe)

    Workspace(cfg, H)
    experiment = cfg.state_dir / 'hypotheses' / H.id / 'experiment.json'
    stored = json.loads(experiment.read_text())
    stored['id'] = 'different-id'
    experiment.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match='identity mismatch'):
        Workspace(cfg, H)


def test_context_claims_do_not_reduce_uncertainty(tmp_path):
    from loam.steering import project

    ledger = tmp_path / 'ledger.jsonl'
    ledger.write_text(json.dumps({'hypothesis_id': 'h', 'group': 'source:background',
                                  'stance': 'context', 'delta': 0.0}) + '\n')
    [row] = project(ledger, {'h': 1200})
    assert row['uncertainty'] == 1.0
    assert row['independent_supports'] == row['independent_contradictions'] == 0


def hpc_config_values(tmp_path):
    script = tmp_path / 'scripts' / 'serve.py'
    script.parent.mkdir(exist_ok=True)
    script.write_text('# managed server')
    values = make_cfg(tmp_path).model_dump(mode='json')
    values['rounds'] = 3
    values['execution'] = {
        'mode': 'hpc',
        'workdir': '/campaign/work',
        'setup': '',
        'hpc': {
            'endpoint_id': 'endpoint-uuid',
            'campaign_id': 'campaign-17',
            'system': 'aurora',
            'exchange_url': 'https://exchange.example',
            'base_dir': '/campaign',
            'worker_code_dir': '/campaign/loam',
            'inspect_roots': ['/datasets', '/campaign/results'],
            'pool_size': 5,
            'server_script': 'scripts/serve.py',
            'host': '0.0.0.0',
            'port': 8765,
            'startup_timeout': 4,
            'allocation': {
                'account': 'project',
                'queue': 'debug',
                'walltime': '01:02:03',
                'nodes': 2,
                'init_blocks': 1,
                'min_blocks': 1,
                'max_blocks': 3,
                'max_idletime': 90,
                'endpoint_options': {'worker_init': 'module load tools'},
            },
        },
    }
    return values, script


def test_config_requires_positive_top_level_rounds(tmp_path):
    for rounds in (0, -1):
        values = make_cfg(tmp_path).model_dump()
        values['rounds'] = rounds
        with pytest.raises(ValueError, match='greater than 0'):
            Config.model_validate(values)


def test_managed_hpc_yaml_resolves_script_and_generates_canonical_environment(tmp_path):
    import yaml

    from loam.config import load

    values, script = hpc_config_values(tmp_path)
    config_path = tmp_path / 'campaign.yaml'
    config_path.write_text(yaml.safe_dump(values))

    cfg = load(config_path)
    deployment = cfg.execution.hpc
    assert deployment is not None
    assert cfg.rounds == 3
    assert deployment.server_script == script.resolve()

    environment = deployment.environment()
    assert environment == {
        'HPC_GLOBUS_ENDPOINT_ID': 'endpoint-uuid',
        'HPC_CAMPAIGN_ID': 'campaign-17',
        'HPC_SYSTEM': 'aurora',
        'ACADEMY_EXCHANGE_URL': 'https://exchange.example',
        'HPC_BASE_DIR': '/campaign',
        'HPC_LOAM_HOME': '/campaign/loam',
        'HPC_INSPECT_ROOTS': '/datasets:/campaign/results',
        'HPC_POOL_SIZE': '5',
        # Cold-endpoint retry bounds: config-driven, defaulted here. They are
        # NOT in HPC_ENDPOINT_CONFIG_JSON on purpose -- changing them must not
        # fork a new user endpoint, since that would forfeit a running block.
        'HPC_COLD_ENDPOINT_TIMEOUT': '300.0',
        'HPC_COLD_ENDPOINT_BACKOFF': '5.0',
        'HPC_ENDPOINT_CONFIG_JSON': json.dumps({
            'worker_init': 'module load tools',
            'account': 'project',
            'queue': 'debug',
            'walltime': '01:02:03',
            'nodes_per_block': 2,
            'init_blocks': 1,
            'min_blocks': 1,
            'max_blocks': 3,
            'max_idletime': 90,
            'available_accelerators': 12,
            'max_workers_per_node': 12,
            'base_dir': '/campaign/loam',
        }, sort_keys=True, separators=(',', ':')),
        'HPC_NNODES': '2',
        'HPC_MCP_TRANSPORT': 'http',
        'HPC_MCP_HOST': '0.0.0.0',
        'HPC_MCP_PORT': '8765',
    }
    assert not any(token in key.upper() for key in environment for token in ('SECRET', 'TOKEN', 'PASSWORD', 'KEY'))


@pytest.mark.parametrize(('change', 'message'), [
    (lambda values: values['execution']['hpc']['allocation']['endpoint_options'].update(account='other'),
     'cannot override typed fields: account'),
    (lambda values: values['execution']['hpc']['allocation'].update(min_blocks=2, init_blocks=1),
     'min_blocks <= init_blocks <= max_blocks'),
    (lambda values: values['execution']['hpc']['allocation'].update(walltime='1:00'),
     'walltime must have format HH:MM:SS'),
    (lambda values: values['execution'].update(workdir='/elsewhere/work'),
     'workdir must be under execution.hpc.base_dir'),
])
def test_managed_hpc_rejects_invalid_allocation_and_paths(tmp_path, change, message):
    values, _ = hpc_config_values(tmp_path)
    change(values)
    with pytest.raises(ValueError, match=message):
        Config.model_validate(values)


@pytest.mark.parametrize(('execution', 'message'), [
    ({'mode': 'mcp', 'mcp_url': 'https://mcp.example/mcp', 'workdir': '/remote', 'hpc': {}},
     'execution.hpc is only valid in HPC mode'),
    ({'mode': 'local', 'mcp_url': 'https://mcp.example/mcp', 'workdir': 'work'},
     'execution.hpc and execution.mcp_url are not valid in local mode'),
])
def test_external_and_local_modes_reject_incompatible_fields(tmp_path, execution, message):
    values, _ = hpc_config_values(tmp_path)
    if execution.get('hpc') == {}:
        execution = {**execution, 'hpc': values['execution']['hpc']}
    values['execution'] = execution
    with pytest.raises(ValueError, match=message):
        Config.model_validate(values)


def test_managed_launcher_runs_until_ready_then_interrupts_cleanly(tmp_path, monkeypatch):
    import loam.launcher as launcher

    values, script = hpc_config_values(tmp_path)
    cfg = Config.model_validate(values)
    cfg.execution.hpc.server_script = script
    calls = []

    class Process:
        returncode = None

        async def wait(self):
            if self.returncode is None:
                await asyncio.Future()
            return self.returncode

        def send_signal(self, sent):
            calls.append(('signal', sent))
            self.returncode = 0

    process = Process()

    async def spawn(*args, **kwargs):
        calls.append(('spawn', args, kwargs))
        return process

    readiness_attempts = 0

    async def check_tools(checked_cfg):
        nonlocal readiness_attempts
        readiness_attempts += 1
        calls.append(('ready', checked_cfg.execution.mcp_url))
        if readiness_attempts == 1:
            raise ConnectionError('starting')

    monkeypatch.setattr(launcher.asyncio, 'create_subprocess_exec', spawn)
    monkeypatch.setattr(launcher, 'check_mcp_tools', check_tools)

    async def exercise():
        async with launcher.managed_hpc_server(cfg):
            calls.append(('running', cfg.execution.mcp_url))

    asyncio.run(exercise())

    assert cfg.execution.mcp_url == 'http://127.0.0.1:8765/mcp'
    assert readiness_attempts == 2
    spawn_call = calls[0]
    assert spawn_call[1] == (sys.executable, str(script))
    assert spawn_call[2]['env']['HPC_GLOBUS_ENDPOINT_ID'] == 'endpoint-uuid'
    assert spawn_call[2]['env']['HPC_MCP_PORT'] == '8765'
    assert ('running', cfg.execution.mcp_url) in calls
    assert calls[-1] == ('signal', signal.SIGINT)


@pytest.mark.parametrize('patch, message', [
    ({'execution': {'mode': 'mcp', 'mcp_url': 'file:///tmp/mcp', 'workdir': '/cluster/work'}},
     'mcp_url'),
    ({'execution': {'mode': 'mcp', 'mcp_url': 'https://mcp.example/mcp', 'workdir': 'relative'}},
     'absolute POSIX'),
    ({'execution': {'mode': 'mcp', 'mcp_url': 'https://mcp.example/mcp', 'workdir': '/cluster/../work'}},
     'absolute POSIX'),
    ({'execution': {'mode': 'mcp', 'mcp_url': 'https://mcp.example/mcp', 'workdir': '/cluster/work'},
      'workspace_inputs': Path('/inputs')}, 'workspace_inputs'),
    ({'concurrency': 0}, 'greater than 0'),
])
def test_config_rejects_unsafe_execution_constraints(tmp_path, patch, message):
    values = make_cfg(tmp_path).model_dump()
    values.update(patch)
    with pytest.raises(ValueError, match=message):
        Config.model_validate(values)


def test_config_rejects_unknown_keys(tmp_path):
    values = make_cfg(tmp_path).model_dump()
    values['concurency'] = 2
    with pytest.raises(ValueError, match='Extra inputs'):
        Config.model_validate(values)

def test_config_requires_every_model_role_and_positive_limits(tmp_path):
    values = make_cfg(tmp_path).model_dump()
    values['models'].pop('reviewer')
    with pytest.raises(ValueError, match='missing: reviewer'):
        Config.model_validate(values)

    for field in ('concurrency', 'max_code_calls', 'max_requests', 'model_retries'):
        invalid = make_cfg(tmp_path).model_dump()
        invalid[field] = 0
        with pytest.raises(ValueError, match='greater than 0'):
            Config.model_validate(invalid)


def test_workspace_input_initialization_retries_after_copy_failure(tmp_path, monkeypatch):
    import loam.workspace as workspace

    cfg = make_cfg(tmp_path)
    cfg.workspace_inputs = tmp_path / 'inputs'
    cfg.workspace_inputs.mkdir()
    (cfg.workspace_inputs / 'sample.txt').write_text('ready')
    real_copytree = workspace.shutil.copytree
    attempts = 0

    def flaky_copytree(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            Path(destination).mkdir()
            (Path(destination) / 'partial.txt').write_text('partial')
            raise OSError('copy interrupted')
        return real_copytree(source, destination)

    monkeypatch.setattr(workspace.shutil, 'copytree', flaky_copytree)
    with pytest.raises(OSError, match='interrupted'):
        workspace.Workspace(cfg, H)
    ws = workspace.Workspace(cfg, H)

    assert attempts == 2
    assert (Path(ws.root) / 'data' / 'sample.txt').read_text() == 'ready'
    assert not (Path(ws.root) / 'data' / 'partial.txt').exists()



def test_batch_receipts_survive_restart(tmp_path):
    from loam.workspace import Workspace

    cfg = make_cfg(tmp_path)
    ws = Workspace(cfg, H)
    run_id, _ = ws.open_run(1, 'sweep')
    ws.add_batch(run_id, 'b-123')
    ws.add_batch(run_id, 'b-123')
    assert Workspace(cfg, H).batch_ids() == ['b-123']
    run = json.loads((ws.records / 'runs' / f'{run_id}.json').read_text())
    assert run['batches'] == ['b-123']

def test_validator_can_poll_a_batch_from_a_crashed_code_generator(tmp_path):
    from loam.execution import Deps, OwnedMCP
    from loam.workspace import Workspace

    class Server:  # stands in for the MCP server: run_batch returns a receipt, others echo their args
        async def call_tool(self, name, args, ctx, tool):
            return {'batch_id': 'b-1'} if name == 'run_batch' else args

    cfg = make_cfg(tmp_path)
    ws = Workspace(cfg, H)
    run_id, run_dir = ws.open_run(1, 'sweep')
    validator = Deps(cfg, {}, ws, 1, 'h')
    mcp = OwnedMCP(Server())
    asyncio.run(mcp.call_tool('run_batch', {'script_path': 'run.py'},
                              SimpleNamespace(deps=replace(validator, run_dir=run_dir)), None))
    ws.close_run(run_id, None, error='timed out')                        # the code-generator died
    poll = asyncio.run(mcp.call_tool('batch_status', {'batch_id': 'b-1'}, SimpleNamespace(deps=validator), None))
    assert poll == {'batch_id': 'b-1'}
    with pytest.raises(ToolFailed, match='does not belong'):
        asyncio.run(mcp.call_tool('batch_status', {'batch_id': 'b-other'}, SimpleNamespace(deps=validator), None))

def test_read_only_mcp_tools_allow_pool_status():
    from loam.execution import MCP_FULL, MCP_READ

    assert 'pool_status' in MCP_READ
    # Read roles get tool DISCOVERY so a validator can prefer a stage an earlier
    # round already proved, but not tool CREATION.
    assert 'list_tools' in MCP_READ
    assert 'register_tool' not in MCP_READ
    assert MCP_FULL == MCP_READ | {'write_files', 'run_batch', 'run_code',
                                   'register_tool'}


def test_remote_labelled_batches_are_adopted_only_for_owned_runs(tmp_path):
    from loam.execution import Deps, OwnedMCP
    from loam.workspace import Workspace

    class Server:
        async def call_tool(self, name, args, ctx, tool):
            assert name == 'list_batches'
            assert args['hypothesis_id'] == H.id
            return [
                {'batch_id': 'owned', 'labels': {'hypothesis_id': H.id, 'run_id': run_id}},
                {'batch_id': 'foreign-hypothesis',
                 'labels': {'hypothesis_id': 'other', 'run_id': run_id}},
                {'batch_id': 'foreign-run',
                 'labels': {'hypothesis_id': H.id, 'run_id': 'r999-01'}},
                {'batch_id': 'unlabelled'},
            ]

    cfg = make_cfg(tmp_path)
    cfg.execution.mode = 'mcp'
    cfg.execution.workdir = '/cluster/work'
    ws = Workspace(cfg, H)
    run_id, _ = ws.open_run(1, 'sweep')
    deps = Deps(cfg, {}, ws, 1, 'h')
    result = asyncio.run(OwnedMCP(Server()).call_tool(
        'list_batches', {}, SimpleNamespace(deps=deps), None))

    assert len(result) == 4
    assert ws.batch_ids() == ['owned']

def test_mcp_submission_and_execution_are_derived_from_observed_calls(tmp_path):
    from loam.execution import Deps, OwnedMCP
    from loam.workspace import Workspace

    class Server:
        async def call_tool(self, name, args, ctx, tool):
            if name == 'run_batch':
                return {'batch_id': 'b-1'}
            return {'batch_id': args['batch_id'], 'complete': True, 'ok': True,
                    'labels': {'hypothesis_id': H.id}}

    cfg = make_cfg(tmp_path)
    cfg.execution.mode = 'mcp'
    cfg.execution.workdir = '/cluster/work'
    ws = Workspace(cfg, H)
    _, run_dir = ws.open_run(1, 'sweep')
    deps = Deps(cfg, {}, ws, 1, 'h', run_dir=run_dir)
    ctx = SimpleNamespace(deps=deps)
    mcp = OwnedMCP(Server())

    asyncio.run(mcp.call_tool('run_batch', {'script_path': 'run.py'}, ctx, None))
    assert deps.submitted_batches == {'b-1'} and deps.executed_batches == set()
    asyncio.run(mcp.call_tool('batch_status', {'batch_id': 'b-1'}, ctx, None))
    assert deps.executed_batches == {'b-1'}


def test_generation_timeout_aborts_before_ranking_or_save(tmp_path, monkeypatch):
    from loam.tournament import GenerationTimedOut, Tournament

    calls = []

    class CoScientist:
        def __init__(self, **kwargs):
            self.memory = SimpleNamespace(save=lambda: calls.append('save'))
        def get_all_hypotheses(self):
            return []

        set_research_goal = start = stop = generate_hypotheses = lambda self, *a, **k: None

        def wait_for_completion(self, timeout):
            return False

    fakes = {'jnana.protognosis.core.coscientist': SimpleNamespace(CoScientist=CoScientist),
             'jnana.protognosis.core.multi_llm_config': SimpleNamespace(LLMConfig=lambda **k: k),
             'jnana.protognosis.periodic_tournament': SimpleNamespace(
                 run_incremental_tournament=lambda **k: calls.append('rank'))}
    for name in ['jnana', 'jnana.protognosis', 'jnana.protognosis.core', *fakes]:
        monkeypatch.setitem(sys.modules, name, fakes.get(name, SimpleNamespace()))

    cfg = make_cfg(tmp_path)
    with pytest.raises(GenerationTimedOut, match='ranking was not started'):
        Tournament(cfg.tournament, cfg.state_dir).run('goal', initial=True)
    assert calls == []


def test_steering_discounts_repeats_from_one_group(tmp_path):
    from loam.steering import project

    ledger = tmp_path / 'ledger.jsonl'
    rows = [{'hypothesis_id': 'a', 'group': 'run:r1', 'delta': -0.392}] * 3 + \
           [{'hypothesis_id': 'b', 'group': f'source:{i}', 'delta': 0.576} for i in range(2)]
    ledger.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    table = {r['hypothesis_id']: r for r in project(ledger, {'a': 1300, 'b': 1200, 'c': 1250})}
    assert table['a']['independent_contradictions'] == 1                      # three claims, one run
    assert abs((table['a']['evidence_rating'] - 1200) / (400 / math.log(10)) + 0.392 * 1.35) < 1e-3
    assert table['b']['independent_supports'] == 2
    assert max(table.values(), key=lambda r: r['priority'])['hypothesis_id'] == 'b'


def test_skills():
    from loam import skills

    found = skills.discover([SKILLS])
    assert 'chai' in found
    assert 'Chai-1' in skills.read(found, 'chai')
    assert 'No file' in skills.read(found, 'chai', '../../../etc/passwd')


# --- failures stay contained ------------------------------------------------

def test_failed_tool_call_is_a_result_not_a_crash(tmp_path):
    """Tool failures are returned to the reviewer, including URL policy rejections."""
    cfg = make_cfg(tmp_path)
    seen = []
    claim = {'claim': 'x', 'source': 'outputs/r001-01/missing.json', 'stance': 'supports'}
    record = run_round(cfg, build(cfg), validator=scripted(report([claim])), reviewer=scripted(
        ('read_file', {'path': 'outputs/r001-01/missing.json'}),
        ('read_file', {'path': '../../etc/passwd'}),
        ('fetch_url', {'url': 'http://127.0.0.1:9/'}),
        ('fetch_url', {'url': 'https://127.0.0.1:9/'}),
        (None, {'verdicts': [{'index': 0, 'verdict': 'rejected', 'checked': 'file missing'}]}), seen=seen))
    thread = record['threads'][0]
    assert thread['error'] is None and thread['review']['verdicts'][0]['verdict'] == 'rejected'
    assert 'FileNotFoundError' in seen[0] and 'outside' in seen[1]
    assert 'only HTTPS URLs' in seen[2]
    assert 'non-public destination' in seen[3]


def test_tool_timeout_kills_the_command_and_its_children(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.timeouts.tool = 1
    seen = []
    record = run_round(cfg, build(cfg),
        validator=scripted(('code_generator', {'instruction': 'run it'}), report()),
        code_generator=scripted(('run_shell', {'command': 'sleep 37.5 | cat'}),
                                (None, {'status': 'blocked', 'code': '', 'result': 'timed out'}), seen=seen))
    assert record['threads'][0]['error'] is None
    assert 'timed out after 1s' in seen[0]
    assert subprocess.run(['pgrep', '-f', 'sleep 37.5'], capture_output=True).returncode == 1  # nothing left


def test_code_generator_timeout_is_reported_to_the_validator(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.timeouts.code_generator = 1
    seen = []
    record = run_round(cfg, build(cfg),
        validator=scripted(('code_generator', {'instruction': 'run it'}), report(), seen=seen),
        code_generator=scripted(('run_shell', {'command': 'sleep 5'}),
                                (None, {'status': 'executed', 'code': '', 'result': ''})))
    thread = record['threads'][0]
    assert thread['error'] is None and thread['report']['outcome'] == 'inconclusive'
    assert 'ended without a result' in seen[0]
    assert thread['code_results'][0]['status'] == 'crashed'
    run = json.loads((cfg.state_dir / 'hypotheses' / 'h-abc123' / 'runs' / 'r001-01.json').read_text())
    assert run['state'] == 'crashed'


@pytest.mark.parametrize('steps, claimed, observed', [
    ((), 'executed', 'blocked'),
    ((('run_shell', {'command': 'true'}),), 'blocked', 'executed'),
])
def test_local_run_state_is_derived_from_observed_shell_calls(tmp_path, steps, claimed, observed):
    cfg = make_cfg(tmp_path)
    code_generator = scripted(
        *steps, (None, {'status': claimed, 'code': 'true', 'result': 'model claim'}))
    record = run_round(
        cfg, build(cfg),
        validator=scripted(('code_generator', {'instruction': 'run it'}), report()),
        code_generator=code_generator)

    result = record['threads'][0]['code_results'][0]
    stored = json.loads((cfg.state_dir / 'hypotheses' / H.id / 'runs' / 'r001-01.json').read_text())
    assert result['status'] == observed
    assert stored['state'] == observed

def test_failed_validator_leaves_the_round_intact(tmp_path):
    cfg = make_cfg(tmp_path)
    record = run_round(cfg, build(cfg), validator=scripted(('no_such_tool', {}), ('no_such_tool', {}),
                                                           ('no_such_tool', {}), ('no_such_tool', {}), report()))
    assert record['threads'][0]['error'] and record['brief']['tournament_brief'] == 'next'

def test_reviewer_failure_preserves_raw_report_but_leaves_round_untested(tmp_path):
    cfg = make_cfg(tmp_path)
    claim = {'claim': 'x', 'source': 'https://example.test/x', 'stance': 'supports'}
    broken_reviewer = scripted(*[('no_such_tool', {})] * 4)

    record = run_round(cfg, build(cfg), validator=scripted(report([claim], outcome='supported')),
                       reviewer=broken_reviewer)
    thread = record['threads'][0]
    saved = json.loads((cfg.state_dir / 'hypotheses' / H.id / 'experiment.json').read_text())

    assert thread['raw_report']['outcome'] == 'supported'
    assert thread['report'] is None and thread['review'] is None and thread['accepted'] == []
    assert thread['error']
    assert saved['rounds'][0]['outcome'] is None


def test_tournament_failure_falls_back_to_the_pool(tmp_path):
    cfg = make_cfg(tmp_path)
    record = run_round(cfg, build(cfg), tournament=BrokenTournament(), validator=scripted(report()))
    assert [h['id'] for h in record['rankings']] == ['h-abc123']


def test_malformed_ledger_tail_is_atomically_removed(tmp_path):
    from loam.reconcile import drop_round

    ledger = tmp_path / 'ledger.jsonl'
    keep = {'round': 2, 'hypothesis_id': 'keep', 'group': 'run:r002-01', 'delta': 0.2}
    ledger.write_text(
        json.dumps({'round': 1, 'hypothesis_id': 'retry', 'group': 'run:r001-01', 'delta': -0.39}) + '\n'
        + json.dumps(keep) + '\n'
        + '{"round": 1, "hypothesis_id": "cut-short"')

    drop_round(ledger, 1)

    assert ledger.read_text() == json.dumps(keep) + '\n'
    assert not ledger.with_name('ledger.jsonl.tmp').exists()


def test_mcp_mode_builds(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.execution.mode, cfg.execution.mcp_url = 'mcp', 'http://127.0.0.1:1/mcp'
    build(cfg)


def test_owned_artifact_resolves_outputs_batches_and_locators(tmp_path):
    """Ownership comes from run RECORDS, not from the shape of the path.

    Three regressions in one, all measured 2026-11-11:
      * a reviewer citing WHERE it checked (`... lines 2, 5-6`) must still match
        the claim -- 34 of 80 reviewed pairs were dropped on that alone;
      * a claim on a run_batch BATCH dir must be citable when the owning run
        recorded that batch -- width-path evidence was uncitable before;
      * a path an agent invented (a sibling `outputs/rNNN-01/` it created
        itself) must NOT be citable, because no record reserved it.
    """
    import json as _json

    from loam.reconcile import _owned_artifact

    class _WS:
        root = '/base/work/h-1'
        records = None

    ws = _WS()
    ws.records = tmp_path
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'runs' / 'r001-01.json').write_text(_json.dumps(
        {'run_id': 'r001-01', 'state': 'executed', 'batches': ['b-abc123']}))

    # its own reserved run dir, absolute and workspace-relative
    want = ('r001-01', 'outputs/r001-01/result.json')
    assert _owned_artifact('/base/work/h-1/outputs/r001-01/result.json', ws) == want
    assert _owned_artifact('outputs/r001-01/result.json', ws) == want
    # ...with a reviewer's line locator appended
    for suffix in (' lines 2, 5-6', ' line 9', ', line 12', ':12', ':12-18', '#L9-13'):
        assert _owned_artifact(f'outputs/r001-01/result.json{suffix}', ws) == want, suffix
    # ...without eating filenames that merely look like locators
    assert _owned_artifact('outputs/r001-01/line_counts.json', ws) is not None

    # a batch the run recorded, living OUTSIDE the workspace under base_dir
    owned_batch = _owned_artifact('/base/batches/b-abc123/run_01/out.cif', ws)
    assert owned_batch == ('r001-01', 'batches/b-abc123/run_01/out.cif')

    # not owned: unrecorded run dir, unrecorded batch, escape
    assert _owned_artifact('outputs/r011-01/feasibility_result.json', ws) is None
    assert _owned_artifact('/base/batches/b-nope/run_01/out.cif', ws) is None
    assert _owned_artifact('outputs/r001-01/../../etc/passwd', ws) is None


def _recorded_ws(tmp_path, batches=('b-abc123',)):
    """A workspace whose only recorded run is r001-01, which submitted `batches`."""
    class _WS:
        root = '/base/work/h-1'
        records = tmp_path

    (tmp_path / 'runs').mkdir()
    (tmp_path / 'runs' / 'r001-01.json').write_text(json.dumps(
        {'run_id': 'r001-01', 'state': 'executed', 'batches': list(batches)}))
    return _WS()


def test_unowned_sources_flags_invented_paths_and_skips_urls(tmp_path):
    """The question the orchestrator asks before the reviewer runs.

    A URL claim is verified against the review's source_ref, not against run
    ownership, so it must never be reported as unowned -- otherwise every
    literature claim would trigger a pointless corrective turn.
    """
    from loam.reconcile import citable_runs, owned_claim_count, unowned_sources
    from loam.schemas import Claim, ValidationReport

    ws = _recorded_ws(tmp_path)

    def claim(source):
        return Claim(claim='x', source=source, stance='supports')

    rep = ValidationReport(
        outcome='inconclusive', evidence_score=0, ledger='', next_step='',
        claims=[claim('outputs/r001-01/result.json'),              # its own run dir
                claim('/base/batches/b-abc123/run_01/out.cif'),    # a batch that run submitted
                claim('https://www.rcsb.org/structure/1ABC'),      # a URL: ownership N/A
                claim('outputs/audit-01/prerequisite_audit.json'),  # invented by the agent
                claim('outputs/r011-01/feasibility_result.json')])  # another round's leftovers

    assert unowned_sources(rep, ws) == [(3, 'outputs/audit-01/prerequisite_audit.json'),
                                        (4, 'outputs/r011-01/feasibility_result.json')]
    assert owned_claim_count(rep, ws) == 3
    assert citable_runs(ws) == [{'run_id': 'r001-01',
                                 'run_dir': '/base/work/h-1/outputs/r001-01',
                                 'batches': ['b-abc123']}]


def _provenance_validator(first_claims, second_claims, captured):
    """A validator that cites `first_claims`, then `second_claims` once corrected.

    Branches on the corrective prompt rather than on turn count, because the
    re-cite is a NEW run: a turn-counting script would simply repeat itself.
    """
    def respond(messages, info: AgentInfo):
        text = ' '.join(str(getattr(p, 'content', '')) for m in messages
                        if isinstance(m, ModelRequest) for p in m.parts)
        if 'PROVENANCE CHECK FAILED' in text:
            captured.append(text)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
                'outcome': 'contradicted', 'evidence_score': -1, 'ledger': '| 1 | search | n=10 |',
                'claims': list(second_claims), 'next_step': 'none'})])
        if sum(isinstance(m, ModelResponse) for m in messages) == 0:
            return ModelResponse(parts=[ToolCallPart('code_generator', {
                'instruction': 'Exhaustively compute best n; write result.json.'})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            'outcome': 'contradicted', 'evidence_score': -1, 'ledger': '| 1 | search | n=10 |',
            'claims': list(first_claims), 'next_step': 'none'})])
    return FunctionModel(respond)


CODEGEN_SCRIPT = ('import json; json.dump({"best_n": 7}, open("result.json", "w"))')


def _codegen():
    return scripted(
        ('write_files', {'files': {'search.py': CODEGEN_SCRIPT}}),
        ('run_shell', {'command': 'python3 search.py && cat result.json'}),
        (None, {'status': 'executed', 'code': CODEGEN_SCRIPT, 'result': '{"best_n": 7}',
                'artifacts': ['result.json']}))


def test_validator_is_told_in_round_when_a_claim_cites_an_unowned_artifact(tmp_path):
    """The feedback gap: reconciliation runs after the reviewer, so a validator
    that cites an unowned path never learned, and the round silently yielded
    nothing (round 12, 2026-11-11: 7 of 7 claims dropped this way).

    Now it gets one corrective turn naming the offending source and the runs it
    may cite, and the recovered claim reaches the ledger.
    """
    cfg = make_cfg(tmp_path)
    agents = build(cfg)
    captured: list[str] = []

    invented = {'claim': 'best_n is 7; gate: exhaustive search',
                'source': 'outputs/audit-01/prerequisite_audit.json', 'stance': 'contradicts'}
    owned = {'claim': 'best_n is 7; gate: exhaustive search',
             'source': 'outputs/r001-01/result.json', 'stance': 'contradicts'}

    reviewer = scripted(
        (None, {'verdicts': [{'index': 0, 'verdict': 'accepted', 'verification': 'artifact_attested',
                              'source_ref': 'outputs/r001-01/result.json',
                              'checked': 'read result.json: best_n 7'}]}))

    record = run_round(cfg, agents,
                       validator=_provenance_validator([invented], [owned], captured),
                       code_generator=_codegen(), reviewer=reviewer)

    thread = record['threads'][0]
    assert thread['error'] is None, thread['error']
    # the corrective turn happened, and said what was wrong and what was citable
    assert len(captured) == 1, captured
    assert 'outputs/audit-01/prerequisite_audit.json' in captured[0]
    assert 'r001-01' in captured[0]
    # the recovered claim was reviewed and accepted
    assert [c['source'] for c in thread['accepted']] == ['outputs/r001-01/result.json']
    assert thread['raw_report']['claims'][0]['source'] == 'outputs/r001-01/result.json'


def test_recite_keeps_the_first_report_when_the_retry_is_no_better(tmp_path):
    """A corrective turn must not be able to lose ground.

    A validator that panics and drops every claim would otherwise look like an
    improvement (nothing left to be unowned), so the retry is kept only when it
    raises the number of CITABLE claims.
    """
    cfg = make_cfg(tmp_path)
    agents = build(cfg)
    captured: list[str] = []

    invented = {'claim': 'best_n is 7; gate: exhaustive search',
                'source': 'outputs/audit-01/prerequisite_audit.json', 'stance': 'contradicts'}

    record = run_round(cfg, agents,
                       validator=_provenance_validator([invented], [], captured),
                       code_generator=_codegen(),
                       # one verdict per claim, as the schema demands; it is dropped
                       # anyway because the source it attests is still unowned
                       reviewer=scripted((None, {'verdicts': [
                           {'index': 0, 'verdict': 'accepted', 'verification': 'artifact_attested',
                            'source_ref': 'outputs/audit-01/prerequisite_audit.json',
                            'checked': 'read prerequisite_audit.json'}]})))

    thread = record['threads'][0]
    assert thread['error'] is None, thread['error']
    assert len(captured) == 1, 'the validator should still have been corrected once'
    # first report kept: the retry surrendered its only claim instead of re-citing it
    assert [c['source'] for c in thread['raw_report']['claims']] == [invented['source']]
    assert thread['accepted'] == []
