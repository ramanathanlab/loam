"""Cross-campaign lesson memory (issue #34): config, round behaviour, failure policy. Offline: no Hindsight."""

import asyncio
import contextlib
import json

import pytest

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from loam.agents import build
from loam.config import Config
from loam.memory import Lesson, MemoryUnavailable
from loam.orchestrator import Campaign
from test_round import H, StubTournament, make_cfg, scripted

ARGO = 'http://127.0.0.1:61612/v1'


def memory_values(**overrides) -> dict:
    return {'campaign': 'alkbh1-pmd',
            'llm': {'model': 'argo:gpt-5.6-luna', 'base_url': ARGO},
            'embeddings': {'model': 'argo:text-embedding-3-small', 'base_url': ARGO},
            **overrides}


def with_memory(tmp_path, **overrides) -> Config:
    values = make_cfg(tmp_path).model_dump()
    values['memory'] = memory_values(**overrides)
    return Config.model_validate(values)


# ---- config ---------------------------------------------------------------------------------------

def test_memory_is_off_unless_configured(tmp_path):
    assert make_cfg(tmp_path).memory is None


def test_memory_defaults(tmp_path):
    memory = with_memory(tmp_path).memory
    assert memory.bank == 'loam'
    assert memory.campaign == 'alkbh1-pmd'
    assert memory.url is None
    assert memory.instance == 'loam-memory'   # loam-owned embedded server's database
    assert memory.recall_tokens == 1000
    assert memory.drain_timeout == 600
    assert memory.llm.base_url == ARGO and memory.embeddings.model == 'argo:text-embedding-3-small'


@pytest.mark.parametrize('missing', ['campaign', 'llm', 'embeddings'])
def test_memory_requires_campaign_and_both_models(tmp_path, missing):
    values = make_cfg(tmp_path).model_dump()
    values['memory'] = memory_values()
    values['memory'].pop(missing)
    with pytest.raises(ValueError, match=missing):
        Config.model_validate(values)


@pytest.mark.parametrize('overrides, message', [
    ({'campaign': ''}, 'at least 1 character'),
    ({'bank': ''}, 'at least 1 character'),
    ({'campagin': 'typo'}, 'Extra inputs'),
    ({'llm': {'model': 'anthropic:claude-opus-5-5'}}, 'memory.llm needs a base_url'),
    ({'embeddings': {'model': 'text-embedding-3-small'}}, 'memory.embeddings needs a base_url'),
    ({'url': 'file:///tmp/hindsight'}, 'memory.url must be an HTTP'),
    ({'url': 'http://127.0.0.1:8888', 'instance': 'other'}, 'instance only applies'),
    ({'instance': 'Bad Name!'}, 'instance'),
    ({'recall_tokens': 0}, 'greater than 0'),
    ({'drain_timeout': 0}, 'greater than 0'),
    ({'timeout': 0}, 'greater than 0'),
])
def test_memory_rejects_invalid_blocks(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        with_memory(tmp_path, **overrides)


def test_memory_with_url_has_no_instance(tmp_path):
    memory = with_memory(tmp_path, url='http://127.0.0.1:8888').memory
    assert memory.url == 'http://127.0.0.1:8888' and memory.instance is None


def test_memory_block_loads_from_yaml(tmp_path):
    import yaml

    from loam.config import load

    values = make_cfg(tmp_path).model_dump(mode='json')
    values['memory'] = memory_values(bank='lessons', recall_tokens=500)
    path = tmp_path / 'campaign.yaml'
    path.write_text(yaml.safe_dump(values))
    memory = load(path).memory
    assert (memory.bank, memory.recall_tokens, memory.instance) == ('lessons', 500, 'loam-memory')


# ---- interface --------------------------------------------------------------------------------------

def test_no_memory_is_a_silent_no_op(tmp_path):
    import asyncio

    from loam.memory import Lesson, NoMemory, from_config

    memory = from_config(make_cfg(tmp_path))
    assert isinstance(memory, NoMemory)

    async def exercise():
        await memory.start()
        assert await memory.recall('anything', kinds=['procedural']) == ''
        await memory.retain([Lesson('review', 'text', 1, 'h1')])
        await memory.close()

    asyncio.run(exercise())


# ---- campaign wiring (#36): through Campaign, with a fake memory ------------------------------------

class FakeMemory:
    """Records every call; recall answers with a marker naming the kinds asked for."""

    def __init__(self):
        self.calls: list[str] = []
        self.recalls: list[tuple[str, list | None]] = []
        self.retained: list[Lesson] = []

    async def start(self):
        self.calls.append('start')

    async def recall(self, query, *, kinds=None):
        self.calls.append('recall')
        self.recalls.append((query, None if kinds is None else list(kinds)))
        return 'PROCEDURAL-LESSON: pin the solver version' if kinds == ['procedural'] else 'ANY-LESSON: check pLDDT'

    async def retain(self, lessons):
        self.calls.append('retain')
        self.retained.extend(lessons)

    async def close(self):
        self.calls.append('close')


def capturing(model: FunctionModel, prompts: list[str]) -> FunctionModel:
    """Wrap a scripted model so the user prompt of every fresh agent run is collected."""
    def respond(messages, info: AgentInfo):
        if len(messages) == 1:
            prompts.extend(str(p.content) for p in messages[0].parts if isinstance(p, UserPromptPart))
        return model.function(messages, info)
    return FunctionModel(respond)


def code_generator_by_instruction():
    """Instruction says BLOCK -> returns without running; CRASH -> raises; otherwise runs a command."""
    def respond(messages, info: AgentInfo):
        prompt = ''.join(str(p.content) for p in messages[0].parts if isinstance(p, UserPromptPart))
        if 'CRASH' in prompt:
            raise RuntimeError('solver segfaulted')
        if 'BLOCK' not in prompt and not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(parts=[ToolCallPart('run_shell', {'command': 'true'})])
        blocked = 'BLOCK' in prompt
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            'status': 'blocked' if blocked else 'executed', 'code': '',
            'result': 'chai-1 weights are absent' if blocked else 'ok'})])
    return FunctionModel(respond)


def campaign_round(cfg, memory=None, prompts=None, **models) -> dict:
    """One round through Campaign.run(1); every agent's fresh prompts are collected per role."""
    prompts = prompts if prompts is not None else {}
    agents = build(cfg)
    models.setdefault('reasoner', scripted((None, {'report': '# synthesis', 'tournament_brief': 'next'})))
    models.setdefault('code_generator', code_generator_by_instruction())
    with contextlib.ExitStack() as stack:
        for role, model in models.items():
            stack.enter_context(getattr(agents, role).override(
                model=capturing(model, prompts.setdefault(role, []))))
        kwargs = {} if memory is None else {'memory': memory}
        asyncio.run(Campaign(cfg, agents=agents, tournament=StubTournament(), **kwargs).run(1))
    return json.loads((cfg.state_dir / 'rounds' / 'round-001.json').read_text())


CLAIMS = [{'claim': 'pLDDT above 80; gate 70', 'source': 'https://example.test/a', 'stance': 'supports'},
          {'claim': 'ipTM above 0.8; gate 0.6', 'source': 'https://example.test/b', 'stance': 'supports'},
          {'claim': 'binder is monomeric in SEC', 'source': 'https://example.test/c', 'stance': 'supports'}]


def full_round_models() -> dict:
    """A blocked run, a crashed run, an executed run; a blocked report; accepted/rejected/unverifiable."""
    validator = scripted(
        ('code_generator', {'instruction': 'BLOCK: fold the complex with chai-1'}),
        ('code_generator', {'instruction': 'CRASH: minimise with the solver'}),
        ('code_generator', {'instruction': 'score the interface'}),
        (None, {'outcome': 'blocked', 'evidence_score': 0, 'ledger': '', 'claims': CLAIMS,
                'next_step': 'needs chai-1 weights installed on the node'}))
    reviewer = scripted((None, {'verdicts': [
        {'index': 0, 'verdict': 'accepted', 'verification': 'source_confirmed',
         'source_ref': 'https://example.test/a#plddt', 'checked': 'read the record: pLDDT 85'},
        {'index': 1, 'verdict': 'rejected', 'checked': 'record shows ipTM 0.41'},
        {'index': 2, 'verdict': 'unverifiable', 'checked': 'no SEC trace in the outputs'}]}))
    return {'validator': validator, 'reviewer': reviewer}


def test_recalled_lessons_reach_the_validator_prompt(tmp_path):
    memory, prompts = FakeMemory(), {}
    campaign_round(make_cfg(tmp_path), memory, prompts, **full_round_models())
    assert (H.summary, None) in memory.recalls
    assert any('ANY-LESSON: check pLDDT' in p for p in prompts['validator'])


def test_code_generator_gets_procedural_lessons_and_reviewer_and_reasoner_get_none(tmp_path):
    memory, prompts = FakeMemory(), {}
    campaign_round(make_cfg(tmp_path), memory, prompts, **full_round_models())
    assert ('score the interface', ['procedural']) in memory.recalls
    [scored] = [p for p in prompts['code_generator'] if 'score the interface' in p]
    assert 'PROCEDURAL-LESSON: pin the solver version' in scored and 'ANY-LESSON' not in scored
    for role in ('reviewer', 'reasoner'):
        assert prompts[role] and not any('LESSON' in p for p in prompts[role]), role


def test_round_retains_exactly_its_failure_lessons(tmp_path):
    memory = FakeMemory()
    record = campaign_round(make_cfg(tmp_path), memory, **full_round_models())
    assert [c['status'] for c in record['threads'][0]['code_results']] == ['blocked', 'crashed', 'executed']

    provenance = sorted((lesson.kind, lesson.round, lesson.hypothesis_id, lesson.run_id)
                        for lesson in memory.retained)
    assert provenance == [('procedural', 1, H.id, ''),           # the blocked validator report
                          ('procedural', 1, H.id, 'r001-01'),    # the blocked run
                          ('procedural', 1, H.id, 'r001-02'),    # the crashed run
                          ('review', 1, H.id, ''), ('review', 1, H.id, '')]
    procedural = {lesson.run_id: lesson.text for lesson in memory.retained if lesson.kind == 'procedural'}
    assert 'fold the complex with chai-1' in procedural['r001-01']
    assert 'chai-1 weights are absent' in procedural['r001-01']
    assert 'solver segfaulted' in procedural['r001-02']
    assert 'needs chai-1 weights installed on the node' in procedural['']
    reviews = [lesson.text for lesson in memory.retained if lesson.kind == 'review']
    assert any('ipTM above 0.8; gate 0.6' in t and 'record shows ipTM 0.41' in t for t in reviews)
    assert any('binder is monomeric in SEC' in t and 'no SEC trace in the outputs' in t for t in reviews)
    every = ' '.join(lesson.text for lesson in memory.retained)
    assert 'pLDDT above 80' not in every and 'score the interface' not in every   # accepted / executed
    assert 'r001-0' not in every                                                  # ids ride in fields
    assert memory.calls[0] == 'start' and memory.calls[-1] == 'close' and 'retain' in memory.calls


def test_a_crashed_validator_is_not_a_lesson(tmp_path):
    def crash(messages, info):
        raise RuntimeError('validator model endpoint down')

    memory = FakeMemory()
    record = campaign_round(make_cfg(tmp_path), memory, validator=FunctionModel(crash))
    assert record['threads'][0]['error']
    assert memory.retained == []


class BrokenMemory(FakeMemory):
    """Starts, then raises on every other call."""

    async def recall(self, query, *, kinds=None):
        raise ConnectionError('memory server went away')

    async def retain(self, lessons):
        raise ConnectionError('memory server went away')

    async def close(self):
        raise ConnectionError('memory server went away')


def test_a_memory_that_raises_cannot_change_the_round(tmp_path):
    from loam.memory import NoMemory

    records = {}
    for name, memory in (('none', NoMemory()), ('broken', BrokenMemory())):
        record = campaign_round(make_cfg(tmp_path / name), memory, **full_round_models())
        records[name] = json.loads(json.dumps(record).replace(str(tmp_path / name), '<tmp>'))
    assert records['broken'] == records['none']
    assert records['none']['threads'][0]['error'] is None


def test_unavailable_memory_stops_the_campaign_before_round_one(tmp_path):
    class Unavailable(FakeMemory):
        async def start(self):
            raise MemoryUnavailable('server never became healthy')

    cfg = make_cfg(tmp_path)
    with pytest.raises(MemoryUnavailable):
        campaign_round(cfg, Unavailable(), **full_round_models())
    assert not list((cfg.state_dir / 'rounds').glob('round-*.json'))


def test_without_a_memory_block_prompts_are_unchanged(tmp_path):
    prompts = {}
    campaign_round(make_cfg(tmp_path), None, prompts, **full_round_models())
    hypothesis = f'{H.summary}\n\n{H.content}'
    assert prompts['validator'][0] == f'Validate this hypothesis (Jnana id {H.id}, Elo 1250):\n\n{hypothesis}'
    assert prompts['code_generator'][0] == (f'HYPOTHESIS UNDER TEST:\n{hypothesis}\n\n'
                                            'RUN r001-01. INSTRUCTION:\nBLOCK: fold the complex with chai-1')


def test_lesson_text_is_capped():
    from loam.memory import lessons

    record = {'round': 2, 'threads': [{
        'hypothesis': {'id': 'h1'}, 'report': None, 'review': None, 'error': None,
        'code_results': [{'run_id': 'r002-01', 'status': 'crashed',
                          'instruction': 'x' * 50_000, 'error': 'y' * 50_000}]}]}
    [lesson] = lessons(record)
    assert lesson.run_id == 'r002-01' and len(lesson.text) <= 2000
