"""Cross-campaign lesson memory (issue #34): config, round behaviour, failure policy. Offline: no Hindsight."""

import pytest

from loam.config import Config
from test_round import make_cfg

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
