"""Hindsight memory backend (issue #35): pure helpers only. No server, no network."""

from types import SimpleNamespace

import pytest

from loam.config import MemoryConfig
from loam.memory import Lesson, MemoryUnavailable

pytest.importorskip('hindsight_client')   # the `memory` extra

from loam import hindsight  # noqa: E402

ARGO = 'http://127.0.0.1:61612/v1'


def memory(**overrides) -> MemoryConfig:
    return MemoryConfig.model_validate({
        'campaign': 'alkbh1-pmd',
        'llm': {'model': 'argo:gpt-5.6-luna', 'base_url': ARGO},
        'embeddings': {'model': 'argo:text-embedding-3-small', 'base_url': 'http://embed.example/v1'},
        **overrides})


# ---- server environment ---------------------------------------------------------------------------

def test_server_env_points_hindsight_at_the_configured_endpoints(monkeypatch):
    monkeypatch.setenv('ALCF_TOKEN', 'tok-123')
    cfg = memory(instance='camp-a',
                 llm={'model': 'gpt-x', 'base_url': ARGO, 'api_key_env': 'ALCF_TOKEN'})

    env = hindsight.server_env(cfg, port=8123)

    assert env['HINDSIGHT_API_DATABASE_URL'] == 'pg0://camp-a'
    assert env['HINDSIGHT_API_PORT'] == '8123'
    assert env['HINDSIGHT_API_LLM_PROVIDER'] == 'openai'
    assert env['HINDSIGHT_API_LLM_BASE_URL'] == ARGO
    assert env['HINDSIGHT_API_LLM_MODEL'] == 'gpt-x'
    assert env['HINDSIGHT_API_LLM_API_KEY'] == 'tok-123'
    assert env['HINDSIGHT_API_LLM_STRICT_SCHEMA_RETAIN'] == 'true'
    assert env['HINDSIGHT_API_EMBEDDINGS_PROVIDER'] == 'openai'
    assert env['HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL'] == 'http://embed.example/v1'
    assert env['HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL'] == 'argo:text-embedding-3-small'
    assert env['HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY'] == 'none'   # no api_key_env: proxy ignores it
    assert env['HINDSIGHT_API_RERANKER_PROVIDER'] == 'rrf'
    assert env['HINDSIGHT_API_WORKER_ID']                             # fixed, so queued work resumes
    assert env['HINDSIGHT_API_WORKER_ID'] == hindsight.server_env(cfg, port=9999)['HINDSIGHT_API_WORKER_ID']


@pytest.mark.parametrize('which', ['llm', 'embeddings'])
def test_a_named_but_unset_api_key_env_is_a_misconfiguration(monkeypatch, which):
    monkeypatch.delenv('LOAM_TEST_UNSET_KEY', raising=False)
    cfg = memory(**{which: {'model': 'm', 'base_url': ARGO, 'api_key_env': 'LOAM_TEST_UNSET_KEY'}})

    with pytest.raises(MemoryUnavailable, match='LOAM_TEST_UNSET_KEY'):
        hindsight.server_env(cfg, port=8123)


# ---- embedding-model guard ------------------------------------------------------------------------

def test_guard_records_the_model_on_first_use_and_accepts_it_again(tmp_path):
    cfg = memory(instance='camp-a')

    hindsight.check_embedding_model(cfg, tmp_path)
    hindsight.check_embedding_model(cfg, tmp_path)   # same model: still fine


def test_guard_refuses_a_different_embedding_model_for_the_same_instance(tmp_path):
    hindsight.check_embedding_model(memory(instance='camp-a'), tmp_path)
    other = memory(instance='camp-a', embeddings={'model': 'argo:text-embedding-ada-002', 'base_url': ARGO})

    with pytest.raises(MemoryUnavailable, match='new memory.instance'):
        hindsight.check_embedding_model(other, tmp_path)


def test_guard_is_per_instance(tmp_path):
    hindsight.check_embedding_model(memory(instance='camp-a'), tmp_path)
    other = memory(instance='camp-b', embeddings={'model': 'argo:text-embedding-ada-002', 'base_url': ARGO})

    hindsight.check_embedding_model(other, tmp_path)


# ---- retain batches -------------------------------------------------------------------------------

def test_lessons_are_batched_per_round_and_hypothesis_one_document_per_kind():
    lessons = [
        Lesson('procedural', 'Chai-1 ran out of memory above 1,200 residues.', 2, 'h7', run_id='r2-1'),
        Lesson('review', 'A single-seed ipTM does not support a binding claim.', 2, 'h3'),
        Lesson('review', 'Contacts from one model are not an interface.', 2, 'h7'),
        Lesson('procedural', 'The MD skill needs a protonated input structure.', 2, 'h7', run_id='r2-3'),
    ]

    batches = hindsight.retain_batches(lessons, campaign='alkbh1-pmd')

    assert len(batches) == 2                       # one async retain per (round, hypothesis)
    h7 = {item['document_id']: item for item in next(items for items in batches if len(items) == 2)}
    # Hindsight's async path refuses duplicate document_ids in one batch: one item per kind, and a
    # re-run round replaces each of them.
    assert set(h7) == {'alkbh1-pmd:r2:h7:procedural', 'alkbh1-pmd:r2:h7:review'}
    procedural = h7['alkbh1-pmd:r2:h7:procedural']
    assert 'Chai-1 ran out of memory' in procedural['content']
    assert 'protonated input structure' in procedural['content']
    assert procedural['tags'] == ['campaign:alkbh1-pmd', 'kind:procedural']
    assert procedural['metadata'] == {'round': '2', 'hypothesis_id': 'h7', 'run_id': 'r2-1,r2-3',
                                      'kind': 'procedural'}
    review = h7['alkbh1-pmd:r2:h7:review']
    assert review['tags'] == ['campaign:alkbh1-pmd', 'kind:review']
    assert review['metadata'] == {'round': '2', 'hypothesis_id': 'h7', 'run_id': '', 'kind': 'review'}
    assert all(item['update_mode'] == 'replace' for item in h7.values())


def test_no_lessons_means_no_batches():
    assert hindsight.retain_batches([], campaign='alkbh1-pmd') == []


# ---- recall formatting ----------------------------------------------------------------------------

def fact(text, tags, metadata=None):
    return SimpleNamespace(text=text, tags=tags, metadata=metadata, type='world', document_id='d')


def test_recalled_lessons_are_labelled_with_their_campaign_from_the_tag():
    results = [
        fact('Chai-1 runs out of memory above 1,200 residues on a 40 GB GPU.',
             ['campaign:alkbh1-pmd', 'kind:procedural'], metadata=None),   # split facts lose metadata
        fact('Single-seed ipTM does not support a binding claim.',
             ['kind:review', 'campaign:kras-g12d'], metadata={'kind': 'review'}),
    ]

    text = hindsight.format_recall(results)

    first, second = text.splitlines()
    assert 'alkbh1-pmd' in first and 'Chai-1 runs out of memory' in first
    assert 'kras-g12d' in second and 'Single-seed ipTM' in second


def test_a_fact_without_a_campaign_tag_is_still_recalled():
    text = hindsight.format_recall([fact('Use five seeds.', tags=None)])

    assert 'Use five seeds.' in text


def test_nothing_recalled_is_empty_text():
    assert hindsight.format_recall([]) == ''


def test_recall_filters_by_kind_tags_only_when_asked():
    assert hindsight.recall_tags(None) is None
    assert hindsight.recall_tags(['procedural']) == ['kind:procedural']
