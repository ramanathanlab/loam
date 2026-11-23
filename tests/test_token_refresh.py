"""ALCF inference tokens are refreshed only for the endpoints configured to need it."""
import asyncio
import os

import pytest
from pydantic import ValidationError

from loam.config import Config
from loam.token_refresh import refresh_tokens

ROLES = ['validator', 'code_generator', 'reviewer', 'reasoner']
ALCF = {'model': 'openai/gpt-oss-120b', 'base_url': 'https://inference.example/v1',
        'api_key_env': 'ALCF_INFERENCE_TOKEN', 'token_refresh': 'alcf'}


def campaign(tmp_path, **models):
    test = {'model': 'test'}
    return Config(goal='goal', state_dir=tmp_path / 'state', models={role: models.get(role, test) for role in ROLES},
                  tournament={'jnana_root': tmp_path, 'generation': models.get('generation', test),
                              'ranking': test})


def test_an_endpoint_configured_for_alcf_gets_a_fresh_token(tmp_path, alcf_tokens):
    asyncio.run(refresh_tokens(campaign(tmp_path, reasoner=ALCF, generation=ALCF)))
    assert os.environ['ALCF_INFERENCE_TOKEN'] == 'refreshed-token'
    assert alcf_tokens() == ['get-token inference']  # one call per variable, however many roles share it


def test_no_refresh_runs_when_no_endpoint_asks_for_one(tmp_path, alcf_tokens):
    asyncio.run(refresh_tokens(campaign(tmp_path, reasoner={**ALCF, 'token_refresh': None})))
    assert os.environ['ALCF_INFERENCE_TOKEN'] == 'stale-token'
    assert alcf_tokens() == []


def test_a_failed_refresh_stops_the_campaign_with_the_login_command(tmp_path, alcf_tokens, monkeypatch):
    monkeypatch.setenv('FAIL', '1')
    with pytest.raises(RuntimeError, match='alcf-tokens login inference'):
        asyncio.run(refresh_tokens(campaign(tmp_path, reasoner=ALCF)))
    assert os.environ['ALCF_INFERENCE_TOKEN'] == 'stale-token'


def test_without_alcf_tokens_installed_the_error_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.delenv('ALCF_TOKENS_BIN', raising=False)
    monkeypatch.setattr('shutil.which', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match=r'loam\[alcf-auth\]'):
        asyncio.run(refresh_tokens(campaign(tmp_path, reasoner=ALCF)))


def test_token_refresh_needs_the_variable_it_refreshes(tmp_path):
    with pytest.raises(ValidationError, match='api_key_env'):
        campaign(tmp_path, reasoner={**ALCF, 'api_key_env': None})
