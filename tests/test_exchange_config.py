"""Exchange selection and authentication survive shipping a factory to workers."""
import importlib.util
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

PATH = Path(__file__).resolve().parents[1] / 'skills/hpc-deployment/scripts/exchange_config.py'
SPEC = importlib.util.spec_from_file_location('exchange_config', PATH)
config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(config)


def test_http_default_preserves_globus_exchange():
    with patch.dict('os.environ', {'ACADEMY_EXCHANGE_URL': 'https://example.test/v1'}, clear=True), \
         patch.object(config, 'HttpExchangeFactory') as factory:
        config.create_exchange_factory()
        factory.assert_called_once_with('https://example.test/v1', auth_method='globus',
                                        retries=3, request_timeout_s=20)


def test_redis_factory_preserves_credentials_after_serialization(tmp_path):
    password = tmp_path / 'password'
    password.write_text('test-password\n')
    with patch.dict('os.environ', {'ACADEMY_EXCHANGE_TYPE': 'redis',
                                  'ACADEMY_REDIS_HOST': '10.128.1.8',
                                  'ACADEMY_REDIS_PORT': '50857',
                                  'ACADEMY_REDIS_DB': '3',
                                  'ACADEMY_REDIS_PASSWORD_FILE': str(password)}, clear=True):
        factory = pickle.loads(pickle.dumps(config.create_exchange_factory()))
    assert factory.redis_info.hostname == '10.128.1.8'
    assert factory.redis_info.port == 50857
    assert factory.redis_info.kwargs['password'] == 'test-password'
    assert factory.redis_info.kwargs['db'] == 3
    assert factory.redis_info.kwargs['socket_timeout'] is None


@pytest.mark.parametrize('values,exception', [
    ({'ACADEMY_EXCHANGE_TYPE': 'invalid'}, ValueError),
    ({'ACADEMY_EXCHANGE_TYPE': 'redis'}, RuntimeError),
    ({'ACADEMY_EXCHANGE_TYPE': 'redis', 'ACADEMY_REDIS_HOST': 'host'}, RuntimeError),
    ({'ACADEMY_EXCHANGE_TYPE': 'redis', 'ACADEMY_REDIS_HOST': 'host',
      'ACADEMY_REDIS_PORT': '0'}, ValueError),
])
def test_invalid_exchange_configuration_fails_before_launch(values, exception):
    with patch.dict('os.environ', values, clear=True), pytest.raises(exception):
        config.create_exchange_factory()
