from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from loam.rate_limits import EndpointCooldown, endpoint_cooldown, retry_after, jnana_coscientist


def response(status, headers=None):
    return SimpleNamespace(status_code=status, headers=headers or {})


def test_retry_after_seconds_milliseconds_and_http_dates():
    assert retry_after({'retry-after': '42'}) == 42
    assert retry_after({'retry-after-ms': '1250'}) == 1.25
    date = format_datetime(datetime.fromtimestamp(110, timezone.utc), usegmt=True)
    assert retry_after({'retry-after': date}, now=100) == 10
    assert retry_after({'retry-after': 'nonsense'}) is None
    assert retry_after({'retry-after': 'nan'}) is None
    assert retry_after({'retry-after': '-1'}) == 0


def test_rate_limited_call_staggers_all_clients_and_rechecks_new_limits():
    now = [100.0]
    gate = EndpointCooldown('tara', clock=lambda: now[0], jitter=lambda: 0)
    assert gate.delay() == 0
    gate.observe(response(429, {'retry-after': '30'}))
    assert gate.delay() == 30
    now[0] = 130
    assert gate.delay() == 0
    assert gate.delay() == 2  # Another caller cannot retry simultaneously.
    gate.observe(response(429, {'retry-after': '50'}))
    assert gate.delay() == 50  # A new server cooldown supersedes a reserved slot.
    now[0] = 180
    assert gate.delay() == 0
    assert gate.delay() == 2


def test_missing_retry_after_uses_exponential_backoff_and_auth_errors_do_not_throttle():
    gate = EndpointCooldown('tara', clock=lambda: 100, jitter=lambda: 0)
    gate.observe(response(401))
    gate.observe(response(403))
    gate.observe(response(503))
    assert gate.delay() == 0
    gate.observe(response(429))
    assert gate.delay() == 5
    gate.observe(response(429))
    assert gate.delay() == 10


def test_service_overload_retry_after_is_respected():
    gate = EndpointCooldown('tara', clock=lambda: 0, jitter=lambda: 0)
    gate.observe(response(503, {'retry-after': '90'}))
    assert gate.delay() == 90


def test_endpoint_sharing_and_isolation():
    assert endpoint_cooldown('https://tara.test/v1/') is endpoint_cooldown('https://tara.test/v1')
    assert endpoint_cooldown('https://tara.test/v1') is not endpoint_cooldown('https://argo.test/v1')


def test_jnana_clients_get_shared_hooks_and_campaign_retry_budget(monkeypatch):
    from loam import rate_limits
    calls = []
    class Client:
        def with_options(self, **kwargs):
            calls.append(kwargs)
            return self
        def close(self):
            calls.append('closed')
    llm = SimpleNamespace(client=Client())
    config = SimpleNamespace(provider='openai', base_url='https://tara.test/v1')
    class Scientist:
        llm_configs = SimpleNamespace(get_config_for_agent=lambda *args: config)
        def _get_llm_for_agent(self, *args):
            return llm
    monkeypatch.setattr(rate_limits, 'sync_http_client', lambda url: 'hooked-client')
    scientist = jnana_coscientist(Scientist, retries=10, timeout=600)()
    assert scientist._get_llm_for_agent('ranking', 'ranking-0') is llm
    scientist._get_llm_for_agent('ranking', 'ranking-1')
    assert calls == [{'http_client': 'hooked-client', 'max_retries': 10, 'timeout': 600}, 'closed']


@pytest.mark.parametrize('asynchronous', [False, True])
def test_sdk_retries_with_cooldown_hooks_and_keeps_request_body(asynchronous):
    import asyncio
    import httpx2
    import openai
    calls = []
    gate = EndpointCooldown('mock-service', jitter=lambda: 0, spacing=0.01)
    def handler(request):
        calls.append(request.content)
        if len(calls) == 1:
            return httpx2.Response(429, headers={'Retry-After': '0.01'},
                                   json={'error': {'message': 'rate limited', 'type': 'rate_limit_error'}})
        return httpx2.Response(200, json={'id': 'test', 'object': 'chat.completion',
            'created': 0, 'model': 'test', 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': 'OK'}}]})
    body = {'model': 'test', 'messages': [{'role': 'user', 'content': 'test'}]}
    if asynchronous:
        async def exercise():
            http = openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler),
                trust_env=False, event_hooks={'request': [gate.wait_async], 'response': [gate.observe_async]})
            async with openai.AsyncOpenAI(api_key='test', base_url='https://mock.test/v1',
                                         max_retries=1, http_client=http) as client:
                return await client.chat.completions.create(**body)
        result = asyncio.run(exercise())
    else:
        http = openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler),
            trust_env=False, event_hooks={'request': [gate.wait], 'response': [gate.observe]})
        with openai.OpenAI(api_key='test', base_url='https://mock.test/v1',
                           max_retries=1, http_client=http) as client:
            result = client.chat.completions.create(**body)
    assert result.choices[0].message.content == 'OK'
    assert len(calls) == 2 and calls[0] == calls[1]
