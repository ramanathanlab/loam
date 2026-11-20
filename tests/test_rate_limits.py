from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from loam.config import RateLimits
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
    monkeypatch.setattr(rate_limits, 'sync_http_client', lambda url, limits=None: 'hooked-client')
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
    gate = EndpointCooldown('mock-service', RateLimits(recovery_spacing=0.01), jitter=lambda: 0)
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


def campaign_config(tmp_path, **extra):
    from loam.config import Config
    test = {'model': 'test'}
    return Config(goal='goal', state_dir=tmp_path / 'state',
                  models=dict.fromkeys(['validator', 'code_generator', 'reviewer', 'reasoner'], test),
                  tournament={'jnana_root': tmp_path, 'generation': test, 'ranking': test}, **extra)


def test_campaign_rate_limits_cap_and_recovery_shape_the_endpoint_gate(tmp_path):
    cfg = campaign_config(tmp_path, rate_limits={'requests_per_minute': 30, 'recovery_spacing': 7,
                                                 'recovery_window': 100, 'backoff': 3, 'max_backoff': 4})
    gate = endpoint_cooldown('https://capped.test/v1', cfg.rate_limits)
    assert gate is endpoint_cooldown('https://capped.test/v1/', cfg.rate_limits)
    now = [0.0]
    gate.clock, gate.jitter = (lambda: now[0]), (lambda: 0)
    assert gate.delay() == 0
    assert gate.delay() == 2  # 30 per minute: every call is spaced, throttled or not.
    now[0] = 2
    assert gate.delay() == 0
    gate.observe(response(429))
    assert gate.delay() == 3  # first backoff
    gate.observe(response(429))
    assert gate.delay() == 4  # doubling, capped at max_backoff
    now[0] = 6
    assert gate.delay() == 0
    assert gate.delay() == 7  # recovery spacing beats the steady cap inside the window
    now[0] = 106
    assert gate.delay() == 0
    assert gate.delay() == 2  # window over: back to the steady cap


def test_campaign_rate_limits_default_to_uncapped_and_reject_inverted_backoff(tmp_path):
    from pydantic import ValidationError
    gate = endpoint_cooldown('https://uncapped.test/v1', campaign_config(tmp_path).rate_limits)
    assert gate.delay() == 0 and gate.delay() == 0
    with pytest.raises(ValidationError, match='backoff'):
        campaign_config(tmp_path, rate_limits={'backoff': 200})


def test_campaign_agents_send_model_requests_through_the_endpoint_gate(tmp_path):
    import asyncio
    import openai
    from loam.agents import build
    url = 'http://127.0.0.1:1/v1'  # nothing listens: the request fails after passing the gate
    cfg = campaign_config(tmp_path, rate_limits={'requests_per_minute': 1})
    cfg.models['reasoner'] = cfg.models['reasoner'].model_copy(update={'model': 'm', 'base_url': url})
    client = build(cfg).reasoner.model.client.with_options(max_retries=0)
    with pytest.raises(openai.APIConnectionError):
        asyncio.run(client.models.retrieve('m'))
    assert endpoint_cooldown(url, cfg.rate_limits).delay() > 50  # that request took this minute's slot


def test_tournament_generation_and_ranking_clients_use_the_campaign_gate(tmp_path, monkeypatch):
    import sys
    from loam import rate_limits
    from loam.tournament import Tournament
    seen = []

    class Client:
        def with_options(self, **kwargs):
            seen.append(kwargs)
            return self
        def close(self):
            pass

    class CoScientist:
        def __init__(self, **kwargs):
            self.memory = SimpleNamespace(save=lambda: None)
            config = SimpleNamespace(provider='openai', base_url='https://tara.test/v1')
            self.llm_configs = SimpleNamespace(get_config_for_agent=lambda *args: config)
        def _get_llm_for_agent(self, agent_type, agent_id):
            return SimpleNamespace(client=Client())
        def generate_hypotheses(self, **kwargs):
            self._get_llm_for_agent('generation', 'generation-0')
        def get_all_hypotheses(self):
            return []
        def wait_for_completion(self, timeout):
            return True
        set_research_goal = start = stop = lambda self, *a, **k: None

    periodic = SimpleNamespace(CoScientist=CoScientist)
    def rank(**kwargs):  # Jnana builds its own CoScientist from the module global
        periodic.CoScientist()._get_llm_for_agent('ranking', 'ranking-0')
        return {'top_hypotheses': []}
    periodic.run_incremental_tournament = rank
    fakes = {'jnana.protognosis.core.coscientist': SimpleNamespace(CoScientist=CoScientist),
             'jnana.protognosis.core.multi_llm_config': SimpleNamespace(LLMConfig=lambda **k: k),
             'jnana.protognosis.periodic_tournament': periodic}
    for name in ['jnana', 'jnana.protognosis', 'jnana.protognosis.core', *fakes]:
        monkeypatch.setitem(sys.modules, name, fakes.get(name, SimpleNamespace()))
    monkeypatch.setattr(rate_limits, 'sync_http_client', lambda url, limits=None: ('hooked', url, limits))

    cfg = campaign_config(tmp_path, model_retries=3, timeouts={'model_request': 90},
                          rate_limits={'requests_per_minute': 12})
    Tournament.for_campaign(cfg).run('goal', initial=True)
    hooked = {'http_client': ('hooked', 'https://tara.test/v1', cfg.rate_limits), 'max_retries': 3, 'timeout': 90}
    assert seen == [hooked, hooked]
    assert periodic.CoScientist is CoScientist  # Jnana's module is restored after ranking
