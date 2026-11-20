"""Coordinate rate-limit recovery across async agents and threaded Jnana calls.

The SDK owns bounded retries. HTTP hooks share one gate per endpoint, so those retries and new
requests do not all hit a throttled service at once. Only OpenAI-compatible endpoints (a ModelSpec
with a base_url) go through a gate; provider-string models keep their SDK's own behaviour.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import threading
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from .config import RateLimits

logger = logging.getLogger('loam.rate_limits')


def retry_after(headers, *, now: float | None = None) -> float | None:
    """Accept Retry-After seconds/HTTP dates and the SDK's millisecond variant."""
    now = time.time() if now is None else now
    for key, scale in (('retry-after-ms', 0.001), ('retry-after', 1.0)):
        value = headers.get(key)
        if value is None:
            continue
        try:
            seconds = float(value) * scale
        except (ValueError, TypeError):
            if key != 'retry-after':
                continue
            try:
                seconds = parsedate_to_datetime(value).timestamp() - now
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(seconds):
            return max(0.0, seconds)
    return None


class EndpointCooldown:
    """One endpoint's request gate, shared by every client in this process that calls it.

    A throttle response (429/529, or 503 with Retry-After) sets a cooldown that every caller waits
    out; without Retry-After it backs off exponentially from `backoff` up to `max_backoff`. For
    `recovery_window` seconds after the cooldown, calls are spaced `recovery_spacing` apart so the
    pool does not stampede back. `requests_per_minute` spaces every call, throttled or not.
    """

    def __init__(self, endpoint: str, limits: RateLimits = RateLimits(), *, clock=time.monotonic, jitter=None):
        self.endpoint = endpoint
        self.limits = limits
        self.clock = clock
        self.jitter = jitter or (lambda: random.uniform(0.0, 1.0))
        self.interval = 60.0 / limits.requests_per_minute if limits.requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._not_before = self._next_slot = self._stagger_until = 0.0
        self._failures = 0

    def observe(self, response) -> None:
        status = response.status_code
        delay = retry_after(response.headers)
        # Keep transport/authentication failures separate from throttling.
        limited = status in {429, 529} or (status == 503 and delay is not None)
        with self._lock:
            if not limited:
                if status < 400:
                    self._failures = 0
                return
            self._failures += 1
            if delay is None:
                delay = min(self.limits.max_backoff, self.limits.backoff * 2 ** min(self._failures - 1, 30))
            delay += self.jitter()
            now = self.clock()
            self._not_before = max(self._not_before, now + delay)
            self._stagger_until = max(self._stagger_until, self._not_before + self.limits.recovery_window)
        logger.warning('Rate limited by %s (HTTP %s); shared cooldown %.1fs, then %.1fs between calls',
                       self.endpoint, status, delay, self.limits.recovery_spacing)

    def delay(self) -> float:
        """Reserve a request slot only when it can run, rechecking new cooldowns."""
        with self._lock:
            now = self.clock()
            wait = max(self._not_before, self._next_slot) - now
            if wait > 0:
                return wait
            gap = max(self.interval, self.limits.recovery_spacing if now < self._stagger_until else 0.0)
            self._next_slot = now + gap if gap else 0.0
            return 0.0

    def wait(self, request=None) -> None:
        while delay := self.delay():
            time.sleep(delay)

    async def wait_async(self, request=None) -> None:
        while delay := self.delay():
            await asyncio.sleep(delay)

    async def observe_async(self, response) -> None:
        self.observe(response)


_registry: dict[tuple, EndpointCooldown] = {}
_registry_lock = threading.Lock()


def endpoint_cooldown(base_url: str, limits: RateLimits = RateLimits()) -> EndpointCooldown:
    """The process-wide gate for this endpoint and these limits."""
    parsed = urlsplit(base_url)
    # Never include credentials or query strings in logs or registry keys.
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    endpoint = f'{parsed.scheme}://{parsed.hostname}:{port}{parsed.path.rstrip("/")}'
    with _registry_lock:
        if (endpoint, limits) not in _registry:
            _registry[endpoint, limits] = EndpointCooldown(endpoint, limits)
        return _registry[endpoint, limits]


def sync_http_client(base_url: str, limits: RateLimits = RateLimits()):
    from openai import DefaultHttpxClient
    gate = endpoint_cooldown(base_url, limits)
    return DefaultHttpxClient(event_hooks={'request': [gate.wait], 'response': [gate.observe]})


def async_http_client(base_url: str, limits: RateLimits = RateLimits()):
    from openai import DefaultAsyncHttpxClient
    gate = endpoint_cooldown(base_url, limits)
    return DefaultAsyncHttpxClient(event_hooks={'request': [gate.wait_async], 'response': [gate.observe_async]})


def jnana_coscientist(base_class, *, retries: int, timeout: float, limits: RateLimits = RateLimits()):
    """Adapt only this process's Jnana clients; leave the shared install intact."""
    class RateLimitedCoScientist(base_class):
        def _get_llm_for_agent(self, agent_type, agent_id):
            llm = super()._get_llm_for_agent(agent_type, agent_id)
            config = self.llm_configs.get_config_for_agent(agent_type, agent_id)
            if (config.provider in {'openai', 'openai_compatible', 'custom'}
                    and config.base_url and not getattr(llm, '_loam_cooldown', False)):
                old_client = llm.client
                llm.client = old_client.with_options(
                    http_client=sync_http_client(config.base_url, limits), max_retries=retries, timeout=timeout,
                )
                old_client.close()
                llm._loam_cooldown = True
            return llm
    return RateLimitedCoScientist
