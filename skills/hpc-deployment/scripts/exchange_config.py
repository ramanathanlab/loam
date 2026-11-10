"""Choose the Academy exchange shared by the MCP manager and its remote agents."""
from __future__ import annotations

import os
from pathlib import Path

from academy.exchange import HttpExchangeFactory, RedisExchangeFactory


def create_exchange_factory():
    kind = os.environ.get('ACADEMY_EXCHANGE_TYPE', 'http').strip().lower()
    if kind == 'http':
        url = os.environ.get('ACADEMY_EXCHANGE_URL', '').strip()
        if not url:
            raise RuntimeError('ACADEMY_EXCHANGE_URL is required for the HTTP exchange')
        timeout = float(os.environ.get('HPC_EXCHANGE_REQUEST_TIMEOUT', '20'))
        if timeout <= 0:
            raise ValueError('HPC_EXCHANGE_REQUEST_TIMEOUT must be positive')
        return HttpExchangeFactory(url, auth_method='globus', retries=3, request_timeout_s=timeout)
    if kind != 'redis':
        raise ValueError('ACADEMY_EXCHANGE_TYPE must be http or redis')
    host = os.environ.get('ACADEMY_REDIS_HOST', '').strip()
    if not host:
        raise RuntimeError('ACADEMY_REDIS_HOST is required for the Redis exchange')
    port = int(os.environ.get('ACADEMY_REDIS_PORT', '6379'))
    db = int(os.environ.get('ACADEMY_REDIS_DB', '0'))
    if not 1 <= port <= 65535 or db < 0:
        raise ValueError('Invalid Redis port or database number')
    password_file = os.environ.get('ACADEMY_REDIS_PASSWORD_FILE', '').strip()
    password = (Path(password_file).read_text().strip() if password_file
                else os.environ.get('ACADEMY_REDIS_PASSWORD', ''))
    if not password:
        raise RuntimeError('Set ACADEMY_REDIS_PASSWORD_FILE or ACADEMY_REDIS_PASSWORD')
    # The serialized factory lets each worker connect to the same internal Redis.
    return RedisExchangeFactory(hostname=host, port=port, db=db, password=password,
                                socket_connect_timeout=10, socket_timeout=None)
