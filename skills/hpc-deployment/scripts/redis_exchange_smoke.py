"""Exercise real Academy request/response messaging across separate processes."""
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
import json
import os
import socket

from academy.agent import Agent, action
from academy.manager import Manager
import redis

from exchange_config import create_exchange_factory


class EchoAgent(Agent):
    @action
    async def echo(self, nonce: str) -> dict:
        return {'nonce': nonce, 'hostname': socket.gethostname(), 'pid': os.getpid()}


async def main():
    factory = create_exchange_factory()
    host, port, _ = factory.redis_info
    unauthorized = redis.Redis(host=host, port=port, socket_connect_timeout=2)
    try:
        unauthorized.ping()
    except redis.exceptions.AuthenticationError:
        pass
    else:
        raise RuntimeError('Redis accepted an unauthenticated connection')
    finally:
        unauthorized.close()
    with ProcessPoolExecutor(max_workers=1) as executor:
        async with await Manager.from_exchange_factory(
            factory=factory, executors={'process': executor}, default_executor='process'
        ) as manager:
            handle = await manager.launch(EchoAgent())
            # Redis 8 defaults to a five-second socket timeout. Ensure the
            # Academy listener survives a longer idle interval before sending.
            await asyncio.sleep(8)
            result = await asyncio.wait_for(handle.echo('redis-exchange-smoke'), timeout=30)
            assert result['nonce'] == 'redis-exchange-smoke'
            assert result['pid'] != os.getpid()
            print(json.dumps({'status': 'passed', 'transport': 'RedisExchangeFactory',
                              'authentication_required': True, 'manager_pid': os.getpid(),
                              'worker': result}))


if __name__ == '__main__':
    asyncio.run(main())
