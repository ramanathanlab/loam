#!/usr/bin/env python3
"""Start the private login-node Redis service used by Academy on Jupiter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import time

import redis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', required=True, help='Compute-reachable internal login-node address')
    parser.add_argument('--port', type=int, default=50857)
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'software/redis')
    args = parser.parse_args()
    root = args.root.resolve()
    state = root / 'state'
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    password_file = state / 'password'
    if not password_file.exists():
        fd = os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(secrets.token_hex(32) + '\n')
    password = password_file.read_text().strip()
    client = redis.Redis(host=args.bind, port=args.port, password=password,
                         socket_connect_timeout=2, socket_timeout=2)
    try:
        alive = client.ping()
    except redis.exceptions.ConnectionError:
        alive = False
    if not alive:
        config = state / 'redis.conf'
        config.write_text('\n'.join([
            f'bind 127.0.0.1 {args.bind}', f'port {args.port}', 'protected-mode yes',
            f'requirepass {password}', 'daemonize no', f'dir {state}',
            'save ""', 'appendonly no', 'maxmemory 256mb',
            'maxmemory-policy noeviction', f'logfile {state / "redis.log"}',
        ]) + '\n')
        config.chmod(0o600)
        with (state / 'launcher.log').open('ab') as log:
            process = subprocess.Popen([str(root / 'bin/redis-server'), str(config)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        (state / 'redis.pid').write_text(str(process.pid) + '\n')
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if client.ping():
                    break
            except redis.exceptions.ConnectionError:
                if process.poll() is not None:
                    raise RuntimeError('Redis exited; inspect software/redis/state logs')
                time.sleep(0.2)
        else:
            raise RuntimeError('Redis did not become ready; inspect its logs')
    # Passwords stay in a private file, outside the repository and command argv.
    settings = {'ACADEMY_EXCHANGE_TYPE': 'redis', 'ACADEMY_REDIS_HOST': args.bind,
                'ACADEMY_REDIS_PORT': str(args.port), 'ACADEMY_REDIS_DB': '0',
                'ACADEMY_REDIS_PASSWORD_FILE': str(password_file)}
    (root / 'env.sh').write_text(''.join(f'export {key}={shlex.quote(value)}\n'
                                       for key, value in settings.items()))
    print(f'Authenticated Redis ready on internal address {args.bind}:{args.port}')
    print(f'Source {root / "env.sh"} for Academy connection settings')


if __name__ == '__main__':
    main()
