#!/usr/bin/env python3
"""Run an authenticated HTTP/HTTPS forward proxy on Jupiter's internal login IP."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import quote


def authentication_required(host: str, port: int) -> bool:
    with socket.create_connection((host, port), timeout=2) as connection:
        connection.sendall(b'CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n')
        return connection.recv(4096).split(b'\r\n', 1)[0].split()[1:2] == [b'407']


def private_write(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(content)
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', required=True)
    parser.add_argument('--port', type=int, default=50858)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2] / 'software/internet-proxy')
    parser.add_argument('--serve', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    root = args.root.resolve()
    state = root / 'state'
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    state.chmod(0o700)
    credentials = state / 'credentials.json'
    if not credentials.exists():
        private_write(credentials, json.dumps({'username': 'loam', 'password': secrets.token_hex(32)}))
    auth = json.loads(credentials.read_text())
    if args.serve:
        from proxy import Proxy
        stop = threading.Event()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stop.set())
        with Proxy(input_args=['--hostname', args.bind, '--port', str(args.port),
                               '--num-acceptors', '1', '--num-workers', '1',
                               '--log-level', 'w', '--data-dir', str(state)],
                   basic_auth=auth['username'] + ':' + auth['password']):
            stop.wait()
        return
    try:
        ready = authentication_required(args.bind, args.port)
    except OSError:
        ready = False
    if not ready:
        # Refuse an occupied port rather than replacing another service.
        with socket.socket() as probe:
            probe.bind((args.bind, args.port))
        with (state / 'proxy.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--serve',
                                        '--bind', args.bind, '--port', str(args.port), '--root', str(root)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        (state / 'proxy.pid').write_text(str(process.pid) + '\n')
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if authentication_required(args.bind, args.port):
                    break
            except OSError:
                pass
            if process.poll() is not None:
                raise RuntimeError(f'Proxy exited; inspect {state / "proxy.log"}')
            time.sleep(0.2)
        else:
            raise RuntimeError('Proxy did not become ready')
    url = f"http://{quote(auth['username'], safe='')}:{quote(auth['password'], safe='')}@{args.bind}:{args.port}"
    bypass = 'localhost,127.0.0.1,::1,' + args.bind + ',.jupiter.internal'
    settings = {key: url for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY')}
    settings.update(no_proxy=bypass, NO_PROXY=bypass)
    private_write(root / 'env.sh', ''.join(f'export {key}={shlex.quote(value)}\n' for key, value in settings.items()))
    private_write(root / 'client.json', json.dumps(settings))
    print(f'Authenticated internet proxy ready at {args.bind}:{args.port}')
    print(f'Source {root / "env.sh"}; credentials are private and not printed.')


if __name__ == '__main__':
    main()
