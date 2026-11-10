"""The login proxy rejects anonymous clients and forwards authenticated requests."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT = WORKSPACE / 'loam/scripts/start_jupiter_proxy.py'
PYTHON = WORKSPACE / 'software/internet-proxy/.venv/bin/python'


def test_authenticated_forwarding_and_private_credentials(tmp_path, monkeypatch):
    if not PYTHON.is_file():
        pytest.skip('Jupiter proxy environment is not installed')
    monkeypatch.setenv('no_proxy', '')
    monkeypatch.setenv('NO_PROXY', '')
    spec = importlib.util.spec_from_file_location('jupiter_proxy', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    credentials = tmp_path / 'state/credentials.json'
    credentials.parent.mkdir()
    module.private_write(credentials, json.dumps({'username': 'test', 'password': 'private-test-secret'}))
    assert credentials.stat().st_mode & 0o777 == 0o600
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'forwarded-through-login')

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen([str(PYTHON), str(SCRIPT), '--serve', '--bind', '127.0.0.1',
                                '--port', str(port), '--root', str(tmp_path)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                if module.authentication_required('127.0.0.1', port):
                    break
            except OSError:
                pass
            assert time.monotonic() < deadline, 'Proxy startup failed'
            time.sleep(0.05)
        assert not seen, 'Anonymous request must not reach upstream'
        proxy = f'http://test:private-test-secret@127.0.0.1:{port}'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy}))
        with opener.open(f'http://127.0.0.1:{upstream.server_port}/probe', timeout=5) as response:
            assert response.read() == b'forwarded-through-login'
        assert seen == ['/probe']
    finally:
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
