"""Loam's MCP error contract works with the installed stock Hermes 0.20.6."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
SERVER_PYTHON = WORKSPACE / 'software/globus-compute/.venv/bin/python'
HERMES_PYTHON = WORKSPACE / 'software/hermes-agent/.venv/bin/python'


def test_stock_hermes_distinguishes_inspection_results_from_pool_failure(tmp_path):
    if not SERVER_PYTHON.is_file() or not HERMES_PYTHON.is_file():
        pytest.skip('Jupiter service environments are not installed')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with (tmp_path / 'server.log').open('w') as log:
        server = subprocess.Popen([str(SERVER_PYTHON), str(FIXTURES / 'mcp_inspection_probe_server.py'),
                                   '--port', str(port)], stdout=log, stderr=log, cwd=tmp_path)
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    break
            except OSError:
                assert server.poll() is None, (tmp_path / 'server.log').read_text()
                assert time.monotonic() < deadline, 'Fixture MCP server startup timed out'
                time.sleep(0.05)
        env = dict(os.environ, HERMES_HOME=str(tmp_path / 'hermes-home'))
        result = subprocess.run([str(HERMES_PYTHON), str(FIXTURES / 'hermes_mcp_inspection_probe.py'),
                                 '--port', str(port)], capture_output=True, text=True,
                                env=env, cwd=tmp_path, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        assert receipt['status'] == 'passed'
    finally:
        server.terminate()
        server.wait(timeout=15)
