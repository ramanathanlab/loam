"""Exercise the installed, unmodified Hermes MCP client against the Loam fixture."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

from hermes_cli.config import atomic_config_write

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()
assert version('hermes-agent') == '0.20.6'
home = Path(os.environ['HERMES_HOME'])
# Trust only this test profile's static loopback fixture. No production
# configuration is read or modified, and the fixture cannot execute user code.
atomic_config_write(home / 'config.yaml', {'mcp_servers': {'loam_probe': {
    'url': f'http://127.0.0.1:{args.port}/mcp', 'timeout': 10, 'trust': 'full'}}})
from tools import mcp_tool  # noqa: E402 - configure isolated home before discovery

try:
    tools = mcp_tool.discover_mcp_tools()
    assert any('inspect_filesystem' in tool for tool in tools), tools
    inspect = mcp_tool._make_tool_handler('loam_probe', 'inspect_filesystem', 10)
    run = mcp_tool._make_tool_handler('loam_probe', 'run_code', 10)
    failure = mcp_tool._make_tool_handler('loam_probe', 'pool_failure', 10)
    for _ in range(3):
        result = json.loads(inspect({'operation': 'ls', 'path': '/fixture/missing'}))
        assert 'error' not in result, result
        body = result.get('structuredContent') or json.loads(result['result'])
        assert body['status'] == 'not_found', result
    assert mcp_tool._server_error_counts.get('loam_probe', 0) == 0
    result = json.loads(run({}))
    assert 'error' not in result, result
    body = result.get('structuredContent') or json.loads(result['result'])
    assert body['stdout'] == 'LOAM_DISPATCH_OK\n', result
    for _ in range(3):
        assert 'error' in json.loads(failure({}))
    assert mcp_tool._server_error_counts.get('loam_probe', 0) >= 3
    assert 'unreachable' in json.loads(run({}))['error']
    print(json.dumps({'status': 'passed', 'hermes_version': version('hermes-agent'),
                      'missing_inspections': 3, 'subsequent_run_code': 'passed',
                      'real_pool_failures_still_open_breaker': True}))
finally:
    mcp_tool.shutdown_mcp_servers()
