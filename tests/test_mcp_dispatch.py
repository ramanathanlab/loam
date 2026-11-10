"""Application-level inspection refusals do not imply an unavailable worker."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/hpc-deployment/scripts'))
import hpc_agent
from mcp_dispatch import dispatch_action


def pool_for(action):
    return SimpleNamespace(next_agent=AsyncMock(return_value=SimpleNamespace(
        handle=SimpleNamespace(action=action))))


def test_three_missing_inspections_then_valid_execution(tmp_path):
    missing = tmp_path / 'not-created'

    async def action(name, *args, **kwargs):
        if name == 'inspect_filesystem':
            return hpc_agent._inspect(args[0], args[1], 1, 100, str(tmp_path), [])
        return hpc_agent._run_subprocess("print('DISPATCH_OK')", str(tmp_path), 10)

    pool = pool_for(action)
    server = FastMCP('dispatch-regression')

    @server.tool
    async def inspect_filesystem(operation: str, path: str) -> dict:
        return await dispatch_action(pool, None, 'inspect_filesystem', operation, path)

    @server.tool
    async def run_code() -> dict:
        return await dispatch_action(pool, None, 'run_code')

    async def check():
        async with Client(server) as client:
            for _ in range(3):
                result = await client.call_tool('inspect_filesystem', {'operation': 'ls', 'path': str(missing)})
                assert not result.is_error
                assert result.data['status'] == 'not_found'
                assert result.data['exists'] is False
                assert not missing.exists(), 'Read-only inspection must not create directories'
            result = await client.call_tool('run_code', {})
            assert not result.is_error
            assert result.data['stdout'] == 'DISPATCH_OK\n'
            assert result.data['returncode'] == '0'
    asyncio.run(check())


def test_expected_refusals_preserve_read_only_security(tmp_path):
    text = tmp_path / 'text'
    text.write_text('preserve me')
    binary = tmp_path / 'binary'
    binary.write_bytes(b'\0private contents')
    escape = tmp_path / 'escape'
    escape.symlink_to('/etc/passwd')

    async def action(name, *args, **kwargs):
        return hpc_agent._inspect(args[0], args[1], 1, 100, str(tmp_path), [])

    async def check():
        pool = pool_for(action)
        for operation, path in [('remove', text), ('cat', binary), ('cat', escape)]:
            result = await dispatch_action(pool, None, 'inspect_filesystem', operation, str(path))
            assert result['status'] == 'invalid_request'
            assert 'lines' not in result and 'entries' not in result
            assert 'private contents' not in str(result)
        assert text.read_text() == 'preserve me'
    asyncio.run(check())


@pytest.mark.parametrize('failure', [ConnectionError('pool disconnected'), FileNotFoundError('pool state missing')])
def test_pool_failures_are_not_converted_to_inspection_results(failure):
    pool = SimpleNamespace(next_agent=AsyncMock(side_effect=failure))
    with pytest.raises(ToolError, match='worker unavailable'):
        asyncio.run(dispatch_action(pool, None, 'inspect_filesystem', 'ls', '/missing'))


def test_unexpected_worker_failure_remains_mcp_error():
    pool = pool_for(AsyncMock(side_effect=RuntimeError('worker crashed')))
    with pytest.raises(ToolError, match='worker crashed'):
        asyncio.run(dispatch_action(pool, None, 'inspect_filesystem', 'ls', '/missing'))


def test_permission_denial_is_explicit_without_exposing_content():
    pool = pool_for(AsyncMock(side_effect=PermissionError('access refused')))
    result = asyncio.run(dispatch_action(pool, None, 'inspect_filesystem', 'cat', '/private'))
    assert result['status'] == 'denied'
    assert 'lines' not in result
