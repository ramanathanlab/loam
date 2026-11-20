"""Real installed MCP handshakes; no inference, HPC, scheduler, or network calls."""
import ast
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import FastMCP
from mcp import ClientSession
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
import pydantic_ai.mcp as installed_mcp

HERE = Path(__file__).parent


def module(which):
    from loam import execution
    return execution


def config(init_timeout=60):
    return SimpleNamespace(execution=SimpleNamespace(mode='mcp', mcp_url='http://fixture.invalid/mcp'),
                           timeouts=SimpleNamespace(tool=6000, mcp_init=init_timeout))


def server_with(names):
    server = FastMCP('initialization-fixture')
    for name in names:
        def tool():
            return 'CPU-only fixture; no scientific evidence'
        server.tool(name=name)(tool)
    return server


def route_to_inprocess(monkeypatch, server, clients, *, first_timeout=None):
    real = installed_mcp.MCPToolset

    def factory(url, **kwargs):
        assert url == 'http://fixture.invalid/mcp'
        # Keep the installed PydanticAI constructor and its timeout forwarding.
        # Only the transport is in-process and protocol pinned to initialize.
        toolset = real(server, **kwargs)
        toolset.client.mode = 'legacy'
        if not clients and first_timeout is not None:
            toolset.client._init_timeout = first_timeout
        clients.append((toolset.client, kwargs))
        return toolset

    monkeypatch.setattr(installed_mcp, 'MCPToolset', factory)


def test_initialization_timeout_has_a_positive_configurable_bound():
    from loam.config import Timeouts
    assert Timeouts().mcp_init == 60
    assert Timeouts(mcp_init=120).mcp_init == 120
    with pytest.raises(ValueError):
        Timeouts(mcp_init=0)


def test_real_handshake_over_five_seconds_fails_old_default_and_passes_candidate(monkeypatch):
    candidate, baseline = module('candidate'), module('baseline')
    clients = []
    route_to_inprocess(monkeypatch, server_with(candidate.MCP_FULL), clients)
    original = ClientSession.initialize

    async def delayed(self, *args, **kwargs):
        await asyncio.sleep(5.2)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ClientSession, 'initialize', delayed)

    async def run():
        with pytest.raises(RuntimeError, match='Failed to initialize server session'):
            await baseline.check_mcp_tools(config(5))
        await candidate.check_mcp_tools(config())

    asyncio.run(run())
    assert clients[0][0]._init_timeout == 5
    assert clients[1][0]._init_timeout == 60
    assert not any(client.is_connected() for client, _ in clients)


@pytest.mark.parametrize('full', [False, True])
def test_parallel_runs_keep_separate_sessions_and_exact_role_allowlists(monkeypatch, full):
    candidate = module('candidate')
    clients, seen = [], []
    route_to_inprocess(monkeypatch, server_with(candidate.MCP_FULL | {'forbidden_admin'}), clients)

    def respond(messages, info):
        seen.append({tool.name for tool in info.function_tools})
        return ModelResponse(parts=[TextPart('metadata-only fixture complete')])

    agent = Agent(FunctionModel(respond), toolsets=[candidate.execution_tools(config(), full)])

    async def run():
        return await asyncio.gather(*(agent.run('discover only') for _ in range(24)))

    results = asyncio.run(run())
    assert len(results) == len(clients) == len(seen) == 24
    assert len({id(client) for client, _ in clients}) == 24
    assert all(names == (candidate.MCP_FULL if full else candidate.MCP_READ) for names in seen)
    assert all(kwargs == {'init_timeout': 60, 'read_timeout': 6000} for _, kwargs in clients)
    assert not any(client.is_connected() for client, _ in clients)


def test_one_timed_out_session_does_not_abort_other_parallel_runs(monkeypatch):
    candidate = module('candidate')
    clients = []
    route_to_inprocess(monkeypatch, server_with(candidate.MCP_FULL), clients, first_timeout=0.01)
    original = ClientSession.initialize

    async def delayed(self, *args, **kwargs):
        await asyncio.sleep(0.08)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ClientSession, 'initialize', delayed)

    def respond(messages, info):
        return ModelResponse(parts=[TextPart('surviving session')])

    agent = Agent(FunctionModel(respond), toolsets=[candidate.execution_tools(config(), False)])

    async def run():
        return await asyncio.gather(*(agent.run('discover only') for _ in range(8)), return_exceptions=True)

    results = asyncio.run(run())
    errors = [x for x in results if isinstance(x, BaseException)]
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert 'Failed to initialize server session' in str(errors[0])
    assert len([x for x in results if not isinstance(x, BaseException)]) == 7
    assert len(clients) == 8
    assert not any(client.is_connected() for client, _ in clients)


def test_startup_still_rejects_a_missing_required_tool(monkeypatch):
    candidate = module('candidate')
    clients = []
    route_to_inprocess(monkeypatch, server_with(candidate.MCP_FULL - {'run_batch'}), clients)
    with pytest.raises(RuntimeError, match='missing required tools: run_batch'):
        asyncio.run(candidate.check_mcp_tools(config()))
    assert len(clients) == 1 and clients[0][1] == {'init_timeout': 60}
