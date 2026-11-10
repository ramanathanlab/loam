"""Batch staging must not turn extra inspection roots into writable software."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client, FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/hpc-deployment/scripts"))
from hpc_agent import HpcAgent
from mcp_dispatch import dispatch_action


def test_readable_software_is_not_writable_but_campaign_staging_works(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    software = tmp_path / "software"
    software.mkdir()
    victim = software / "model.py"
    victim.write_text("keep installed software\n")
    agent = HpcAgent("staging-test", str(campaign), inspect_roots=[str(software)])

    async def check():
        inspected = await agent.inspect_filesystem("cat", str(victim))
        assert "keep installed software" in json.dumps(inspected)
        with pytest.raises(ValueError):
            await agent.write_files({str(victim): "overwrite\n"})
        assert victim.read_text() == "keep installed software\n"
        staged = campaign / "plan" / "script.py"
        result = await agent.write_files({str(staged): "print('staged')\n"})
        assert result["count"] == 1
        assert staged.read_text() == "print('staged')\n"

    asyncio.run(check())


def test_expected_staging_refusals_do_not_mark_healthy_mcp_transport_failed(tmp_path):
    agent = HpcAgent("staging-refusal", str(tmp_path))

    async def action(name, files):
        return await agent.write_files(files)

    pool = SimpleNamespace(next_agent=AsyncMock(return_value=SimpleNamespace(handle=SimpleNamespace(action=action))))
    server = FastMCP("staging-regression")

    @server.tool
    async def write_files(files: dict) -> dict:
        return await dispatch_action(pool, None, "write_files", files)

    async def check():
        async with Client(server) as client:
            for _ in range(3):
                result = await client.call_tool("write_files", {"files": {"../escape.py": "private contents"}})
                assert not result.is_error
                assert result.data["status"] == "invalid_request"
                assert "private contents" not in json.dumps(result.data)
            result = await client.call_tool("write_files", {"files": {"good/script.py": "print(1)\n"}})
            assert not result.is_error
            assert result.data["count"] == 1
            assert (tmp_path / "good/script.py").read_text() == "print(1)\n"

    asyncio.run(check())
