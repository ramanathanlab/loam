"""Smoke-test the inexpensive surface of a running cluster-mode server."""
from __future__ import annotations

import asyncio
import os

from fastmcp import Client


async def main() -> None:
    url = os.environ.get("HPC_MCP_URL")
    if not url:
        raise RuntimeError("set HPC_MCP_URL, for example http://127.0.0.1:8000/mcp")
    async with Client(url, timeout=180) as client:
        names = {tool.name for tool in await client.list_tools()}
        expected = {"run_code", "write_files", "inspect_filesystem", "run_batch",
                    "batch_status", "list_batches", "pool_status"}
        if names != expected:
            raise RuntimeError(
                f"cluster-mode tools differ: missing={expected - names}, unexpected={names - expected}"
            )
        # Do not submit science (or even a batch) here. Listing the seven-tool
        # contract proves run_batch was registered after the full pool started.
        probe = "print('hpc smoke ok')"
        code = await client.call_tool("run_code", {"payload": probe})
        print("RUN_CODE", code.content[0].text)
        root = os.environ["HPC_BASE_DIR"]
        listing = await client.call_tool("inspect_filesystem", {"operation": "ls", "path": root})
        print("INSPECT", listing.content[0].text)
        capacity = await client.call_tool("pool_status", {})
        print("POOL", capacity.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
