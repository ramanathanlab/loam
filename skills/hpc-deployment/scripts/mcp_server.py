"""Local FastMCP server: GCExecutor → PBSPro allocation → pinned Academy agents.

Required environment:
  HPC_GLOBUS_ENDPOINT_ID, HPC_POOL_SIZE, HPC_ENDPOINT_CONFIG_JSON,
  HPC_BASE_DIR, HPC_CAMPAIGN_ID, ACADEMY_EXCHANGE_URL

Run locally with the directory containing this file on PYTHONPATH. The endpoint
must make this directory/module importable on workers, or use an endpoint-
approved Globus Compute code-shipping serializer.
"""
from __future__ import annotations

import json
import os
import signal
import sys
from contextlib import asynccontextmanager

from academy.exchange import HttpExchangeFactory
from academy.handle import Handle
from academy.manager import Manager
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from globus_compute_sdk import Executor
from hpc_agent import HpcAgent, HpcClientAgent, HpcClusterAgent
from pool_manager import CampaignPool, ClusterPool


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


ENDPOINT_ID = required("HPC_GLOBUS_ENDPOINT_ID")
POOL_SIZE = int(required("HPC_POOL_SIZE"))
BASE_DIR = required("HPC_BASE_DIR")
CAMPAIGN_ID = required("HPC_CAMPAIGN_ID")
EXCHANGE_URL = required("ACADEMY_EXCHANGE_URL")
ENDPOINT_CONFIG = json.loads(required("HPC_ENDPOINT_CONFIG_JSON"))
if not isinstance(ENDPOINT_CONFIG, dict):
    raise RuntimeError("HPC_ENDPOINT_CONFIG_JSON must encode an object")
if POOL_SIZE < 1:
    raise RuntimeError("HPC_POOL_SIZE must be positive")
INSPECT_ROOTS = [item for item in os.environ.get("HPC_INSPECT_ROOTS", "").split(":") if item]
# Selects the EL SystemConfig and the GPU-pinning variable the cluster agent
# hands to tasks: ZE_AFFINITY_MASK on Aurora, CUDA_VISIBLE_DEVICES on Polaris.
SYSTEM = os.environ.get("HPC_SYSTEM", "polaris")
if SYSTEM not in ("aurora", "polaris"):
    raise RuntimeError("HPC_SYSTEM must be 'aurora' or 'polaris'")


# The GC executor is the only allocation launcher. Passing the user endpoint
# config here causes the endpoint's Parsl PBSProProvider to render the PBS job.
def make_executor() -> Executor:
    return Executor(endpoint_id=ENDPOINT_ID, user_endpoint_config=ENDPOINT_CONFIG)


def create_server(cluster:bool=False,ckpt_dir=None, nnodes:int | None = None) -> FastMCP:
    # ckpt_dir only seeds the first cluster generation; ClusterPool mints a
    # fresh one for every rebuild so a new ClusterClient can never bind to the
    # dead master's checkpoint record.
    if cluster:
        assert nnodes is not None, "create server needs nnodes when using a EL cluster"

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        # Values are serialized into every agent constructor; worker-side setup
        # must make BASE_DIR available. Do not mkdir it locally: it may be a
        # remote-only HPC filesystem path.
        factory = HttpExchangeFactory(EXCHANGE_URL, auth_method="globus")
        gc_executor = make_executor()
        async with await Manager.from_exchange_factory(factory=factory, executors={"globus": gc_executor}, default_executor="globus") as manager:
            # In cluster mode EL owns placement and fanout, so the pool is a
            # fixed set of interchangeable submitters built once at startup;
            # the client agents are unpinned by design. Direct mode keeps
            # CampaignPool, which pins one accelerator per agent.
            pool = ClusterPool(CAMPAIGN_ID, POOL_SIZE, ckpt_dir) if cluster else CampaignPool(
                manager, HpcAgent, CAMPAIGN_ID, POOL_SIZE)

            # Handles are rebound to the exchange client: the pool probes them
            # for the life of the server, past the launch call that made them.
            async def launch_one() -> Handle:
                # The agent process is submitted through GCExecutor. This is
                # deliberately not a local executor and has no PBS commands.
                handle = await manager.launch(
                    HpcAgent(CAMPAIGN_ID, BASE_DIR, INSPECT_ROOTS),
                    executor="globus", name=f"{CAMPAIGN_ID}-agent"
                )
                return Handle(handle.agent_id, exchange=manager.exchange_client)

            async def launch_client(generation_ckpt_dir: str) -> Handle:
                handle = await manager.launch(
                    HpcClientAgent(CAMPAIGN_ID, BASE_DIR, generation_ckpt_dir, INSPECT_ROOTS),
                    executor="globus", name=f"{CAMPAIGN_ID}-agent"
                )
                return Handle(handle.agent_id, exchange=manager.exchange_client)

            async def launch_cluster(generation_ckpt_dir: str) -> Handle:
                handle = await manager.launch(HpcClusterAgent(
                    BASE_DIR, generation_ckpt_dir, nnodes, SYSTEM,
                ))
                return Handle(handle.agent_id, exchange=manager.exchange_client)

            async def run_code(payload: str, workdir: str | None = None, timeout: float = 300.0, venv_path: str | None = None) -> dict[str, str]:
                """Run Python on one campaign-owned, GPU/tile-pinned Academy agent."""
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action("run_code", payload, workdir=workdir or BASE_DIR, timeout=timeout, venv_path = venv_path)
                except Exception as exc:
                    raise ToolError(f"run_code failed: {type(exc).__name__}: {exc}") from exc

            async def inspect_filesystem(operation: str, path: str, offset: int = 1, limit: int = 100) -> dict:
                """Bounded read-only ls/cat/tail inspection on a pinned campaign worker."""
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action("inspect_filesystem", operation, path, offset=offset, limit=limit)
                except Exception as exc:
                    raise ToolError(f"inspect_filesystem failed: {type(exc).__name__}: {exc}") from exc

            # Build the pool before serving: a tool call must never have to
            # launch or verify an agent, or concurrent calls serialize. The
            # pool brings the EL cluster up first, then the clients that
            # handshake with its checkpoint dir.
            if cluster:
                await pool.start(launch_cluster, launch_client)

            server.add_tool(run_code)
            server.add_tool(inspect_filesystem)
            try:
                yield
            finally:
                # Stop health checking before Manager.close() shuts the agents
                # down, or the reaper sees them die and relaunches into a
                # cluster that is itself on the way out.
                if cluster:
                    await pool.stop()
        gc_executor.shutdown(wait=False, cancel_futures=False)

    return FastMCP(name="hpc-deployment", instructions=(
        "run_code executes Python on a pinned accelerator. inspect_filesystem is bounded read-only ls/cat/tail; it is not a shell."
    ), lifespan=lifespan, on_duplicate="replace")


if __name__ == "__main__":
    # uvicorn's capture_signals restores the original SIGTERM handler and then
    # re-raises the signal once serve() returns. Python's default SIGTERM
    # handler kills the process on the spot, so the FastMCP lifespan never
    # resumes past its yield: Manager.close() is skipped, no agent is asked to
    # shut down, and HpcClusterAgent.agent_on_shutdown never calls el.stop() --
    # orphaning a live EL cluster that keeps its allocation until walltime.
    # Raising KeyboardInterrupt instead lets that re-raise unwind the lifespan
    # stack, so the cluster still dies with its agent. SIGINT already does this.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    # Must match the node count requested in HPC_ENDPOINT_CONFIG_JSON: it sets
    # the EL policy's leaf count, not the allocation size.
    create_server(cluster=True, nnodes=int(os.environ.get("HPC_NNODES", "1"))).run(
        transport=os.environ.get("HPC_MCP_TRANSPORT", "stdio"),
    )
