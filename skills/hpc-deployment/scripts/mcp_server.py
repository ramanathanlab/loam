"""Local FastMCP server: GCExecutor → PBSPro allocation → pinned Academy agents.

Required environment:
  HPC_GLOBUS_ENDPOINT_ID, HPC_POOL_SIZE, HPC_ENDPOINT_CONFIG_JSON,
  HPC_BASE_DIR, HPC_CAMPAIGN_ID; see exchange_config.py for exchange settings

Run locally with the directory containing this file on PYTHONPATH. The endpoint
must make this directory/module importable on workers, or use an endpoint-
approved Globus Compute code-shipping serializer.
"""
from __future__ import annotations

import json
import logging
import os
import signal
from contextlib import asynccontextmanager

from academy.handle import Handle
from academy.manager import Manager
from fastmcp import FastMCP
from globus_compute_sdk import Executor
from exchange_config import create_exchange_factory
from mcp_dispatch import dispatch_action
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
ENDPOINT_CONFIG = json.loads(required("HPC_ENDPOINT_CONFIG_JSON"))
if not isinstance(ENDPOINT_CONFIG, dict):
    raise RuntimeError("HPC_ENDPOINT_CONFIG_JSON must encode an object")
if POOL_SIZE < 1:
    raise RuntimeError("HPC_POOL_SIZE must be positive")
INSPECT_ROOTS = [item for item in os.environ.get("HPC_INSPECT_ROOTS", "").split(":") if item]
# Selects the EL SystemConfig and the GPU-pinning variable the cluster agent
# hands to tasks: ZE_AFFINITY_MASK on Aurora, CUDA_VISIBLE_DEVICES on Polaris
# and CSCS (Alps GH200). 'cscs' also makes a single-node cluster launch its
# leaf child via multiprocessing instead of mpiexec (Alps has no mpiexec).
SYSTEM = os.environ.get("HPC_SYSTEM", "polaris")
if SYSTEM not in ("aurora", "polaris", "alps", "cscs", "jupiter"):
    raise RuntimeError("HPC_SYSTEM must be 'aurora', 'polaris' or 'alps' ('cscs' alias) or 'jupiter'")


# The GC executor is the only allocation launcher. Passing the user endpoint
# config here causes the endpoint's Parsl PBSProProvider to render the PBS job.
def make_executor() -> Executor:
    return Executor(endpoint_id=ENDPOINT_ID, user_endpoint_config=ENDPOINT_CONFIG)


def create_server(cluster: bool = False, ckpt_dir: str | None = None,
                  nnodes: int | None = None) -> FastMCP:
    # ckpt_dir only seeds the first cluster generation; ClusterPool mints a
    # fresh one for every rebuild so a new ClusterClient can never bind to the
    # dead master's checkpoint record.
    if cluster and nnodes is None:
        raise ValueError("create_server(cluster=True) needs nnodes")

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        # Values are serialized into every agent constructor; worker-side setup
        # must make BASE_DIR available. Do not mkdir it locally: it may be a
        # remote-only HPC filesystem path.
        factory = create_exchange_factory()
        gc_executor = make_executor()
        served = False
        try:
            async with _serve(factory, gc_executor, server):
                served = True
                yield
        except Exception:  # noqa: BLE001 - shutdown must finish even if an agent died badly
            if not served:
                raise  # a startup failure must still fail the server loudly
            # Manager.close() asks every agent to shut down first, then awaits
            # each one and re-raises the first failure. After a cluster rebuild
            # the previous generation's agents always fail (ManagerLost at
            # walltime), so without this the exit skipped the executor shutdown
            # below and its non-daemon threads kept the process alive forever.
            logging.getLogger(__name__).exception("agent failures during shutdown (agents were already asked to stop)")
        finally:
            gc_executor.shutdown(wait=False, cancel_futures=True)

    @asynccontextmanager
    async def _serve(factory, gc_executor, server: FastMCP):
        async with await Manager.from_exchange_factory(factory=factory, executors={"globus": gc_executor}, default_executor="globus") as manager:
            # In cluster mode EL owns placement and fanout, so the pool is a
            # fixed set of interchangeable submitters built once at startup;
            # the client agents are unpinned by design. Direct mode keeps
            # CampaignPool, which pins one accelerator per agent.
            pool = ClusterPool(CAMPAIGN_ID, POOL_SIZE, ckpt_dir) if cluster else CampaignPool(
                manager, HpcAgent, CAMPAIGN_ID, POOL_SIZE)

            # Every agent process is submitted through GCExecutor (the
            # manager's default executor) -- deliberately not a local executor,
            # and no PBS commands. The handle is rebound to the exchange
            # client: the pool probes it for the life of the server, past the
            # launch call that made it.
            async def launch(agent, **kwargs) -> Handle:
                handle = await manager.launch(agent, **kwargs)
                return Handle(handle.agent_id, exchange=manager.exchange_client)

            named = {"executor": "globus", "name": f"{CAMPAIGN_ID}-agent"}

            async def launch_one() -> Handle:
                return await launch(HpcAgent(CAMPAIGN_ID, BASE_DIR, INSPECT_ROOTS), **named)

            async def launch_client(generation_ckpt_dir: str) -> Handle:
                return await launch(HpcClientAgent(
                    CAMPAIGN_ID, BASE_DIR, generation_ckpt_dir, INSPECT_ROOTS), **named)

            async def launch_cluster(generation_ckpt_dir: str) -> Handle:
                return await launch(HpcClusterAgent(BASE_DIR, generation_ckpt_dir, nnodes, SYSTEM))

            # Worker/pool failures remain MCP errors. Expected inspection
            # refusals return explicit application status so healthy transport
            # is not misclassified as an outage by the harness.
            async def dispatch(action: str, *args, **kwargs):
                return await dispatch_action(pool, launch_one, action, *args, **kwargs)

            async def run_code(payload: str, workdir: str | None = None, timeout: float = 300.0, venv_path: str | None = None) -> dict[str, str]:
                """Run Python on one campaign-owned, GPU/tile-pinned Academy agent."""
                return await dispatch("run_code", payload, workdir=workdir or BASE_DIR,
                                      timeout=timeout, venv_path=venv_path)

            async def inspect_filesystem(operation: str, path: str, offset: int = 1, limit: int = 100) -> dict:
                """Bounded read-only ls/cat/tail inspection on a pinned worker.

                Expected refusals return status not_found, denied, or invalid_request;
                they do not mean the server is unreachable. Inspection never creates
                directories. run_code creates its assigned workdir when executing.
                """
                return await dispatch("inspect_filesystem", operation, path, offset=offset, limit=limit)

            async def write_files(files: dict) -> dict:
                """Write text files onto the cluster filesystem, creating parent directories.

                Pass `files` as {absolute_path: file_contents}; several in one call.
                This is how you stage a batch: `run_batch` needs its script_path and
                inputs_path to exist already, so write both here, then submit. Nothing
                is executed, so use it for the script, the inputs JSONL, and any small
                config a row reads -- not for data a row can generate itself. Paths are
                confined to the campaign tree; a relative path resolves inside it.
                Returns the resolved path and byte count per file. Prefer this over
                `run_code` for every file you need to create.
                """
                return await dispatch("write_files", files)

            async def run_batch(script_path: str | None = None, inputs_path: str | None = None,
                                timeout_each: float = 900.0, gpus: float = 1,
                                venv_path: str | None = None, batch_id: str | None = None,
                                labels: dict | None = None) -> dict:
                """Run one Python script once per row of a JSONL file, in parallel across the cluster's GPUs.

                Give script_path. Each row runs as `python script.py input.json` in its
                own folder, where input.json is that row (an "id" field is optional).
                Write results to files in that folder and exit 0 on success. The script and
                inputs file are copied when the batch is created, so later edits do not
                affect it. gpus is the number of GPUs/tiles EACH ROW gets, not the batch:
                rows run side by side on all free tiles, so a script that uses one device
                (e.g. xpu:0 or cuda:0) needs gpus=1 and 12 rows then fill a 12-tile node at
                once. Raise it only when a single row itself spans several devices.
                Returns at once with a batch_id: poll batch_status, never wait.
                To resume, pass only batch_id; finished runs are skipped. Use this rather
                than run_code for more than a handful of runs.
                """
                return await dispatch(
                    "run_batch", script_path=script_path, inputs_path=inputs_path,
                    timeout_each=timeout_each, gpus=gpus, venv_path=venv_path,
                    batch_id=batch_id, labels=labels,
                )

            async def batch_status(batch_id: str) -> dict:
                """Progress of one run_batch batch: total, succeeded, failed, unfinished, and the first failures' stderr."""
                return await dispatch("batch_status", batch_id)

            async def list_batches(plan_id: str | None = None, hypothesis_id: str | None = None,
                                   limit: int = 50) -> list[dict]:
                """Batches in the registry, newest first, with their last known status."""
                return await dispatch("list_batches", plan_id=plan_id,
                                      hypothesis_id=hypothesis_id, limit=limit)

            async def register_tool(name: str, script_path: str, params: dict, description: str,
                                    example: dict, defaults: dict | None = None,
                                    call: str = "json", venv_path: str | None = None,
                                    skip_smoke: bool = False, labels: dict | None = None,
                                    expects: list[str] | None = None) -> dict:
                """Save a reusable script as a named, versioned tool that run_batch(tool=name) can run.

                params maps each input name to one of int, float, str, bool, list, dict;
                defaults fills params a row leaves out. example is one input row: it is run
                once as a smoke test and the tool is registered only if it exits 0 (use
                skip_smoke for a script that needs a GPU; it is then marked tested=false).
                call="json" runs `python script.py input.json`; call="argv" runs
                `python script.py --name value ...`. The script (at most 1 MB) is copied, so
                later edits need a new registration, which becomes the next version.
                Registering an unchanged script returns the existing version. Tools cannot
                be deleted. Call list_tools first and reuse a tool that fits.
                expects lists output filenames/globs (relative to each run's folder) the
                script must write; a run that exits 0 without them counts as failed.
                """
                return await dispatch(
                    "register_tool", name, script_path, params, description, example,
                    defaults=defaults, call=call, venv_path=venv_path,
                    skip_smoke=skip_smoke, labels=labels, expects=expects,
                )

            async def list_tools(query: str | None = None, name: str | None = None,
                                 all_versions: bool = False, limit: int = 50) -> list[dict]:
                """Registered tools: params, description, example, how often batches used them, last clean batch."""
                return await dispatch("list_tools", query=query, name=name,
                                      all_versions=all_versions, limit=limit)

            async def pool_status() -> dict:
                """Live compute budget for sizing a run_batch fan-out: nodes, addressable GPU tiles, queue, walltime.

                Read this BEFORE launching a large batch so you fan out to the tiles
                you actually have rather than guessing. `total_tiles` is the number of
                run_batch(gpus=1) rows that run side by side at once; each job must
                finish inside `walltime`, and on debug / debug-scaling queues the block
                is reclaimed about once an hour, so keep per-job work well under the
                hour. This reads local allocation config only -- it reports CAPACITY,
                not current free/busy; combine with list_batches / batch_status to see
                what is already in flight. Instant, never queues behind worker calls.
                """
                per_node = {"aurora": 12, "polaris": 4, "alps": 4, "cscs": 4, "jupiter": 4}.get(SYSTEM, 12)
                total_tiles = (nnodes * per_node) if cluster else POOL_SIZE
                cfg = ENDPOINT_CONFIG

                def pick(*keys):
                    for k in keys:
                        v = cfg.get(k)
                        if v not in (None, ""):
                            return v
                    return None

                return {
                    "system": SYSTEM,
                    "mode": "cluster" if cluster else "direct",
                    "nodes": nnodes if cluster else None,
                    "tiles_per_node": per_node,
                    "total_tiles": total_tiles,
                    "pool_size": POOL_SIZE,
                    "queue": pick("queue"),
                    "walltime": pick("walltime"),
                    "nodes_per_block": pick("nodes_per_block", "select"),
                    "account": pick("account", "project"),
                    "campaign_id": CAMPAIGN_ID,
                    "guidance": (
                        f"Up to {total_tiles} run_batch(gpus=1) rows run concurrently. "
                        "Size a fan-out to this; each job must finish "
                        "within walltime (debug/debug-scaling reclaim ~hourly). "
                        "This is capacity, not live free tiles -- check list_batches for in-flight work."
                    ),
                }

            # run_batch needs EL; status/listing only read disk and register_tool's
            # smoke run is a local subprocess, so any agent serves those.
            # --- Tool registry DISABLED 2026-11-02 (register_tool / list_tools) ---
            # The registry's "reuse a tool that fits" nudge (register_tool docstring +
            # list_tools discovery) steered code-generators to reuse prior, broken
            # registered tools -- e.g. a chai fold that skipped the skill's chai_xpu.py
            # launcher and silently produced 0 structures, yet was marked last_ok_batch.
            # Un-exposing both leaves run_batch(script_path=...) as the only path: a fresh
            # script built from the skill each time. The functions remain defined above.
            # To re-enable, add  register_tool, list_tools  back into this list.
            tools = [run_code, write_files, inspect_filesystem, batch_status, list_batches, pool_status]
            if cluster:
                # Build the pool before serving: a tool call must never have to
                # launch or verify an agent, or concurrent calls serialize. The
                # pool brings the EL cluster up first, then the clients that
                # handshake with its checkpoint dir.
                await pool.start(launch_cluster, launch_client)
                tools.append(run_batch)
            for tool in tools:
                server.add_tool(tool)
            try:
                yield
            finally:
                # Stop health checking before Manager.close() shuts the agents
                # down, or the reaper sees them die and relaunches into a
                # cluster that is itself on the way out.
                if cluster:
                    await pool.stop()

    return FastMCP(name="hpc-deployment", instructions=(
        "run_code executes Python on a pinned accelerator. write_files creates files on the cluster filesystem "
        "without executing anything -- use it to stage a batch's script and inputs rather than running code that writes them. "
        "inspect_filesystem is bounded read-only ls/cat/tail; it is not a shell. "
        "For many runs of one script use run_batch, then poll batch_status; list_batches shows past batches. "
        "Call pool_status to see the compute budget (nodes, addressable tiles, queue, walltime) before sizing a large fan-out, "
        "so you match the number of parallel run_batch rows to the tiles actually available."
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
