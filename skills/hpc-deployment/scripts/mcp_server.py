"""FastMCP scientific tools in a Globus-managed or already granted allocation.

Required environment:
  HPC_GLOBUS_ENDPOINT_ID, HPC_POOL_SIZE, HPC_ENDPOINT_CONFIG_JSON,
  HPC_BASE_DIR, HPC_CAMPAIGN_ID, ACADEMY_EXCHANGE_URL

Run locally with the directory containing this file on PYTHONPATH. The endpoint
must make this directory/module importable on workers, or use an endpoint-
approved Globus Compute code-shipping serializer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from contextlib import asynccontextmanager

from exchange_config import create_exchange_factory
from academy.handle import Handle
from academy.manager import Manager
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from globus_compute_sdk import Executor
from hpc_agent import HpcAgent, HpcClientAgent, HpcClusterAgent, enable_registry_index
from pool_manager import CampaignPool, ClusterPool

import inspect as _inspect
from fastmcp.tools import Tool as _Tool


def _add_tolerant(server: FastMCP, fn) -> None:
    """Register fn as a tool that DROPS unknown keyword arguments instead of
    rejecting the whole call.

    Weak tool-calling models (e.g. nemotron-super-vl) frequently hallucinate
    extra kwargs; FastMCP's generated schema sets additionalProperties:false,
    so every such call failed with 'unexpected_keyword_argument' and no HPC
    work was ever dispatched. We keep the schema FastMCP derived from the real
    signature -- required arguments and types are STILL enforced -- but flip
    additionalProperties to true and strip any argument the function does not
    declare before calling it, so a call that merely carries junk extras
    succeeds while a genuinely malformed one (missing a required arg) still
    errors.
    """
    tool = _Tool.from_function(fn)
    try:
        if isinstance(tool.parameters, dict):
            tool.parameters["additionalProperties"] = True
    except Exception:
        pass
    valid = set(_inspect.signature(fn).parameters)
    orig = tool.fn

    async def _tolerant(*args, **kwargs):
        return await orig(*args, **{k: v for k, v in kwargs.items() if k in valid})

    tool.fn = _tolerant
    server.add_tool(tool)


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


LAUNCH_MODE = os.environ.get("HPC_LAUNCH_MODE", "globus")
if LAUNCH_MODE not in ("globus", "existing"):
    raise RuntimeError("HPC_LAUNCH_MODE must be globus or existing")
if LAUNCH_MODE == "existing" and (not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("HPC_SCIENCE_NODELIST")):
    raise RuntimeError("Existing-allocation mode requires an active Slurm job and explicit science node list")
ENDPOINT_ID = os.environ.get("HPC_GLOBUS_ENDPOINT_ID", "") if LAUNCH_MODE == "existing" else required("HPC_GLOBUS_ENDPOINT_ID")
POOL_SIZE = int(required("HPC_POOL_SIZE"))
SCIENCE_WORKERS_PER_GPU = int(os.environ.get('HPC_SCIENCE_WORKERS_PER_GPU', '1'))
if SCIENCE_WORKERS_PER_GPU not in (1, 2):
    raise RuntimeError('HPC_SCIENCE_WORKERS_PER_GPU must be one or two')
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
if SYSTEM not in ("aurora", "polaris", "jupiter"):
    raise RuntimeError("HPC_SYSTEM must be 'aurora', 'polaris' or 'jupiter'")
# Longest a listen stream stays open before a clean reconnect. The exchange
# server blocks in a destructive pop for this long on each listen; when the
# ALCF proxy severs our side, that stale reader lingers and can take (and
# lose) the next message while our reconnected stream waits beside it. A
# shorter window clears stale readers sooner; the cost is a few extra listen
# requests per agent per minute. Academy's default is 60s.
EXCHANGE_REQUEST_TIMEOUT_S = float(os.environ.get("HPC_EXCHANGE_REQUEST_TIMEOUT", "20"))
if EXCHANGE_REQUEST_TIMEOUT_S <= 0:
    raise RuntimeError("HPC_EXCHANGE_REQUEST_TIMEOUT must be positive")


# Default mode delegates allocation to the configured Globus endpoint.
# Existing mode only attaches to an allocation already granted by Slurm.
def make_executor() -> Executor | None:
    # Academy can host its asynchronous control agents in this event loop.
    # EL still executes science on explicitly assigned Slurm nodes via srun.
    # No second allocation, Globus submission, or compute-node internet is used.
    return None if LAUNCH_MODE == "existing" else Executor(endpoint_id=ENDPOINT_ID, user_endpoint_config=ENDPOINT_CONFIG)


logger = logging.getLogger("hpc-deployment")

_COLD_ENDPOINT_CODE = "ENDPOINT_NOT_ONLINE"
COLD_ENDPOINT_TIMEOUT = float(os.environ.get("HPC_COLD_ENDPOINT_TIMEOUT", "300"))
COLD_ENDPOINT_BACKOFF = float(os.environ.get("HPC_COLD_ENDPOINT_BACKOFF", "5"))
# How long one probe waits for an answer before concluding the uep took the
# task. Must stay well above a web round-trip (a 422 arrives in ~1s) and well
# below any queue wait -- it is NOT a budget for the block to start.
COLD_ENDPOINT_PROBE_WAIT = float(os.environ.get("HPC_COLD_ENDPOINT_PROBE_WAIT", "30"))


def _is_cold_endpoint(exc: BaseException) -> bool:
    """Known startup races: endpoint offline (422) or registration conflict (409).

    A new per-config endpoint can already have submitted its Slurm block while
    the service still reports RESOURCE_CONFLICT. Retry the identical config;
    never change the allocation namespace to bypass this race.
    """
    codes = {_COLD_ENDPOINT_CODE, "RESOURCE_CONFLICT"}
    if getattr(exc, "code", None) in codes:
        return True
    return any(code in str(exc) for code in codes)


async def await_endpoint_ready() -> None:
    """Push one trivial task through a throwaway Executor until the uep accepts it.

    Readiness means THE SUBMIT WAS ACCEPTED, not that the task ran: the block is
    usually still queued, and waiting for a result would wait out the queue. A
    cold uep refuses with 422 in one web round-trip, so an attempt that goes
    quiet for COLD_ENDPOINT_PROBE_WAIT has been accepted and queued -- that is
    the success signal.

    Raises if known startup races persist after COLD_ENDPOINT_TIMEOUT,
    which means it is failing to start rather than merely waking up. Any other
    error propagates immediately -- this must not become a general retry that
    hides a misconfiguration behind five minutes of silence.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + COLD_ENDPOINT_TIMEOUT
    delay, attempt = COLD_ENDPOINT_BACKOFF, 0
    while True:
        attempt += 1
        executor = make_executor()
        try:
            future = executor.submit(os.getpid)
            await asyncio.wait_for(asyncio.wrap_future(future),
                                   timeout=COLD_ENDPOINT_PROBE_WAIT)
            logger.info("user endpoint accepted work on attempt %d (task ran)", attempt)
            return
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, TimeoutError):
            # Accepted and queued: nothing refused us, there is just no worker
            # yet. Returning here is the whole point -- the queue wait belongs to
            # PBS, not to this probe.
            logger.info("user endpoint accepted work on attempt %d (task queued)", attempt)
            return
        except Exception as exc:  # noqa: BLE001 - re-raised below unless cold-start
            remaining = deadline - loop.time()
            if not _is_cold_endpoint(exc) or remaining <= 0:
                if _is_cold_endpoint(exc):
                    raise RuntimeError(
                        f"user endpoint still refused work with {_COLD_ENDPOINT_CODE} after "
                        f"{attempt} attempt(s) over {COLD_ENDPOINT_TIMEOUT:g}s; it is probably "
                        f"failing to start -- check the endpoint manager log"
                    ) from exc
                raise
            logger.info("user endpoint not ready yet (%s), attempt %d; retrying in %.0fs",
                        _COLD_ENDPOINT_CODE, attempt, min(delay, remaining))
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, max(COLD_ENDPOINT_TIMEOUT / 4, delay))
        finally:
            try:
                # wait=False: a blocking shutdown cannot drain a task whose
                # block has not started, and hung the process past SIGTERM.
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass


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
            if gc_executor is not None:
                gc_executor.shutdown(wait=False, cancel_futures=True)

    @asynccontextmanager
    async def _serve(factory, gc_executor, server: FastMCP):
        # The uep may have just been forked; prove it takes work before Academy
        # launches an agent, whose 422 would otherwise arrive as an opaque
        # AgentTerminatedError from the pool's first probe.
        if LAUNCH_MODE != "existing":
            await await_endpoint_ready()
        executor_name = "allocation" if LAUNCH_MODE == "existing" else "globus"
        if cluster and LAUNCH_MODE == "existing" and SYSTEM == "jupiter":
            # All local agents share this process. Publish registry updates
            # through one index instead of rescanning every JSON per lookup.
            enable_registry_index(BASE_DIR)
        async with await Manager.from_exchange_factory(factory=factory, executors={executor_name: gc_executor}, default_executor=executor_name) as manager:
            # In cluster mode EL owns placement and fanout, so the pool is a
            # fixed set of interchangeable submitters built once at startup;
            # the client agents are unpinned by design. Direct mode keeps
            # CampaignPool, which pins one accelerator per agent.
            pool = ClusterPool(CAMPAIGN_ID, POOL_SIZE, ckpt_dir) if cluster else CampaignPool(
                manager, HpcAgent, CAMPAIGN_ID, POOL_SIZE)

            # Globus mode delegates agents to the endpoint; existing mode
            # hosts async control agents here without another allocation.
            # EL owns scientific placement in both modes. Rebind the handle
            # to the exchange for probes throughout the server lifetime.
            async def launch(agent, **kwargs) -> Handle:
                handle = await manager.launch(agent, **kwargs)
                return Handle(handle.agent_id, exchange=manager.exchange_client)

            named = {"executor": executor_name, "name": f"{CAMPAIGN_ID}-agent"}

            async def launch_one() -> Handle:
                return await launch(HpcAgent(CAMPAIGN_ID, BASE_DIR, INSPECT_ROOTS), **named)

            async def launch_client(generation_ckpt_dir: str) -> Handle:
                return await launch(HpcClientAgent(
                    CAMPAIGN_ID, BASE_DIR, generation_ckpt_dir, INSPECT_ROOTS,
                    science_workers_per_gpu=SCIENCE_WORKERS_PER_GPU), **named)

            async def launch_cluster(generation_ckpt_dir: str) -> Handle:
                return await launch(HpcClusterAgent(BASE_DIR, generation_ckpt_dir, nnodes, SYSTEM,
                    science_workers_per_gpu=SCIENCE_WORKERS_PER_GPU))

            # Every tool forwards to the same-named action on the next pool
            # agent. Failures come back as ToolError so the harness gets a
            # readable message rather than an internal server error.
            async def dispatch(action: str, *args, **kwargs):
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action(action, *args, **kwargs)
                except Exception as exc:
                    logger.warning("HPC tool %s failed: %s: %s", action, type(exc).__name__, exc)
                    raise ToolError(f"{action} failed: {type(exc).__name__}: {exc}") from exc

            async def run_code(payload: str, workdir: str | None = None, timeout: float = 300.0, venv_path: str | None = None) -> dict[str, str]:
                """Run CPU preparation, polling and analysis on a campaign worker.

                GPU science uses run_batch. Use declared software installations;
                avoid recursive searches of project and scratch filesystems.
                Inference already runs at the service configured in campaign.yaml;
                preserve its model settings rather than installing or launching
                another model server.
                Write derived analysis in your run workspace; preserve registered-tool outputs.
                """
                return await dispatch("run_code", payload, workdir=workdir or BASE_DIR,
                                      timeout=timeout, venv_path=venv_path)

            async def inspect_filesystem(operation: str, path: str, offset: int = 1, limit: int = 100) -> dict:
                """Bounded read-only ls/cat/tail inspection on a pinned campaign worker."""
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
                                labels: dict | None = None, tool: str | None = None,
                                tool_version: int | None = None) -> dict:
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

                Prefer tool=<name> over script_path once a script is stable:
                register_tool freezes it with an `expects` output contract, and a
                batch submitted against a registered tool inherits that contract,
                so a row that exits 0 without writing its outputs is counted as
                failed instead of travelling downstream as a datapoint.
                """
                return await dispatch(
                    "run_batch", script_path=script_path, inputs_path=inputs_path,
                    timeout_each=timeout_each, gpus=gpus, venv_path=venv_path,
                    batch_id=batch_id, labels=labels, tool=tool,
                    tool_version=tool_version,
                )

            async def register_tool(name: str, script_path: str, params: dict,
                                    description: str, example: dict,
                                    defaults: dict | None = None,
                                    call: str = "json", venv_path: str | None = None,
                                    skip_smoke: bool = False, labels: dict | None = None,
                                    expects: list[str] | None = None) -> dict:
                """Freeze a script as the next version of a named tool, after one smoke run.

                `expects` is the output contract: filenames/globs, relative to each
                run's folder, that the script MUST produce. A run that exits 0 without
                them is counted as failed (L3) -- the only thing separating a row that
                did the work from one that wrote nothing. Declare it here ONCE; every
                batch submitted with tool=<name> inherits it. skip_smoke registers
                without the smoke run for scripts needing a GPU the client node may
                lack (recorded as tested=False).
                """
                return await dispatch(
                    "register_tool", name=name, script_path=script_path, params=params,
                    description=description, example=example, defaults=defaults,
                    call=call, venv_path=venv_path, skip_smoke=skip_smoke,
                    labels=labels, expects=expects,
                )

            async def list_tools(query: str | None = None, name: str | None = None,
                                 all_versions: bool = False, limit: int = 50) -> list[dict]:
                """Registered tools, newest version of each unless all_versions."""
                return await dispatch("list_tools", query=query, name=name,
                                      all_versions=all_versions, limit=limit)

            async def batch_status(batch_id: str) -> dict:
                """Progress of one run_batch batch: total, succeeded, failed, unfinished, and the first failures' stderr."""
                return await dispatch("batch_status", batch_id)

            async def list_batches(plan_id: str | None = None, hypothesis_id: str | None = None,
                                   limit: int = 50) -> list[dict]:
                """Batches in the registry, newest first, with their last known status."""
                return await dispatch("list_batches", plan_id=plan_id,
                                      hypothesis_id=hypothesis_id, limit=limit)


            async def pool_status() -> dict:
                """Live compute budget for sizing a run_batch fan-out: nodes, addressable GPU tiles, queue, walltime.

                Read this BEFORE launching a large batch so you fan out to the tiles
                you actually have rather than guessing. `total_tiles` counts physical
                devices; `concurrent_scientific_rows` includes bounded sharing. Each row must
                finish inside `walltime`, and on debug / debug-scaling queues the block
                is reclaimed about once an hour, so keep per-job work well under the
                hour. This reads local allocation config only -- it reports CAPACITY,
                not current free/busy; combine with list_batches / batch_status to see
                what is already in flight. Instant, never queues behind worker calls.
                """
                per_node = int(os.environ.get('HPC_GPUS_PER_NODE', {"aurora": 12, "polaris": 4, "jupiter": 4}.get(SYSTEM, 12)))
                total_tiles = (nnodes * per_node) if cluster else POOL_SIZE
                row_capacity = total_tiles * SCIENCE_WORKERS_PER_GPU if cluster else total_tiles
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
                    "science_workers_per_gpu": SCIENCE_WORKERS_PER_GPU,
                    "concurrent_scientific_rows": row_capacity,
                    "pool_size": POOL_SIZE,
                    "queue": pick("queue"),
                    "walltime": pick("walltime"),
                    "nodes_per_block": pick("nodes_per_block", "select"),
                    "account": pick("account", "project"),
                    "campaign_id": CAMPAIGN_ID,
                    "guidance": (
                        f"Up to {row_capacity} run_batch(gpus=1) rows run concurrently on {total_tiles} physical devices. "
                        "Size a fan-out to this; each job must finish "
                        "within walltime (debug/debug-scaling reclaim ~hourly). "
                        "This is capacity, not live free tiles -- check list_batches for in-flight work."
                    ),
                }

            # run_batch needs EL; the other tools are also useful in direct mode.
            tools = [run_code, write_files, inspect_filesystem, batch_status,
                     register_tool, list_tools,
                     list_batches, pool_status]
            if cluster:
                # Build the pool before serving: a tool call must never have to
                # launch or verify an agent, or concurrent calls serialize. The
                # pool brings the EL cluster up first, then the clients that
                # handshake with its checkpoint dir.
                await pool.start(launch_cluster, launch_client)
                tools.append(run_batch)
            for tool in tools:
                _add_tolerant(server, tool)
            try:
                yield
            finally:
                # Stop health checking before Manager.close() shuts the agents
                # down, or the reaper sees them die and relaunches into a
                # cluster that is itself on the way out.
                if cluster:
                    await pool.stop()

    return FastMCP(name="hpc-deployment", instructions=(
        "run_code executes CPU preparation, polling and analysis; GPU science uses run_batch. "
        "Use declared software installations rather than recursively searching project or scratch filesystems. "
        "Inference already runs at the service configured in campaign.yaml; preserve its model settings. "
        "Write derived analysis in your run workspace; preserve registered-tool outputs. "
        "write_files creates files on the cluster filesystem "
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
    transport = os.environ.get("HPC_MCP_TRANSPORT", "stdio")
    options = {}
    if transport == "http":
        options = {
            "host": os.environ.get("HPC_MCP_HOST", "127.0.0.1"),
            "port": int(os.environ.get("HPC_MCP_PORT", "8000")),
        }
    create_server(cluster=True, nnodes=int(os.environ.get("HPC_NNODES", "1"))).run(
        transport=transport,
        **options,
    )
