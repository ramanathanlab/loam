"""Campaign-scoped Academy-agent lifecycle for hpc-deployment."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass

from academy.handle import Handle

logger = logging.getLogger(__name__)


def _setting(value, env_name: str, default: str, cast=float):
    """An explicit constructor argument, else the environment, else the default."""
    return value if value is not None else cast(os.environ.get(env_name, default))


def _reraise_cancel(results: list) -> None:
    """gather(return_exceptions=True) swallows cancellation; re-raise it so stop() works."""
    for item in results:
        if isinstance(item, asyncio.CancelledError):
            raise item


# A multi-user endpoint forks the user endpoint on demand, and that child
# answers the first submit with 422 ENDPOINT_NOT_ONLINE (the SDK does not retry
# it). A cold first submit is normal for a not-yet-warm endpoint, so it must be
# retried rather than propagated; everything else still fails immediately.
_COLD_ENDPOINT_CODE = "ENDPOINT_NOT_ONLINE"


def _is_cold_endpoint(exc: BaseException) -> bool:
    """True only for the 422 a not-yet-ready user endpoint returns on first submit.

    Matched on ComputeAPIError.code with a substring fallback, so this module
    doesn't import globus_sdk just to name one error class.
    """
    if getattr(exc, "code", None) == _COLD_ENDPOINT_CODE:
        return True
    return _COLD_ENDPOINT_CODE in str(exc)


@dataclass
class AgentInfo:
    handle: Handle
    pin: str


class CampaignPool:
    """Discover/adopt only this campaign's agents; then launch a shortfall."""

    def __init__(self, manager, agent_class, campaign_id: str, want: int, timeout_s: float = 60.0,
                 startup_timeout_s: float | None = None, require_pin: bool = True):
        self.manager = manager
        self.agent_class = agent_class
        self.campaign_id = campaign_id
        self.want = want
        # Direct mode pins one accelerator per agent (unique non-empty pin
        # required); cluster mode lets EL own placement, so the agent has none.
        self.require_pin = require_pin
        # Liveness probes fail fast; a fresh launch gets a longer deadline to
        # clear scheduler wait and agent_on_startup.
        self.timeout_s = timeout_s
        self.startup_timeout_s = _setting(startup_timeout_s, "HPC_AGENT_STARTUP_TIMEOUT", "600")
        self._agents: list[AgentInfo] = []
        self._lock = asyncio.Lock()
        self._cursor = 0

    async def _probe(self, handle: Handle, timeout_s: float | None = None) -> AgentInfo | None:
        try:
            result = await asyncio.wait_for(handle.action("probe"), timeout=timeout_s or self.timeout_s)
            if result.get("campaign_id") != self.campaign_id:
                logger.warning("agent %s rejected: campaign_id=%r (want %r)",
                               handle.agent_id, result.get("campaign_id"), self.campaign_id)
                return None
            if self.require_pin and not result.get("device_pin"):
                logger.warning("agent %s rejected: empty device_pin; endpoint config must set "
                               "available_accelerators", handle.agent_id)
                return None
            return AgentInfo(handle, result.get("device_pin", ""))
        except Exception as exc:  # noqa: BLE001 - an unhealthy agent must never propagate
            logger.warning("probe of agent %s failed: %s: %s", handle.agent_id, type(exc).__name__, exc)
            return None

    async def _discover(self) -> list[AgentInfo]:
        ids = await self.manager.exchange_client.discover(self.agent_class, allow_subclasses=True)
        checks = [self._probe(Handle(agent_id, exchange=self.manager.exchange_client)) for agent_id in ids]
        return [item for item in await asyncio.gather(*checks) if item is not None]

    async def ensure(self, launch_one) -> list[AgentInfo]:
        async with self._lock:
            survivors = [item for item in await asyncio.gather(*(self._probe(a.handle) for a in self._agents)) if item]
            seen = {str(item.handle.agent_id) for item in survivors}
            for item in await self._discover():
                if str(item.handle.agent_id) not in seen:
                    survivors.append(item)
                    seen.add(str(item.handle.agent_id))
                if len(survivors) >= self.want:
                    break
            while len(survivors) < self.want:
                handle = await launch_one()
                checked = await self._probe(handle, timeout_s=self.startup_timeout_s)
                if checked is None:
                    raise RuntimeError(
                        "new Academy agent did not become healthy within "
                        f"{self.startup_timeout_s}s (see probe warnings above)"
                    )
                survivors.append(checked)
            pins = [agent.pin for agent in survivors]
            if self.require_pin and len(set(pins)) != len(pins):
                raise RuntimeError(f"agent device pins are not unique: {pins}")
            self._agents = survivors
            return survivors

    async def next_agent(self, launch_one) -> AgentInfo:
        pool = await self.ensure(launch_one)
        agent = pool[self._cursor % len(pool)]
        self._cursor += 1
        return agent


class ClusterUnrecoverable(RuntimeError):
    """The EL cluster could not be rebuilt within the allowed attempts."""


class ClusterPool:
    """Owns the EL cluster agent and the client agents that submit into it.

    In cluster mode EL owns placement and fanout, so the client agents are
    interchangeable submitters and hold no accelerator. The pool is built once
    by start() and handed out by next_agent() with no work per call; a
    background task re-probes it on an interval and repairs what died.

    The cluster agent defines what "dead" means: its shutdown stops the EL
    cluster, so losing it leaves every client bound to a cluster that no longer
    exists. A lost cluster therefore replaces the whole pool; a lost client is
    replaced on its own. Repairing here (not in the server) keeps the MCP server
    and its sessions up across a cluster restart.

    Unlike CampaignPool, repair runs on a timer rather than inside next_agent(),
    keeping it off the request path, and there is no adoption: Manager.close()
    shuts down every agent it launched. next_agent() keeps CampaignPool's async
    signature so the tool bodies work with either pool; launch_one is ignored.
    """

    def __init__(self, campaign_id: str, want: int, ckpt_dir: str | None = None,
                 startup_timeout_s: float | None = None, probe_interval_s: float | None = None,
                 probe_timeout_s: float | None = None, cluster_timeout_s: float | None = None,
                 cluster_grace: int | None = None, max_restarts: int | None = None,
                 wait_timeout_s: float | None = None, client_grace: int | None = None,
                 cold_endpoint_timeout_s: float | None = None,
                 cold_endpoint_backoff_s: float | None = None):
        self.campaign_id = campaign_id
        self.want = want
        # How long to retry a cold-endpoint (ENDPOINT_NOT_ONLINE) submit, plus
        # the initial backoff (doubles, capped at a quarter of the window).
        # Covers uep fork/registration only, not the scheduler's queue.
        self.cold_endpoint_timeout_s = _setting(
            cold_endpoint_timeout_s, "HPC_COLD_ENDPOINT_TIMEOUT", "300")
        self.cold_endpoint_backoff_s = _setting(
            cold_endpoint_backoff_s, "HPC_COLD_ENDPOINT_BACKOFF", "5")
        # A new client has to clear scheduler wait and agent_on_startup (the EL
        # ClusterClient checkpoint handshake) before it answers a probe.
        self.startup_timeout_s = _setting(startup_timeout_s, "HPC_AGENT_STARTUP_TIMEOUT", "600")
        # The cluster agent's first probe waits out the scheduler's queue, not just
        # process startup, so it gets a much larger budget than anything else.
        self.cluster_timeout_s = _setting(cluster_timeout_s, "HPC_CLUSTER_STARTUP_TIMEOUT", "3000")
        self.probe_interval_s = _setting(probe_interval_s, "HPC_POOL_PROBE_INTERVAL", "60")
        # Liveness probes must fail fast; only a cold launch gets the long one.
        self.probe_timeout_s = _setting(probe_timeout_s, "HPC_POOL_PROBE_TIMEOUT", "180")
        # Consecutive strikes before a rebuild: tearing down a live cluster over
        # one dropped probe costs a restart and maybe a fresh allocation.
        self.cluster_grace = _setting(cluster_grace, "HPC_CLUSTER_PROBE_STRIKES", "3", int)
        # Clients get strikes too: a probe can fail on a healthy agent (a severed
        # listen stream or a 5xx burst), and replacing one on a single miss piles
        # relaunch load onto the exchange exactly while it's struggling.
        self.client_grace = _setting(client_grace, "HPC_CLIENT_PROBE_STRIKES", "3", int)
        # Consecutive failed rebuilds before self-healing gives up for good.
        self.max_restarts = _setting(max_restarts, "HPC_CLUSTER_MAX_RESTARTS", "3", int)
        # How long a tool call waits for a restart. Unset means wait as long as
        # it takes and let the MCP client's own timeout be the bound.
        if wait_timeout_s is None:
            raw = os.environ.get("HPC_POOL_WAIT_TIMEOUT", "")
            wait_timeout_s = float(raw) if raw else None
        self.wait_timeout_s = wait_timeout_s

        self._cluster: AgentInfo | None = None
        self._agents: list[AgentInfo] = []
        self._cursor = 0
        self._started = False
        self._failed = False
        self._cluster_misses = 0
        # Consecutive failed probes per client agent id; cleared on any success.
        self._client_misses: dict[str, int] = {}
        self._restart_failures = 0
        # Caller-supplied dir seeds generation 0; later generations mint their
        # own so a new ClusterClient can't bind the dead master's record. Old
        # dirs are left as-is (remote HPC storage, and evidence of the failure).
        self._seed_ckpt_dir = ckpt_dir
        self._ckpt_dir: str | None = None
        self._launch_cluster = None
        self._launch_one = None
        self._reaper: asyncio.Task | None = None
        # Clear while the cluster is being rebuilt; tool calls wait on it
        # rather than failing, so a restart is invisible to a patient caller.
        self._ready = asyncio.Event()

    # -- probing ---------------------------------------------------------

    async def _probe(self, agent: AgentInfo, check_campaign: bool = True) -> AgentInfo | None:
        try:
            result = await asyncio.wait_for(
                agent.handle.action("probe"), timeout=self.probe_timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - an unhealthy agent must never propagate
            logger.warning("agent %s failed liveness probe: %s: %s",
                           agent.handle.agent_id, type(exc).__name__, exc)
            return None
        # HpcClusterAgent.probe reports pid/device_pin only; the campaign check
        # applies to client agents.
        if check_campaign and result.get("campaign_id") != self.campaign_id:
            logger.warning("agent %s rejected: campaign_id=%r (want %r)",
                           agent.handle.agent_id, result.get("campaign_id"), self.campaign_id)
            return None
        return agent

    # -- launching -------------------------------------------------------

    def _next_ckpt_dir(self) -> str:
        if self._seed_ckpt_dir is not None:
            seeded, self._seed_ckpt_dir = self._seed_ckpt_dir, None
            return seeded
        return f"ckpt_{uuid.uuid4().hex}"

    async def _launch_cluster_warm(self, ckpt_dir: str):
        """`_launch_cluster`, retrying only while the user endpoint is still cold.

        Bounded by cold_endpoint_timeout_s. Any other failure, and cancellation,
        propagate on the first attempt rather than masking a misconfiguration.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.cold_endpoint_timeout_s
        delay = self.cold_endpoint_backoff_s
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._launch_cluster(ckpt_dir)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - re-raised below unless cold-start
                remaining = deadline - loop.time()
                if not _is_cold_endpoint(exc) or remaining <= 0:
                    if _is_cold_endpoint(exc):
                        raise RuntimeError(
                            f'user endpoint still refused tasks with {_COLD_ENDPOINT_CODE} after '
                            f'{attempt} attempt(s) over {self.cold_endpoint_timeout_s:g}s; it may '
                            f'be failing to start -- check the endpoint manager log'
                        ) from exc
                    raise
                wait = min(delay, remaining)
                logger.info(
                    "user endpoint not ready yet (%s), attempt %d; retrying in %.1fs",
                    _COLD_ENDPOINT_CODE, attempt, wait,
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, max(self.cold_endpoint_timeout_s / 4, delay))

    async def _start_cluster(self) -> None:
        self._ckpt_dir = self._next_ckpt_dir()
        handle = await self._launch_cluster_warm(self._ckpt_dir)
        try:
            result = await asyncio.wait_for(
                handle.action("probe"), timeout=self.cluster_timeout_s
            )
        except BaseException:
            try:
                await asyncio.wait_for(handle.shutdown(), timeout=self.probe_timeout_s)
            except BaseException:
                logger.debug("cleanup of failed cluster launch %s also failed", handle.agent_id)
            raise
        self._cluster = AgentInfo(handle, result.get("device_pin", ""))
        self._cluster_misses = 0
        logger.info("EL cluster agent %s up (pid %s, ckpt_dir=%s)",
                    handle.agent_id, result.get("pid"), self._ckpt_dir)

    async def _launch_client(self) -> AgentInfo:
        handle = await self._launch_one(self._ckpt_dir)
        try:
            result = await asyncio.wait_for(handle.action("probe"), timeout=self.startup_timeout_s)
            if result.get("campaign_id") != self.campaign_id:
                raise RuntimeError(
                    f"agent {handle.agent_id} reports campaign_id="
                    f"{result.get('campaign_id')!r} (want {self.campaign_id!r})"
                )
            # EL assigns GPUs to tasks; the client agent is unpinned by design.
            return AgentInfo(handle, result.get("device_pin", ""))
        except BaseException:
            try:
                await asyncio.wait_for(handle.shutdown(), timeout=self.probe_timeout_s)
            except BaseException:
                logger.debug("cleanup of failed client launch %s also failed", handle.agent_id)
            raise

    async def _launch_clients(self, count: int) -> list[AgentInfo]:
        launched = await asyncio.gather(
            *(self._launch_client() for _ in range(count)), return_exceptions=True,
        )
        _reraise_cancel(launched)
        for item in launched:
            if isinstance(item, BaseException):
                logger.error("client agent launch failed: %s: %s", type(item).__name__, item)
        return [item for item in launched if isinstance(item, AgentInfo)]

    async def _shutdown_all(self, agents: list[AgentInfo]) -> None:
        """Release the worker slots the replacements are about to need.

        The endpoint is sized at one slot per client (+1 for the cluster), so
        stale agents left running leave the new ones nowhere to land. Best
        effort: the agent being replaced usually already stopped answering.
        """
        if not agents:
            return
        logger.info("shutting down %d stale agent(s) to free worker slots", len(agents))

        async def one(agent: AgentInfo) -> None:
            try:
                await asyncio.wait_for(agent.handle.shutdown(), timeout=self.probe_timeout_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - it is probably already gone
                logger.debug("shutdown of %s failed: %s: %s",
                             agent.handle.agent_id, type(exc).__name__, exc)

        _reraise_cancel(await asyncio.gather(*(one(a) for a in agents), return_exceptions=True))

    # -- repair ----------------------------------------------------------

    async def _restart_cluster(self) -> None:
        """Rebuild the cluster and every client that was bound to the old one."""
        logger.error("EL cluster agent lost after %d probes; rebuilding cluster and "
                     "%d client agent(s)", self.cluster_grace, len(self._agents))
        self._ready.clear()
        stale = ([self._cluster] if self._cluster is not None else []) + self._agents
        # Stop serving agents wired to the dead cluster before anything slow.
        self._cluster = None
        self._agents = []
        self._client_misses.clear()
        await self._shutdown_all(stale)
        try:
            await self._start_cluster()
            self._agents = await self._launch_clients(self.want)
            if len(self._agents) != self.want:
                raise RuntimeError(
                    f"only {len(self._agents)} of {self.want} client agents came up "
                    "against the new EL cluster"
                )
        except asyncio.CancelledError:
            partial = ([self._cluster] if self._cluster is not None else []) + self._agents
            self._cluster = None
            self._agents = []
            await self._shutdown_all(partial)
            raise
        except Exception as exc:  # noqa: BLE001 - decide whether to keep healing
            partial = ([self._cluster] if self._cluster is not None else []) + self._agents
            self._cluster = None
            self._agents = []
            await self._shutdown_all(partial)
            self._restart_failures += 1
            logger.error("cluster restart attempt %d/%d failed: %s: %s",
                         self._restart_failures, self.max_restarts, type(exc).__name__, exc)
            if self._restart_failures >= self.max_restarts:
                self._failed = True
                # Release anyone waiting; they get the terminal error instead.
                self._ready.set()
                raise ClusterUnrecoverable(
                    f"EL cluster unrecoverable after {self.max_restarts} restart attempts"
                ) from exc
            return  # try again on the next sweep
        self._restart_failures = 0
        self._ready.set()
        logger.info("cluster rebuilt: %d client agent(s), ckpt_dir=%s",
                    len(self._agents), self._ckpt_dir)

    async def _repair(self) -> None:
        if self._cluster is None or await self._probe(self._cluster, check_campaign=False) is None:
            self._cluster_misses += 1
            logger.warning("EL cluster agent probe failed [%d/%d]",
                           self._cluster_misses, self.cluster_grace)
            if self._cluster_misses >= self.cluster_grace:
                await self._restart_cluster()
            return
        self._cluster_misses = 0
        try:
            await self._repair_clients()
        finally:
            # A rebuild that brought the cluster up but no clients returns with
            # _ready cleared; once a sweep restores clients, release waiters.
            if self._agents and not self._ready.is_set():
                self._restart_failures = 0
                self._ready.set()

    async def _repair_clients(self) -> None:
        probed = await asyncio.gather(*(self._probe(agent) for agent in self._agents))
        survivors, dropped = [], []
        for agent, ok in zip(self._agents, probed):
            key = str(agent.handle.agent_id)
            if ok is not None:
                self._client_misses.pop(key, None)
                survivors.append(agent)
                continue
            misses = self._client_misses.get(key, 0) + 1
            if misses < self.client_grace:
                # Keep serving it: a dead client costs a few failed tool calls
                # until its strikes run out; a false drop costs a relaunch.
                self._client_misses[key] = misses
                logger.warning("client agent %s probe failed [%d/%d]",
                               key, misses, self.client_grace)
                survivors.append(agent)
            else:
                self._client_misses.pop(key, None)
                dropped.append(agent)
        if len(survivors) < len(self._agents):
            # Publish survivors before relaunching: a dead agent must stop
            # being handed out even if its replacement cannot be launched.
            logger.warning("cluster pool lost %d client agent(s)",
                           len(self._agents) - len(survivors))
            self._agents = survivors
        shortfall = self.want - len(survivors)
        if shortfall <= 0:
            return
        # Free the dropped agents' slots BEFORE launching replacements: a hung
        # client still holds its slot, and with one slot per client the
        # replacement would otherwise block the whole sweep for startup_timeout_s.
        if dropped:
            await self._shutdown_all(dropped)
        fresh = await self._launch_clients(shortfall)
        if fresh:
            self._agents = survivors + fresh
            logger.info("cluster pool restored to %d client agent(s)", len(self._agents))

    async def _repair_loop(self) -> None:
        while True:
            await asyncio.sleep(self.probe_interval_s)
            try:
                await self._repair()
            except asyncio.CancelledError:
                raise
            except ClusterUnrecoverable as exc:
                logger.error("%s; self-healing stopped. The MCP server stays up and every "
                             "tool call will report this.", exc)
                return
            except Exception:  # noqa: BLE001 - one bad sweep must not end health checking
                logger.exception("cluster pool repair sweep failed; retrying next interval")

    # -- lifecycle -------------------------------------------------------

    async def start(self, launch_cluster, launch_client) -> list[AgentInfo]:
        """Bring up the complete cluster/client pool, then start health checks."""
        self._launch_cluster = launch_cluster
        self._launch_one = launch_client
        try:
            await self._start_cluster()
            self._agents = await self._launch_clients(self.want)
            if len(self._agents) != self.want:
                raise RuntimeError(
                    f"only {len(self._agents)} of {self.want} client agents could be launched"
                )
        except BaseException:
            partial = ([self._cluster] if self._cluster is not None else []) + self._agents
            self._cluster = None
            self._agents = []
            await self._shutdown_all(partial)
            raise
        self._started = True
        self._ready.set()
        self._reaper = asyncio.create_task(self._repair_loop(), name="cluster-pool-repair")
        logger.info("campaign %s cluster pool ready: %d client agent(s), probing every %ss",
                    self.campaign_id, len(self._agents), self.probe_interval_s)
        return self._agents

    async def stop(self) -> None:
        """Stop health checking. Must run before the Manager shuts agents down,
        or the reaper races the shutdown and rebuilds what it just lost."""
        if self._reaper is None:
            return
        self._reaper.cancel()
        try:
            await self._reaper
        except asyncio.CancelledError:
            pass
        self._reaper = None

    async def next_agent(self, launch_one=None) -> AgentInfo:
        if not self._started:
            raise RuntimeError("ClusterPool.start() has not run")
        if not self._ready.is_set():
            logger.info("tool call waiting for EL cluster rebuild")
            if self.wait_timeout_s is None:
                await self._ready.wait()
            else:
                await asyncio.wait_for(self._ready.wait(), timeout=self.wait_timeout_s)
        if self._failed:
            raise RuntimeError(
                f"EL cluster unrecoverable after {self.max_restarts} restart attempts; "
                "self-healing has stopped and the server must be restarted"
            )
        if not self._agents:
            raise RuntimeError("cluster pool has no healthy client agents")
        agent = self._agents[self._cursor % len(self._agents)]
        self._cursor += 1
        return agent
