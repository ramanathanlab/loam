"""Campaign-scoped Academy-agent lifecycle for hpc-deployment."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from academy.handle import Handle

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    handle: Handle
    pin: str


class CampaignPool:
    """Discover/adopt only this campaign's agents; then launch a shortfall."""

    def __init__(self, manager, agent_class, campaign_id: str, want: int, timeout_s: float = 60.0,
                 startup_timeout_s: float | None = None):
        self.manager = manager
        self.agent_class = agent_class
        self.campaign_id = campaign_id
        self.want = want
        # Liveness probes of pool members must fail fast. A freshly launched
        # agent instead has to clear scheduler wait and agent_on_startup (the
        # EL ClusterClient checkpoint handshake), so it gets a longer deadline.
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s if startup_timeout_s is not None else float(
            os.environ.get("HPC_AGENT_STARTUP_TIMEOUT", "600")
        )
        self._agents: list[AgentInfo] = []
        self._lock = asyncio.Lock()
        self._cursor = 0

    async def _probe(self, handle: Handle, timeout_s: float | None = None) -> AgentInfo | None:
        try:
            result = await asyncio.wait_for(handle.action("probe"), timeout=timeout_s or self.timeout_s)
            if result.get("campaign_id") != self.campaign_id or not result.get("device_pin"):
                logger.warning("agent %s rejected: campaign_id=%r device_pin=%r",
                               handle.agent_id, result.get("campaign_id"), result.get("device_pin"))
                return None
            return AgentInfo(handle, result["device_pin"])
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
            if len(set(pins)) != len(pins):
                raise RuntimeError(f"agent device pins are not unique: {pins}")
            self._agents = survivors
            return survivors

    async def next_agent(self, launch_one) -> AgentInfo:
        pool = await self.ensure(launch_one)
        agent = pool[self._cursor % len(pool)]
        self._cursor += 1
        return agent
