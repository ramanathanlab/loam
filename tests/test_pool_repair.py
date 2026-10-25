"""ClusterPool self-repair ordering (M8): a dropped client is shut down to free
its worker slot BEFORE its replacement is launched, so a hung-but-alive client
can't make the relaunch block the whole repair sweep."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "hpc-deployment" / "scripts"
HAVE_DEPS = importlib.util.find_spec("academy") is not None


@unittest.skipUnless(HAVE_DEPS, "academy not installed")
class ClusterRepairOrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import pool_manager
        self.pm = pool_manager

    def tearDown(self) -> None:
        sys.path.remove(str(SCRIPTS))

    def test_dropped_client_is_shut_down_before_relaunch(self) -> None:
        pm = self.pm

        class FakeHandle:
            def __init__(self, name: str) -> None:
                self.agent_id = name

        pool = pm.ClusterPool("camp", want=2)
        cluster_marker = object()
        pool._cluster = cluster_marker
        good = pm.AgentInfo(FakeHandle("good"), "")
        bad = pm.AgentInfo(FakeHandle("bad"), "")
        pool._agents = [good, bad]
        events: list[tuple] = []

        async def fake_probe(agent, check_campaign=True):
            if agent is cluster_marker:   # cluster stays healthy
                return agent
            return agent if agent is good else None   # 'bad' client is dropped

        async def fake_shutdown_all(agents):
            events.append(("shutdown", tuple(a.handle.agent_id for a in agents)))

        async def fake_launch_clients(n):
            events.append(("launch", n))
            return [pm.AgentInfo(FakeHandle(f"new{i}"), "") for i in range(n)]

        pool._probe = fake_probe
        pool._shutdown_all = fake_shutdown_all
        pool._launch_clients = fake_launch_clients

        asyncio.run(pool._repair())

        # The dropped client's slot is freed first, then the replacement launches.
        self.assertEqual(events[0], ("shutdown", ("bad",)))
        self.assertEqual(events[1], ("launch", 1))

    def test_repair_releases_waiters_once_clients_are_restored(self) -> None:
        # H1: a rebuild that brought the cluster up but launched no clients left
        # _ready cleared. The next sweep restored the clients through the normal
        # shortfall path without touching _ready, so next_agent() waited forever
        # on a fully healthy pool.
        pm = self.pm

        class FakeHandle:
            def __init__(self, name: str) -> None:
                self.agent_id = name

        pool = pm.ClusterPool("camp", want=1)
        cluster_marker = object()
        pool._cluster = cluster_marker
        pool._agents = []
        pool._ready.clear()
        pool._restart_failures = 1

        async def fake_probe(agent, check_campaign=True):
            return agent

        async def fake_launch_clients(n):
            return [pm.AgentInfo(FakeHandle(f"new{i}"), "") for i in range(n)]

        pool._probe = fake_probe
        pool._launch_clients = fake_launch_clients

        asyncio.run(pool._repair())

        self.assertEqual(len(pool._agents), 1)
        self.assertTrue(pool._ready.is_set())
        self.assertEqual(pool._restart_failures, 0)


if __name__ == "__main__":
    unittest.main()
