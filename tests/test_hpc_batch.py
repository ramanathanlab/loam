"""run_batch / batch_status / list_batches on disk, with a fake EL client.

Needs the hpc-deployment dependencies (academy-py, aiohttp); EL itself is not
required: Task and the ClusterClient are replaced by stand-ins.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "hpc-deployment" / "scripts"
HAVE_DEPS = all(importlib.util.find_spec(name) for name in ("academy", "aiohttp"))

SIM = """\
import json, sys
row = json.load(open(sys.argv[1]))
if row.get("fail"):
    print("boom", file=sys.stderr)
    sys.exit(3)
json.dump({"x2": row["x"] * 2}, open("result.json", "w"))
"""


@dataclass
class FakeTask:
    task_id: str
    nnodes: int
    ppn: int
    ngpus_per_process: float
    executable: object
    args: tuple
    kwargs: dict = field(default_factory=dict)


class FakeClient:
    """Records submitted tasks; run() executes them in-process like EL would.

    Its lifecycle method is teardown() with NO stop(), mirroring the real EL
    ClusterClient API, so a regression to self._client.stop() in
    agent_on_shutdown fails loudly here instead of only in production.
    """

    def __init__(self) -> None:
        self.submitted: list[FakeTask] = []
        self.torn_down = False

    def submit_batch(self, tasks):
        self.submitted.extend(tasks)
        return [object() for _ in tasks]

    def run(self, only=None) -> None:
        for task in self.submitted[:only]:
            task.executable(*task.args)
        self.submitted.clear()

    def teardown(self) -> None:
        self.torn_down = True


@unittest.skipUnless(HAVE_DEPS, "hpc-deployment dependencies not installed")
class RunBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import hpc_agent

        self.hpc = hpc_agent
        self.hpc.Task = FakeTask
        # These tests exercise freezing, resume, leases and status -- not batch
        # WIDTH -- and the fixture below deliberately uses four rows. The
        # production floor (_MIN_BATCH_ROWS, default 32) would refuse every one
        # of them, so pin it to 1 here and let BatchWidthFloorTests own the
        # floor's behaviour. Restored in tearDown so it cannot leak.
        self._floor = hpc_agent._MIN_BATCH_ROWS
        hpc_agent._MIN_BATCH_ROWS = 1
        self.base = Path(tempfile.mkdtemp())
        os.environ.pop("HPC_BATCH_ROOT", None)
        (self.base / "sim.py").write_text(SIM)
        self.inputs = self.base / "inputs.jsonl"
        self.inputs.write_text("".join(json.dumps(row) + "\n" for row in (
            {"id": "a", "x": 1}, {"id": "b", "x": 2}, {"id": "c", "x": 3, "fail": True}, {"x": 4},
        )))
        self.agent = hpc_agent.HpcClientAgent("campaign", str(self.base), "ckpt")
        self.client = FakeClient()
        self.agent._client = self.client
        self.labels = {"plan_id": "P1", "hypothesis_id": "H7"}

    def tearDown(self) -> None:
        self.hpc._MIN_BATCH_ROWS = self._floor
        sys.path.remove(str(SCRIPTS))

    def _call(self, action_name, /, *args, **kwargs):
        # Academy's @action wraps the coroutine; call the undecorated function.
        method = getattr(type(self.agent), action_name)
        method = getattr(method, "__wrapped__", method)
        return asyncio.run(method(self.agent, *args, **kwargs))

    def _new_batch(self):
        return self._call("run_batch", script_path=str(self.base / "sim.py"),
                          inputs_path=str(self.inputs), labels=self.labels)

    def test_agent_on_shutdown_tears_down_the_client(self) -> None:
        # agent_on_shutdown must release the ClusterClient via its real API,
        # teardown(). FakeClient exposes teardown() and NO stop(), so a
        # regression to self._client.stop() raises AttributeError here (the real
        # EL ClusterClient has no stop(), and the normal suite runs without EL).
        self.assertFalse(self.client.torn_down)
        asyncio.run(self.agent.agent_on_shutdown())
        self.assertTrue(self.client.torn_down)

    def test_new_batch_freezes_inputs_and_registers(self) -> None:
        result = self._new_batch()
        batch_dir = self.base / "batches" / result["batch_id"]
        self.assertEqual((result["total"], result["submitted"], result["already_done"]), (4, 4, 0))
        self.assertEqual(len(self.client.submitted), 4)
        self.assertEqual({t.ngpus_per_process for t in self.client.submitted}, {1})
        self.assertEqual((batch_dir / "inputs.jsonl").read_text(), self.inputs.read_text())
        entry = json.loads((self.base / "batches" / "registry" / f"{result['batch_id']}.json").read_text())
        self.assertEqual(entry["labels"], self.labels)
        self.assertEqual(len(entry["inputs_sha256"]), 64)

    def test_status_and_resume_use_frozen_copy(self) -> None:
        batch_id = self._new_batch()["batch_id"]
        self.client.run(only=3)  # a, b succeed; c fails; row 4 never runs
        self.inputs.write_text('{"id": "z", "x": 9}\n')  # source edited after submission

        status = self._call("batch_status", batch_id)
        self.assertEqual((status["succeeded"], status["failed"], status["unfinished"]), (2, 1, 1))
        self.assertEqual(status["failures"][0]["run"], "run_c")
        self.assertIn("boom", status["failures"][0]["stderr_tail"])
        self.assertEqual(
            json.loads((self.base / "batches" / batch_id / "run_a" / "result.json").read_text()), {"x2": 2})

        resumed = self._call("run_batch", batch_id=batch_id, labels=self.labels)
        self.assertEqual((resumed["submitted"], resumed["already_done"]), (2, 2))
        self.assertEqual(sorted(t.task_id for t in self.client.submitted),
                         [f"{batch_id}-4", f"{batch_id}-c"])

    def test_resume_keeps_frozen_gpus_and_timeout(self) -> None:
        # H4: a batch submitted with gpus=2 / timeout=1800 must resume with the
        # same, not revert to the defaults (gpus=1 / timeout=900).
        batch_id = self._call("run_batch", script_path=str(self.base / "sim.py"),
                              inputs_path=str(self.inputs), labels=self.labels,
                              gpus=2, timeout_each=1800.0)["batch_id"]
        self.assertEqual({t.ngpus_per_process for t in self.client.submitted}, {2})
        self.assertEqual({t.args[3] for t in self.client.submitted}, {1800.0})
        self.client.run(only=1)  # finish run a
        self.client.submitted.clear()
        # Bare resume: nothing passed -> frozen values, not the call defaults.
        self._call("run_batch", batch_id=batch_id, labels=self.labels)
        self.assertTrue(self.client.submitted)
        self.assertEqual({t.ngpus_per_process for t in self.client.submitted}, {2})
        self.assertEqual({t.args[3] for t in self.client.submitted}, {1800.0})

    def test_complete_is_true_but_ok_false_when_a_run_failed(self) -> None:
        # H3: a batch where a run crashed is complete (all terminal) but not ok.
        self.inputs.write_text('{"id": "a", "x": 1}\n{"id": "b", "x": 2, "fail": true}\n')
        batch_id = self._new_batch()["batch_id"]
        self.client.run()
        status = self._call("batch_status", batch_id)
        self.assertEqual((status["succeeded"], status["failed"], status["unfinished"]), (1, 1, 0))
        self.assertTrue(status["complete"])
        self.assertFalse(status["ok"])

    def test_failed_attempt_does_not_leave_result_at_top_level(self) -> None:
        # M6: a failed run must not promote a result artifact to the run-folder
        # top level; a later successful attempt promotes cleanly.
        folder = self.base / "runM6"
        # Writes result.json in its cwd, then fails unless an ok_flag exists one
        # level up (so attempt 1 fails, attempt 2 succeeds).
        script = self.base / "flaky.py"
        script.write_text(
            "import json, os, sys\n"
            "json.dump({'junk': True}, open('result.json', 'w'))\n"
            "sys.exit(0 if os.path.exists(os.path.join(os.pardir, 'ok_flag')) else 3)\n")
        out1 = self.hpc._run_one(str(script), {"x": 1}, str(folder), 5.0, expects=["result.json"])
        self.assertNotEqual(out1["returncode"], 0)
        self.assertFalse((folder / "result.json").exists())           # not promoted
        self.assertTrue((folder / "attempt_1" / "result.json").exists())  # kept for debugging
        (folder / "ok_flag").write_text("")  # make attempt 2 succeed
        out2 = self.hpc._run_one(str(script), {"x": 1}, str(folder), 5.0, expects=["result.json"])
        self.assertEqual((out2["returncode"], out2["attempt"]), (0, 2))
        self.assertTrue((folder / "result.json").exists())            # promoted

    def test_resume_refuses_other_hypothesis(self) -> None:
        batch_id = self._new_batch()["batch_id"]
        with self.assertRaisesRegex(ValueError, "another hypothesis_id"):
            self._call("run_batch", batch_id=batch_id,
                       labels={"plan_id": "P1", "hypothesis_id": "H9"})

    def test_resume_skips_in_flight_lease_and_reclaims_stale(self) -> None:
        # C3: a resume overlapping an in-flight row must not re-dispatch it.
        import time

        batch_id = self._new_batch()["batch_id"]
        batch_dir = self.base / "batches" / batch_id
        self.client.run(only=2)  # a, b finish cleanly; c and row 4 remain todo
        # Simulate run_c as still running: folder + fresh lease, no done file.
        run_c = batch_dir / "run_c"
        run_c.mkdir()
        run_c.joinpath("lease.json").write_text(
            json.dumps({"pid": 999999, "host": "other", "started_at": time.time()})
        )

        resumed = self._call("run_batch", batch_id=batch_id, labels=self.labels)
        # Only row 4 is re-dispatched; the in-flight c is left to its owner.
        self.assertEqual([t.task_id for t in self.client.submitted], [f"{batch_id}-4"])
        self.assertEqual(resumed["submitted"], 1)

        # Age the lease past timeout + margin -> c is now reclaimable on resume.
        self.client.submitted.clear()
        frozen = json.loads((batch_dir.parent / "registry" / f"{batch_id}.json").read_text())
        run_c.joinpath("lease.json").write_text(json.dumps({
            "pid": 999999, "host": "other",
            "started_at": time.time() - (frozen["timeout_each"] + self.hpc.LEASE_STALE_MARGIN + 10),
        }))
        self._call("run_batch", batch_id=batch_id, labels=self.labels)
        self.assertIn(f"{batch_id}-c", [t.task_id for t in self.client.submitted])

    def test_run_one_backs_off_when_folder_is_leased(self) -> None:
        # C3: a second task started on a live-leased folder must not clobber it.
        import time

        folder = self.base / "leased"
        folder.mkdir()
        folder.joinpath("lease.json").write_text(
            json.dumps({"pid": 1, "host": "other", "started_at": time.time()})
        )
        folder.joinpath("stdout.txt").write_text("PRIOR")  # the owner's output
        out = self.hpc._run_one("/nonexistent/script.py", {"x": 1}, str(folder), 5.0)
        self.assertIn("leased", out.get("skipped", ""))
        self.assertEqual(folder.joinpath("stdout.txt").read_text(), "PRIOR")  # preserved
        self.assertFalse(folder.joinpath("done").exists())

    def test_list_batches_filters_by_labels(self) -> None:
        batch_id = self._new_batch()["batch_id"]
        self.assertEqual([b["batch_id"] for b in self._call("list_batches", hypothesis_id="H7")], [batch_id])
        self.assertEqual(self._call("list_batches", hypothesis_id="H9"), [])

    def test_list_batches_survives_a_corrupt_entry_and_a_missing_created_at(self) -> None:
        # H1: an unguarded json.loads broke every listing permanently, and a
        # pre-`created_at` entry made the sort raise on None.
        batch_id = self._new_batch()["batch_id"]
        registry = self.base / "batches" / "registry"
        (registry / "torn.json").write_text("{not json")
        legacy = json.loads((registry / f"{batch_id}.json").read_text())
        legacy.pop("created_at")
        (registry / "legacy.json").write_text(json.dumps(legacy))
        listed = self._call("list_batches")
        self.assertEqual(sorted(b["batch_id"] for b in listed), [batch_id, batch_id])

    def test_batch_status_tolerates_a_missing_stderr_file(self) -> None:
        # H1: an unguarded whole-file read aborted the whole batch's status.
        self.inputs.write_text('{"id": "a", "x": 1, "fail": true}\n')
        batch_id = self._new_batch()["batch_id"]
        self.client.run()
        (self.base / "batches" / batch_id / "run_a" / "stderr.txt").unlink()
        status = self._call("batch_status", batch_id)
        self.assertEqual((status["failed"], status["failures"][0]["stderr_tail"]), (1, []))

    def test_batch_id_cannot_escape_the_batch_root(self) -> None:
        # H1: "." and ".." passed the id check and were joined onto the root.
        for bad in (".", ".."):
            with self.assertRaisesRegex(ValueError, "bad batch_id"):
                self._call("run_batch", script_path=str(self.base / "sim.py"),
                           inputs_path=str(self.inputs), batch_id=bad)
            with self.assertRaisesRegex(ValueError, "bad batch_id"):
                self._call("batch_status", bad)

    def test_rejects_paths_outside_roots_and_bad_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            self._call("run_batch", script_path="/etc/hosts", inputs_path=str(self.inputs))
        self.inputs.write_text('{"id": "a"}\n{"id": "a"}\n')
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            self._new_batch()
        self.assertFalse((self.base / "batches" / "registry").exists()
                         and any((self.base / "batches" / "registry").iterdir()))

    # -- tool library --------------------------------------------------------

    def _register(self, script="sim.py", **overrides):
        args = dict(name="doubler", script_path=str(self.base / script),
                    params={"x": "int", "fail": "bool"}, defaults={"fail": False},
                    description="Doubles x.", example={"x": 5}, labels=self.labels)
        args.update(overrides)
        return self._call("register_tool", **args)

    def test_register_tool_freezes_smoke_tests_and_dedupes(self) -> None:
        first = self._register()
        self.assertEqual((first["version"], first["tested"], first["existing"]), (1, True, False))
        version_dir = self.base / "tools" / "doubler" / "v1"
        self.assertEqual(json.loads((version_dir / "smoke" / "result.json").read_text()), {"x2": 10})
        self.assertEqual(self._register()["existing"], True)
        (self.base / "sim.py").write_text(SIM + "\n# changed\n")
        self.assertEqual(self._register()["version"], 2)

    def test_register_tool_refuses_failing_smoke_and_bad_specs(self) -> None:
        with self.assertRaisesRegex(ValueError, "boom"):
            self._register(example={"x": 1, "fail": True})
        self.assertFalse((self.base / "tools" / "doubler" / "v1" / "tool.json").exists())
        self.assertEqual(self._register()["version"], 2)  # v1 stays an unregistered leftover
        with self.assertRaisesRegex(ValueError, "type must be"):
            self._register(params={"x": "number"})
        with self.assertRaisesRegex(ValueError, "missing params"):
            self._register(example={})
        with self.assertRaisesRegex(ValueError, "x must be int"):
            self._register(example={"x": True})
        with self.assertRaisesRegex(ValueError, "tool name"):
            self._register(name="Bad-Name")

    def test_output_contract_marks_rc0_without_output_as_failed(self) -> None:
        # L3: a tool can declare required outputs; rc0 without them = failed, at
        # registration time (smoke) and per run.
        (self.base / "noout.py").write_text("import sys; sys.exit(0)\n")
        with self.assertRaisesRegex(ValueError, "declared outputs"):
            self._register(name="needs_out", script="noout.py", params={"x": "int"},
                           defaults={}, example={"x": 1}, expects=["result.json"])
        # sim.py DOES write result.json, so the same contract passes smoke.
        self._register(name="with_out", expects=["result.json"])
        self.inputs.write_text('{"id": "a", "x": 1}\n')
        batch_id = self._call("run_batch", tool="with_out", inputs_path=str(self.inputs))["batch_id"]
        self.client.run()
        status = self._call("batch_status", batch_id)
        self.assertEqual((status["succeeded"], status["ok"]), (1, True))

    def test_corrupt_tool_json_raises_not_silent_fallback(self) -> None:
        # M15: a present-but-unparseable tool.json must raise, not silently fall
        # back to an older version that runs different science.
        self._register()  # v1
        (self.base / "sim.py").write_text(SIM + "\n# v2\n")
        self._register()  # v2
        (self.base / "tools" / "doubler" / "v2" / "tool.json").write_text("{ broken")
        with self.assertRaisesRegex(ValueError, "unreadable tool.json"):
            self._call("list_tools", name="doubler")

    def test_skip_smoke_registers_untested(self) -> None:
        result = self._register(skip_smoke=True, example={"x": 1, "fail": True})
        self.assertEqual((result["tested"], result["smoke"]), (False, None))
        self.assertFalse((self.base / "tools" / "doubler" / "v1" / "smoke").exists())

    def test_list_tools_newest_query_and_usage(self) -> None:
        self._register()
        (self.base / "sim.py").write_text(SIM + "\n# changed\n")
        self._register()
        self.assertEqual([t["version"] for t in self._call("list_tools")], [2])
        self.assertEqual([t["version"] for t in self._call("list_tools", all_versions=True)], [2, 1])
        self.assertEqual(self._call("list_tools", query="triples"), [])
        self.inputs.write_text('{"id": "a", "x": 1}\n')
        batch_id = self._call("run_batch", tool="doubler", inputs_path=str(self.inputs))["batch_id"]
        self.client.run()
        self._call("batch_status", batch_id)
        listed = self._call("list_tools", query="DOUBLES")[0]
        self.assertEqual((listed["uses"], listed["last_ok_batch"]), (1, batch_id))
        self.assertNotIn("script", listed)

    def test_run_batch_with_tool_checks_rows_and_records_tool(self) -> None:
        self._register()
        self.inputs.write_text('{"id": "a", "x": 1}\n{"id": "b", "x": 2, "y": 3}\n')
        with self.assertRaisesRegex(ValueError, r"row 'b': unknown params \['y'\]"):
            self._call("run_batch", tool="doubler", inputs_path=str(self.inputs))
        self.assertFalse((self.base / "batches").exists()
                         and any(p.name.startswith("b-") for p in (self.base / "batches").iterdir()))
        self.inputs.write_text('{"id": "a", "x": 1}\n{"id": "b", "x": 2, "fail": true}\n')
        result = self._call("run_batch", tool="doubler", inputs_path=str(self.inputs))
        entry = json.loads((self.base / "batches" / "registry" / f"{result['batch_id']}.json").read_text())
        self.assertEqual((entry["tool"], entry["tool_version"]), ("doubler", 1))
        self.assertEqual(entry["tool_sha256"], entry["script_sha256"])
        task = next(t for t in self.client.submitted if t.task_id.endswith("-a"))
        self.assertEqual(task.args[1], {"id": "a", "x": 1, "fail": False})  # default filled in
        with self.assertRaisesRegex(ValueError, "exactly one of"):
            self._call("run_batch", tool="doubler", script_path=str(self.base / "sim.py"),
                       inputs_path=str(self.inputs))
        with self.assertRaisesRegex(ValueError, "no registered tool"):
            self._call("run_batch", tool="doubler", tool_version=7, inputs_path=str(self.inputs))

    def test_argv_tool_and_local_staging(self) -> None:
        (self.base / "argv_sim.py").write_text(
            "import argparse, json\n"
            "p = argparse.ArgumentParser(); p.add_argument('--x', type=int); p.add_argument('--tag')\n"
            "a = p.parse_args(); json.dump({'x2': a.x * 2, 'tag': a.tag}, open('result.json', 'w'))\n")
        stage = self.base / "stage"
        os.environ["HPC_LOCAL_STAGE"] = str(stage)
        self.addCleanup(os.environ.pop, "HPC_LOCAL_STAGE", None)
        self._register(name="argv_doubler", script="argv_sim.py", params={"x": "int", "tag": "str"},
                       defaults={"tag": "t"}, example={"x": 2}, call="argv")
        self.inputs.write_text('{"id": "a", "x": 4}\n')
        batch_id = self._call("run_batch", tool="argv_doubler", inputs_path=str(self.inputs))["batch_id"]
        self.client.run()
        result = json.loads((self.base / "batches" / batch_id / "run_a" / "result.json").read_text())
        self.assertEqual(result, {"x2": 8, "tag": "t"})
        sha = json.loads((self.base / "batches" / "registry" / f"{batch_id}.json").read_text())["script_sha256"]
        self.assertTrue((stage / sha / "script.py").is_file())


@unittest.skipUnless(HAVE_DEPS, "hpc-deployment dependencies not installed")
class ExchangeListenerBudgetTests(unittest.TestCase):
    """The exchange-listener give-up budget is time-based, not count-based, so
    the ALCF proxy's routine stream drops can't zombie a live cluster."""

    def setUp(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import hpc_agent
        self.hpc = hpc_agent

    def tearDown(self) -> None:
        sys.path.remove(str(SCRIPTS))

    def test_backoff_is_exponential_and_capped(self) -> None:
        self.assertEqual(self.hpc._listen_backoff(1), 1.0)
        self.assertEqual(self.hpc._listen_backoff(2), 2.0)
        self.assertLessEqual(self.hpc._listen_backoff(50), self.hpc._LISTEN_BACKOFF_CAP)

    def test_never_gives_up_while_messages_arrive(self) -> None:
        # first_failure=None means a message arrived since the last drop.
        self.assertFalse(self.hpc._listen_should_give_up(None, 1e9))

    def test_many_brief_drops_within_window_do_not_give_up(self) -> None:
        # Sustained-silence budget is what matters, not the number of drops.
        self.assertFalse(self.hpc._listen_should_give_up(100.0, 100.0 + 1199, giveup_s=1200))

    def test_gives_up_only_after_sustained_silence(self) -> None:
        self.assertTrue(self.hpc._listen_should_give_up(100.0, 100.0 + 1200, giveup_s=1200))


def _response_error(status: int):
    import aiohttp
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL
    url = URL("https://exchange.example/v1/listen")
    info = aiohttp.RequestInfo(url, "GET", CIMultiDictProxy(CIMultiDict()), url)
    return aiohttp.ClientResponseError(info, (), status=status, message="x")


@unittest.skipUnless(HAVE_DEPS, "hpc-deployment dependencies not installed")
class ExchangeTransientErrorTests(unittest.TestCase):
    """A 429/5xx burst from the exchange is ridden out like a severed stream,
    rather than killing the listener (and the EL cluster) or the heartbeat."""

    def setUp(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import hpc_agent
        from academy.exchange.cloud.client import HttpExchangeTransport
        self.hpc = hpc_agent
        self.transport = HttpExchangeTransport
        saved = (HttpExchangeTransport.listen, HttpExchangeTransport.update_heartbeat)
        self.addCleanup(self._restore, saved)

    def tearDown(self) -> None:
        sys.path.remove(str(SCRIPTS))

    def _restore(self, saved) -> None:
        self.transport.listen, self.transport.update_heartbeat = saved

    def test_classification(self) -> None:
        import aiohttp
        transient = self.hpc._is_transient_exchange_error
        for status in (429, 500, 502, 503, 504):
            self.assertTrue(transient(_response_error(status)), status)
        for status in (400, 401, 403, 404, 413):
            self.assertFalse(transient(_response_error(status)), status)
        self.assertTrue(transient(aiohttp.ClientPayloadError("cut")))
        self.assertTrue(transient(aiohttp.ServerDisconnectedError()))
        self.assertFalse(transient(ValueError("x")))

    def _harden_listen(self, script):
        """Install a fake original listen that replays `script`, then harden it."""
        if not hasattr(self.hpc, "_harden_exchange_listener"):
            self.skipTest("exchange hardening currently disabled (commented out in hpc_agent)")
        calls = []

        async def fake_listen(transport, timeout=None):
            step = script[len(calls)]
            calls.append(step)
            if isinstance(step, BaseException):
                raise step
            for message in step:
                yield message

        self.transport.listen = fake_listen
        self.hpc._harden_exchange_listener()
        return calls

    def _drain(self):
        async def run():
            sleeps = []
            real_sleep = asyncio.sleep

            async def no_sleep(seconds):
                sleeps.append(seconds)
                await real_sleep(0)

            self.hpc.asyncio.sleep = no_sleep
            try:
                return [m async for m in self.transport.listen(object())], sleeps
            finally:
                self.hpc.asyncio.sleep = real_sleep
        return asyncio.run(run())

    def test_listener_reconnects_through_a_5xx_burst(self) -> None:
        calls = self._harden_listen([_response_error(500), _response_error(503), ["m1", "m2"]])
        messages, sleeps = self._drain()
        self.assertEqual(messages, ["m1", "m2"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_listener_still_raises_non_transient_errors(self) -> None:
        self._harden_listen([_response_error(403)])
        with self.assertRaises(Exception) as caught:
            self._drain()
        self.assertEqual(getattr(caught.exception, "status", None), 403)

    def test_client_startup_does_not_block_the_event_loop(self) -> None:
        # ClusterClient's constructor polls for the checkpoint with time.sleep.
        # Run on the event loop, it froze the exchange listener and heartbeat.
        import time as _time
        hpc = self.hpc

        class SlowClusterClient:
            def __init__(self, checkpoint_dir, checkpoint_timeout):
                _time.sleep(0.5)   # stands in for _wait_for_path

            def start(self):
                pass

        saved = (getattr(hpc, "ClusterClient", None), hpc.EL_AVAILABLE)
        hpc.ClusterClient, hpc.EL_AVAILABLE = SlowClusterClient, True

        def restore():
            hpc.ClusterClient, hpc.EL_AVAILABLE = saved
        self.addCleanup(restore)

        agent = hpc.HpcClientAgent("camp", tempfile.mkdtemp(), "ckpt")

        async def run():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.02)
                    ticks += 1

            task = asyncio.create_task(ticker())
            await agent.agent_on_startup()
            task.cancel()
            return ticks

        ticks = asyncio.run(run())
        self.assertIsInstance(agent._client, SlowClusterClient)
        self.assertGreater(ticks, 10)   # ~25 if the loop stayed free; 0 if blocked

    def test_heartbeat_skips_a_transient_failure_and_raises_others(self) -> None:
        if not hasattr(self.hpc, "_harden_exchange_heartbeat"):
            self.skipTest("exchange hardening currently disabled (commented out in hpc_agent)")
        outcomes = iter([_response_error(503), None, _response_error(401)])

        async def fake_heartbeat(transport):
            outcome = next(outcomes)
            if outcome is not None:
                raise outcome

        self.transport.update_heartbeat = fake_heartbeat
        self.hpc._harden_exchange_heartbeat()
        beat = self.transport.update_heartbeat
        asyncio.run(beat(object()))       # 503: logged and swallowed
        asyncio.run(beat(object()))       # success
        with self.assertRaises(Exception) as caught:
            asyncio.run(beat(object()))   # 401: not ours to hide
        self.assertEqual(getattr(caught.exception, "status", None), 401)


@unittest.skipUnless(HAVE_DEPS, "hpc-deployment dependencies not installed")
class BatchWidthFloorTests(unittest.TestCase):
    """A new batch must carry enough rows to be worth an allocation.

    Every case here is a submission actually observed on a 1024-node
    (12,288-device) reservation on 2026-11-09, where peak utilisation was ~1
    device: one-row batches, and a zero-row batch that was created SUCCESSFULLY
    because `inputs.is_file()` passes for an empty file and the row loop does
    nothing.

    The floor lives on the cluster because nothing else can see the file: the
    policy hook and mcp_server.py both run on the operator's laptop and cannot
    stat a path under /flare, so neither can count rows.
    """

    def setUp(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import hpc_agent

        self.hpc = hpc_agent
        self.hpc.Task = FakeTask
        self._floor = hpc_agent._MIN_BATCH_ROWS
        hpc_agent._MIN_BATCH_ROWS = 32
        self.base = Path(tempfile.mkdtemp())
        os.environ.pop("HPC_BATCH_ROOT", None)
        (self.base / "sim.py").write_text(SIM)
        self.agent = hpc_agent.HpcClientAgent("campaign", str(self.base), "ckpt")
        self.agent._client = FakeClient()

    def tearDown(self) -> None:
        self.hpc._MIN_BATCH_ROWS = self._floor
        sys.path.remove(str(SCRIPTS))

    def _submit(self, n_rows: int):
        path = self.base / "inputs.jsonl"
        path.write_text("".join(json.dumps({"id": f"r{i}", "x": i}) + "\n" for i in range(n_rows)))
        method = getattr(type(self.agent), "run_batch")
        method = getattr(method, "__wrapped__", method)
        return asyncio.run(method(self.agent, script_path=str(self.base / "sim.py"),
                                  inputs_path=str(path)))

    def test_a_one_row_batch_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._submit(1)
        message = str(caught.exception)
        self.assertIn("1 row", message)
        self.assertIn("32", message)

    def test_an_empty_inputs_file_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._submit(0)

    def test_a_refused_batch_leaves_no_directory_or_registry_entry(self) -> None:
        # The check must precede mkdir, or a refusal leaves a resumable stub and
        # the next call "resumes" a batch that never had rows.
        with self.assertRaises(ValueError):
            self._submit(1)
        batches = self.base / "batches"
        created = [p.name for p in batches.iterdir() if p.name != "registry"] if batches.exists() else []
        self.assertEqual(created, [], "a refused batch must not leave a batch directory")
        registry = batches / "registry"
        self.assertEqual(list(registry.glob("*.json")) if registry.exists() else [], [],
                         "a refused batch must not leave a registry entry")

    def test_a_wide_batch_is_accepted(self) -> None:
        self.assertEqual(self._submit(32)["total"], 32)

    def test_the_refusal_tells_the_agent_to_widen_not_to_retry(self) -> None:
        # The message is all the child sees; one that reads as transient gets
        # resubmitted unchanged or abandoned for run_code. Both were observed.
        with self.assertRaises(ValueError) as caught:
            self._submit(2)
        message = str(caught.exception)
        self.assertIn("write_files", message)
        self.assertIn("run_code", message)
        self.assertIn("Do not retry this batch unchanged", message)

    def test_the_floor_is_operator_controlled_not_agent_controlled(self) -> None:
        # run_batch takes no floor argument: an agent that could set its own
        # would set 1. Only HPC_MIN_BATCH_ROWS moves it.
        import inspect

        method = getattr(type(self.agent), "run_batch")
        method = getattr(method, "__wrapped__", method)
        params = set(inspect.signature(method).parameters)
        self.assertNotIn("min_rows", params)

    def test_resume_is_not_blocked_by_the_floor(self) -> None:
        # Resuming passes only batch_id and must not be re-validated for width:
        # those rows were already accepted when the batch was created.
        #
        # The batch is created WITH labels on purpose. Resume also requires
        # provenance (see ResumeProvenanceTests), so an unlabelled fixture would
        # be refused here for a different reason and this test would stop
        # exercising the floor exemption at all.
        self.hpc._MIN_BATCH_ROWS = 1
        path = self.base / "inputs.jsonl"
        path.write_text("".join(json.dumps({"id": f"r{i}", "x": i}) + "\n" for i in range(2)))
        method = getattr(type(self.agent), "run_batch")
        method = getattr(method, "__wrapped__", method)
        created = asyncio.run(method(self.agent, script_path=str(self.base / "sim.py"),
                                     inputs_path=str(path),
                                     labels={"plan_id": "P1", "hypothesis_id": "H7"}))
        self.hpc._MIN_BATCH_ROWS = 32
        resumed = asyncio.run(method(self.agent, batch_id=created["batch_id"]))
        self.assertEqual(resumed["batch_id"], created["batch_id"])


@unittest.skipUnless(HAVE_DEPS, "hpc-deployment dependencies not installed")
class BatchDistinctnessTests(unittest.TestCase):
    """Rows must describe DIFFERENT work, not just be numerous.

    Observed 2026-11-09 on the 1024-node reservation: once _MIN_BATCH_ROWS
    refused 1-row batches, agents widened to exactly 32 and padded. Of 352 rows
    across 11 batches only 47 payloads were distinct (13.4%); 8 of the 11 had
    all 32 rows identical after id/out_dir were stripped, with the stripped
    payload literally `{}`. The shapes below are those real batches.
    """

    def setUp(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import hpc_agent

        self.hpc = hpc_agent
        self.hpc.Task = FakeTask
        self._floor = hpc_agent._MIN_BATCH_ROWS
        self._distinct = hpc_agent._MIN_BATCH_DISTINCT
        hpc_agent._MIN_BATCH_ROWS = 1          # isolate the distinctness rule
        hpc_agent._MIN_BATCH_DISTINCT = 2
        self.base = Path(tempfile.mkdtemp())
        os.environ.pop("HPC_BATCH_ROOT", None)
        (self.base / "sim.py").write_text(SIM)
        self.agent = hpc_agent.HpcClientAgent("campaign", str(self.base), "ckpt")
        self.agent._client = FakeClient()

    def tearDown(self) -> None:
        self.hpc._MIN_BATCH_ROWS = self._floor
        self.hpc._MIN_BATCH_DISTINCT = self._distinct
        sys.path.remove(str(SCRIPTS))

    def _submit(self, rows):
        path = self.base / "inputs.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        method = getattr(type(self.agent), "run_batch")
        method = getattr(method, "__wrapped__", method)
        return asyncio.run(method(self.agent, script_path=str(self.base / "sim.py"),
                                  inputs_path=str(path)))

    def test_rows_with_no_parameters_at_all_are_refused(self) -> None:
        # b-8f8c0f6439fe: {"id":"s01"} .. {"id":"s32"}
        with self.assertRaises(ValueError) as caught:
            self._submit([{"id": f"s{i:02d}"} for i in range(1, 33)])
        message = str(caught.exception)
        self.assertIn("32 rows but only 1 distinct", message)
        self.assertIn("NO parameters at all", message)

    def test_varying_only_the_output_dir_is_refused(self) -> None:
        # b-1a4cacdcba83: gate01..gate32, each with its own run0N out_dir.
        # Same computation filed separately is still one computation.
        with self.assertRaises(ValueError) as caught:
            self._submit([{"id": f"gate{i:02d}", "out_dir": f"/flare/x/run{i:02d}"}
                          for i in range(1, 33)])
        self.assertIn("Varying only id or out_dir is NOT variation", str(caught.exception))

    def test_identical_rows_sharing_one_output_dir_are_refused(self) -> None:
        # b-1fec34bdfed4: every row wrote to the SAME out_dir, so rows raced.
        with self.assertRaises(ValueError):
            self._submit([{"id": f"egfr_structural_gate_{i:02d}", "out_dir": "/flare/x/h1"}
                          for i in range(1, 33)])

    def test_a_real_seed_sweep_is_accepted(self) -> None:
        # b-8c88eceb5702: strategy+seed per row -- the one real sweep of the 11.
        rows = [{"id": f"junction_{i:02d}", "strategy": "junction", "seed": 100 + i}
                for i in range(1, 33)]
        self.assertEqual(self._submit(rows)["total"], 32)

    def test_a_small_but_genuine_parameter_sweep_is_accepted(self) -> None:
        # b-3666210247e5 shape: few distinct topologies across many rows. Thin,
        # but it does describe different work, so it must not be refused.
        rows = [{"id": f"preorganized_{i:02d}",
                 "topology": ["two_module", "three_module", "rigid"][i % 3]}
                for i in range(32)]
        self.assertEqual(self._submit(rows)["total"], 32)

    def test_explicit_replicate_seeds_pass_where_bare_replicates_fail(self) -> None:
        # The refusal names this fix, so it has to actually work: bare
        # replicate_01..N is refused, the same replicates with per-row seeds pass.
        with self.assertRaises(ValueError):
            self._submit([{"id": f"replicate_{i:02d}"} for i in range(1, 33)])
        rows = [{"id": f"replicate_{i:02d}", "seed": 1000 + i} for i in range(1, 33)]
        self.assertEqual(self._submit(rows)["total"], 32)

    def test_a_single_row_batch_is_not_judged_on_distinctness(self) -> None:
        # One row cannot be "all the same as itself"; the ROW FLOOR is what
        # governs that case, and these two rules must not double-refuse.
        self.assertEqual(self._submit([{"id": "only", "seq": "MKT"}])["total"], 1)

    def test_a_refused_batch_leaves_no_directory_or_registry_entry(self) -> None:
        with self.assertRaises(ValueError):
            self._submit([{"id": f"s{i}"} for i in range(32)])
        batches = self.base / "batches"
        created = [p.name for p in batches.iterdir() if p.name != "registry"] if batches.exists() else []
        self.assertEqual(created, [], "a refused batch must not leave a batch directory")

    def test_non_dict_rows_do_not_crash_the_check(self) -> None:
        # _read_rows may yield a non-mapping; _row_signature must tolerate it
        # rather than raising inside the guard and failing open.
        self.assertEqual(self.hpc._row_signature(["a", 1]), repr(["a", 1]))
        self.assertEqual(self.hpc._row_signature("x"), repr("x"))

    def test_identity_key_matching_is_case_insensitive(self) -> None:
        with self.assertRaises(ValueError):
            self._submit([{"ID": f"s{i}", "OUT_DIR": f"/flare/x/{i}"} for i in range(32)])


@unittest.skipUnless(HAVE_DEPS, "hpc-deployment dependencies not installed")
class ResumeProvenanceTests(unittest.TestCase):
    """Resume must not become a way around the width guards or provenance.

    Resume is exempt from the row floor and the distinctness check because the
    rows were validated at creation -- that exemption is what lets a
    walltime-interrupted batch finish. On 2026-11-09 a child reactivated
    b-2f95dbc8caab, a ONE-ROW batch created the previous day with `labels: {}`,
    bypassing both guards; the pre-existing conflict check never fired because
    an EMPTY stored label set conflicts with nothing.
    """

    def setUp(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import hpc_agent

        self.hpc = hpc_agent
        self.hpc.Task = FakeTask
        self._floor = hpc_agent._MIN_BATCH_ROWS
        self._distinct = hpc_agent._MIN_BATCH_DISTINCT
        self._age = hpc_agent._MAX_RESUME_AGE_SECONDS
        hpc_agent._MIN_BATCH_ROWS = 1
        hpc_agent._MIN_BATCH_DISTINCT = 1
        self.base = Path(tempfile.mkdtemp())
        os.environ.pop("HPC_BATCH_ROOT", None)
        (self.base / "sim.py").write_text(SIM)
        self.inputs = self.base / "inputs.jsonl"
        self.inputs.write_text(json.dumps({"id": "a", "x": 1}) + "\n")
        self.agent = hpc_agent.HpcClientAgent("campaign", str(self.base), "ckpt")
        self.agent._client = FakeClient()
        self.labels = {"plan_id": "P1", "hypothesis_id": "H7"}

    def tearDown(self) -> None:
        self.hpc._MIN_BATCH_ROWS = self._floor
        self.hpc._MIN_BATCH_DISTINCT = self._distinct
        self.hpc._MAX_RESUME_AGE_SECONDS = self._age
        sys.path.remove(str(SCRIPTS))

    def _run_batch(self, **kwargs):
        method = getattr(type(self.agent), "run_batch")
        method = getattr(method, "__wrapped__", method)
        return asyncio.run(method(self.agent, **kwargs))

    def _create(self, labels):
        return self._run_batch(script_path=str(self.base / "sim.py"),
                               inputs_path=str(self.inputs), labels=labels)

    def _registry(self, batch_id):
        return self.base / "batches" / "registry" / f"{batch_id}.json"

    def test_an_unlabelled_batch_cannot_be_resumed_even_with_no_labels_passed(self) -> None:
        """The exact 2026-11-09 case, and the form that actually slipped through.

        `run_batch(batch_id=...)` with NO labels is the documented resume call,
        and it skips the conflict loop entirely (`labels.get(key)` is falsy), so
        any batch on disk could be reactivated by anyone. The guard therefore has
        to be a property of the BATCH, not of the call.
        """
        created = self._create(None)
        entry = json.loads(self._registry(created["batch_id"]).read_text())
        self.assertEqual(entry["labels"], {}, "fixture must reproduce the empty label set")
        with self.assertRaises(ValueError) as caught:
            self._run_batch(batch_id=created["batch_id"])      # no labels: the real path
        message = str(caught.exception)
        self.assertIn("no plan_id or hypothesis_id", message)
        self.assertIn("must not be attributed", message)

    def test_an_unlabelled_batch_is_also_refused_when_labels_are_passed(self) -> None:
        created = self._create(None)
        with self.assertRaises(ValueError):
            self._run_batch(batch_id=created["batch_id"], labels=self.labels)

    def test_resuming_this_plans_own_batch_still_works(self) -> None:
        # The legitimate case must not regress: same plan, same hypothesis.
        created = self._create(self.labels)
        resumed = self._run_batch(batch_id=created["batch_id"], labels=self.labels)
        self.assertEqual(resumed["batch_id"], created["batch_id"])

    def test_resume_without_labels_works_for_a_LABELLED_batch(self) -> None:
        # Passing only batch_id is the documented resume form and must keep
        # working -- for a batch that carries provenance. An earlier version of
        # this test allowed it for ANY batch, which is precisely the hole that
        # let an unlabelled day-old batch be reactivated.
        created = self._create(self.labels)
        resumed = self._run_batch(batch_id=created["batch_id"])
        self.assertEqual(resumed["batch_id"], created["batch_id"])

    def test_a_conflicting_label_is_still_refused(self) -> None:
        created = self._create(self.labels)
        with self.assertRaises(ValueError) as caught:
            self._run_batch(batch_id=created["batch_id"],
                            labels={"plan_id": "P2", "hypothesis_id": "H7"})
        self.assertIn("belongs to another plan_id", str(caught.exception))

    def test_a_stale_batch_is_refused_by_age(self) -> None:
        created = self._create(self.labels)
        path = self._registry(created["batch_id"])
        entry = json.loads(path.read_text())
        entry["created_at"] = "2026-11-08T21:14:27+00:00"   # the real stale batch's timestamp
        path.write_text(json.dumps(entry))
        with self.assertRaises(ValueError) as caught:
            self._run_batch(batch_id=created["batch_id"], labels=self.labels)
        message = str(caught.exception)
        self.assertIn("too old to resume", message)
        self.assertIn("stale code on stale inputs", message)

    def test_the_age_guard_can_be_disabled(self) -> None:
        created = self._create(self.labels)
        path = self._registry(created["batch_id"])
        entry = json.loads(path.read_text())
        entry["created_at"] = "2026-11-08T21:14:27+00:00"
        path.write_text(json.dumps(entry))
        self.hpc._MAX_RESUME_AGE_SECONDS = 0
        resumed = self._run_batch(batch_id=created["batch_id"], labels=self.labels)
        self.assertEqual(resumed["batch_id"], created["batch_id"])

    def test_an_unparseable_created_at_does_not_block_resume(self) -> None:
        # A corrupt timestamp must not strand a batch: fail OPEN on age, since
        # the labels check already carries the provenance guarantee.
        created = self._create(self.labels)
        path = self._registry(created["batch_id"])
        entry = json.loads(path.read_text())
        entry["created_at"] = "not-a-timestamp"
        path.write_text(json.dumps(entry))
        resumed = self._run_batch(batch_id=created["batch_id"], labels=self.labels)
        self.assertEqual(resumed["batch_id"], created["batch_id"])


if __name__ == "__main__":
    unittest.main()
