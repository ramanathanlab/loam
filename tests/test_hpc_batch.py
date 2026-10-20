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
    """Records submitted tasks; run() executes them in-process like EL would."""

    def __init__(self) -> None:
        self.submitted: list[FakeTask] = []

    def submit_batch(self, tasks):
        self.submitted.extend(tasks)
        return [object() for _ in tasks]

    def run(self, only=None) -> None:
        for task in self.submitted[:only]:
            task.executable(*task.args)
        self.submitted.clear()


@unittest.skipUnless(HAVE_DEPS, "hpc-deployment dependencies not installed")
class RunBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import hpc_agent

        self.hpc = hpc_agent
        self.hpc.Task = FakeTask
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
        sys.path.remove(str(SCRIPTS))

    def _call(self, action_name, /, *args, **kwargs):
        # Academy's @action wraps the coroutine; call the undecorated function.
        method = getattr(type(self.agent), action_name)
        method = getattr(method, "__wrapped__", method)
        return asyncio.run(method(self.agent, *args, **kwargs))

    def _new_batch(self):
        return self._call("run_batch", script_path=str(self.base / "sim.py"),
                          inputs_path=str(self.inputs), labels=self.labels)

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

    def test_resume_refuses_other_hypothesis(self) -> None:
        batch_id = self._new_batch()["batch_id"]
        with self.assertRaisesRegex(ValueError, "another hypothesis_id"):
            self._call("run_batch", batch_id=batch_id,
                       labels={"plan_id": "P1", "hypothesis_id": "H9"})

    def test_list_batches_filters_by_labels(self) -> None:
        batch_id = self._new_batch()["batch_id"]
        self.assertEqual([b["batch_id"] for b in self._call("list_batches", hypothesis_id="H7")], [batch_id])
        self.assertEqual(self._call("list_batches", hypothesis_id="H9"), [])

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


if __name__ == "__main__":
    unittest.main()
