"""`write_files` stages a batch without handing anyone an interpreter.

`run_batch` needs its script_path and inputs_path to exist on the cluster
already, and until now no HPC tool could create a file -- so the only way to
stage was to run code that wrote one. That made "never run the science in
run_code" impossible to obey, and measurably it was ignored: 62 code-generator
children made 524 `run_code` calls and 15 `run_batch` calls.

Writing is not executing, so this closes the gap without widening what a child
may run. The confinement tests matter most: the path need not exist yet, which
is exactly the case `_safe_path` could not handle, so the containment check is
new code rather than a reuse of a proven one.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "skills" / "hpc-deployment" / "scripts" / "hpc_agent.py"


def _load_write_files():
    """Import the module's helpers without importing the whole agent stack.

    hpc_agent imports academy/aiohttp at module scope, which the test env need
    not carry, and nothing here touches the Agent class.
    """

    source = AGENT.read_text(encoding="utf-8")
    namespace: dict = {}
    preamble = (
        "from __future__ import annotations\n"
        "import os, json\n"
        "from pathlib import Path\n"
        "from typing import Any, Literal, Mapping\n"
    )
    wanted = ("_MAX_WRITE_FILES", "_MAX_WRITE_BYTES", "_MAX_WRITE_TOTAL_BYTES")
    lines = [line for line in source.splitlines() if line.split("=")[0].strip() in wanted]
    body = _extract(source, "def _roots(") + _extract(source, "def _safe_write_path(") \
        + _extract(source, "def _write_files(")
    exec(preamble + "\n".join(lines) + "\n\n" + body, namespace)  # noqa: S102 - test harness
    return namespace


def _extract(source: str, header: str) -> str:
    start = source.index(header)
    rest = source[start:].splitlines(keepends=True)
    out = [rest[0]]
    for line in rest[1:]:
        if line.strip() and not line.startswith((" ", "\t", ")")) and not line.startswith("#"):
            break
        out.append(line)
    return "".join(out) + "\n\n"


class WriteFilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ns = _load_write_files()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = str(Path(self._tmp.name).resolve())

    def _write(self, files, roots=None):
        return self.ns["_write_files"](files, self.base, roots or [])

    def test_writes_several_files_in_one_call(self) -> None:
        result = self._write({
            f"{self.base}/plan/design.py": "print('hi')\n",
            f"{self.base}/plan/inputs.jsonl": '{"id": 1}\n',
        })
        self.assertEqual(result["count"], 2)
        self.assertEqual((Path(self.base) / "plan" / "design.py").read_text(), "print('hi')\n")
        self.assertEqual((Path(self.base) / "plan" / "inputs.jsonl").read_text(), '{"id": 1}\n')

    def test_creates_missing_parent_directories(self) -> None:
        self._write({f"{self.base}/a/b/c/script.py": "x = 1\n"})
        self.assertTrue((Path(self.base) / "a" / "b" / "c" / "script.py").is_file())

    def test_a_relative_path_resolves_inside_the_campaign_tree(self) -> None:
        self._write({"stage/run.py": "x = 1\n"})
        self.assertTrue((Path(self.base) / "stage" / "run.py").is_file())

    def test_a_path_outside_the_roots_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "escape.py"
            with self.assertRaises(ValueError) as caught:
                self._write({str(target): "x = 1\n"})
            # The refusal names the roots, so a wrong guess is corrected once
            # rather than retried blindly.
            self.assertIn("write only within", str(caught.exception))
            self.assertFalse(target.exists())

    def test_traversal_out_of_the_roots_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._write({f"{self.base}/../escaped.py": "x = 1\n"})

    def test_a_symlinked_parent_cannot_be_used_to_escape(self) -> None:
        # Non-strict resolve() expands the symlink, so containment is judged on
        # the real destination rather than the path as written.
        with tempfile.TemporaryDirectory() as outside:
            (Path(self.base) / "door").symlink_to(outside)
            with self.assertRaises(ValueError):
                self._write({f"{self.base}/door/escape.py": "x = 1\n"})
            self.assertFalse((Path(outside) / "escape.py").exists())

    def test_refusing_to_overwrite_through_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            victim = Path(outside) / "victim.txt"
            victim.write_text("original\n")
            (Path(self.base) / "link.txt").symlink_to(victim)
            with self.assertRaises(ValueError):
                self._write({f"{self.base}/link.txt": "overwritten\n"})
            self.assertEqual(victim.read_text(), "original\n")

    def test_a_directory_target_is_refused(self) -> None:
        (Path(self.base) / "adir").mkdir()
        with self.assertRaises(ValueError):
            self._write({f"{self.base}/adir": "x = 1\n"})

    def test_validation_happens_before_any_write(self) -> None:
        # All-or-nothing: a bad second path must not leave the first file on
        # disk, or run_batch would stage against a half-written job.
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ValueError):
                self._write({
                    f"{self.base}/good.py": "x = 1\n",
                    f"{outside}/bad.py": "x = 2\n",
                })
            self.assertFalse((Path(self.base) / "good.py").exists())

    def test_an_oversized_file_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._write({f"{self.base}/huge.bin": "x" * (self.ns["_MAX_WRITE_BYTES"] + 1)})
        self.assertIn("per-file limit", str(caught.exception))

    def test_too_many_files_is_refused(self) -> None:
        files = {f"{self.base}/f{i}.py": "x\n" for i in range(self.ns["_MAX_WRITE_FILES"] + 1)}
        with self.assertRaises(ValueError):
            self._write(files)

    def test_a_bad_shape_is_refused(self) -> None:
        for files in ({}, [], "script.py", None, {"": "x"}, {"ok.py": 3}):
            with self.subTest(files=files):
                with self.assertRaises(ValueError):
                    self._write(files)

    def test_an_extra_inspect_root_is_writable(self) -> None:
        with tempfile.TemporaryDirectory() as extra:
            root = str(Path(extra).resolve())
            self._write({f"{root}/staged.py": "x = 1\n"}, roots=[root])
            self.assertTrue((Path(root) / "staged.py").is_file())


class ToolSurfaceTests(unittest.TestCase):
    """The tool must be reachable, and must not widen the read-only roles."""

    def setUp(self) -> None:
        self.server = (ROOT / "skills" / "hpc-deployment" / "scripts" / "mcp_server.py").read_text()
        if str(ROOT / ".loam" / "hooks") not in sys.path:
            sys.path.insert(0, str(ROOT / ".loam" / "hooks"))
        spec = importlib.util.spec_from_file_location(
            "loam_orch_writefiles", ROOT / ".loam" / "hooks" / "hypothesis_orchestration.py"
        )
        assert spec and spec.loader
        self.orchestration = importlib.util.module_from_spec(spec)
        sys.modules["loam_orch_writefiles"] = self.orchestration
        spec.loader.exec_module(self.orchestration)

    def test_the_tool_is_registered_on_every_server_mode(self) -> None:
        # Registered in the base list, not under `if cluster`, so a non-cluster
        # server can stage too.
        self.assertIn("tools = [run_code, write_files,", self.server)

    def test_the_server_instructions_point_at_it_for_staging(self) -> None:
        self.assertIn("without executing anything", self.server)

    def test_the_policy_knows_the_tool_name(self) -> None:
        names = self.orchestration.hpc_tool_names(
            {"tool_name": "mcp__hpc__run_code", "function_name": "run_code"}
        )
        self.assertEqual(names["write_files"], "mcp__hpc__write_files")

    def test_write_files_is_not_granted_to_the_read_only_roles(self) -> None:
        source = (ROOT / ".loam" / "hooks" / "hypothesis_orchestration.py").read_text()
        # Both role gates list their tools explicitly; adding a writable tool to
        # hpc_tool_names must not have leaked into either allowlist.
        self.assertIn("observation = ('inspect_filesystem', 'batch_status', 'list_batches')", source)
        self.assertIn("observation = ('batch_status', 'inspect_filesystem', 'list_batches')", source)


if __name__ == "__main__":
    unittest.main()
