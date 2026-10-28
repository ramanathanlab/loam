"""Tests for the Pando-style per-hypothesis workspace."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
WORKSPACE = HOOKS / "hypothesis_workspace.py"
CONFIG = ROOT / ".loam" / "adapters" / "hermes" / "plugin" / "loam_config.py"


def _load(name: str, path: Path = WORKSPACE):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _payload(base: Path) -> dict:
    return {
        "workflow_id": "research-v1",
        "loam_run_id": "run-1",
        "loam_workspace": {"enabled": True, "root": str(base / "hypotheses"), "strict": True},
    }


def _profile_yaml(tmp: Path, workspace: dict | None) -> Path:
    profile = {"workflow_id": "research-v1", "state_dir": str(tmp / "state")}
    if workspace is not None:
        profile["workspace"] = workspace
    path = tmp / "loam.yaml"
    path.write_text(json.dumps({"enabled": True, "active_profile": "research", "profiles": {"research": profile}}))
    return path


class WorkspaceConfigTests(unittest.TestCase):
    def test_workspace_is_off_unless_enabled(self) -> None:
        config = _load("loam_config_workspace_off", CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            for block in (None, {"enabled": False}, {"root": "/x"}):
                path = _profile_yaml(Path(tmp), block)
                with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(path)}, clear=False):
                    profile = config.load_profile()
                assert profile is not None
                self.assertNotIn("loam_workspace", config.payload_values(profile))

    def test_workspace_root_defaults_under_state_dir_and_follows_override(self) -> None:
        config = _load("loam_config_workspace_on", CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            path = _profile_yaml(Path(tmp), {"enabled": True})
            with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(path)}, clear=False):
                profile = config.load_profile()
            assert profile is not None
            self.assertEqual(
                config.payload_values(profile)["loam_workspace"],
                {"enabled": True, "root": str(Path(tmp) / "state" / "hypotheses"), "strict": True},
            )
            override = Path(tmp) / "run-42"
            with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(path), "LOAM_STATE_DIR": str(override)}, clear=False):
                profile = config.load_profile()
            assert profile is not None
            self.assertEqual(config.payload_values(profile)["loam_workspace"]["root"], str(override / "hypotheses"))

    def test_explicit_root_and_non_strict(self) -> None:
        config = _load("loam_config_workspace_explicit", CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            path = _profile_yaml(Path(tmp), {"enabled": True, "root": "/abs/ws", "strict": False})
            with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(path)}, clear=False):
                profile = config.load_profile()
            assert profile is not None
            self.assertEqual(
                config.payload_values(profile)["loam_workspace"], {"enabled": True, "root": "/abs/ws", "strict": False}
            )


class WorkspaceLayoutTests(unittest.TestCase):
    def test_disabled_payload_has_no_settings(self) -> None:
        workspace = _load("loam_workspace_disabled")
        self.assertIsNone(workspace.settings({}))
        self.assertIsNone(workspace.settings({"loam_workspace": {"enabled": False, "root": "/x"}}))
        with self.assertRaises(ValueError):
            workspace.workspace_root({}, "plan-1", "h-1")

    def test_unsafe_ids_are_refused(self) -> None:
        workspace = _load("loam_workspace_unsafe")
        with tempfile.TemporaryDirectory() as tmp:
            payload = _payload(Path(tmp))
            for bad in (os.pardir, ".", "../etc", "a/b", "", ".hidden", "x" * 65):
                with self.assertRaises(ValueError, msg=bad):
                    workspace.workspace_root(payload, bad, "h-1")
                with self.assertRaises(ValueError, msg=bad):
                    workspace.workspace_root(payload, "plan-1", bad)

    def test_ensure_workspace_creates_layout_once(self) -> None:
        workspace = _load("loam_workspace_ensure")
        with tempfile.TemporaryDirectory() as tmp:
            payload = _payload(Path(tmp))
            root = workspace.ensure_workspace(payload, "plan-1", "h-1", "Claim.", "validator-1")
            again = workspace.ensure_workspace(payload, "plan-1", "h-1", "Claim.", "validator-1")

            self.assertEqual(root, again)
            self.assertEqual(root, Path(tmp) / "hypotheses" / "plan-1" / "h-1")
            for name in workspace.LAYOUT_DIRS:
                self.assertTrue((root / name).is_dir(), name)
            experiment = workspace.read_json(root / "experiment.json")
            self.assertEqual(experiment["state"], "open")
            self.assertEqual(experiment["statement"], "Claim.")
            self.assertEqual(experiment["runs"], [])
            entries = [json.loads(line) for line in (root / "log.jsonl").read_text().splitlines()]
            self.assertEqual([e["kind"] for e in entries], ["note"])

    def test_allocate_run_numbers_runs_and_logs_plans(self) -> None:
        workspace = _load("loam_workspace_allocate")
        with tempfile.TemporaryDirectory() as tmp:
            root = workspace.ensure_workspace(_payload(Path(tmp)), "plan-1", "h-1")
            first = workspace.allocate_run(root, "child-1", "fold it", "validator asked")
            second = workspace.allocate_run(root, "child-2", "dock it")

            self.assertEqual((first, second), ("r1", "r2"))
            for run_id in ("r1", "r2"):
                self.assertTrue((root / "outputs" / run_id).is_dir())
                self.assertTrue((root / "scratch" / run_id).is_dir())
            run = workspace.read_json(root / "runs" / "r1" / "run.json")
            self.assertEqual((run["state"], run["child_session_id"], run["goal"]), ("started", "child-1", "fold it"))
            self.assertEqual(workspace.read_json(root / "experiment.json")["runs"], ["r1", "r2"])
            entries = [json.loads(line) for line in (root / "log.jsonl").read_text().splitlines()]
            self.assertEqual([(e["kind"], e.get("run")) for e in entries], [("note", None), ("plan", "r1"), ("plan", "r2")])

    def test_allocate_run_refuses_a_closed_workspace(self) -> None:
        workspace = _load("loam_workspace_closed")
        with tempfile.TemporaryDirectory() as tmp:
            root = workspace.ensure_workspace(_payload(Path(tmp)), "plan-1", "h-1")
            experiment = workspace.read_json(root / "experiment.json")
            (root / "experiment.json").write_text(json.dumps({**experiment, "state": "closed"}))
            with self.assertRaises(ValueError):
                workspace.allocate_run(root, "child-1", "late child")


class WorkspaceLogTests(unittest.TestCase):
    def _root_with_entries(self, workspace, tmp: Path) -> Path:
        root = workspace.ensure_workspace(_payload(tmp), "plan-1", "h-1")
        workspace.append_log(root, "note", {"what": "two"})
        workspace.append_log(root, "note", {"what": "three"})
        return root

    def test_intact_log_verifies(self) -> None:
        workspace = _load("loam_workspace_log_ok")
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_entries(workspace, Path(tmp))
            self.assertEqual(workspace.verify_log(root), [])
            seqs = [json.loads(line)["seq"] for line in (root / "log.jsonl").read_text().splitlines()]
            self.assertEqual(seqs, [1, 2, 3])

    def test_edited_line_breaks_the_chain(self) -> None:
        workspace = _load("loam_workspace_log_edit")
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_entries(workspace, Path(tmp))
            path = root / "log.jsonl"
            path.write_text(path.read_text().replace('"two"', '"TWO"'))
            self.assertEqual(len(workspace.verify_log(root)), 1)
            self.assertIn("line 3", workspace.verify_log(root)[0])

    def test_removed_line_breaks_the_chain(self) -> None:
        workspace = _load("loam_workspace_log_remove")
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_entries(workspace, Path(tmp))
            path = root / "log.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join([lines[0], lines[2]]) + "\n")
            self.assertTrue(workspace.verify_log(root))

    def test_unknown_kind_is_refused(self) -> None:
        workspace = _load("loam_workspace_log_kind")
        with tempfile.TemporaryDirectory() as tmp:
            root = workspace.ensure_workspace(_payload(Path(tmp)), "plan-1", "h-1")
            with self.assertRaises(ValueError):
                workspace.append_log(root, "rewrite", {})



ORCHESTRATION = HOOKS / "hypothesis_orchestration.py"
VALIDATOR_GOAL = "Validate [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]"


class _OrchestrationHelpers:
    """Payloads and lifecycle calls shared by the orchestration-driven tests."""

    def _payload(self, base: Path, mode: str = "remote_mcp", workspace: bool = True) -> dict:
        state_file = base / "jnana.json"
        state_file.write_text(json.dumps({"hypotheses": [{"hypothesis_id": "h-1", "content": "Binder B binds T."}]}))
        execution = {"mode": mode, "code_source": "generated", "max_children": 3, "runtime": "python3.12"}
        if mode == "remote_mcp":
            execution.update(
                tool_name="mcp_hpc_run_code", endpoint="http://127.0.0.1:8000/mcp", function_name="run_code",
                input_schema="hpc-run-code-v1", workdir_root="/lus/campaign/work",
            )
        payload = {
            "repo_root": str(ROOT),
            "workflow_id": "research-v1",
            "loam_run_id": "run-1",
            "loam_state_dir": str(base / "state"),
            "loam_hypothesis_tournament": {"state_file": str(state_file)},
            "loam_execution": execution,
        }
        if workspace:
            payload["loam_workspace"] = {"enabled": True, "root": str(base / "hypotheses"), "strict": True}
        return payload

    def _rows(self, payload: dict) -> list[dict]:
        path = Path(payload["loam_state_dir"]) / "hypothesis-orchestration.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def _start(self, orchestration, payload: dict, session: str, goal: str) -> dict:
        orchestration.record_start({**payload, "child_subagent_id": session, "child_session_id": session, "child_goal": goal})
        return self._rows(payload)[-1]

    def _delegate(self, orchestration, payload: dict, goals: list[str]) -> dict:
        return orchestration.canonical_delegation_args(
            {**payload, "session_id": "validator-1", "arguments": {"tasks": [{"goal": g} for g in goals]}}
        )



class WorkspaceStartTests(_OrchestrationHelpers, unittest.TestCase):
    """Group 2: record_start and child delegation populate the workspace."""

    def test_validator_start_creates_workspace_with_statement(self) -> None:
        orchestration = _load("loam_orch_ws_validator", ORCHESTRATION)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            row = self._start(orchestration, payload, "validator-1", VALIDATOR_GOAL)

            root = Path(tmp) / "hypotheses" / "plan-1" / "h-1"
            self.assertEqual(row["workspace"], str(root))
            experiment = json.loads((root / "experiment.json").read_text())
            self.assertEqual((experiment["state"], experiment["statement"]), ("open", "Binder B binds T."))
            self.assertEqual(experiment["validator_session_id"], "validator-1")

    def test_delegation_reserves_runs_and_names_them_in_goal_and_context(self) -> None:
        orchestration = _load("loam_orch_ws_reserve", ORCHESTRATION)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            self._start(orchestration, payload, "validator-1", VALIDATOR_GOAL)
            tasks = self._delegate(orchestration, payload, ["Fold B with T.", "Score the interface."])["tasks"]

            root = Path(tmp) / "hypotheses" / "plan-1" / "h-1"
            for task, run_id in zip(tasks, ("r1", "r2")):
                self.assertIn(f"[loam-hypothesis-id=h-1] [loam-run-id={run_id}]", task["goal"])
                self.assertIn(f"/lus/campaign/work/plan-1/h-1/outputs/{run_id}/", task["context"])
                self.assertIn(f"/lus/campaign/work/plan-1/h-1/scratch/{run_id}/", task["context"])
                self.assertIn("never /tmp", task["context"])
                run = json.loads((root / "runs" / run_id / "run.json").read_text())
                self.assertEqual(run["state"], "reserved")
            # Reserving is not starting: no plan is logged until a child begins.
            kinds = [json.loads(line)["kind"] for line in (root / "log.jsonl").read_text().splitlines()]
            self.assertEqual(kinds, ["note"])

    def test_child_start_moves_reserved_run_to_started_and_logs_plan(self) -> None:
        orchestration = _load("loam_orch_ws_child", ORCHESTRATION)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            self._start(orchestration, payload, "validator-1", VALIDATOR_GOAL)
            task = self._delegate(orchestration, payload, ["Fold B with T."])["tasks"][0]
            row = self._start(orchestration, payload, "child-1", task["goal"])

            root = Path(tmp) / "hypotheses" / "plan-1" / "h-1"
            self.assertEqual((row["role"], row["run"], row["workspace"]), ("code-generator", "r1", str(root)))
            run = json.loads((root / "runs" / "r1" / "run.json").read_text())
            self.assertEqual((run["state"], run["child_session_id"]), ("started", "child-1"))
            self.assertNotIn("[loam-", run["goal"])
            entries = [json.loads(line) for line in (root / "log.jsonl").read_text().splitlines()]
            self.assertEqual([(e["kind"], e.get("run")) for e in entries], [("note", None), ("plan", "r1")])

    def test_local_mode_layout_points_at_the_local_workspace(self) -> None:
        orchestration = _load("loam_orch_ws_local", ORCHESTRATION)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            self._start(orchestration, payload, "validator-1", VALIDATOR_GOAL)
            task = self._delegate(orchestration, payload, ["Fold B with T."])["tasks"][0]

            root = Path(tmp) / "hypotheses" / "plan-1" / "h-1"
            self.assertIn(f"{root}/outputs/r1/", task["context"])
            self.assertIn(f"{root}/src/", task["context"])

    def test_child_without_reserved_id_gets_a_run_at_start(self) -> None:
        orchestration = _load("loam_orch_ws_unreserved", ORCHESTRATION)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            self._start(orchestration, payload, "validator-1", VALIDATOR_GOAL)
            goal = "Check [loam-agent-role=code-generator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1]"
            row = self._start(orchestration, payload, "child-1", goal)
            self.assertEqual(row["run"], "r1")

    def test_invalid_batch_reserves_nothing(self) -> None:
        orchestration = _load("loam_orch_ws_invalid", ORCHESTRATION)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            self._start(orchestration, payload, "validator-1", VALIDATOR_GOAL)
            self.assertIsNone(self._delegate(orchestration, payload, ["Fold B.", " "]))
            self.assertEqual(list((Path(tmp) / "hypotheses" / "plan-1" / "h-1" / "runs").iterdir()), [])

    def test_workspace_failure_keeps_the_agent_started_row(self) -> None:
        orchestration = _load("loam_orch_ws_failure", ORCHESTRATION)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            # No validator start, so the child's workspace does not exist and
            # the forged run id cannot be started.
            goal = "Check [loam-agent-role=code-generator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1] [loam-run-id=r7]"
            row = self._start(orchestration, payload, "child-1", goal)
            self.assertEqual((row["event"], row["role"]), ("agent_started", "code-generator"))
            self.assertIn("workspace_error", row)
            self.assertNotIn("run", row)

    def test_disabled_workspace_changes_nothing(self) -> None:
        orchestration = _load("loam_orch_ws_disabled", ORCHESTRATION)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), workspace=False)
            row = self._start(orchestration, payload, "validator-1", VALIDATOR_GOAL)
            task = self._delegate(orchestration, payload, ["Fold B with T."])["tasks"][0]

            self.assertFalse({"workspace", "run", "workspace_error"} & row.keys())
            self.assertNotIn("loam-run-id", task["goal"])
            self.assertNotIn("WORKSPACE LAYOUT", task["context"])
            self.assertFalse((Path(tmp) / "hypotheses").exists())



class WorkspacePolicyTests(_OrchestrationHelpers, unittest.TestCase):
    """Group 3: pre-tool enforcement (fs_policy) and code capture."""

    REMOTE = "/lus/campaign/work/plan-1/h-1"

    def _child(self, tmp: Path, mode: str = "local", strict: bool = True):
        """A started validator + one started child (run r1). Returns (orchestration, payload, root)."""

        orchestration = _load(f"loam_orch_ws_policy_{self.id().rsplit('.', 1)[-1]}", ORCHESTRATION)
        payload = self._payload(tmp, mode=mode)
        payload["loam_workspace"]["strict"] = strict
        payload["loam_hypothesis_orchestration"] = True
        self._start(orchestration, payload, "validator-1", VALIDATOR_GOAL)
        task = self._delegate(orchestration, payload, ["Fold B with T."])["tasks"][0]
        self._start(orchestration, payload, "child-1", task["goal"])
        return orchestration, payload, tmp / "hypotheses" / "plan-1" / "h-1"

    def _call(self, orchestration, base: dict, tool_name: str, /, session: str = "child-1", **arguments):
        # Positional-only, so tool arguments named `payload` or `tool` don't collide.
        return orchestration.workspace_policy({**base, "session_id": session, "tool_name": tool_name, "arguments": arguments})

    def assertBlocked(self, verdict, fragment: str) -> None:
        self.assertIsNotNone(verdict, f"expected a block mentioning {fragment!r}")
        self.assertEqual(verdict["action"], "block")
        self.assertIn(fragment, verdict["message"])

    # local file tools ------------------------------------------------------------
    def test_writes_inside_the_run_layout_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            for target in ("src/fold.py", "outputs/r1/scores.csv", "scratch/r1/tmp.txt", "src/lib/util.py"):
                self.assertIsNone(self._call(orchestration, payload, "write_file", path=str(root / target), content="x"), target)

    def test_record_files_are_hook_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            for target in ("log.jsonl", "experiment.json", ".workspace.lock", "runs/r1/run.json"):
                verdict = self._call(orchestration, payload, "write_file", path=str(root / target), content="x")
                self.assertBlocked(verdict, "record file")

    def test_other_runs_data_and_stray_names_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            cases = {
                "outputs/r2/x.csv": "not your run",
                "scratch/r9/x": "not your run",
                "outputs/x.csv": "not your run",
                "data/target.fasta": "read-only inputs",
                "checkpoints/model.pt": "not part of the workspace layout",
                "notes.md": "not part of the workspace layout",
            }
            for target, fragment in cases.items():
                verdict = self._call(orchestration, payload, "write_file", path=str(root / target), content="x")
                self.assertBlocked(verdict, fragment)

    def test_relative_outside_and_sibling_writes_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            self.assertBlocked(self._call(orchestration, payload, "write_file", path="outputs/r1/x", content="x"), "absolute path")
            self.assertBlocked(self._call(orchestration, payload, "write_file", path=f"{tmp}/elsewhere.txt", content="x"), "outside")
            sibling = root.parent / "h-2" / "src" / "x.py"
            self.assertBlocked(self._call(orchestration, payload, "write_file", path=str(sibling), content="x"), "another hypothesis")
            dotdot = f"{root}/src/../../h-2/src/x.py"
            self.assertBlocked(self._call(orchestration, payload, "write_file", path=dotdot, content="x"), "another hypothesis")

    def test_symlink_escape_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            (root.parent / "h-2").mkdir()
            (root / "src" / "link").symlink_to(root.parent / "h-2")
            verdict = self._call(orchestration, payload, "write_file", path=str(root / "src" / "link" / "x.py"), content="x")
            self.assertBlocked(verdict, "another hypothesis")
            (root / "src" / "log").symlink_to(root / "log.jsonl")
            verdict = self._call(orchestration, payload, "write_file", path=str(root / "src" / "log"), content="x")
            self.assertBlocked(verdict, "record file")

    def test_patch_paths_are_checked_in_both_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            ok = self._call(orchestration, payload, "patch", mode="replace", path=str(root / "src" / "a.py"), old_string="a", new_string="b")
            self.assertIsNone(ok)
            text = f"*** Begin Patch\n*** Update File: {root}/src/a.py\n*** Update File: {root}/log.jsonl\n*** End Patch"
            self.assertBlocked(self._call(orchestration, payload, "patch", mode="patch", patch=text), "record file")
            move = f"*** Move File: {root}/src/a.py -> {root}/data/a.py"
            self.assertBlocked(self._call(orchestration, payload, "patch", mode="patch", patch=move), "read-only inputs")

    def test_reads_are_blocked_only_in_other_hypotheses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            self.assertIsNone(self._call(orchestration, payload, "read_file", path=str(root / "log.jsonl")))
            self.assertIsNone(self._call(orchestration, payload, "read_file", path="/etc/hosts"))
            self.assertIsNone(self._call(orchestration, payload, "search_files", path=str(root / "outputs")))
            sibling = root.parent / "h-2" / "outputs" / "r1" / "x"
            self.assertBlocked(self._call(orchestration, payload, "read_file", path=str(sibling)), "another hypothesis")

    # terminal ----------------------------------------------------------------------
    def test_terminal_needs_a_workdir_in_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            self.assertBlocked(self._call(orchestration, payload, "terminal", command="ls"), "pass workdir")
            self.assertBlocked(self._call(orchestration, payload, "terminal", command="ls", workdir=str(root / "data")), "workdir")
            self.assertIsNone(self._call(orchestration, payload, "terminal", command="python fold.py", workdir=str(root / "scratch" / "r1")))

    def test_terminal_write_targets_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            workdir = str(root / "scratch" / "r1")
            blocked = {
                "echo hi > ../../log.jsonl": "record file",
                "echo hi >> ../../experiment.json": "record file",
                f"sed -i -e s,a,b, {root}/experiment.json": "record file",
                f"sed -i 's/a/b/' {root}/outputs/r2/x": "not your run",
                f"python fold.py && cp out.csv {root}/data/": "read-only inputs",
                f"rm -rf {root}/runs": "record file",
                f"cat {root.parent}/h-2/outputs/r1/x": "another hypothesis",
            }
            for command, fragment in blocked.items():
                self.assertBlocked(self._call(orchestration, payload, "terminal", command=command, workdir=workdir), fragment)
            allowed = (
                f"cp {root}/data/target.fasta {root}/scratch/r1/",
                f"cp {root}/outputs/r1/a.csv ../../outputs/r1/b.csv",
                "python fold.py > fold.log 2>&1",
                f"cat {root}/log.jsonl | tail -1",
            )
            for command in allowed:
                self.assertIsNone(self._call(orchestration, payload, "terminal", command=command, workdir=workdir), command)

    # run state, roles, strictness -------------------------------------------------
    def test_finished_run_is_frozen_but_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            path = root / "runs" / "r1" / "run.json"
            path.write_text(json.dumps({**json.loads(path.read_text()), "state": "completed"}))
            verdict = self._call(orchestration, payload, "write_file", path=str(root / "outputs" / "r1" / "x"), content="x")
            self.assertBlocked(verdict, "frozen")
            self.assertIsNone(self._call(orchestration, payload, "read_file", path=str(root / "outputs" / "r1" / "x")))

    def test_only_code_generators_are_governed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            outside = f"{tmp}/elsewhere.txt"
            self.assertIsNone(self._call(orchestration, payload, "write_file", session="validator-1", path=outside, content="x"))
            self.assertIsNone(self._call(orchestration, payload, "write_file", session="someone-else", path=outside, content="x"))

    def test_child_without_workspace_may_only_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            goal = "Check [loam-agent-role=code-generator] [loam-plan-id=plan-1] [loam-hypothesis-id=h-1] [loam-run-id=r7]"
            self._start(orchestration, payload, "child-2", goal)
            verdict = self._call(orchestration, payload, "write_file", session="child-2", path=str(root / "src" / "x"), content="x")
            self.assertBlocked(verdict, "no workspace run")
            self.assertIsNone(self._call(orchestration, payload, "read_file", session="child-2", path=str(root / "log.jsonl")))

    def test_violations_are_logged_and_non_strict_allows_them(self) -> None:
        workspace = _load("loam_workspace_policy_log")
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp), strict=False)
            self.assertIsNone(self._call(orchestration, payload, "write_file", path=str(root / "data" / "x"), content="x"))
            last = json.loads((root / "log.jsonl").read_text().splitlines()[-1])
            self.assertEqual((last["kind"], last["run"], last["blocked"]), ("violation", "r1", False))
            self.assertIn("data/", last["what"])
            self.assertEqual(workspace.verify_log(root), [])

    def test_disabled_workspace_is_never_consulted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp))
            off = {key: value for key, value in payload.items() if key != "loam_workspace"}
            self.assertIsNone(self._call(orchestration, off, "write_file", path=str(root / "log.jsonl"), content="x"))

    # remote (HPC) tools -------------------------------------------------------------
    def test_run_code_needs_a_workdir_in_the_runs_hpc_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, _ = self._child(Path(tmp), mode="remote_mcp")
            tool = "mcp_hpc_run_code"
            self.assertBlocked(self._call(orchestration, payload, tool, payload="print(1)"), "pass workdir")
            self.assertIsNone(self._call(orchestration, payload, tool, payload="print(1)", workdir=f"{self.REMOTE}/scratch/r1"))
            self.assertIsNone(self._call(orchestration, payload, tool, payload="print(1)", workdir=f"{self.REMOTE}/outputs/r1"))
            self.assertBlocked(
                self._call(orchestration, payload, tool, payload="print(1)", workdir=f"{self.REMOTE}/outputs/r2"), "not your run"
            )
            self.assertBlocked(
                self._call(orchestration, payload, tool, payload="print(1)", workdir="/lus/campaign/work/plan-1/h-2/scratch/r1"),
                "another hypothesis",
            )
            self.assertBlocked(self._call(orchestration, payload, tool, payload="print(1)", workdir="/home/me"), "outside")
            code = "open('/lus/campaign/work/plan-1/h-2/outputs/r1/scores.csv').read()"
            self.assertBlocked(
                self._call(orchestration, payload, tool, payload=code, workdir=f"{self.REMOTE}/scratch/r1"), "another hypothesis"
            )

    def test_batch_scripts_must_live_in_the_hypothesis_hpc_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, _ = self._child(Path(tmp), mode="remote_mcp")
            ok = self._call(
                orchestration, payload, "mcp_hpc_run_batch",
                script_path=f"{self.REMOTE}/src/fold.py", inputs_path=f"{self.REMOTE}/scratch/r1/rows.jsonl",
            )
            self.assertIsNone(ok)
            self.assertIsNone(self._call(orchestration, payload, "mcp_hpc_run_batch", tool="chai_fold", inputs_path=f"{self.REMOTE}/data/rows.jsonl"))
            self.assertBlocked(
                self._call(orchestration, payload, "mcp_hpc_run_batch", script_path="/home/me/fold.py"), "outside"
            )
            self.assertBlocked(
                self._call(orchestration, payload, "mcp_hpc_register_tool", name="t", script_path="/lus/campaign/work/plan-1/h-2/src/t.py"),
                "another hypothesis",
            )

    def test_inspect_filesystem_and_status_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, _ = self._child(Path(tmp), mode="remote_mcp")
            self.assertIsNone(self._call(orchestration, payload, "mcp_hpc_inspect_filesystem", operation="ls", path=f"{self.REMOTE}/outputs/r1"))
            self.assertIsNone(self._call(orchestration, payload, "mcp_hpc_inspect_filesystem", operation="ls", path="/lus/models"))
            self.assertBlocked(
                self._call(orchestration, payload, "mcp_hpc_inspect_filesystem", operation="cat", path="/lus/campaign/work/plan-1/h-2/x"),
                "another hypothesis",
            )
            self.assertIsNone(self._call(orchestration, payload, "mcp_hpc_batch_status", batch_id="b-1"))
            self.assertIsNone(self._call(orchestration, payload, "mcp_hpc_list_tools"))

    # capture ------------------------------------------------------------------------
    def test_run_code_payload_is_saved_to_src_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration, payload, root = self._child(Path(tmp), mode="remote_mcp")
            for code in ("print('fold')", "print('score')\n"):
                orchestration.workspace_capture(
                    {**payload, "session_id": "child-1", "tool_name": "mcp_hpc_run_code",
                     "arguments": {"payload": code, "workdir": f"{self.REMOTE}/scratch/r1"}}
                )
            orchestration.workspace_capture(
                {**payload, "session_id": "child-1", "tool_name": "mcp_hpc_run_batch",
                 "arguments": {"script_path": f"{self.REMOTE}/src/fold.py", "inputs_path": f"{self.REMOTE}/scratch/r1/rows.jsonl"}}
            )
            orchestration.workspace_capture({**payload, "session_id": "child-1", "tool_name": "mcp_hpc_list_tools", "arguments": {}})

            self.assertEqual((root / "src" / "r1_01.py").read_text(), "print('fold')\n")
            self.assertEqual((root / "src" / "r1_02.py").read_text(), "print('score')\n")
            calls = json.loads((root / "runs" / "r1" / "run.json").read_text())["calls"]
            self.assertEqual([c.get("src") for c in calls], ["src/r1_01.py", "src/r1_02.py", None])
            self.assertEqual(calls[2]["script_path"], f"{self.REMOTE}/src/fold.py")
            self.assertEqual(len(calls[0]["sha256"]), 64)

    # end to end through the real pre-tool.py script -------------------------------------
    def _pre_tool(self, payload: dict) -> str:
        import subprocess

        completed = subprocess.run(
            [sys.executable, str(HOOKS / "pre-tool.py")], input=json.dumps({**payload, "event": "pre_tool_call"}),
            text=True, capture_output=True, cwd=str(ROOT), check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout.strip()

    def test_pre_tool_blocks_and_skips_capture_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, payload, root = self._child(Path(tmp), mode="remote_mcp")
            call = {**payload, "session_id": "child-1", "tool_name": "mcp_hpc_run_code",
                    "arguments": {"payload": "print(1)", "workdir": "/lus/campaign/work/plan-1/h-2/scratch/r1"}}
            verdict = json.loads(self._pre_tool(call))
            self.assertEqual(verdict["action"], "block")
            self.assertIn("another hypothesis", verdict["message"])
            self.assertEqual(list((root / "src").iterdir()), [])

    def test_pre_tool_allows_and_captures_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, payload, root = self._child(Path(tmp), mode="remote_mcp")
            call = {**payload, "session_id": "child-1", "tool_name": "mcp_hpc_run_code",
                    "arguments": {"payload": "print(1)", "workdir": f"{self.REMOTE}/scratch/r1"}}
            self.assertEqual(self._pre_tool(call), "")
            self.assertEqual((root / "src" / "r1_01.py").read_text(), "print(1)\n")



class _StopHelpers(_OrchestrationHelpers):
    """Lifecycle helpers shared by the stop-time and verification tests."""

    REMOTE = "/lus/campaign/work/plan-1"

    def _orchestration(self):
        return _load(f"loam_orch_ws_stop_{self.id().rsplit('.', 1)[-1]}", ORCHESTRATION)

    def _hypothesis(self, orchestration, payload: dict, hyp: str, goals: list[str]) -> tuple[Path, list[dict]]:
        """Start validator-<hyp> and delegate ``goals``; children are not started. Returns (root, tasks)."""

        goal = f"Validate [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-1] [loam-hypothesis-id={hyp}]"
        self._start(orchestration, payload, f"validator-{hyp}", goal)
        tasks = orchestration.canonical_delegation_args(
            {**payload, "session_id": f"validator-{hyp}", "arguments": {"tasks": [{"goal": g} for g in goals]}}
        )["tasks"]
        return Path(payload["loam_workspace"]["root"]) / "plan-1" / hyp, tasks

    def _stop(self, orchestration, payload: dict, session: str, status: str = "completed", summary: str = "done") -> None:
        stop = {**payload, "child_subagent_id": session, "child_session_id": session,
                "child_status": status, "child_summary": summary}
        orchestration.record_stop(stop)
        orchestration.workspace_stop(stop)

    def _log(self, root: Path) -> list[dict]:
        return [json.loads(line) for line in (root / "log.jsonl").read_text().splitlines()]

    def _run(self, root: Path, run_id: str) -> dict:
        return json.loads((root / "runs" / run_id / "run.json").read_text())

    def _completed_local_run(self, orchestration, payload: dict, hyp: str, files: dict[str, str]) -> Path:
        root, tasks = self._hypothesis(orchestration, payload, hyp, ["Make results."])
        self._start(orchestration, payload, f"child-{hyp}", tasks[0]["goal"])
        for name, text in files.items():
            (root / "outputs" / "r1" / name).write_text(text)
        self._stop(orchestration, payload, f"child-{hyp}")
        return root



class WorkspaceStopTests(_StopHelpers, unittest.TestCase):
    """Group 4: finalize runs, close the workspace, audit, cross-hypothesis inputs."""

    # finalize_run --------------------------------------------------------------------
    def test_child_stop_completes_hashes_logs_and_freezes_the_run(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold B with T."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            (root / "outputs" / "r1" / "deep").mkdir()
            (root / "outputs" / "r1" / "scores.csv").write_text("iptm,0.8\n")
            (root / "outputs" / "r1" / "deep" / "model.cif").write_text("data_x\n")
            (root / "scratch" / "r1" / "tmp.bin").write_text("ignored")
            self._stop(orchestration, payload, "child-1", summary="ipTM 0.8")

            run = self._run(root, "r1")
            self.assertEqual((run["state"], run["child_status"], run["manifest_status"]), ("completed", "completed", "recorded"))
            self.assertEqual([m["path"] for m in run["manifest"]], ["outputs/r1/deep/model.cif", "outputs/r1/scores.csv"])
            self.assertEqual(run["manifest"][1]["bytes"], len("iptm,0.8\n"))
            result = self._log(root)[-1]
            self.assertEqual((result["kind"], result["run"], result["state"]), ("result", "r1", "completed"))
            self.assertEqual(result["results"], "ipTM 0.8")
            self.assertEqual(result["artifacts"], ["outputs/r1/deep/model.cif", "outputs/r1/scores.csv"])
            verdict = orchestration.workspace_policy(
                {**payload, "session_id": "child-1", "tool_name": "write_file",
                 "arguments": {"path": str(root / "outputs" / "r1" / "late.csv"), "content": "x"}}
            )
            self.assertIn("frozen", verdict["message"])

    def test_child_status_maps_to_run_state(self) -> None:
        workspace = _load("loam_workspace_run_state")
        self.assertEqual(workspace.run_state("completed"), "completed")
        for status in ("interrupted", "cancelled"):
            self.assertEqual(workspace.run_state(status), "dropped")
        for status in ("error", "failed", "timeout", ""):
            self.assertEqual(workspace.run_state(status), "failed")

    def test_finalize_is_idempotent(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            self._stop(orchestration, payload, "child-1", status="error")
            orchestration.workspace_stop({**payload, "child_subagent_id": "child-1", "child_session_id": "child-1",
                                          "child_status": "completed", "child_summary": "again"})
            self.assertEqual(self._run(root, "r1")["state"], "failed")
            self.assertEqual([e["kind"] for e in self._log(root)].count("result"), 1)

    # remote manifest -----------------------------------------------------------------
    def test_remote_manifest_snippet_round_trips(self) -> None:
        """Run the exact program the child is given, then parse its output."""

        import subprocess

        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "remote" / "outputs" / "r1"
            (outputs / "sub").mkdir(parents=True)
            (outputs / "a.csv").write_text("1\n")
            (outputs / "sub" / "b.cif").write_text("data_b\n")
            block = orchestration._layout_block(str(Path(tmp) / "remote"), "r1", remote=True)
            program = block.split("paste its complete output verbatim at the end of your report:\n", 1)[1]
            printed = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True, check=True).stdout

            workspace = _load("loam_workspace_remote_manifest")
            entries, status = workspace.remote_manifest(f"Report.\n{printed}\nThanks.", "r1")
            self.assertEqual(status, "recorded")
            self.assertEqual([e["path"] for e in entries], ["outputs/r1/a.csv", "outputs/r1/sub/b.cif"])
            self.assertEqual(entries[0]["sha256"], hashlib.sha256(b"1\n").hexdigest())

    def test_remote_manifest_missing_or_invalid_is_never_guessed(self) -> None:
        workspace = _load("loam_workspace_remote_manifest_bad")
        good = json.dumps({"path": "a.csv", "sha256": "0" * 64, "bytes": 1})
        self.assertEqual(workspace.remote_manifest("no block here", "r1"), ([], "missing"))
        for line in (json.dumps({"path": "../../x", "sha256": "0" * 64, "bytes": 1}),
                     json.dumps({"path": "/etc/passwd", "sha256": "0" * 64, "bytes": 1}),
                     json.dumps({"path": "a.csv", "sha256": "nothex", "bytes": 1}),
                     "not json"):
            self.assertEqual(workspace.remote_manifest(f"LOAM_MANIFEST\n{line}\nEND_LOAM_MANIFEST", "r1"), ([], "invalid"))
        # A run with no output files prints an empty block: recorded, not missing.
        self.assertEqual(workspace.remote_manifest("LOAM_MANIFEST\nEND_LOAM_MANIFEST\n", "r1"), ([], "recorded"))
        # The LAST block wins, so a recap earlier in the report doesn't count.
        two = f"LOAM_MANIFEST\nnot json\nEND_LOAM_MANIFEST\n...\nLOAM_MANIFEST\n{good}\nEND_LOAM_MANIFEST"
        self.assertEqual(workspace.remote_manifest(two, "r1")[1], "recorded")

    def test_remote_child_stop_records_the_reported_manifest(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self.assertIn("LOAM_MANIFEST", tasks[0]["context"])
            self.assertIn(f"root = '{self.REMOTE}/h-1/outputs/r1'", tasks[0]["context"])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            line = json.dumps({"path": "fold.cif", "sha256": "a" * 64, "bytes": 10})
            self._stop(orchestration, payload, "child-1", summary=f"ok\nLOAM_MANIFEST\n{line}\nEND_LOAM_MANIFEST")
            run = self._run(root, "r1")
            self.assertEqual(run["manifest"], [{"path": "outputs/r1/fold.cif", "sha256": "a" * 64, "bytes": 10}])

    # close_workspace -----------------------------------------------------------------
    def test_validator_stop_closes_workspace_and_drops_unstarted_runs(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold.", "Score."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])  # r2 is never started
            self._stop(orchestration, payload, "child-1")
            self._stop(orchestration, payload, "validator-h-1", summary="### Evidence Score\n+0.40")

            experiment = json.loads((root / "experiment.json").read_text())
            self.assertEqual(experiment["state"], "closed")
            # Copied from the validator_completed row record_stop wrote, not re-decided.
            self.assertEqual((experiment["evidence_score"], experiment["evidence_score_status"]), (0.4, "reported"))
            self.assertEqual(self._run(root, "r2")["state"], "dropped")
            self.assertEqual(self._run(root, "r2")["dropped_reason"], "child never started")
            kinds = [e["kind"] for e in self._log(root)]
            self.assertEqual(kinds[-1], "report")
            self.assertIn("note", kinds)
            # The completion row itself is unchanged.
            completed = [r for r in self._rows(payload) if r.get("event") == "validator_completed"][-1]
            self.assertNotIn("workspace", completed)

    def test_close_is_idempotent_and_blocks_new_runs(self) -> None:
        orchestration = self._orchestration()
        workspace = _load("loam_workspace_close_twice")
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            root, _ = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._stop(orchestration, payload, "validator-h-1")
            lines = (root / "log.jsonl").read_text()
            self.assertEqual(workspace.close_workspace(root, {"status": "completed"}, remote=False), [])
            self.assertEqual((root / "log.jsonl").read_text(), lines)
            with self.assertRaises(ValueError):
                workspace.reserve_runs(root, 1)

    # audit ---------------------------------------------------------------------------
    def test_audit_reports_tampering_and_strays_without_repairing(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            root = self._completed_local_run(orchestration, payload, "h-1", {"a.csv": "1", "b.csv": "2"})
            (root / "outputs" / "r1" / "a.csv").write_text("edited")
            (root / "outputs" / "r1" / "b.csv").unlink()
            (root / "outputs" / "r1" / "c.csv").write_text("late")
            (root / "checkpoints").mkdir()
            (root / "outputs" / "r9").mkdir()
            lines = (root / "log.jsonl").read_text().splitlines()
            (root / "log.jsonl").write_text("\n".join([lines[0].replace("dispatched", "DISPATCHED"), *lines[1:]]) + "\n")
            self._stop(orchestration, payload, "validator-h-1")

            problems = [e["what"] for e in self._log(root) if e["kind"] == "violation"]
            for fragment in ("outputs/r1/a.csv changed", "outputs/r1/b.csv was removed", "outputs/r1/c.csv was added",
                             "stray top-level entry checkpoints", "outputs/r9 is not a run", "log.jsonl line 2"):
                self.assertTrue(any(fragment in p for p in problems), (fragment, problems))
            self.assertTrue((root / "checkpoints").is_dir())  # reported, not removed
            campaign = Path(payload["loam_state_dir"]) / "hypothesis-campaign.jsonl"
            rows = [json.loads(line) for line in campaign.read_text().splitlines()]
            violation = [r for r in rows if r["event"] == "workspace_violation"][-1]
            self.assertEqual((violation["hypothesis_id"], violation["workspace"]), ("h-1", str(root)))

    def test_clean_workspace_audits_clean(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            root = self._completed_local_run(orchestration, payload, "h-1", {"a.csv": "1"})
            self._stop(orchestration, payload, "validator-h-1")
            self.assertEqual([e for e in self._log(root) if e["kind"] == "violation"], [])
            campaign = Path(payload["loam_state_dir"]) / "hypothesis-campaign.jsonl"
            self.assertNotIn("workspace_violation", campaign.read_text())

    # cross-hypothesis inputs ---------------------------------------------------------
    def test_completed_outputs_of_another_hypothesis_are_readable_and_recorded(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            source = self._completed_local_run(orchestration, payload, "h-2", {"scores.csv": "iptm,0.8\n"})
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Reuse h-2 scores."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            call = {**payload, "session_id": "child-1", "tool_name": "read_file",
                    "arguments": {"path": str(source / "outputs" / "r1" / "scores.csv")}}
            self.assertIsNone(orchestration.workspace_policy(call))
            orchestration.workspace_capture(call)
            orchestration.workspace_capture(call)  # recorded once

            inputs = self._run(root, "r1")["inputs"]
            self.assertEqual(len(inputs), 1)
            self.assertEqual(
                {k: inputs[0][k] for k in ("plan_id", "hypothesis_id", "run", "path")},
                {"plan_id": "plan-1", "hypothesis_id": "h-2", "run": "r1", "path": "outputs/r1/scores.csv"},
            )
            self.assertEqual(inputs[0]["sha256"], hashlib.sha256(b"iptm,0.8\n").hexdigest())
            note = [e for e in self._log(root) if e["kind"] == "note" and "input" in e][-1]
            self.assertIn("h-2/outputs/r1/scores.csv", note["what"])

    def test_only_completed_outputs_of_other_hypotheses_are_readable(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            source = self._completed_local_run(orchestration, payload, "h-2", {"scores.csv": "1"})
            h3, h3_tasks = self._hypothesis(orchestration, payload, "h-3", ["Still running."])
            self._start(orchestration, payload, "child-h-3", h3_tasks[0]["goal"])  # h-3/r1 is started, not completed
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Reuse."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])

            def verdict(tool: str, **arguments):
                return orchestration.workspace_policy({**payload, "session_id": "child-1", "tool_name": tool, "arguments": arguments})

            self.assertIn("not completed", verdict("read_file", path=str(h3 / "outputs" / "r1" / "x"))["message"])
            self.assertIn("only outputs/rK/", verdict("read_file", path=str(source / "scratch" / "r1" / "x"))["message"])
            self.assertIn("only outputs/rK/", verdict("read_file", path=str(source / "log.jsonl"))["message"])
            self.assertIn("read-only to you", verdict("write_file", path=str(source / "outputs" / "r1" / "x"), content="x")["message"])
            workdir = str(root / "scratch" / "r1")
            self.assertIsNone(verdict("terminal", command=f"cp {source}/outputs/r1/scores.csv {workdir}/", workdir=workdir))
            self.assertIn("read-only to you", verdict("terminal", command=f"echo x > {source}/outputs/r1/x", workdir=workdir)["message"])

    def test_remote_code_may_read_completed_outputs_across_plans(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            # A completed run of h-0 in an EARLIER plan, made directly on disk.
            earlier = Path(payload["loam_workspace"]["root"]) / "plan-0" / "h-0"
            (earlier / "runs" / "r1").mkdir(parents=True)
            (earlier / "runs" / "r1" / "run.json").write_text(json.dumps(
                {"state": "completed", "manifest_status": "recorded",
                 "manifest": [{"path": "outputs/r1/lib.fasta", "sha256": "b" * 64, "bytes": 3}]}
            ))
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Reuse the earlier library."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            code = "open('/lus/campaign/work/plan-0/h-0/outputs/r1/lib.fasta').read()"
            call = {**payload, "session_id": "child-1", "tool_name": "mcp_hpc_run_code",
                    "arguments": {"payload": code, "workdir": f"{self.REMOTE}/h-1/scratch/r1"}}
            self.assertIsNone(orchestration.workspace_policy(call))
            orchestration.workspace_capture(call)
            inputs = self._run(root, "r1")["inputs"]
            self.assertEqual((inputs[0]["plan_id"], inputs[0]["hypothesis_id"], inputs[0]["sha256"]), ("plan-0", "h-0", "b" * 64))
            blocked = {**call, "arguments": {**call["arguments"], "payload": "open('/lus/campaign/work/plan-0/h-0/scratch/r1/t')"}}
            self.assertIn("only outputs/rK/", orchestration.workspace_policy(blocked)["message"])

    # end to end through the real subagent-stop.py script ------------------------------------
    def test_subagent_stop_script_finalizes_and_closes(self) -> None:
        import subprocess

        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            (root / "outputs" / "r1" / "a.csv").write_text("1")
            for session, summary in (("child-1", "done"), ("validator-h-1", "### Evidence Score\n0.10")):
                stop = {**payload, "event": "subagent_stop", "child_subagent_id": session, "child_session_id": session,
                        "child_status": "completed", "child_summary": summary}
                completed = subprocess.run([sys.executable, str(HOOKS / "subagent-stop.py")], input=json.dumps(stop),
                                           text=True, capture_output=True, cwd=str(ROOT), check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(self._run(root, "r1")["state"], "completed")
            experiment = json.loads((root / "experiment.json").read_text())
            self.assertEqual((experiment["state"], experiment["evidence_score"]), ("closed", 0.1))



VERIFIER = ROOT / ".loam" / "reconcile" / "verify_workspace.py"


class WorkspaceVerificationTests(_StopHelpers, unittest.TestCase):
    """Group 5: HPC path aliases, same-hypothesis inputs, unverified runs, the HPC verifier."""

    ALIAS = "/alias/campaign"  # another spelling of /lus/campaign

    def _aliased(self, payload: dict) -> dict:
        payload["loam_execution"]["workdir_aliases"] = {self.ALIAS: "/lus/campaign"}
        return payload

    def _policy(self, orchestration, base: dict, session: str, tool_name: str, /, **arguments):
        # Positional-only, so tool arguments named `payload` or `tool` don't collide.
        return orchestration.workspace_policy({**base, "session_id": session, "tool_name": tool_name, "arguments": arguments})

    def _fake_hpc(self, remote_base: Path):
        """A run_code that runs the program locally, with /lus/campaign/work mapped to ``remote_base``."""

        import subprocess

        calls = []

        def run_code(program: str, workdir: str) -> str:
            calls.append(workdir)
            program = program.replace("/lus/campaign/work", str(remote_base))
            return subprocess.run([sys.executable, "-c", program], text=True, capture_output=True, check=True).stdout

        return run_code, calls

    # config -----------------------------------------------------------------------
    def test_workdir_aliases_reach_the_payload_only_when_set(self) -> None:
        config = _load("loam_config_aliases", CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            raw = {"enabled": True, "active_profile": "research", "profiles": {"research": {
                "workflow_id": "research-v1", "state_dir": str(Path(tmp) / "state"),
                "hypothesis_orchestration": {"enabled": True},
                "workspace": {"enabled": True, "verifier_python": "/venv/bin/python",
                              "verifier_mcp_url": "http://127.0.0.1:8000/mcp"},
                "execution": {"mode": "local", "runtime": "python3.11",
                              "workdir_aliases": {"/lus/flare/projects/X/": "/flare/X", "relative": "/x"}},
            }}}
            path = Path(tmp) / "loam.yaml"
            path.write_text(json.dumps(raw))
            with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(path)}, clear=False):
                profile = config.load_profile()
            values = config.payload_values(profile)
            self.assertEqual(values["loam_execution"]["workdir_aliases"], {"/lus/flare/projects/X": "/flare/X"})
            self.assertEqual(values["loam_workspace"]["verifier_mcp_url"], "http://127.0.0.1:8000/mcp")
            del raw["profiles"]["research"]["execution"]["workdir_aliases"]
            path.write_text(json.dumps(raw))
            with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(path)}, clear=False):
                self.assertNotIn("workdir_aliases", config.payload_values(config.load_profile())["loam_execution"])

    # aliases ----------------------------------------------------------------------
    def test_aliased_hpc_spelling_is_the_same_place(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._aliased(self._payload(Path(tmp)))
            self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            task = orchestration.canonical_delegation_args(
                {**payload, "session_id": "validator-h-1", "arguments": {"tasks": [{"goal": "Again."}]}})["tasks"][0]
            self._start(orchestration, payload, "child-1", task["goal"])  # run r2
            aliased = f"{self.ALIAS}/work/plan-1/h-1"
            ok = self._policy(orchestration, payload, "child-1", "mcp_hpc_run_code", payload="print(1)", workdir=f"{aliased}/scratch/r2")
            self.assertIsNone(ok)
            sibling = self._policy(orchestration, payload, "child-1", "mcp_hpc_run_code", payload="print(1)",
                                   workdir=f"{self.ALIAS}/work/plan-1/h-2/scratch/r1")
            self.assertIn("another hypothesis", sibling["message"])
            code = f"open('{self.ALIAS}/work/plan-1/h-2/scratch/r1/x')"
            blocked = self._policy(orchestration, payload, "child-1", "mcp_hpc_run_code", payload=code, workdir=f"{aliased}/scratch/r2")
            self.assertIn("another hypothesis", blocked["message"])

    # same-hypothesis inputs ---------------------------------------------------------
    def test_earlier_runs_of_the_same_hypothesis_follow_the_read_rule(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp), mode="local")
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["First.", "Second."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])  # r1
            self._start(orchestration, payload, "child-2", tasks[1]["goal"])  # r2
            (root / "outputs" / "r1" / "a.csv").write_text("1\n")
            target = str(root / "outputs" / "r1" / "a.csv")

            running = self._policy(orchestration, payload, "child-2", "read_file", path=target)
            self.assertIn("not completed", running["message"])
            self.assertIn("this hypothesis", running["message"])
            self._stop(orchestration, payload, "child-1")
            call = {**payload, "session_id": "child-2", "tool_name": "read_file", "arguments": {"path": target}}
            self.assertIsNone(orchestration.workspace_policy(call))
            orchestration.workspace_capture(call)
            inputs = self._run(root, "r2")["inputs"]
            self.assertEqual((inputs[0]["hypothesis_id"], inputs[0]["run"], inputs[0]["path"]), ("h-1", "r1", "outputs/r1/a.csv"))
            self.assertEqual(inputs[0]["sha256"], hashlib.sha256(b"1\n").hexdigest())
            scratch = self._policy(orchestration, payload, "child-2", "read_file", path=str(root / "scratch" / "r1" / "x"))
            self.assertIn("only outputs/rK/", scratch["message"])
            # Own run, src/ and the record stay freely readable.
            for own in ("outputs/r2/x", "src/a.py", "log.jsonl", "data/t.fasta"):
                self.assertIsNone(self._policy(orchestration, payload, "child-2", "read_file", path=str(root / own)), own)

    # unverified runs ----------------------------------------------------------------
    def test_completed_remote_run_without_manifest_is_unreadable_until_verified(self) -> None:
        orchestration = self._orchestration()
        workspace = sys.modules["hypothesis_workspace"]
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            source, tasks = self._hypothesis(orchestration, payload, "h-2", ["Make."])
            self._start(orchestration, payload, "child-h-2", tasks[0]["goal"])
            self._stop(orchestration, payload, "child-h-2", summary="done, no manifest")
            self.assertEqual(self._run(source, "r1")["manifest_status"], "missing")
            _, tasks = self._hypothesis(orchestration, payload, "h-1", ["Reuse."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            code = "open('/lus/campaign/work/plan-1/h-2/outputs/r1/x.csv')"
            args = {"payload": code, "workdir": "/lus/campaign/work/plan-1/h-1/scratch/r1"}
            blocked = self._policy(orchestration, payload, "child-1", "mcp_hpc_run_code", **args)
            self.assertIn("no verified output manifest", blocked["message"])
            workspace.record_verified_manifest(source, "r1", [{"path": "outputs/r1/x.csv", "sha256": "c" * 64, "bytes": 1}])
            self.assertIsNone(self._policy(orchestration, payload, "child-1", "mcp_hpc_run_code", **args))

    # the verifier -------------------------------------------------------------------
    def test_fill_hashes_outputs_on_hpc_and_records_them(self) -> None:
        orchestration = self._orchestration()
        verifier = _load("loam_verify_fill", VERIFIER)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            self._stop(orchestration, payload, "child-1", summary="forgot the manifest")
            remote = Path(tmp) / "hpc"
            (remote / "plan-1" / "h-1" / "outputs" / "r1").mkdir(parents=True)
            (remote / "plan-1" / "h-1" / "outputs" / "r1" / "fold.cif").write_text("data_x\n")
            run_code, calls = self._fake_hpc(remote)

            self.assertEqual(verifier.fill(root, "r1", "/lus/campaign/work/plan-1/h-1", run_code), "recorded")
            run = self._run(root, "r1")
            self.assertEqual(run["manifest_status"], "recorded_by_loam")
            self.assertIn("verified_at", run)
            self.assertEqual(run["manifest"], [{"path": "outputs/r1/fold.cif", "sha256": hashlib.sha256(b"data_x\n").hexdigest(), "bytes": 7}])
            self.assertEqual(calls, ["/lus/campaign/work/plan-1/h-1"])
            self.assertIn("recorded by the Loam verifier", self._log(root)[-1]["what"])
            self.assertEqual(verifier.fill(root, "r1", "/lus/campaign/work/plan-1/h-1", run_code), "skipped")
            self.assertEqual(sys.modules["hypothesis_workspace"].verify_log(root), [])

    def test_fill_failures_are_logged_never_guessed(self) -> None:
        orchestration = self._orchestration()
        verifier = _load("loam_verify_fill_fail", VERIFIER)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            self._stop(orchestration, payload, "child-1")

            def down(program: str, workdir: str) -> str:
                raise ConnectionError("server unreachable")

            self.assertEqual(verifier.fill(root, "r1", "/lus/x", down), "failed")
            self.assertIn("server unreachable", self._log(root)[-1]["what"])
            self.assertEqual(verifier.fill(root, "r1", "/lus/x", lambda p, w: "garbage"), "failed")
            self.assertEqual(self._run(root, "r1")["manifest_status"], "missing")

    def test_recheck_reports_remote_outputs_changed_after_freeze(self) -> None:
        orchestration = self._orchestration()
        verifier = _load("loam_verify_recheck", VERIFIER)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            self._stop(orchestration, payload, "child-1")
            outputs = Path(tmp) / "hpc" / "plan-1" / "h-1" / "outputs" / "r1"
            outputs.mkdir(parents=True)
            (outputs / "a.csv").write_text("1")
            run_code, _ = self._fake_hpc(Path(tmp) / "hpc")
            verifier.fill(root, "r1", "/lus/campaign/work/plan-1/h-1", run_code)
            self.assertEqual(verifier.recheck(root, "/lus/campaign/work/plan-1/h-1", run_code), [])
            (outputs / "a.csv").write_text("edited")
            (outputs / "b.csv").write_text("late")
            problems = verifier.recheck(root, "/lus/campaign/work/plan-1/h-1", run_code)
            self.assertEqual(problems, ["outputs/r1/a.csv changed after run r1 finished", "outputs/r1/b.csv was added after run r1 finished"])
            self.assertEqual([e["what"] for e in self._log(root)[-2:]], problems)

    def test_stop_hook_launches_the_verifier_detached(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            payload["loam_workspace"].update(verifier_python="/venv/bin/python", verifier_mcp_url="http://127.0.0.1:8000/mcp")
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            with patch("subprocess.Popen") as popen:
                self._stop(orchestration, payload, "child-1", summary="no manifest")
                self._stop(orchestration, payload, "validator-h-1")
            commands = [call.args[0] for call in popen.call_args_list]
            self.assertEqual([c[2] for c in commands], ["fill", "recheck"])
            self.assertEqual(commands[0][:2], ["/venv/bin/python", str(VERIFIER)])
            self.assertIn("r1", commands[0])
            for flag, value in (("--workspace", str(root)), ("--remote-root", "/lus/campaign/work/plan-1/h-1"),
                                ("--mcp-url", "http://127.0.0.1:8000/mcp"), ("--tool", "run_code")):
                self.assertEqual(commands[0][commands[0].index(flag) + 1], value)
            self.assertTrue(all(call.kwargs["start_new_session"] for call in popen.call_args_list))
            self.assertIn("verifier launched: fill --run r1", [e.get("what") for e in self._log(root)])

    def test_no_verifier_configured_leaves_a_note(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            with patch("subprocess.Popen") as popen:
                self._stop(orchestration, payload, "child-1", summary="no manifest")
            popen.assert_not_called()
            self.assertTrue(any("verifier not configured" in e.get("what", "") for e in self._log(root)))

    def test_child_with_manifest_does_not_launch_fill(self) -> None:
        orchestration = self._orchestration()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            payload["loam_workspace"].update(verifier_python="/venv/bin/python", verifier_mcp_url="http://x/mcp")
            _, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            line = json.dumps({"path": "a.cif", "sha256": "a" * 64, "bytes": 1})
            with patch("subprocess.Popen") as popen:
                self._stop(orchestration, payload, "child-1", summary=f"LOAM_MANIFEST\n{line}\nEND_LOAM_MANIFEST")
            popen.assert_not_called()

    def test_verifier_cli_runs_fill(self) -> None:
        import contextlib
        import io

        orchestration = self._orchestration()
        verifier = _load("loam_verify_cli", VERIFIER)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(Path(tmp))
            root, tasks = self._hypothesis(orchestration, payload, "h-1", ["Fold."])
            self._start(orchestration, payload, "child-1", tasks[0]["goal"])
            self._stop(orchestration, payload, "child-1")
            (Path(tmp) / "hpc" / "plan-1" / "h-1" / "outputs" / "r1").mkdir(parents=True)
            run_code, _ = self._fake_hpc(Path(tmp) / "hpc")
            out = io.StringIO()
            with patch.object(verifier, "mcp_run_code", return_value=run_code), contextlib.redirect_stdout(out):
                code = verifier.main(["fill", "--workspace", str(root), "--run", "r1", "--remote-root",
                                      "/lus/campaign/work/plan-1/h-1", "--mcp-url", "http://x/mcp"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.getvalue())["result"], "recorded")


if __name__ == "__main__":
    unittest.main()
