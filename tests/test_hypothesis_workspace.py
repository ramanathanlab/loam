"""Tests for the Pando-style per-hypothesis workspace."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
