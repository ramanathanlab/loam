"""The `metrics:` W&B exporter: config parsing and the in-process engine.

Two layers are covered. The config layer (loam_config) is held to the same
loud-on-typo contract as `execution`/`campaign`: a malformed block raises, an
absent one is inert, and a valid one reaches `payload_values` without a secret.
The engine (wandb_metrics) is exercised with wandb BOTH absent (it must stay
inert and never raise) and present-via-a-fake (token/tool attribution lands on
the right agent role, aggregates sum, the local JSONL is durable, and the run
is finished once).
"""

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
PLUGIN = ROOT / ".loam" / "adapters" / "hermes" / "plugin"
CONFIG = PLUGIN / "loam_config.py"
ENGINE = PLUGIN / "wandb_metrics.py"


def _load(name: str, path: Path):
    if str(PLUGIN) not in sys.path:
        sys.path.insert(0, str(PLUGIN))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _profile_with_metrics(config, metrics: object | None) -> object | None:
    research: dict[str, object] = {
        "workflow_id": "research-v1",
        "state_dir": "/tmp/state",
        "hypothesis_orchestration": {"enabled": True},
        "execution": {"mode": "local", "runtime": "python3.11"},
    }
    if metrics is not None:
        research["metrics"] = metrics
    raw = {"enabled": True, "active_profile": "research", "profiles": {"research": research}}
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "loam.yaml"
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(config_path)}, clear=False):
            return config.load_profile()


# --------------------------------------------------------------------------
# Config layer
# --------------------------------------------------------------------------


class MetricsConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _load("loam_config_metrics", CONFIG)

    def test_absent_block_is_inert(self) -> None:
        profile = _profile_with_metrics(self.config, None)
        assert profile is not None
        self.assertIsNone(profile.metrics)
        self.assertNotIn("loam_metrics", self.config.payload_values(profile))

    def test_disabled_block_is_inert(self) -> None:
        profile = _profile_with_metrics(self.config, {"enabled": False, "project": "x"})
        assert profile is not None
        self.assertIsNone(profile.metrics)

    def test_valid_block_parses_and_reaches_payload(self) -> None:
        profile = _profile_with_metrics(
            self.config,
            {
                "enabled": True,
                "project": "alkbh1-metrics",
                "entity": "ramanathan-lab",
                "mode": "offline",
                "tags": ["alkbh1", "alkbh1", "pilot"],
                "api_key_env": "MY_WANDB_KEY",
                "table_log_every": 10,
            },
        )
        assert profile is not None and profile.metrics is not None
        self.assertEqual(profile.metrics.project, "alkbh1-metrics")
        self.assertEqual(profile.metrics.mode, "offline")
        # Tags are de-duplicated preserving order.
        self.assertEqual(profile.metrics.tags, ("alkbh1", "pilot"))
        payload = self.config.payload_values(profile)["loam_metrics"]
        self.assertEqual(payload["project"], "alkbh1-metrics")
        self.assertEqual(payload["api_key_env"], "MY_WANDB_KEY")
        # The payload carries the variable NAME, never a resolved secret value.
        self.assertNotIn("api_key", payload)

    def test_api_key_env_defaults(self) -> None:
        profile = _profile_with_metrics(self.config, {"enabled": True, "project": "p"})
        assert profile is not None and profile.metrics is not None
        self.assertEqual(profile.metrics.api_key_env, "WANDB_API_KEY")
        self.assertEqual(profile.metrics.mode, "online")

    def test_missing_project_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.config._metrics_config({"enabled": True})

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.config._metrics_config({"enabled": True, "project": "p", "mode": "wandb-cloud"})

    def test_bad_tags_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.config._metrics_config({"enabled": True, "project": "p", "tags": "alkbh1"})


# --------------------------------------------------------------------------
# Engine layer
# --------------------------------------------------------------------------


class _Cfg:
    enabled = True
    project = "p"
    mode = "offline"  # avoids the online no-key refusal in init
    entity = ""
    run_name = ""
    group = ""
    tags = ()
    api_key_env = "WANDB_API_KEY"
    base_url = ""
    table_log_every = 2


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.summary: dict = {}
        self.finished = 0

    def log(self, data, step=None):  # noqa: D401, ANN001
        self.logged.append({"step": step, "data": dict(data)})

    def finish(self):  # noqa: D401
        self.finished += 1


class _FakeWandb:
    def __init__(self) -> None:
        self.runs: list[_FakeRun] = []
        self.init_kwargs: dict | None = None

    class Settings:  # mirror wandb.Settings(silent=...)
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

    class Table:
        def __init__(self, columns, data):  # noqa: ANN001
            self.columns = columns
            self.data = data

    def init(self, **kwargs):  # noqa: ANN003
        self.init_kwargs = kwargs
        run = _FakeRun()
        self.runs.append(run)
        return run


class MetricsEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wm = _load("wandb_metrics_under_test", ENGINE)
        self.wm._reset_for_tests()
        self.addCleanup(self.wm._reset_for_tests)
        self.addCleanup(lambda: sys.modules.pop("wandb", None))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state_dir = Path(tmp.name)

    def _events(self) -> list[dict]:
        path = self.state_dir / "metrics-events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_inert_without_wandb_never_raises(self) -> None:
        # Setting sys.modules["wandb"] = None makes `import wandb` raise
        # ImportError, simulating wandb genuinely not installed -> every W&B
        # path is a no-op, but the durable JSONL still records every event.
        sys.modules["wandb"] = None  # type: ignore[assignment]
        self.wm.configure(_Cfg(), state_dir=self.state_dir, run_id="r1", workflow_id="wf")
        self.wm.note_subagent_start("child-sess", "code-generator")
        self.wm.record_api_request(
            {"session_id": "child-sess", "model": "m", "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}
        )
        self.wm.record_tool_call({"session_id": "child-sess", "tool_name": "run_code", "duration_ms": 1500, "status": "ok"})
        self.wm.finish()
        events = self._events()
        self.assertEqual(len(events), 2)
        self.assertEqual(self.wm._tokens_by_role["code-generator"]["total"], 120)

    def test_attribution_and_aggregation_with_fake_wandb(self) -> None:
        fake = _FakeWandb()
        sys.modules["wandb"] = fake
        self.wm.configure(_Cfg(), state_dir=self.state_dir, run_id="run/42", workflow_id="wf")

        self.wm.note_subagent_start("val-sess", "hypothesis-validator")
        self.wm.note_subagent_start("gen-sess", "code-generator")

        # coordinator (no registered session) + two children
        self.wm.record_api_request({"session_id": "", "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}})
        self.wm.record_api_request(
            {"session_id": "gen-sess", "usage": {"input_tokens": 200, "output_tokens": 50, "reasoning_tokens": 7, "total_tokens": 257}}
        )
        self.wm.record_api_request(
            {"session_id": "val-sess", "usage": {"input_tokens": 80, "output_tokens": 8, "total_tokens": 88}}
        )
        self.wm.record_tool_call({"session_id": "gen-sess", "tool_name": "run_code", "duration_ms": 1000, "status": "ok"})
        self.wm.record_tool_call({"session_id": "gen-sess", "tool_name": "run_code", "duration_ms": 3000, "status": "error"})
        self.wm.note_subagent_stop("gen-sess", "code-generator", 60000, "completed")
        self.wm.finish()

        # Token attribution by role
        self.assertEqual(self.wm._tokens_by_role["coordinator"]["total"], 12)
        self.assertEqual(self.wm._tokens_by_role["code-generator"]["total"], 257)
        self.assertEqual(self.wm._tokens_by_role["code-generator"]["reasoning"], 7)
        self.assertEqual(self.wm._tokens_by_role["hypothesis-validator"]["total"], 88)
        # Tool timing by tool and by agent
        self.assertEqual(self.wm._tool_by_name["run_code"]["count"], 2)
        self.assertEqual(self.wm._tool_by_name["run_code"]["time_ms"], 4000)
        self.assertEqual(self.wm._tool_by_name["run_code"]["errors"], 1)
        self.assertEqual(self.wm._tool_by_role["code-generator"]["time_ms"], 4000)
        self.assertEqual(self.wm._agent_wall_ms["code-generator"], 60000)

        # A run was created, finished exactly once, with id slashes sanitized.
        self.assertEqual(len(fake.runs), 1)
        self.assertEqual(fake.runs[0].finished, 1)
        self.assertEqual(fake.init_kwargs["id"], "run_42")
        self.assertEqual(fake.init_kwargs["mode"], "offline")
        self.assertEqual(fake.runs[0].summary["tokens_total"], 12 + 257 + 88)
        self.assertEqual(fake.runs[0].summary["tool_time_ms_total"], 4000)
        # All five events are in the durable log.
        self.assertEqual(len(self._events()), 5)

    def test_online_without_credential_stays_inert(self) -> None:
        fake = _FakeWandb()
        sys.modules["wandb"] = fake
        cfg = _Cfg()
        cfg.mode = "online"
        # No env key and no netrc login -> refuse online init (would otherwise
        # hang on a prompt). Patch the credential check so the result does not
        # depend on whether the test machine happens to have a ~/.netrc.
        with patch.object(self.wm, "_has_online_credential", return_value=False):
            self.wm.configure(cfg, state_dir=self.state_dir, run_id="r", workflow_id="wf")
            self.wm.record_api_request({"session_id": "s", "usage": {"input_tokens": 1, "total_tokens": 1}})
            self.wm.finish()
        # No run created, but the durable JSONL still kept the event.
        self.assertEqual(len(fake.runs), 0)
        self.assertEqual(len(self._events()), 1)

    def test_online_with_credential_initializes(self) -> None:
        # A credential present (env key OR a `wandb login` in ~/.netrc) lets the
        # online run init -- a netrc login must not require also exporting the key.
        fake = _FakeWandb()
        sys.modules["wandb"] = fake
        cfg = _Cfg()
        cfg.mode = "online"
        with patch.object(self.wm, "_has_online_credential", return_value=True):
            self.wm.configure(cfg, state_dir=self.state_dir, run_id="r", workflow_id="wf")
            self.wm.record_api_request({"session_id": "s", "usage": {"total_tokens": 1}})
            self.wm.finish()
        self.assertEqual(len(fake.runs), 1)
        self.assertEqual(fake.init_kwargs["mode"], "online")

    def test_has_online_credential_honours_env_and_netrc(self) -> None:
        wm = self.wm

        class _FakeNetrc:
            def __init__(self, *_a):  # netrc.netrc() takes no args here
                pass

            def authenticators(self, host):  # noqa: ANN001
                return ("user", None, "secret") if host == "api.wandb.ai" else None

        def _no_file(*_a):
            raise FileNotFoundError

        with patch.dict(os.environ, {}, clear=False):
            # Configured env var name holds a key -> credential present.
            os.environ["MY_WANDB_KEY"] = "k"
            self.assertTrue(wm._has_online_credential("MY_WANDB_KEY"))
            os.environ.pop("MY_WANDB_KEY", None)
            os.environ.pop("WANDB_API_KEY", None)

            # No env key, but netrc has the default host -> credential present.
            with patch.object(wm.netrc, "netrc", _FakeNetrc):
                self.assertTrue(wm._has_online_credential("WANDB_API_KEY"))
                # A self-hosted base_url whose host is NOT in netrc -> absent.
                self.assertFalse(
                    wm._has_online_credential("WANDB_API_KEY", base_url="https://wandb.my-lab.org")
                )
            # No env key and no ~/.netrc at all -> absent (no raise).
            with patch.object(wm.netrc, "netrc", _no_file):
                self.assertFalse(wm._has_online_credential("WANDB_API_KEY"))

    def test_finish_is_idempotent(self) -> None:
        fake = _FakeWandb()
        sys.modules["wandb"] = fake
        self.wm.configure(_Cfg(), state_dir=self.state_dir, run_id="r", workflow_id="wf")
        self.wm.record_api_request({"session_id": "s", "usage": {"total_tokens": 5}})
        self.wm.finish()
        self.wm.finish()
        self.assertEqual(fake.runs[0].finished, 1)

    def test_tool_call_is_linked_to_its_inference_with_gap_and_tokens(self) -> None:
        import time as _time

        sys.modules["wandb"] = None  # type: ignore[assignment]
        self.wm.configure(_Cfg(), state_dir=self.state_dir, run_id="r", workflow_id="wf")
        self.wm.note_subagent_start("gen", "code-generator")
        now = _time.time()
        self.wm.record_api_request(
            {
                "session_id": "gen",
                "api_request_id": "req-1",
                "api_duration": 2.5,
                "ended_at": now,
                "usage": {"input_tokens": 500, "output_tokens": 120, "reasoning_tokens": 40, "total_tokens": 620},
            }
        )
        # Tool starts 0.2 s after the inference ended, then runs for 1.8 s.
        self.wm.note_tool_start("gen", "tc-1", "t1")
        # Overwrite the stamped start so the gap is deterministic (not wall time).
        self.wm._tool_start[("gen", "tc-1")] = now + 0.2
        rec = self.wm.record_tool_call(
            {
                "session_id": "gen",
                "tool_call_id": "tc-1",
                "api_request_id": "req-1",
                "tool_name": "run_code",
                "duration_ms": 1800,
                "status": "ok",
            }
        )
        assert rec is not None
        self.assertEqual(rec["duration_ms"], 1800)
        self.assertEqual(rec["inference"]["duration_ms"], 2500.0)
        self.assertEqual(rec["inference"]["tokens"]["total"], 620)
        self.assertEqual(rec["inference"]["tokens"]["reasoning"], 40)
        self.assertAlmostEqual(rec["gap_from_inference_ms"], 200.0, delta=5.0)
        self.assertEqual(rec["inference"]["api_request_id"], "req-1")

    def test_tool_call_without_inference_still_records(self) -> None:
        sys.modules["wandb"] = None  # type: ignore[assignment]
        self.wm.configure(_Cfg(), state_dir=self.state_dir, run_id="r", workflow_id="wf")
        rec = self.wm.record_tool_call({"session_id": "x", "tool_name": "terminal", "duration_ms": 10, "status": "ok"})
        assert rec is not None
        self.assertEqual(rec["duration_ms"], 10)
        self.assertIsNone(rec["gap_from_inference_ms"])
        self.assertIsNone(rec["inference"]["duration_ms"])


class WorkspaceMetricsTests(unittest.TestCase):
    """hypothesis_workspace.record_tool_metrics writes runs/<run_id>/metrics.jsonl."""

    def setUp(self) -> None:
        if str(HOOKS := ROOT / ".loam" / "hooks") not in sys.path:
            sys.path.insert(0, str(HOOKS))
        spec = importlib.util.spec_from_file_location("hw_metrics_test", HOOKS / "hypothesis_workspace.py")
        assert spec and spec.loader
        self.ws = importlib.util.module_from_spec(spec)
        sys.modules["hw_metrics_test"] = self.ws
        spec.loader.exec_module(self.ws)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.payload = {
            "workflow_id": "research-v1",
            "loam_run_id": "run-1",
            "loam_workspace": {"enabled": True, "root": str(self.base / "hypotheses"), "strict": True},
        }
        self.root = self.ws.ensure_workspace(self.payload, "plan-1", "h-1")
        self.run_id = self.ws.allocate_run(self.root, "gen-sess", "fold it", "validator asked")
        self.start = {
            "role": "code-generator",
            "workspace": str(self.root),
            "run": self.run_id,
            "child_session_id": "gen-sess",
        }

    def _metric(self) -> dict:
        return {"kind": "tool_call", "tool_name": "run_code", "duration_ms": 1800, "gap_from_inference_ms": 12.0}

    def test_writes_one_line_per_tool_call_for_a_code_generator(self) -> None:
        self.ws.record_tool_metrics(self.payload, self.start, self._metric())
        self.ws.record_tool_metrics(self.payload, self.start, self._metric())
        path = self.root / "runs" / self.run_id / "metrics.jsonl"
        self.assertTrue(path.is_file())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["run"], self.run_id)
        self.assertEqual(rows[0]["tool_name"], "run_code")
        self.assertIn("observed_at", rows[0])

    def test_no_op_for_non_code_generator_role(self) -> None:
        start = {**self.start, "role": "hypothesis-validator"}
        self.assertIsNone(self.ws.record_tool_metrics(self.payload, start, self._metric()))
        self.assertFalse((self.root / "runs" / self.run_id / "metrics.jsonl").exists())

    def test_no_op_when_run_folder_missing(self) -> None:
        start = {**self.start, "run": "r9"}
        self.assertIsNone(self.ws.record_tool_metrics(self.payload, start, self._metric()))

    def test_metrics_file_is_not_flagged_by_the_workspace_audit(self) -> None:
        self.ws.record_tool_metrics(self.payload, self.start, self._metric())
        experiment = self.ws.read_json(self.root / "experiment.json")
        problems = self.ws._audit_unlocked(self.root, experiment, remote=False)
        self.assertEqual([p for p in problems if "metrics.jsonl" in p], [])


if __name__ == "__main__":
    unittest.main()
