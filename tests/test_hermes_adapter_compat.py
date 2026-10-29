"""Compatibility of the Hermes adapter with newer Hermes hook payloads and runtimes."""

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


def _load(name: str, path: Path):
    if str(PLUGIN) not in sys.path:
        sys.path.insert(0, str(PLUGIN))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _profile_json(state_dir: Path) -> str:
    return json.dumps(
        {
            "enabled": True,
            "active_profile": "compat",
            "profiles": {"compat": {"workflow_id": "compat-wf", "state_dir": str(state_dir), "pre_llm_every": 1}},
        }
    )


class HermesAdapterCompatTests(unittest.TestCase):
    def test_pre_llm_accepts_hermes_parent_session_id_kwarg(self) -> None:
        adapter = _load("loam_adapter_compat_parent", PLUGIN / "__init__.py")
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "loam.yaml"
            config.write_text(_profile_json(Path(tmp) / "state"), encoding="utf-8")
            captured = {}

            def fake_invoke(script, payload, timeout_seconds=0):
                captured.update(payload)
                return None

            with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(config)}, clear=False), patch.object(
                adapter, "_invoke", fake_invoke
            ):
                adapter.pre_llm_call(session_id="coord-1", parent_session_id="hermes-parent-9", is_first_turn=True)

        self.assertEqual(captured["session_id"], "coord-1")
        self.assertEqual(captured["parent_session_id"], "coord-1")
        self.assertEqual(captured["hermes_parent_session_id"], "hermes-parent-9")

    def test_hook_interpreter_can_be_overridden(self) -> None:
        adapter = _load("loam_adapter_compat_python", PLUGIN / "__init__.py")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)

            class Done:
                returncode = 0
                stdout = ""
                stderr = ""

            return Done()

        with patch.object(adapter.subprocess, "run", fake_run):
            with patch.dict(os.environ, {"LOAM_HOOK_PYTHON": "/opt/loam-venv/bin/python"}, clear=False):
                adapter._invoke("pre-llm.py", {"event": "pre_llm_call"})
            env = {k: v for k, v in os.environ.items() if k != "LOAM_HOOK_PYTHON"}
            with patch.dict(os.environ, env, clear=True):
                adapter._invoke("pre-llm.py", {"event": "pre_llm_call"})

        self.assertEqual(calls[0][0], "/opt/loam-venv/bin/python")
        self.assertEqual(calls[1][0], sys.executable)

    def test_override_interpreter_does_not_inherit_hermes_pythonpath(self) -> None:
        adapter = _load("loam_adapter_compat_env", PLUGIN / "__init__.py")
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(kwargs.get("env"))

            class Done:
                returncode = 0
                stdout = ""
                stderr = ""

            return Done()

        hermes_env = {"PYTHONPATH": "/hermes/site-packages", "PYTHONHOME": "/hermes", "KEEP_ME": "1"}
        with patch.object(adapter.subprocess, "run", fake_run):
            with patch.dict(os.environ, {**hermes_env, "LOAM_HOOK_PYTHON": "/opt/loam-venv/bin/python"}, clear=False):
                adapter._invoke("pre-llm.py", {"event": "pre_llm_call"})
            env = {k: v for k, v in os.environ.items() if k != "LOAM_HOOK_PYTHON"}
            with patch.dict(os.environ, {**env, **hermes_env}, clear=True):
                adapter._invoke("pre-llm.py", {"event": "pre_llm_call"})

        override_env, default_env = seen
        self.assertNotIn("PYTHONPATH", override_env)
        self.assertNotIn("PYTHONHOME", override_env)
        self.assertEqual(override_env["KEEP_ME"], "1")
        # Hermes's own interpreter keeps inheriting its environment unchanged.
        self.assertIsNone(default_env)

    def test_profile_loads_as_json_without_pyyaml(self) -> None:
        config_module = _load("loam_config_compat_json", PLUGIN / "loam_config.py")
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "loam.yaml"
            config.write_text(_profile_json(Path(tmp) / "state"), encoding="utf-8")
            with patch.object(config_module, "yaml", None), patch.dict(
                os.environ, {"LOAM_CONFIG_PATH": str(config)}, clear=False
            ):
                profile = config_module.load_profile()

        self.assertIsNotNone(profile)
        self.assertEqual(profile.workflow_id, "compat-wf")


if __name__ == "__main__":
    unittest.main()
