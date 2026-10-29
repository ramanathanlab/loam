"""FALDA block parsing (on by default from .loam/falda.json), and its run-manifest summary."""

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
HOOKS = ROOT / ".loam" / "hooks"
POOL = "efb99ee4-c412-11f1-8af6-0affe8db6a05"


def _load(name: str, path: Path):
    for directory in (PLUGIN, HOOKS):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _block(**overrides):
    block = {"enabled": True, "base_url": "https://falda.example", "pool": POOL}
    block.update(overrides)
    return block


_ABSENT = object()


class FaldaConfigTests(unittest.TestCase):
    def _profile(self, falda=_ABSENT, workflow_id="wf-1", extra_env=None):
        config = _load("loam_config_falda", PLUGIN / "loam_config.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = {"workflow_id": workflow_id, "state_dir": str(root / "state"), "pre_llm_every": 1}
            if falda is not _ABSENT:
                profile["falda"] = falda
            path = root / "loam.yaml"
            path.write_text(
                json.dumps({"enabled": True, "active_profile": "p", "profiles": {"p": profile}}),
                encoding="utf-8",
            )
            env = {key: value for key, value in os.environ.items() if not key.startswith("LOAM_")}
            env["LOAM_CONFIG_PATH"] = str(path)
            env.update(extra_env or {})
            with patch.dict(os.environ, env, clear=True):
                loaded = config.load_profile()
                values = config.payload_values(loaded) if loaded else None
        return loaded, values

    def test_absent_block_uses_repo_default(self) -> None:
        profile, values = self._profile()
        self.assertEqual(profile.falda_status, "enabled")
        falda = values["loam_falda"]
        self.assertEqual(falda["pool"], POOL)
        self.assertEqual(falda["base_url"], "https://falda.cairnscore.ai")
        self.assertEqual(falda["recall_timeout_seconds"], 2)
        # Writes run in the detached flusher; FALDA takes ~1 s per atom, so the write timeout is generous.
        self.assertEqual((falda["recall_budget_chars"], falda["timeout_seconds"]), (1500, 60))
        self.assertEqual((falda["token_env"], falda["key_dir"]), ("FALDA_TOKEN", "~/.falda/keys"))
        for switch in ("recall_generation", "recall_reasoner", "write_validator_outcomes", "reasoner_tools",
                       "auto_mint", "auto_flush"):
            self.assertIs(falda[switch], True, switch)
        self.assertNotIn("tenant_env", falda)
        self.assertEqual(values["loam_falda_status"], "enabled")

    def test_env_switch_disables(self) -> None:
        for value in ("0", "off", "false", "no"):
            with self.subTest(value=value):
                profile, values = self._profile(extra_env={"LOAM_FALDA": value})
                self.assertIsNone(profile.falda)
                self.assertEqual(profile.falda_status, "disabled")
                self.assertNotIn("loam_falda", values)

    def test_profile_block_wins(self) -> None:
        profile, values = self._profile({"enabled": False})
        self.assertIsNone(profile.falda)
        self.assertEqual(values["loam_falda_status"], "disabled")
        profile, values = self._profile(_block(recall_timeout_seconds=1))
        self.assertEqual(values["loam_falda"]["base_url"], "https://falda.example")
        self.assertEqual(values["loam_falda"]["recall_timeout_seconds"], 1)

    def test_switches_are_independent(self) -> None:
        profile, values = self._profile(_block(recall_generation=False, reasoner_tools=False))
        self.assertFalse(values["loam_falda"]["recall_generation"])
        self.assertTrue(values["loam_falda"]["recall_reasoner"])
        self.assertTrue(values["loam_falda"]["write_validator_outcomes"])
        self.assertFalse(values["loam_falda"]["reasoner_tools"])

    def test_non_boolean_switch_rejects_the_block(self) -> None:
        profile, _ = self._profile(_block(reasoner_tools="yes"))
        self.assertIsNone(profile.falda)
        self.assertEqual(profile.falda_status, "rejected:bad_switch")

    def test_missing_or_invalid_pool_rejects_only_the_block(self) -> None:
        for pool in (None, "", "self", "not-a-uuid"):
            with self.subTest(pool=pool):
                block = _block()
                if pool is None:
                    del block["pool"]
                else:
                    block["pool"] = pool
                profile, values = self._profile(block)
                self.assertIsNotNone(profile)
                self.assertIsNone(profile.falda)
                self.assertEqual(profile.falda_status, "rejected:missing_pool")
                self.assertNotIn("loam_falda", values)
                self.assertEqual(values["loam_falda_status"], "rejected:missing_pool")

    def test_missing_base_url_is_rejected(self) -> None:
        profile, _ = self._profile(_block(base_url=""))
        self.assertEqual(profile.falda_status, "rejected:missing_base_url")

    def test_timeouts_must_be_positive(self) -> None:
        profile, _ = self._profile(_block(recall_timeout_seconds=0))
        self.assertEqual(profile.falda_status, "rejected:bad_timeout")

    def test_payload_values_carry_env_names_not_secrets(self) -> None:
        _, values = self._profile(_block(), extra_env={"FALDA_TOKEN": "falda_secret_value"})
        self.assertNotIn("falda_secret_value", json.dumps(values))
        self.assertEqual(values["loam_falda"]["token_env"], "FALDA_TOKEN")

    def test_run_manifest_records_falda_configuration(self) -> None:
        manifest = _load("run_manifest_falda", HOOKS / "run_manifest.py")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                "event": "pre_llm_call",
                "workflow_id": "wf-1",
                "loam_state_dir": temporary,
                "loam_falda_status": "enabled",
                "loam_falda": {
                    "enabled": True,
                    "pool": POOL,
                    "recall_generation": True,
                    "recall_reasoner": False,
                    "write_validator_outcomes": True,
                    "reasoner_tools": True,
                },
            }
            value = json.loads(manifest.ensure(payload).read_text(encoding="utf-8"))
        self.assertEqual(
            value["campaign"]["falda"],
            {
                "status": "enabled",
                "pool": POOL,
                "workflow_tag": "workflow:wf-1",
                "recall_generation": True,
                "recall_reasoner": False,
                "write_validator_outcomes": True,
                "reasoner_tools": True,
            },
        )
        self.assertEqual(value["artifacts"]["falda_calls"], "falda-calls.jsonl")

    def test_run_manifest_records_rejected_status(self) -> None:
        manifest = _load("run_manifest_falda_rejected", HOOKS / "run_manifest.py")
        with tempfile.TemporaryDirectory() as temporary:
            payload = {
                "event": "pre_llm_call",
                "workflow_id": "wf-1",
                "loam_state_dir": temporary,
                "loam_falda_status": "rejected:missing_pool",
            }
            value = json.loads(manifest.ensure(payload).read_text(encoding="utf-8"))
        self.assertEqual(value["campaign"]["falda"], {"status": "rejected:missing_pool"})


if __name__ == "__main__":
    unittest.main()
