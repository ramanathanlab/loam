"""Profile parsing for the optional `campaign:` block.

The block names files (`.loam/campaigns/<goal>.md`, `.loam/rules/<name>.md`) that
the hooks resolve against `repo_root`; this layer only validates and forwards the
names, so these tests cover the names and the payload shape, never file content.
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
HOOKS = ROOT / ".loam" / "hooks"
CONFIG = ROOT / ".loam" / "adapters" / "hermes" / "plugin" / "loam_config.py"


def _load(name: str, path: Path):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _profile(config, campaign: object | None) -> object | None:
    research: dict[str, object] = {
        "workflow_id": "research-v1",
        "state_dir": "/tmp/state",
        "hypothesis_orchestration": {"enabled": True},
        "execution": {"mode": "local", "runtime": "python3.11"},
    }
    if campaign is not None:
        research["campaign"] = campaign
    raw = {"enabled": True, "active_profile": "research", "profiles": {"research": research}}
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "loam.yaml"
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        with patch.dict(os.environ, {"LOAM_CONFIG_PATH": str(config_path)}, clear=False):
            return config.load_profile()


class CampaignConfigTests(unittest.TestCase):
    def test_goal_and_rules_reach_the_hook_payload(self) -> None:
        config = _load("loam_config_campaign", CONFIG)
        profile = _profile(config, {"goal": "alkbh1-binders", "rules": ["design-integrity", "hpc-budget"]})

        assert profile is not None
        assert profile.campaign is not None
        self.assertEqual(profile.campaign.goal, "alkbh1-binders")
        self.assertEqual(profile.campaign.rules, ("design-integrity", "hpc-budget"))
        values = config.payload_values(profile)
        self.assertEqual(
            values["loam_campaign"],
            {"goal": "alkbh1-binders", "rules": ["design-integrity", "hpc-budget"]},
        )

    def test_rules_are_optional(self) -> None:
        config = _load("loam_config_campaign_no_rules", CONFIG)
        profile = _profile(config, {"goal": "alkbh1-binders"})

        assert profile is not None
        assert profile.campaign is not None
        self.assertEqual(profile.campaign.rules, ())
        self.assertEqual(config.payload_values(profile)["loam_campaign"]["rules"], [])

    def test_absent_block_leaves_the_profile_usable(self) -> None:
        """A profile predating this feature must stay live, not go inert.

        Requiring `campaign:` under orchestration would reject the whole profile
        and take every hook down with it; the campaign context simply is not
        injected.
        """

        config = _load("loam_config_campaign_absent", CONFIG)
        profile = _profile(config, None)

        assert profile is not None
        self.assertIsNone(profile.campaign)
        self.assertTrue(profile.hypothesis_orchestration)
        self.assertNotIn("loam_campaign", config.payload_values(profile))

    def test_goal_is_required_when_the_block_exists(self) -> None:
        config = _load("loam_config_campaign_rules_only", CONFIG)
        with self.assertRaises(ValueError):
            _profile(config, {"rules": ["design-integrity"]})

    def test_a_name_may_not_carry_a_path(self) -> None:
        """A name becomes a file name under `.loam/`; a path would escape it."""

        config = _load("loam_config_campaign_traversal", CONFIG)
        for goal in ("../../etc/passwd", "/etc/passwd", "..", "sub/dir", ""):
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                _profile(config, {"goal": goal})
        with self.assertRaises(ValueError):
            _profile(config, {"goal": "ok", "rules": ["../escape"]})

    def test_malformed_rules_raise_rather_than_degrade(self) -> None:
        config = _load("loam_config_campaign_bad_rules", CONFIG)
        with self.assertRaises(ValueError):
            _profile(config, {"goal": "ok", "rules": "design-integrity"})
        with self.assertRaises(ValueError):
            _profile(config, {"goal": "ok", "rules": ["dup", "dup"]})


if __name__ == "__main__":
    unittest.main()
