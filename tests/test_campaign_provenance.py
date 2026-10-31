"""The run manifest records which campaign policy produced a run's results.

Moving the objective and rules out of the hook and into editable files makes a
result harder to attribute than hardcoding did, unless what governed it is
recorded beside it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"


def _load(name: str, filename: str):
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(name, HOOKS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FIXTURE_GOAL = "---\nname: fixture-goal\n---\n\nEstablish whether the fixture subject behaves as predicted.\n"
FIXTURE_PROTOCOL = "---\nname: fixture-protocol\nscope: code-generator\n---\n\n- The decisive check.\n"


def _campaign(root: Path) -> dict:
    """A campaign in a user-owned directory; the repo ships no campaign content."""

    policy = root / "campaign-policy"
    (policy / "campaigns").mkdir(parents=True, exist_ok=True)
    (policy / "rules").mkdir(parents=True, exist_ok=True)
    (policy / "campaigns" / "fixture-goal.md").write_text(FIXTURE_GOAL, encoding="utf-8")
    (policy / "rules" / "fixture-protocol.md").write_text(FIXTURE_PROTOCOL, encoding="utf-8")
    return {"dir": str(policy), "goal": "fixture-goal", "rules": ["fixture-protocol"]}


def _payload(state_dir: Path, campaign: dict | None = None) -> dict:
    payload = {
        "event": "pre_llm_call",
        "repo_root": str(ROOT),
        "workflow_id": "research-v1",
        "session_id": "session-1",
        "loam_state_dir": str(state_dir),
    }
    if campaign is not None:
        payload["loam_campaign"] = campaign
    return payload


class CampaignProvenanceTests(unittest.TestCase):
    def test_policy_is_recorded_with_a_digest(self) -> None:
        orchestration = _load("loam_prov_campaign", "campaign.py")
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            provenance = orchestration.campaign_provenance({"repo_root": str(ROOT), "loam_campaign": campaign})

        self.assertEqual(provenance["goal"], "fixture-goal")
        self.assertEqual(
            provenance["rules"], ["evidence-integrity", "experiment-discipline", "fixture-protocol"]
        )
        # A user-supplied rule is digested like a shipped one: it governed the run.
        self.assertEqual(
            set(provenance["digests"]),
            {"fixture-goal", "evidence-integrity", "experiment-discipline", "fixture-protocol"},
        )
        self.assertEqual(len(provenance["policy_sha256"]), 64)

    def test_the_invariant_rules_are_recorded_without_a_campaign_block(self) -> None:
        """They govern the run whether or not a profile named them."""

        orchestration = _load("loam_prov_absent", "campaign.py")
        provenance = orchestration.campaign_provenance({"repo_root": str(ROOT)})

        self.assertEqual(provenance["goal"], "")
        self.assertEqual(provenance["rules"], ["evidence-integrity", "experiment-discipline"])
        self.assertEqual(len(provenance["policy_sha256"]), 64)

    def test_no_policy_at_all_records_nothing(self) -> None:
        orchestration = _load("loam_prov_nothing", "campaign.py")
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(orchestration.campaign_provenance({"repo_root": temporary}), {})

    def test_an_unreadable_policy_file_is_recorded_not_raised(self) -> None:
        """The injection path must fail loudly; the manifest is written on every event."""

        orchestration = _load("loam_prov_missing", "campaign.py")
        provenance = orchestration.campaign_provenance(
            {"repo_root": str(ROOT), "loam_campaign": {"goal": "no-such-goal"}}
        )

        self.assertIn("error", provenance)
        self.assertNotIn("policy_sha256", provenance)

    def test_the_manifest_carries_and_refreshes_the_policy(self) -> None:
        manifest = _load("loam_prov_manifest", "run_manifest.py")
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            campaign = _campaign(state_dir)
            path = manifest.ensure(_payload(state_dir, campaign))
            first = json.loads(path.read_text(encoding="utf-8"))["campaign"]

            # A later event in the same deployment must not be rejected, and must
            # re-read the policy rather than keep a weeks-old snapshot.
            manifest.ensure(_payload(state_dir, {"dir": campaign["dir"], "goal": "fixture-goal"}))
            second = json.loads(path.read_text(encoding="utf-8"))["campaign"]

        self.assertEqual(
            first["policy"]["rules"], ["evidence-integrity", "experiment-discipline", "fixture-protocol"]
        )
        # The named protocol rule is gone; the invariant ones cannot be dropped.
        self.assertEqual(second["policy"]["rules"], ["evidence-integrity", "experiment-discipline"])
        self.assertNotEqual(first["policy"]["policy_sha256"], second["policy"]["policy_sha256"])

    def test_policy_is_not_part_of_deployment_identity(self) -> None:
        """Editing a rule mid-campaign is recorded, never a fatal identity mismatch."""

        manifest = _load("loam_prov_identity", "run_manifest.py")
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            campaign = _campaign(state_dir)
            manifest.ensure(_payload(state_dir, campaign))
            # No raise: the policy changed, the deployment did not.
            manifest.ensure(_payload(state_dir, {**campaign, "rules": ["evidence-integrity"]}))
            manifest.ensure(_payload(state_dir, None))


if __name__ == "__main__":
    unittest.main()
