"""Campaign memory reaches Jnana's generation goal, and only when Jnana will generate."""

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
for directory in (HOOKS, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import falda_memory  # noqa: E402
from falda_fake import POOL, FakeTransport, falda_block  # noqa: E402

MESSAGE = "Does the mean of [1, 2, 3, 4] exceed 2.4?"
CONTEXT = (
    "Filtered to memories tagged workflow:wf-1; pool-wide episodes, core, and recent observations omitted.\n\n"
    "## Best known results\n[m1] mean is exactly 5/2\n\n"
    "## What did not work\n[m2] float check misleads at the boundary\n\n"
    "## Findings\n[m3] exact arithmetic agrees\n\n"
    "## Next steps and open questions\n[m4] try threshold 2.5\n\n"
    "## Observations from earlier sessions\n[m5] noise"
)
EXPECTED_MEMORY = (
    "Prior campaign memory (advisory; verify before relying on it):\n"
    "## What did not work\n[m2] float check misleads at the boundary\n\n"
    "## Best known results\n[m1] mean is exactly 5/2\n\n"
    "## Next steps and open questions\n[m4] try threshold 2.5\n\n"
    "## Findings\n[m3] exact arithmetic agrees"
)


def _load_orchestration(name: str):
    spec = importlib.util.spec_from_file_location(name, HOOKS / "hypothesis_orchestration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GenerationRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {"FALDA_TOKEN": "falda_test"})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(falda_memory.set_transport, None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.orchestration = _load_orchestration(f"orchestration_generation_{self._testMethodName}")

    def _payload(self, hypotheses: int = 0, falda=None, tournament: bool = True) -> dict:
        state_file = self.root / "jnana.json"
        state_file.write_text(
            json.dumps({"hypotheses": [{"hypothesis_id": f"h-{i}", "content": "c"} for i in range(hypotheses)]}),
            encoding="utf-8",
        )
        payload = {
            "repo_root": str(ROOT),
            "workflow_id": "wf-1",
            "session_id": "coordinator-1",
            "user_message": MESSAGE,
            "loam_state_dir": str(self.root / "loam"),
            "loam_falda": falda or falda_block(),
        }
        if tournament:
            payload["loam_hypothesis_tournament"] = {"state_file": str(state_file), "reports_dir": str(self.root / "reports")}
        return payload

    def test_memory_is_appended_in_section_order_under_the_advisory_heading(self) -> None:
        fake = FakeTransport(recall=[(200, {"recall_id": "r-1", "context": CONTEXT, "items": [{"id": "a"}]})])
        falda_memory.set_transport(fake)
        payload = self._payload()
        self.orchestration.prepare_payload(payload)
        self.assertEqual(payload["user_message"], f"{MESSAGE}\n\n{EXPECTED_MEMORY}")
        [body] = fake.bodies("/recall")
        self.assertEqual((body["query"], body["require_tags"], body["pool"]), (MESSAGE, ["workflow:wf-1"], POOL))

    def test_brief_and_memory_are_both_appended_and_the_brief_joins_the_query(self) -> None:
        fake = FakeTransport(recall=[(200, {"context": CONTEXT, "items": []})])
        falda_memory.set_transport(fake)
        payload = self._payload()
        state = Path(payload["loam_state_dir"])
        state.mkdir(parents=True)
        (state / "hypothesis-orchestration.jsonl").write_text(
            json.dumps({"event": "reasoner_completed", "workflow_id": "wf-1", "summary": "Check boundary 2.5."}) + "\n",
            encoding="utf-8",
        )
        self.orchestration.prepare_payload(payload)
        message = payload["user_message"]
        self.assertLess(message.index("Prior reasoner brief"), message.index("Prior campaign memory"))
        self.assertIn("Check boundary 2.5.", fake.bodies("/recall")[0]["query"])

    def test_no_recall_once_jnana_has_hypotheses_or_without_a_tournament(self) -> None:
        # Build each payload inside the loop: _payload rewrites the shared Jnana state file.
        for kwargs in ({"hypotheses": 2}, {"tournament": False}, {"falda": falda_block(recall_generation=False)}):
            with self.subTest(kwargs=kwargs):
                fake = FakeTransport()
                falda_memory.set_transport(fake)
                payload = self._payload(**kwargs)
                self.orchestration.prepare_payload(payload)
                self.assertEqual(fake.requests, [])
                self.assertEqual(payload["user_message"], MESSAGE)

    def test_jnana_state_is_not_read_when_falda_is_off(self) -> None:
        payload = self._payload()
        del payload["loam_falda"]
        Path(payload["loam_hypothesis_tournament"]["state_file"]).write_text("{not json", encoding="utf-8")
        self.orchestration.prepare_payload(payload)
        self.assertEqual(payload["user_message"], MESSAGE)

    def test_failure_or_empty_memory_leaves_the_goal_unchanged(self) -> None:
        for response in ((503, {"error": "unavailable"}), (200, {"context": "", "items": []})):
            falda_memory.set_transport(FakeTransport(recall=[response]))
            payload = self._payload()
            self.orchestration.prepare_payload(payload)
            self.assertEqual(payload["user_message"], MESSAGE)
        calls = [json.loads(line) for line in (self.root / "loam" / "falda-calls.jsonl").read_text().splitlines()]
        self.assertEqual([(row["consumer"], row["ok"]) for row in calls], [("jnana_generation", False), ("jnana_generation", True)])

    def test_memory_is_cut_to_the_budget(self) -> None:
        falda_memory.set_transport(FakeTransport(recall=[(200, {"context": CONTEXT, "items": []})]))
        payload = self._payload(falda=falda_block(recall_budget_chars=70))
        self.orchestration.prepare_payload(payload)
        memory = payload["user_message"].split("Prior campaign memory (advisory; verify before relying on it):\n", 1)[1]
        self.assertEqual(memory, "## What did not work\n[m2] float check misleads at the boundary")
        self.assertTrue(memory.startswith("## What did not work"))


if __name__ == "__main__":
    unittest.main()
