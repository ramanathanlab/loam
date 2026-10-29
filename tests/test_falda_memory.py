"""The hosted FALDA client: request shape, failure handling, call log."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / ".loam" / "hooks", Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import falda_memory  # noqa: E402
from falda_fake import POOL, FakeTransport, falda_block  # noqa: E402

TOKEN = "falda_test_token_value"


class FaldaMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {"FALDA_TOKEN": TOKEN})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(falda_memory.set_transport, None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = Path(temporary.name)
        self.payload = {
            "workflow_id": "wf-1",
            "loam_run_id": "run-1",
            "loam_state_dir": str(self.state),
            "loam_falda": falda_block(key_dir=str(self.state / "keys")),
        }

    def _calls(self) -> list[dict]:
        path = self.state / "falda-calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []

    def test_recall_sends_pool_tag_and_auth_and_logs(self) -> None:
        context = "## Findings\n[m1] x"
        fake = FakeTransport(recall=[(200, {"recall_id": "r-9", "context": context, "items": [{"id": "a-1"}, {"id": "a-2"}]})])
        falda_memory.set_transport(fake)
        result = falda_memory.recall(self.payload, "mean of 1..4", consumer="test", budget=900)
        self.assertTrue(result.ok)
        self.assertEqual((result.recall_id, result.item_ids, result.context), ("r-9", ("a-1", "a-2"), context))
        request = fake.requests[0]
        self.assertEqual(request["url"], "https://falda.example/recall")
        self.assertEqual(
            request["body"],
            {"query": "mean of 1..4", "require_tags": ["workflow:wf-1"], "budget": 900, "mode": "explicit", "pool": POOL},
        )
        self.assertEqual(request["headers"]["Authorization"], f"Bearer {TOKEN}")
        self.assertEqual(request["headers"]["X-Falda-Pool"], POOL)
        self.assertNotIn("X-Falda-Tenant", request["headers"])
        self.assertEqual(request["timeout"], 2.0)
        [row] = self._calls()
        self.assertEqual(
            (row["op"], row["consumer"], row["ok"], row["recall_id"], row["item_ids"], row["chars"]),
            ("recall", "test", True, "r-9", ["a-1", "a-2"], len(context)),
        )
        self.assertEqual((row["workflow_id"], row["run_id"], row["require_tags"]), ("wf-1", "run-1", ["workflow:wf-1"]))

    def test_upsert_sends_atoms_with_explicit_pool(self) -> None:
        fake = FakeTransport(upsert=[(200, {"items": [{"id": "atom-7"}], "total_count": 1})])
        falda_memory.set_transport(fake)
        atom = {"type": "fact", "content": "c", "confidence": "low", "tags": ["workflow:wf-1"]}
        result = falda_memory.upsert(self.payload, [atom], consumer="validator_outcome", plan_id="plan-1", hypothesis_id="h-1")
        self.assertEqual((result.ok, result.atom_ids), (True, ("atom-7",)))
        self.assertEqual(fake.requests[0]["url"], "https://falda.example/atoms/upsert")
        self.assertEqual(fake.requests[0]["body"], {"atoms": [atom], "pool": POOL})
        [row] = self._calls()
        self.assertEqual(
            (row["op"], row["atom_ids"], row["plan_id"], row["hypothesis_id"], row["ok"]),
            ("upsert", ["atom-7"], "plan-1", "h-1", True),
        )

    def test_server_error_codes_are_returned_and_logged(self) -> None:
        cases = (
            (403, {"error": "membership_stale"}, "membership_stale"),
            (500, {}, "http_500"),
            (502, b"<html>bad gateway</html>", "http_502"),
            (200, {"error": "one of {query, topic} is required"}, "one of {query, topic} is required"),
        )
        for status, body, code in cases:
            with self.subTest(code=code):
                falda_memory.set_transport(FakeTransport(recall=[(status, body)]))
                result = falda_memory.recall(self.payload, "q", consumer="test", budget=100)
                self.assertFalse(result.ok)
                self.assertEqual((result.error, result.context), (code, ""))
        self.assertEqual([row["error"] for row in self._calls()], [code for _, _, code in cases])

    def test_network_failure_never_raises(self) -> None:
        falda_memory.set_transport(FakeTransport(error=TimeoutError("slow")))
        self.assertEqual(falda_memory.recall(self.payload, "q", consumer="t", budget=100).error, "network:TimeoutError")
        self.assertEqual(falda_memory.upsert(self.payload, [{"content": "c"}], consumer="t").error, "network:TimeoutError")
        self.assertEqual([row["ok"] for row in self._calls()], [False, False])

    def test_missing_token_skips_the_request(self) -> None:
        fake = FakeTransport()
        falda_memory.set_transport(fake)
        with patch.dict(os.environ, {"FALDA_TOKEN": ""}):
            result = falda_memory.recall(self.payload, "q", consumer="t", budget=100)
        self.assertEqual(result.error, "missing_token")
        self.assertEqual(fake.requests, [])
        self.assertEqual(self._calls()[0]["error"], "missing_token")

    def test_disabled_config_makes_no_call_and_no_log(self) -> None:
        fake = FakeTransport()
        falda_memory.set_transport(fake)
        payload = {key: value for key, value in self.payload.items() if key != "loam_falda"}
        self.assertEqual(falda_memory.recall(payload, "q", consumer="t", budget=100).error, "disabled")
        self.assertEqual(falda_memory.upsert(payload, [{"content": "c"}], consumer="t").error, "disabled")
        self.assertEqual(fake.requests, [])
        self.assertEqual(self._calls(), [])

    def test_workflow_tag_fallbacks(self) -> None:
        tag = falda_memory.workflow_tag
        self.assertEqual(tag({"workflow_id": "wf-1"}), "workflow:wf-1")
        with patch.dict(os.environ, {"CAIRN_PROJECT": "exp-1"}):
            self.assertEqual(tag({"workflow_id": "environment-workflow"}), "workflow:exp-1")
            self.assertEqual(tag({"workflow_id": "  "}), "workflow:exp-1")
        with patch.dict(os.environ, {"CAIRN_PROJECT": ""}):
            today = datetime.now(timezone.utc).strftime("%Y%m%d")
            self.assertEqual(tag({}), f"workflow:loam-{today}")
        self.assertEqual(len(tag({"workflow_id": "w" * 300})), 128)

    def test_missing_workflow_id_still_enabled_with_fallback_tag(self) -> None:
        fake = FakeTransport()
        falda_memory.set_transport(fake)
        with patch.dict(os.environ, {"CAIRN_PROJECT": "exp-1"}):
            falda_memory.recall({**self.payload, "workflow_id": ""}, "q", consumer="t", budget=100)
        self.assertEqual(fake.requests[0]["body"]["require_tags"], ["workflow:exp-1"])

    def test_recall_is_wall_clock_capped(self) -> None:
        def _slow(url, headers, body, timeout):
            time.sleep(5)
            return 200, b"{}"

        falda_memory.set_transport(_slow)
        start = time.monotonic()
        result = falda_memory.recall(self.payload, "q", consumer="t", budget=100)
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertEqual((result.ok, result.error), (False, "timeout"))
        self.assertEqual(self._calls()[-1]["error"], "timeout")

    def test_cached_key_file_is_used_without_env(self) -> None:
        keys = self.state / "keys"
        keys.mkdir()
        (keys / f"loam-{POOL}.json").write_text(json.dumps({"api_key": "falda_cached", "pool": POOL}))
        fake = FakeTransport()
        falda_memory.set_transport(fake)
        with patch.dict(os.environ, {"FALDA_TOKEN": ""}):
            falda_memory.recall(self.payload, "q", consumer="t", budget=100)
        self.assertEqual(fake.requests[0]["headers"]["Authorization"], "Bearer falda_cached")

    def test_unwritable_call_log_does_not_break_calls(self) -> None:
        blocker = self.state / "not-a-dir"
        blocker.write_text("x")
        falda_memory.set_transport(FakeTransport())
        result = falda_memory.recall({**self.payload, "loam_state_dir": str(blocker)}, "q", consumer="t", budget=100)
        self.assertTrue(result.ok)

    def test_token_never_reaches_state_files(self) -> None:
        falda_memory.set_transport(FakeTransport(recall=[(403, {"error": "not_a_member"})]))
        falda_memory.recall(self.payload, "q", consumer="t", budget=100)
        falda_memory.set_transport(FakeTransport(error=OSError(f"refused with {TOKEN}")))
        falda_memory.upsert(self.payload, [{"content": "c"}], consumer="t")
        files = [path for path in self.state.rglob("*") if path.is_file()]
        self.assertTrue(files)
        for path in files:
            self.assertNotIn(TOKEN, path.read_text())

    def test_select_sections_keeps_named_sections_in_order_within_limit(self) -> None:
        context = (
            "Filtered to memories tagged workflow:wf-1; pool-wide episodes, core, and recent observations omitted.\n\n"
            "## Findings\n[m1] f1\n\n## What did not work\n[m2] n1\n[m3] n2\n\n"
            "## Observations from earlier sessions\n[m4] o1"
        )
        titles = ("What did not work", "Findings")
        self.assertEqual(
            falda_memory.select_sections(context, titles, 1000),
            "## What did not work\n[m2] n1\n[m3] n2\n\n## Findings\n[m1] f1",
        )
        cut = falda_memory.select_sections(context, titles, 30)
        self.assertEqual(cut, "## What did not work\n[m2] n1")
        self.assertEqual(falda_memory.select_sections(context, titles, 22), "")
        self.assertEqual(falda_memory.select_sections("", titles, 100), "")


if __name__ == "__main__":
    unittest.main()
