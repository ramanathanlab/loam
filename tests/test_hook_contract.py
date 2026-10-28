"""The `.loam/manifest.yaml` wire contract, exercised the way the adapter runs hooks.

Each canonical hook is spawned as the Hermes adapter spawns it (`sys.executable`,
one JSON object on stdin, repository cwd, the adapter's timeout) with Loam
inactive, and its exit code and stdout are held to the manifest: observers stay
silent, and a hook that answers emits one JSON object in a supported form.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOAM = ROOT / ".loam"
MANIFEST = LOAM / "manifest.yaml"
ADAPTER = LOAM / "adapters" / "hermes" / "plugin" / "__init__.py"
# Mirrors HOOK_TIMEOUT_SECONDS in the adapter; a slower hook is dropped there.
TIMEOUT_SECONDS = 15
RESPONSE_KEYS = {
    "may_block": {"action", "message"},
    "may_inject_context": {"context"},
    "observe": set(),
}
# A minimal well-formed payload per event, as an inactive profile would send it.
VALID_PAYLOADS = {
    "pre_tool_call": {"tool_name": "terminal", "arguments": {"command": "true"}},
    "pre_llm_call": {"session_id": "contract-session"},
    "subagent_start": {"child_subagent_id": "contract-child"},
    "subagent_stop": {"child_subagent_id": "contract-child", "child_session_id": "contract-child-session"},
}
MALFORMED_INPUTS = ("", "not json", "[]", '{"event": "some_other_event"}')


def _manifest_hooks() -> dict[str, dict]:
    return yaml.safe_load(MANIFEST.read_text())["hooks"]


class HookContractTests(unittest.TestCase):
    def setUp(self) -> None:
        state = tempfile.TemporaryDirectory()
        self.addCleanup(state.cleanup)
        self.state_dir = Path(state.name)

    def _run(self, script: Path, stdin: str) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "LOAM_STATE_DIR": str(self.state_dir)}
        return subprocess.run(
            [sys.executable, str(script)],
            input=stdin,
            text=True,
            capture_output=True,
            cwd=str(ROOT),
            env=env,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )

    def _assert_response(self, behavior: str, stdout: str) -> None:
        output = stdout.strip()
        if not output:
            return
        self.assertNotEqual(behavior, "observe", f"observer wrote to stdout: {output[:200]}")
        response = json.loads(output)
        self.assertIsInstance(response, dict)
        self.assertLessEqual(set(response), RESPONSE_KEYS[behavior])

    def test_manifest_covers_every_event_with_a_known_behavior(self) -> None:
        hooks = _manifest_hooks()
        self.assertEqual(set(hooks), set(VALID_PAYLOADS))
        for event, spec in hooks.items():
            with self.subTest(event=event):
                self.assertIn(spec["behavior"], RESPONSE_KEYS)

    def test_manifest_scripts_are_executable_python(self) -> None:
        for event, spec in _manifest_hooks().items():
            script = LOAM / spec["script"]
            with self.subTest(event=event, script=spec["script"]):
                self.assertTrue(script.is_file(), f"{script} is missing")
                self.assertTrue(os.access(script, os.X_OK), f"{script} is not executable")
                self.assertEqual(script.read_text().splitlines()[0], "#!/usr/bin/env python3")

    def test_adapter_invokes_only_manifest_scripts(self) -> None:
        # A misspelled name is not an error in the adapter: it logs "hook missing"
        # and the event silently becomes a no-op.
        invoked = set(re.findall(r'_invoke\(\s*"([^"]+)"', ADAPTER.read_text()))
        declared = {Path(spec["script"]).name for spec in _manifest_hooks().values()}
        self.assertTrue(invoked)
        self.assertLessEqual(invoked, declared)

    def test_well_formed_payload_succeeds_within_contract(self) -> None:
        for event, spec in _manifest_hooks().items():
            payload = {"event": event, "repo_root": str(ROOT), **VALID_PAYLOADS[event]}
            with self.subTest(event=event):
                completed = self._run(LOAM / spec["script"], json.dumps(payload))
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)
                self._assert_response(spec["behavior"], completed.stdout)

    def test_payload_with_workspace_enabled_stays_within_contract(self) -> None:
        # An active campaign with the per-hypothesis workspace on, but no
        # campaign session yet: every hook must still answer within its
        # behavior, and nothing may be blocked.
        active = {
            "loam_state_dir": str(self.state_dir),
            "workflow_id": "contract-v1",
            "loam_hypothesis_orchestration": True,
            "loam_workspace": {"enabled": True, "root": str(self.state_dir / "hypotheses"), "strict": True},
        }
        for event, spec in _manifest_hooks().items():
            payload = {"event": event, "repo_root": str(ROOT), **VALID_PAYLOADS[event], **active}
            with self.subTest(event=event):
                completed = self._run(LOAM / spec["script"], json.dumps(payload))
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)
                self._assert_response(spec["behavior"], completed.stdout)
                if spec["behavior"] == "may_block":
                    self.assertEqual(completed.stdout.strip(), "")

    def test_inactive_pre_tool_call_is_not_blocked(self) -> None:
        script = LOAM / _manifest_hooks()["pre_tool_call"]["script"]
        payload = {"event": "pre_tool_call", "repo_root": str(ROOT), **VALID_PAYLOADS["pre_tool_call"]}
        completed = self._run(script, json.dumps(payload))
        self.assertEqual(completed.stdout.strip(), "")

    def test_malformed_input_fails_quietly(self) -> None:
        for event, spec in _manifest_hooks().items():
            for stdin in MALFORMED_INPUTS:
                with self.subTest(event=event, stdin=stdin):
                    completed = self._run(LOAM / spec["script"], stdin)
                    # Any exit code is allowed (the adapter treats nonzero as
                    # "proceed"), but the failure must be reported, not crash,
                    # and must never produce a response the adapter would act on.
                    self.assertEqual(completed.stdout.strip(), "")
                    self.assertNotIn("Traceback", completed.stderr)
                    self.assertTrue(completed.stderr.strip())

    def test_malformed_pre_tool_call_never_blocks(self) -> None:
        # Blocking on a parse failure would veto every tool in every session.
        script = LOAM / _manifest_hooks()["pre_tool_call"]["script"]
        for stdin in MALFORMED_INPUTS:
            with self.subTest(stdin=stdin):
                self.assertEqual(self._run(script, stdin).returncode, 0)


if __name__ == "__main__":
    unittest.main()
