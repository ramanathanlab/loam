"""Focused tests for the campaign readiness preflight."""
from __future__ import annotations

import importlib.util
import json
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "campaign_preflight.py"
SPEC = importlib.util.spec_from_file_location("campaign_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
campaign_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign_preflight)


class _Handler(BaseHTTPRequestHandler):
    response_text = "LOAM_PREFLIGHT_OK"
    seen: dict = {}
    models: list = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        type(self).seen = json.loads(self.rfile.read(length))
        type(self).models.append(type(self).seen.get("model"))
        body = json.dumps({"choices": [{"message": {"content": self.response_text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        (self.root / "home").mkdir()
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        port = self.server.server_address[1]
        _Handler.models = []
        (self.root / "home" / "config.yaml").write_text(
            f"model:\n  default: reasoner-model\n  base_url: http://127.0.0.1:{port}/v1\n  api_key: test-key\n"
            f"delegation:\n  base_url: http://127.0.0.1:{port}/v1\n  model: subagent-model\n"
        )
        (self.root / "loam.yaml").write_text(
            "enabled: true\nactive_profile: p\nprofiles:\n  p:\n"
            f"    hypothesis_tournament:\n      enabled: true\n"
            f"      inference_endpoint: http://127.0.0.1:{port}/v1\n"
            "      model: judge-model\n"
            "      generation:\n        model: generator-model\n"
        )

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_every_configured_role_is_smoked(self) -> None:
        report = campaign_preflight.check_inference(
            self.root / "home", self.root / "loam.yaml", 5
        )
        self.assertEqual(
            {role: entry["model"] for role, entry in report.items()},
            {
                "reasoner": "reasoner-model",
                "subagents": "subagent-model",
                "generation": "generator-model",
                "ranking": "judge-model",
            },
        )
        for role, entry in report.items():
            self.assertEqual(entry["response"], "LOAM_PREFLIGHT_OK", role)
        # Each distinct model is called exactly once.
        self.assertEqual(sorted(_Handler.models),
                         ["generator-model", "judge-model", "reasoner-model", "subagent-model"])

    def test_duplicate_role_pairs_are_called_once(self) -> None:
        (self.root / "loam.yaml").write_text(
            (self.root / "loam.yaml").read_text().replace("        model: generator-model", "        model: judge-model")
        )
        report = campaign_preflight.check_inference(
            self.root / "home", self.root / "loam.yaml", 5
        )
        self.assertEqual(_Handler.models.count("judge-model"), 1)
        self.assertIn("same as", report["ranking"]["response"])

    def test_unexpected_content_fails_closed(self) -> None:
        _Handler.response_text = "wrong"
        try:
            with self.assertRaisesRegex(RuntimeError, "unexpected content"):
                campaign_preflight.check_inference(
                    self.root / "home", self.root / "loam.yaml", 5
                )
        finally:
            _Handler.response_text = "LOAM_PREFLIGHT_OK"


class PortTests(unittest.TestCase):
    def test_port_open_detects_listener(self) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        try:
            self.assertTrue(campaign_preflight._port_open("127.0.0.1", sock.getsockname()[1]))
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
