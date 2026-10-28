"""hpc_control: allocation caps and the environment handed to a restarted mcp_server.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "hpc-deployment" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import hpc_control  # noqa: E402

ENV_FILE = """\
export HPC_NNODES=1
export HPC_POOL_SIZE=1
export HPC_ENDPOINT_CONFIG_JSON='{"account": "P", "queue": "debug", "walltime": "00:30:00", "nodes_per_block": '"$HPC_NNODES"', "max_workers_per_node": '"$((HPC_POOL_SIZE + 1))"', "worker_init": "export TMPDIR=/tmp; export PYTHONPATH=/x:$PYTHONPATH"}'
"""

CAPS = {"allowed_queues": ["debug", "capacity"], "max_nodes": 4, "max_walltime": "12:00:00"}


class ValidateTests(unittest.TestCase):
    def test_accepts_within_caps(self) -> None:
        hpc_control.validate("debug", "00:30:00", 2, CAPS)
        hpc_control.validate("capacity", "12:00:00", 4, CAPS)

    def test_refuses_outside_caps_and_queue_limits(self) -> None:
        cases = [
            (("prod", "01:00:00", 1), "not allowed"),
            (("debug", "00:30:00", 3), "cap of 2"),          # debug queue max 2 nodes
            (("debug", "02:00:00", 1), "cap of 01:00:00"),   # debug queue max 1 h
            (("capacity", "24:00:00", 1), "cap of 12:00:00"),  # operator cap
            (("capacity", "01:00:00", 5), "cap of 4"),       # operator cap
            (("debug", "1h", 1), "HH:MM:SS"),
            (("debug", "00:30:00", 0), "positive"),
            (("debug", "00:30:00", True), "positive"),
        ]
        for args, message in cases:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, message):
                hpc_control.validate(*args, CAPS)


class EnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(tempfile.mkdtemp()) / "set_env.sh"
        self.path.write_text(ENV_FILE)
        self.env = hpc_control.sourced_env(str(self.path))

    def test_settings_come_from_env_file_then_saved_choice(self) -> None:
        self.assertEqual(hpc_control.current_settings(self.env, None),
                         {"queue": "debug", "walltime": "00:30:00", "nodes": 1})
        saved = {"queue": "capacity", "nodes": 2, "unrelated": 1}
        self.assertEqual(hpc_control.current_settings(self.env, saved),
                         {"queue": "capacity", "walltime": "00:30:00", "nodes": 2})

    def test_child_env_patches_config_and_keeps_worker_init(self) -> None:
        env = hpc_control.child_env(self.env, {"queue": "capacity", "walltime": "06:00:00", "nodes": 2})
        config = json.loads(env["HPC_ENDPOINT_CONFIG_JSON"])
        self.assertEqual((config["queue"], config["walltime"], config["nodes_per_block"]),
                         ("capacity", "06:00:00", 2))
        self.assertEqual(env["HPC_NNODES"], "2")
        self.assertEqual(config["account"], "P")
        self.assertIn("$PYTHONPATH", config["worker_init"])  # expanded on the worker, not here
        self.assertEqual(env["HPC_MCP_TRANSPORT"], "http")
        self.assertTrue(env["PYTHONPATH"].startswith(str(SCRIPTS)))


if __name__ == "__main__":
    unittest.main()
