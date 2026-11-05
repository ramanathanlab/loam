"""Hooks may use an isolated scientific runtime while Hermes keeps its own deps."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ADAPTER = Path(__file__).resolve().parents[1] / '.loam/adapters/hermes/plugin/__init__.py'
spec = importlib.util.spec_from_file_location('jupiter_hook_adapter', ADAPTER)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_hook_interpreter_override_and_default(tmp_path):
    script = tmp_path / 'probe.py'
    script.write_text('import json,sys; print(json.dumps({"value":json.load(sys.stdin)["value"]}))')
    for override in ('/custom/jnana/python', ''):
        with patch.object(adapter, 'HOOKS_DIR', tmp_path), \
             patch.dict(os.environ, {'LOAM_HOOK_PYTHON': override}), \
             patch.object(adapter.subprocess, 'run', return_value=subprocess.CompletedProcess(
                 [], 0, json.dumps({'value': 42}), '')) as run:
            assert adapter._invoke('probe.py', {'value': 42}) == {'value': 42}
            assert run.call_args.args[0][0] == (override or sys.executable)
