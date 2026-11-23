import stat

import pytest


@pytest.fixture
def alcf_tokens(tmp_path, monkeypatch):
    """A stand-in `alcf-tokens` that records each call and prints a fresh token (or fails, if told to)."""
    calls = tmp_path / 'calls'
    script = tmp_path / 'alcf-tokens'
    script.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n'
                      '[ -n "$FAIL" ] && exit 1\necho refreshed-token\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('ALCF_TOKENS_BIN', str(script))
    monkeypatch.setenv('ALCF_INFERENCE_TOKEN', 'stale-token')
    return lambda: calls.read_text().splitlines() if calls.exists() else []
