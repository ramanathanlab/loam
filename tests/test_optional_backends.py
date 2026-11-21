"""Nemotron and ALCF auth are optional: a plain install brings in neither, and core loam never imports them."""
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())['project']


def names(requirements):
    return {r.split('[')[0].split('=')[0].split('>')[0].split('<')[0].strip().lower() for r in requirements}


def test_a_plain_install_brings_in_no_nemotron_or_alcf_auth_dependencies():
    assert not names(PROJECT['dependencies']) & {'aiohttp', 'alcf-tokens', 'globus-sdk'}


def test_each_backend_has_its_own_extra():
    extras = PROJECT['optional-dependencies']
    assert 'aiohttp' in names(extras['nemotron'])
    assert 'alcf-tokens' in names(extras['alcf-auth'])


def test_running_a_campaign_imports_neither_backend():
    code = ('import sys, loam.orchestrator, loam.launcher, loam.token_refresh; '
            'print(sorted(m for m in ("loam.nemotron", "loam.inference_pool", "alcf_tokens") if m in sys.modules))')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                            cwd=Path(__file__).parents[1])
    assert result.stdout.strip() == '[]'
