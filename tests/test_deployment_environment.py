"""The deployment environment contract between loam and the HPC deployment scripts.

loam writes the HPC_* deployment environment (`HPCDeployment.environment`); the
MCP server and cluster agent read it back without importing loam. Nothing else
keeps the two sides in step, so this test does: every variable the scripts read
is produced by loam for every System, or is a tuning knob with a default; and
every variable loam produces is read by the scripts, or is for the operator only.
"""

import ast
import re
from pathlib import Path

import pytest

from loam.config import SiteProfile
from loam.systems import SYSTEMS
from test_round import hpc_environment

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills' / 'hpc-deployment' / 'scripts'
# The MCP server, its agent pools, and the agents (the cluster agent among them). smoke_test_client.py
# is left out: it is an operator's client, and its HPC_MCP_URL is for the operator to set.
READERS = ('mcp_server.py', 'pool_manager.py', 'hpc_agent.py')

# Read with a default that loam leaves alone; an operator may set one to tune a deployment.
TUNING = {
    'HPC_AGENT_STARTUP_TIMEOUT', 'HPC_BATCH_MAX_ROWS', 'HPC_BATCH_ROOT', 'HPC_CLIENT_PROBE_STRIKES',
    'HPC_CLUSTER_MAX_RESTARTS', 'HPC_CLUSTER_PROBE_STRIKES', 'HPC_CLUSTER_STARTUP_TIMEOUT',
    'HPC_COLD_ENDPOINT_PROBE_WAIT', 'HPC_COPPER_MOUNT', 'HPC_EXCHANGE_AUTH', 'HPC_EXCHANGE_NO_PROXY',
    'HPC_EXCHANGE_REQUEST_TIMEOUT', 'HPC_LOCAL_STAGE', 'HPC_MAX_RESUME_AGE_SECONDS', 'HPC_MIN_BATCH_DISTINCT',
    'HPC_MIN_BATCH_ROWS', 'HPC_POOL_PROBE_INTERVAL', 'HPC_POOL_PROBE_TIMEOUT', 'HPC_POOL_WAIT_TIMEOUT',
    'HPC_RUN_CODE_MAX_OUTPUT', 'HPC_TOOL_ROOT', 'HPC_TOOL_SMOKE_TIMEOUT',
}

# Set by whoever splits an existing allocation into science and inference roles (the single-allocation
# coordinator), not by the campaign config: each allocation's hosts are only known once it runs.
ROLE_ASSIGNMENT = {'HPC_SCIENCE_NODELIST'}

# Produced by loam for the operator, who exports it to workers (skills/hpc-deployment/SKILL.md); no script reads it.
OPERATOR_ONLY = {'HPC_LOAM_HOME'}


def read_by_scripts() -> set[str]:
    """Every HPC_* name the scripts spell out as a string: how each of them reads a variable."""
    names = set()
    for script in READERS:
        for node in ast.walk(ast.parse((SCRIPTS / script).read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and re.fullmatch(r'HPC_[A-Z0-9_]+', node.value):
                names.add(node.value)
    return names


def produced_by_loam(tmp_path, system) -> set[str]:
    # With a Site profile, so its HPC_WORKER_ENV_JSON is produced too.
    environment = hpc_environment(tmp_path, system, SiteProfile(software_root='/soft/loam'))
    return {name for name in environment if name.startswith('HPC_')}


@pytest.mark.parametrize('system', sorted(SYSTEMS))
def test_every_variable_the_scripts_read_is_produced_by_loam_or_is_tuning(tmp_path, system):
    unproduced = read_by_scripts() - produced_by_loam(tmp_path, system) - TUNING - ROLE_ASSIGNMENT
    assert not unproduced, (f'read by {", ".join(READERS)} but not produced by loam for {system}; produce it '
                            f'(loam.config / loam.systems) or, if it has a default, list it in TUNING')


@pytest.mark.parametrize('system', sorted(SYSTEMS))
def test_every_variable_loam_produces_is_read_or_is_for_the_operator(tmp_path, system):
    unread = produced_by_loam(tmp_path, system) - read_by_scripts() - OPERATOR_ONLY
    assert not unread, (f'produced by loam for {system} but read by none of {", ".join(READERS)}; '
                        f'read it, stop producing it, or list it in OPERATOR_ONLY')


@pytest.mark.parametrize('system', sorted(SYSTEMS))
def test_the_exception_lists_are_not_stale(tmp_path, system):
    read, produced = read_by_scripts(), produced_by_loam(tmp_path, system)
    assert TUNING <= read - produced, 'TUNING lists a variable no script reads, or one loam now produces'
    assert ROLE_ASSIGNMENT <= read - produced, \
        'ROLE_ASSIGNMENT lists a variable no script reads, or one loam now produces'
    assert OPERATOR_ONLY <= produced - read, 'OPERATOR_ONLY lists a variable loam no longer produces, or one a script reads'
