"""Attaching to an allocation that is already running: its nodes come from the scheduler, never a guess."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('academy')
pytest.importorskip('ensemble_launcher')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills' / 'hpc-deployment' / 'scripts'))
import hpc_agent  # noqa: E402


def nodefile(tmp_path, *hosts):
    path = tmp_path / 'nodefile'
    path.write_text(''.join(host + '\n' for host in hosts))
    return str(path)


def scontrol(monkeypatch, replies):
    """`scontrol show hostnames <expression>` expands each expression from `replies`."""
    monkeypatch.setattr(hpc_agent.subprocess, 'run', lambda args, **kw: SimpleNamespace(stdout=replies[args[-1]]))


def test_pbs_nodes_come_from_the_node_file(tmp_path):
    environ = {'PBS_NODEFILE': nodefile(tmp_path, 'x4000c0s0b0n0.hostmgmt', 'x4000c0s1b0n0.hostmgmt')}

    assert hpc_agent.allocation_nodes('pbs-nodefile', environ, 2) == ['x4000c0s0b0n0', 'x4000c0s1b0n0']


def test_slurm_nodes_come_from_the_node_list(monkeypatch):
    scontrol(monkeypatch, {'jpbo-[001-002]': 'jpbo-001\njpbo-002\n'})

    assert hpc_agent.allocation_nodes('slurm-nodelist', {'SLURM_JOB_NODELIST': 'jpbo-[001-002]'}, 2) \
        == ['jpbo-001', 'jpbo-002']


@pytest.mark.parametrize(('method', 'variable'), [('pbs-nodefile', 'PBS_NODEFILE'),
                                                  ('slurm-nodelist', 'SLURM_JOB_NODELIST')])
def test_outside_an_allocation_fails_clearly(method, variable):
    # Never this host as a one-node stand-in: that is how Globus mode runs a single worker.
    with pytest.raises(RuntimeError, match=f'not inside an allocation.*{variable}'):
        hpc_agent.allocation_nodes(method, {}, 1)


def test_an_allocation_of_the_wrong_size_is_rejected(tmp_path):
    environ = {'PBS_NODEFILE': nodefile(tmp_path, 'a', 'b', 'c')}

    with pytest.raises(RuntimeError, match='3 nodes; the campaign expects 2'):
        hpc_agent.allocation_nodes('pbs-nodefile', environ, 2)


def test_science_runs_only_on_its_share_of_the_allocation(tmp_path):
    environ = {'PBS_NODEFILE': nodefile(tmp_path, 'a', 'b', 'c'), 'HPC_SCIENCE_NODELIST': 'a,b'}

    assert hpc_agent.allocation_nodes('pbs-nodefile', environ, 2) == ['a', 'b']


# Moeen's test_el_sees_only_science_subset, on the System-neutral discovery.
def test_science_nodes_must_be_a_unique_subset_of_the_allocation(monkeypatch):
    replies = {'all': 'a\nb\nc\n', 'science': 'a\nb\n'}
    scontrol(monkeypatch, replies)
    environ = {'SLURM_JOB_NODELIST': 'all', 'HPC_SCIENCE_NODELIST': 'science'}
    assert hpc_agent.allocation_nodes('slurm-nodelist', environ, 2) == ['a', 'b']
    for invalid in ('a\nc\nforeign\n', 'a\na\n', 'a\n'):
        replies['science'] = invalid
        with pytest.raises(RuntimeError, match='science nodes'):
            hpc_agent.allocation_nodes('slurm-nodelist', environ, 2)


def test_a_timed_out_science_row_takes_its_descendants_with_it(tmp_path):
    # Ported from the moeen branch's single-allocation tests: a row that outlives its timeout
    # must not leave children holding the allocation's GPUs, nor block on their inherited pipe.
    import signal
    import subprocess
    import time
    payload = ('import subprocess,sys,time; '
               'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"]); '
               'print(p.pid,flush=True); time.sleep(10)')
    start = time.monotonic()

    code, stdout, _, timed_out = hpc_agent._run_process([sys.executable, '-c', payload], tmp_path, 0.5,
                                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert timed_out and code == -signal.SIGKILL
    assert time.monotonic() - start < 4
    child = Path('/proc') / stdout.decode().strip() / 'status'
    try:
        state = child.read_text()
    except (FileNotFoundError, ProcessLookupError):
        return  # Already reaped, or no /proc on this platform.
    assert '\nState:\tZ' in state  # Killed, awaiting its new parent's reap.
