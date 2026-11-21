"""How a System submits a single-allocation job, finds its hosts, and runs one step on chosen hosts.

These are the only scheduler-specific commands single-allocation staging and its coordinator use
(ADR 0004): everything else asks the System. Ported from the moeen branch's sbatch and srun calls.
"""
from types import SimpleNamespace

import pytest

from loam import systems
from loam.systems import AURORA, JUPITER, POLARIS


def submission(system, **changes):
    values = dict(script='/jobs/a/job.sh', name='loam-a-1', workdir='/jobs/a', account='test-account',
                  queue='test-partition', nodes=256, walltime='09:00:00', reservation='', dependencies=())
    values.update(changes)
    return system.with_node_shape(gpus=8, cpus=96).batch_submission(**values)


def test_slurm_submits_one_exclusive_job_with_the_whole_node_shape():
    args = submission(JUPITER, reservation='test-reservation', dependencies=(('afterany', '41'),))

    assert args[0] == 'sbatch' and args[-1] == '/jobs/a/job.sh'
    assert {'--parsable', '--exclusive', '--nodes=256', '--account=test-account', '--partition=test-partition',
            '--gres=gpu:8', '--cpus-per-task=96', '--ntasks-per-node=1', '--time=09:00:00',
            '--job-name=loam-a-1', '--chdir=/jobs/a', '--reservation=test-reservation',
            '--dependency=afterany:41', '--export=NIL'} <= set(args)


def test_slurm_omits_an_empty_reservation():
    assert not [arg for arg in submission(JUPITER) if arg.startswith('--reservation')]


@pytest.mark.parametrize('stdout', ['123456\n', '123456;jupiter\n'])
def test_slurm_job_id_is_read_from_the_parsable_acknowledgement(stdout):
    assert JUPITER.submitted_job_id(stdout) == '123456'


@pytest.mark.parametrize('stdout', ['', 'Submitted batch job 12\n', '12a;x\n'])
def test_an_ambiguous_acknowledgement_is_refused(stdout):
    with pytest.raises(RuntimeError, match='Ambiguous'):
        JUPITER.submitted_job_id(stdout)


@pytest.mark.parametrize('system', [AURORA, POLARIS])
def test_systems_loam_cannot_submit_to_say_so(system):
    with pytest.raises(NotImplementedError, match=system.name):
        system.batch_submission(script='/j.sh', name='n', workdir='/', account='a', queue='q', nodes=2,
                                walltime='01:00:00', reservation='', dependencies=())


def test_slurm_hosts_are_expanded_from_the_node_list(monkeypatch):
    calls = []

    def scontrol(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='jpbo-001\njpbo-002.jupiter\n')
    monkeypatch.setattr(systems.subprocess, 'run', scontrol)

    assert JUPITER.allocation_hosts({'SLURM_JOB_NODELIST': 'jpbo-[001-002]'}) == ['jpbo-001', 'jpbo-002']
    assert calls == [['scontrol', 'show', 'hostnames', 'jpbo-[001-002]']]


def test_pbs_hosts_are_read_from_the_node_file(tmp_path):
    nodefile = tmp_path / 'nodes'
    nodefile.write_text('x1.hsn\nx2.hsn\nx1.hsn\n')

    assert POLARIS.allocation_hosts({'PBS_NODEFILE': str(nodefile)}) == ['x1', 'x2']


@pytest.mark.parametrize('system', [JUPITER, POLARIS])
def test_outside_an_allocation_there_are_no_hosts(system):
    with pytest.raises(RuntimeError, match='not inside an allocation'):
        system.allocation_hosts({})


def test_a_slurm_step_shares_the_allocation_on_exactly_the_chosen_hosts():
    step = JUPITER.host_step(['jpbo-003', 'jpbo-004'], ['python', 'serve.py'], cpus=32)

    assert step[0] == 'srun' and step[-2:] == ['python', 'serve.py']
    assert {'--overlap', '--nodes=2', '--ntasks=2', '--ntasks-per-node=1', '--cpus-per-task=32', '--nodelist=jpbo-003,jpbo-004',
            '--gpus-per-node=4', '--gpu-bind=none'} <= set(step)


def test_an_mpich_step_runs_one_process_on_each_chosen_host():
    assert POLARIS.host_step(['x1', 'x2'], ['python', 'serve.py']) == [
        'mpiexec', '-n', '2', '--ppn', '1', '--hosts', 'x1,x2', 'python', 'serve.py']


def test_gpu_telemetry_is_queried_the_way_the_system_reads_it():
    command = JUPITER.gpu_telemetry_query()

    assert command[0] == 'nvidia-smi' and '--format=csv,noheader,nounits' in command
    with pytest.raises(NotImplementedError, match='aurora'):
        AURORA.gpu_telemetry_query()
