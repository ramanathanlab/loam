"""One existing allocation split into roles: science never runs on inference or foreign hosts.

Ported from the moeen branch's single-allocation and overlap tests, with the job id
generalised from Slurm's numeric ids to PBS's dotted ones.
"""
import json

import pytest

from loam.allocation import pooled_overlap, read_allocation, validate_role


def manifest(tmp_path, **changes):
    data = dict(mode='single-allocation', job_id='123', all_nodes=['a', 'b', 'c'],
                science_nodes=['a', 'b'], inference_nodes=['c'])
    data.update(changes)
    path = tmp_path / 'allocation.json'
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize('changes', [
    {'science_nodes': ['a', 'b', 'c']},  # overlapping roles
    {'inference_nodes': ['foreign']},
    {'all_nodes': ['a', 'b', 'c', 'orphan']},
    {'science_nodes': ['a', 'a', 'b']},
    {'inference_nodes': []},
    {'job_id': ''},
    {'job_id': '../123'},
    {'mode': 'globus'},
])
def test_rejects_invalid_role_partition(tmp_path, changes):
    with pytest.raises(RuntimeError):
        read_allocation(manifest(tmp_path, **changes))


def test_enforces_parent_job_and_host_role(tmp_path):
    path = manifest(tmp_path)
    assert validate_role(path, 'science', hostname='a.local', job_id='123')['job_id'] == '123'
    assert validate_role(path, 'inference', hostname='c', job_id='123')['job_id'] == '123'
    for role, host, job in [('science', 'c', '123'), ('inference', 'a', '123'),
                            ('science', 'a', '456'), ('science', 'foreign', '123'), ('login', 'a', '123')]:
        with pytest.raises(RuntimeError, match='assigned job'):
            validate_role(path, role, hostname=host, job_id=job)


def test_a_pbs_job_is_found_from_the_scheduler_environment(tmp_path, monkeypatch):
    job = '4242.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov'
    path = manifest(tmp_path, job_id=job)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setenv('PBS_JOBID', job)

    assert validate_role(path, 'science', hostname='a')['job_id'] == job


def test_outside_any_job_no_role_is_valid(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('PBS_JOBID', raising=False)

    with pytest.raises(RuntimeError, match='assigned job'):
        validate_role(manifest(tmp_path), 'science', hostname='a')


# Simultaneous replicas must not multiply the allocation wall clock.
def test_simultaneous_replica_overlap_is_intersection():
    science = {'start_epoch': 100, 'end_epoch': 200}
    replicas = [{'JobIDRaw': 'a', 'start_epoch': 80, 'end_epoch': 190},
                {'JobIDRaw': 'b', 'start_epoch': 120, 'end_epoch': 220}]
    assert pooled_overlap(science, replicas, 2) == [{'start': 120, 'end': 190, 'seconds': 70,
                                                     'inference_jobs': ['a', 'b']}]


def test_missing_or_nonoverlapping_replica_is_not_full_split():
    science = {'start_epoch': 100, 'end_epoch': 200}
    one = {'JobIDRaw': 'a', 'start_epoch': 80, 'end_epoch': 190}
    assert pooled_overlap(science, [one], 2) == []
    assert pooled_overlap(science, [one, {'JobIDRaw': 'b', 'start_epoch': 191, 'end_epoch': 220}], 2) == []
