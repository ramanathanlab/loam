"""Full allocation cost and missing telemetry cannot be hidden by busy-only averages."""
import dataclasses
import json

import pytest

from loam import systems
from loam.saturation import metrics

# A Slurm System with NVIDIA telemetry, standing in for one loam can audit.
SLURM_CUDA = dataclasses.replace(systems.POLARIS, name='slurm-cuda', scheduler='slurm',
                                 allocation_accounting='sacct', gpu_telemetry='nvidia-smi-csv')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def telemetry(job, gpus, times=(10, 20, 30)):
    write(job / 'allocation.json', {'job_id': '123', 'all_nodes': ['s', 'i'],
                                    'science_nodes': ['s'], 'inference_nodes': ['i']})
    for host, role in [('s', 'science'), ('i', 'inference')]:
        path = job / 'campaign/telemetry' / f'123-{host}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps({
            'job_id': '123', 'host': host, 'role': role, 'timestamp': t,
            'gpu_csv': '\n'.join(f'{g},uuid{g},100,10,1,100,100,50' for g in range(gpus))}) + '\n' for t in times))


def test_missing_time_counts_in_full_job_denominator(tmp_path):
    telemetry(tmp_path, 4)
    summary, nodes = metrics.hardware(tmp_path, 0, 100, system=SLURM_CUDA, gpus_per_node=4)
    assert summary['overall_GPU_busy_percent'] == 30
    assert summary['overall_telemetry_coverage_percent'] == 30
    assert all(n['observed_GPU_busy_percent'] == 100 for n in nodes)
    assert all(n['per_GPU_whole_allocation_busy_lower_bound_percent'] == [30] * 4 for n in nodes)


def test_telemetry_is_read_with_the_systems_node_shape(tmp_path):
    telemetry(tmp_path, 12)
    summary, nodes = metrics.hardware(tmp_path, 0, 100, system=SLURM_CUDA, gpus_per_node=12)
    assert summary['overall_GPU_busy_percent'] == 30
    assert all(len(n['per_GPU_whole_allocation_busy_lower_bound_percent']) == 12 for n in nodes)
    # Samples that do not cover every GPU on the node are not evidence of busy time.
    summary, _ = metrics.hardware(tmp_path, 0, 100, system=SLURM_CUDA, gpus_per_node=4)
    assert summary['overall_telemetry_coverage_percent'] == 0


def test_occupancy_includes_startup_and_idle_tail(tmp_path):
    write(tmp_path / 'plan.json', {'science_count': 1})
    write(tmp_path / 'campaign/batches/b/run_1/status.json', {'process_finished_at': 40})
    audit = {'row_timings': [{'batch_id': 'b', 'row': 'run_1', 'gpus': 1, 'started_at': 20, 'seconds': 20}]}
    result = metrics.occupancy(tmp_path, audit, 0, 100, gpus_per_node=4)
    assert result['science_slot_occupancy_percent'] == 5
    assert result['fraction_time_below_10percent_science_slot_occupancy'] == .8


def test_shared_rows_do_not_multiply_physical_gpu_occupancy(tmp_path):
    write(tmp_path / 'plan.json', {'science_count': 1, 'science_workers_per_gpu': 2})
    rows = []
    for n in range(2):
        folder = tmp_path / 'campaign/batches/b' / f'run_{n}'
        write(folder / 'status.json', {'host': 'science-node', 'gpu': '0', 'process_finished_at': 40})
        rows.append({'batch_id': 'b', 'row': f'run_{n}', 'gpus': 1, 'started_at': 20, 'seconds': 20})
    result = metrics.occupancy(tmp_path, {'row_timings': rows}, 0, 100, gpus_per_node=4)
    assert result['science_slot_occupancy_percent'] == 5
    busy = next(x for x in result['trace'] if x['timestamp'] == 20)
    assert busy['active_physical_GPUs'] == 1 and busy['active_scientific_rows'] == 2


def test_large_submission_gate_requires_clean_feedback_and_science(tmp_path, monkeypatch):
    baseline = {'screen_trials_per_allocated_GPU_hour': 2, 'full_context_trials_per_allocated_GPU_hour': 1,
                'science_GPU_busy_percent': 8}
    current = {'screen_trials_per_allocated_GPU_hour': 4, 'full_context_trials_per_allocated_GPU_hour': 2,
               'science_GPU_busy_percent': 17, 'overall_GPU_busy_percent': 50, 'nodes': 256,
               'scientific_completion_passed': True, 'recorded_rounds': 3, 'clean_synthesis_rounds': 3,
               'minimum_node_telemetry_coverage_percent': 98,
               'fraction_time_below_10percent_science_slot_occupancy': .2}
    old = tmp_path / 'baseline'
    new = tmp_path / 'new'
    monkeypatch.setattr(metrics, 'measure', lambda job: {'summary': baseline if job == old else current})
    assert metrics.evaluate(new, old)['final_gate_passed']
    current['clean_synthesis_rounds'] = 2
    assert not metrics.evaluate(new, old)['principle_gate_passed']
    current['clean_synthesis_rounds'] = 3
    current['scientific_completion_passed'] = False
    assert not metrics.evaluate(new, old)['principle_gate_passed']


def campaign_on(job, system):
    path = job / 'campaign/campaign.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'execution:\n  mode: hpc\n  hpc:\n    system: {system}\n')


def test_a_job_on_a_system_loam_cannot_audit_is_refused(tmp_path):
    campaign_on(tmp_path, 'aurora')
    write(tmp_path / 'plan.json', {'nodes': 20})
    write(tmp_path / 'submission.json', {'job_id': '123', 'job_name': 'owned'})
    with pytest.raises(NotImplementedError, match='aurora'):
        metrics.measure(tmp_path)
    campaign_on(tmp_path, 'nowhere')
    with pytest.raises(ValueError, match="System 'nowhere'"):
        metrics.measure(tmp_path)


def test_requeued_boot_cost_reduces_overall_busy_and_throughput(tmp_path, monkeypatch):
    # The plan records no node shape, so the System's four GPUs per node apply.
    monkeypatch.setitem(systems.SYSTEMS, SLURM_CUDA.name, SLURM_CUDA)
    campaign_on(tmp_path, SLURM_CUDA.name)
    write(tmp_path / 'plan.json', {'nodes': 20, 'science_count': 12, 'inference_count': 8})
    write(tmp_path / 'submission.json', {'job_id': '123', 'job_name': 'owned', 'submitted_at': -100})
    write(tmp_path / 'campaign/scaling-audit.json', {'required_campaign_completion_checks': {},
                                                     'required_campaign_completion_checks_passed': True})
    monkeypatch.setattr(metrics, 'deduplicate', lambda audit: ([{}], [{}]))
    monkeypatch.setattr(systems.System, 'parent_jobs',
                        lambda self, ids: {'123': {'State': 'COMPLETED', 'start_epoch': 0, 'end_epoch': 100}})
    attempts = {'allocation_attempts': 2, 'total_allocated_seconds': 110, 'total_allocated_GPU_seconds': 8800,
                'allocations': [{'NNodes': '20', 'JobName': 'owned', 'State': 'REQUEUED', 'start_epoch': -20},
                                {'NNodes': '20', 'JobName': 'owned', 'State': 'COMPLETED', 'start_epoch': 0}]}
    monkeypatch.setattr(systems.System, 'allocation_attempts',
                        lambda self, job, gpus_per_node: attempts if gpus_per_node == 4 else None)
    monkeypatch.setattr(metrics, 'hardware', lambda *args, **kwargs: (
        {'overall_GPU_busy_percent': 22, 'overall_telemetry_coverage_percent': 100,
         'science_GPU_busy_percent': 10, 'inference_GPU_busy_percent': 40}, []))
    monkeypatch.setattr(metrics, 'occupancy', lambda *args, **kwargs: {
        'science_slot_occupancy_percent': 50, 'fraction_time_below_10percent_science_slot_occupancy': .2,
        'trace': []})
    summary = metrics.measure(tmp_path)['summary']
    assert summary['allocated_GPU_hours'] == pytest.approx(8800 / 3600)
    assert summary['allocated_seconds'] == 110 and summary['final_allocation_seconds'] == 100
    assert summary['prior_allocation_seconds'] == 10
    assert summary['time_from_submission_to_final_start_seconds'] == 100
    assert summary['queue_seconds'] == 90  # Both waiting periods; excludes prior allocated boot time.
    assert summary['overall_GPU_busy_percent'] == pytest.approx(20)
    assert summary['science_GPU_busy_percent'] == 10  # Per-role plot uses final assigned interval.
    assert summary['full_context_trials_per_allocated_GPU_hour'] == pytest.approx(3600 / 8800)
    assert summary['fraction_time_below_10percent_science_slot_occupancy'] == pytest.approx(30 / 110)
    assert summary['prior_attempt_GPU_hours'] == pytest.approx(4 * 20 * 10 / 3600)


def test_audit_command_writes_the_gates_next_to_the_job(tmp_path, monkeypatch, capsys):
    from loam.saturation.__main__ import main
    job = tmp_path / 'trial'
    job.mkdir()
    summary = {'screen_trials_per_allocated_GPU_hour': 2, 'full_context_trials_per_allocated_GPU_hour': 1,
               'science_GPU_busy_percent': 8, 'overall_GPU_busy_percent': 10, 'nodes': 20,
               'scientific_completion_passed': True, 'recorded_rounds': 3, 'clean_synthesis_rounds': 3,
               'minimum_node_telemetry_coverage_percent': 98,
               'fraction_time_below_10percent_science_slot_occupancy': .2}
    monkeypatch.setattr(metrics, 'measure', lambda path: {'summary': summary})
    main([str(job), '--baseline', str(tmp_path / 'baseline')])
    written = json.loads((job / 'saturation-audit.json').read_text())
    assert written['gains']['science_GPU_busy_percent'] == 1
    assert not written['principle_gate_passed']
    printed = json.loads(capsys.readouterr().out)
    assert printed['principle_gate'] == written['principle_gate']
    assert 'current' not in printed
