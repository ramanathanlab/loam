"""The scaling recipe: Moeen's measured Jupiter numbers, and a guard against quietly extrapolating them.

Expected values are the measured operating points in docs/scaling-recipe-book.md, not the formula.
"""

import json

import pytest
import yaml

from loam.config import load
from loam.scaling import UnmeasuredScaling, UnmeasuredScalingWarning, main, recipe
from loam.systems import AURORA, JUPITER, POLARIS
from test_round import hpc_config_values

# A test Site profile: the campaign config leaves the allocation's account and queue to it.
SITE_PROFILE = {'allocation': {'account': 'test-account', 'queue': 'test-partition'}, 'software_root': '/soft/loam'}


def jupiter_campaign_config(tmp_path, monkeypatch, nodes, **settings):
    """A Jupiter campaign config on `nodes` nodes, its account and queue from the test Site profile."""
    profile = tmp_path / 'profiles' / 'jupiter-test.yaml'
    profile.parent.mkdir(parents=True)
    profile.write_text(yaml.safe_dump(SITE_PROFILE))
    monkeypatch.setenv('LOAM_SITE_PROFILE', str(profile))
    values, _ = hpc_config_values(tmp_path)
    hpc = values['execution']['hpc']
    hpc['system'] = 'jupiter'
    hpc['allocation']['nodes'] = nodes
    del hpc['allocation']['account'], hpc['allocation']['queue']
    merge(values, settings)
    path = tmp_path / 'loam.yaml'
    path.write_text(yaml.safe_dump(values))
    return path


def merge(into, settings):
    for key, value in settings.items():
        if isinstance(value, dict):
            merge(into.setdefault(key, {}), value)
        else:
            into[key] = value


@pytest.mark.parametrize(
    'nodes,inference,science,concurrency,top_k,initial_hypotheses,new_per_round,tournament_workers,'
    'synthesis_concurrency', [
        (20, 8, 12, 48, 144, 288, 72, 16, 8),
        (256, 102, 154, 616, 1848, 3696, 924, 204, 102),
    ])
def test_measured_jupiter_configuration_returns_moeens_recipe(nodes, inference, science, concurrency, top_k,
                                                               initial_hypotheses, new_per_round,
                                                               tournament_workers, synthesis_concurrency):
    measured = recipe(JUPITER, nodes)
    assert (measured.inference, measured.science, measured.concurrency, measured.top_k,
            measured.initial_hypotheses, measured.new_per_round, measured.tournament_workers,
            measured.synthesis_concurrency) == (inference, science, concurrency, top_k, initial_hypotheses,
                                                new_per_round, tournament_workers, synthesis_concurrency)
    assert measured.generation_batch_size == 4 and measured.swiss_rounds == 1 and measured.rounds == 3
    assert measured.measured
    assert (measured.system, measured.basis.gpus_per_node, measured.basis.cpus_per_node) == ('jupiter', 4, 288)
    assert nodes in measured.basis.node_counts


@pytest.mark.parametrize('system', [AURORA, POLARIS, JUPITER.with_node_shape(gpus=8), JUPITER.with_node_shape(cpus=96)],
                         ids=['aurora', 'polaris', 'jupiter-8-gpus', 'jupiter-96-cpus'])
def test_unmeasured_system_or_node_shape_is_refused(system):
    with pytest.raises(UnmeasuredScaling, match='measured only on jupiter with 4 GPUs and 288 CPUs per node'):
        recipe(system, 20)


def test_unmeasured_system_or_node_shape_warns_when_accepted():
    with pytest.warns(UnmeasuredScalingWarning, match='polaris'):
        r = recipe(POLARIS, 20, allow_unmeasured=True)
    assert not r.measured
    # Polaris runs one science worker per GPU, so 12 science nodes x 4 GPUs is 48 slots, half of Jupiter's 96.
    assert (r.science, r.science_workers_per_gpu, r.concurrency, r.top_k) == (12, 1, 24, 72)


def test_unmeasured_node_count_on_the_measured_shape_warns():
    with pytest.warns(UnmeasuredScalingWarning, match='64 nodes'):
        r = recipe(JUPITER, 64)
    assert not r.measured
    # 26 : 38 nodes; 38 x 4 GPUs x 2 workers = 304 slots, half of them active.
    assert (r.inference, r.science, r.concurrency, r.top_k) == (26, 38, 152, 456)


def test_measured_configuration_does_not_warn(recwarn):
    recipe(JUPITER, 20)
    assert not recwarn.list


def test_density_tracks_gpus_per_node_and_explicit_overrides():
    eight_gpus = JUPITER.with_node_shape(gpus=8)
    with pytest.warns(UnmeasuredScalingWarning):
        r = recipe(eight_gpus, 10, inference=4, allow_unmeasured=True)
    assert (r.science, r.concurrency, r.top_k) == (6, 48, 144)
    with pytest.warns(UnmeasuredScalingWarning):
        assert recipe(eight_gpus, 10, inference=4, science_workers_per_gpu=1, allow_unmeasured=True).concurrency == 24


def test_departing_from_the_measured_split_or_sharing_warns():
    with pytest.warns(UnmeasuredScalingWarning, match='inference'):
        assert not recipe(JUPITER, 20, inference=4).measured
    with pytest.warns(UnmeasuredScalingWarning, match='science worker'):
        assert recipe(JUPITER, 20, science_workers_per_gpu=1).concurrency == 24
    assert recipe(JUPITER, 20, inference=8, science_workers_per_gpu=2).measured


@pytest.mark.parametrize('nodes,overrides', [
    (1, {}), (0, {}), (True, {}), (20, {'inference': 0}), (20, {'inference': -1}), (20, {'inference': 20}),
    (20, {'science_workers_per_gpu': 0}), (20, {'science_workers_per_gpu': 3}),
])
def test_invalid_budget_is_rejected(nodes, overrides):
    with pytest.raises(ValueError):
        recipe(JUPITER, nodes, **overrides)


@pytest.mark.parametrize('gpus', [0, -1])
def test_a_node_without_gpus_is_rejected(gpus):
    with pytest.raises(ValueError):
        recipe(JUPITER.with_node_shape(gpus=gpus), 20, allow_unmeasured=True)


def test_a_campaign_config_gets_the_recipe_for_its_system_and_nodes_and_takes_its_settings(tmp_path, monkeypatch):
    hpc = load(jupiter_campaign_config(tmp_path, monkeypatch, 20)).execution.hpc
    assert json.loads(hpc.environment()['HPC_ENDPOINT_CONFIG_JSON'])['account'] == 'test-account'

    r = recipe(hpc.system_definition(), hpc.allocation.nodes)
    assert r.measured and r.concurrency == 48

    cfg = load(jupiter_campaign_config(tmp_path / 'with-recipe', monkeypatch, 20, **r.campaign_settings()))
    assert (cfg.rounds, cfg.concurrency, cfg.synthesis_concurrency) == (3, 48, 8)
    tournament = cfg.tournament
    assert (tournament.top_k, tournament.initial_hypotheses, tournament.new_per_round, tournament.gen_batch_size,
            tournament.swiss_rounds, tournament.workers) == (
        144, 288, 72, 4, 1, 16)
    assert cfg.execution.hpc.science_workers_per_gpu == 2


def test_cli_prints_the_measured_recipe_with_where_it_was_measured(capsys):
    main(['--system', 'jupiter', '--nodes', '256'])
    printed = json.loads(capsys.readouterr().out)
    assert printed['measured'] is True
    assert printed['measured_on']['system'] == 'jupiter' and printed['measured_on']['node_counts'] == [20, 256]
    assert printed['recipe']['inference'] == 102 and printed['campaign_settings']['tournament']['top_k'] == 1848


def test_cli_refuses_an_unmeasured_system(capsys):
    with pytest.raises(SystemExit) as exit:
        main(['--system', 'aurora', '--nodes', '20'])
    assert exit.value.code != 0
    assert 'measured only on jupiter' in capsys.readouterr().err


def test_cli_prints_an_accepted_unmeasured_recipe_as_unmeasured(capsys):
    with pytest.warns(UnmeasuredScalingWarning):
        main(['--system', 'jupiter', '--gpus-per-node', '8', '--nodes', '10', '--inference', '4',
              '--allow-unmeasured'])
    printed = json.loads(capsys.readouterr().out)
    assert printed['measured'] is False and printed['recipe']['concurrency'] == 48


def test_cli_reads_the_system_node_shape_and_nodes_from_a_campaign_config(tmp_path, monkeypatch, capsys):
    main(['--config', str(jupiter_campaign_config(tmp_path, monkeypatch, 20))])
    printed = json.loads(capsys.readouterr().out)
    assert printed['measured'] is True and printed['recipe']['top_k'] == 144
