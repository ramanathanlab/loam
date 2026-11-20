"""Test the exact controller hook without importing or contacting production services."""
import ast
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_single_allocation_campaign.py'
tree = ast.parse(SOURCE.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'configure_controller_gc')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
configure = namespace['configure_controller_gc']


@pytest.fixture(autouse=True)
def restore_policy():
    before = gc.get_threshold()
    enabled = gc.isenabled()
    gc.enable()
    gc.set_threshold(4321, 17, 23)
    yield
    gc.set_threshold(*before)
    gc.enable() if enabled else gc.disable()


@pytest.mark.parametrize('plan', [{}, {'controller_gc_threshold0': None}])
def test_default_preserves_process_policy(plan):
    before = gc.get_threshold()
    assert configure(plan) == {'requested_threshold0': None, 'thresholds_before': list(before),
                               'thresholds_applied': list(before), 'automatic_gc_enabled': True}
    assert gc.get_threshold() == before


@pytest.mark.parametrize('threshold', [700, 10000, 100000])
def test_only_young_generation_changes_and_collector_stays_enabled(threshold):
    before = gc.get_threshold()
    receipt = configure({'controller_gc_threshold0': threshold})
    assert gc.get_threshold() == (threshold, *before[1:])
    assert receipt['thresholds_before'] == list(before)
    assert receipt['thresholds_applied'] == [threshold, *before[1:]]
    assert receipt['automatic_gc_enabled'] and gc.isenabled()


@pytest.mark.parametrize('invalid', [0, -1, 699, 100001, True, False, 10000.0, '10000', [], {}])
def test_invalid_policy_fails_without_side_effects(invalid):
    before = gc.get_threshold()
    with pytest.raises(ValueError):
        configure({'controller_gc_threshold0': invalid})
    assert gc.get_threshold() == before and gc.isenabled()


def test_explicit_tuning_rejects_disabled_gc_without_changing_it():
    gc.disable()
    before = gc.get_threshold()
    with pytest.raises(ValueError):
        configure({'controller_gc_threshold0': 10000})
    assert gc.get_threshold() == before and not gc.isenabled()


def test_default_does_not_change_explicitly_disabled_legacy_policy():
    gc.disable()
    before = gc.get_threshold()
    receipt = configure({})
    assert not receipt['automatic_gc_enabled'] and not gc.isenabled()
    assert gc.get_threshold() == before


def test_plan_and_model_settings_are_not_mutated():
    plan = {'controller_gc_threshold0': 10000, 'models': {
        'reasoner': {'max_tokens': 32768, 'enable_thinking': False},
        'validator': {'max_tokens': 8192, 'enable_thinking': False}},
        'science': {'workers_per_gpu': 2, 'input_sha256': 'unchanged'}}
    original = copy.deepcopy(plan)
    configure(plan)
    assert plan == original


def test_fresh_interpreter_policy_is_independent():
    command = [sys.executable, '-c', 'import gc,json;print(json.dumps(gc.get_threshold()))']
    before = json.loads(subprocess.check_output(command, text=True))
    configure({'controller_gc_threshold0': 10000})
    after = json.loads(subprocess.check_output(command, text=True))
    assert before == after
    assert gc.get_threshold() == (10000, 17, 23)


def test_entire_candidate_entrypoint_compiles():
    compile(SOURCE.read_text(), str(SOURCE), 'exec')
