"""Hermes gateway supplies parent_session_id alongside the session identity."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / '.loam/adapters/hermes/plugin/__init__.py'
SPEC = importlib.util.spec_from_file_location('loam_gateway_payload_test', PATH)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def test_gateway_parent_identity_reaches_pre_llm_hook():
    profile = SimpleNamespace(pre_llm_every=1, hypothesis_tournament=None)
    with patch.object(adapter, '_profile', return_value=profile), \
         patch.object(adapter, 'payload_values', return_value={}), \
         patch.object(adapter, '_invoke') as invoke:
        adapter.pre_llm_call(session_id='child', parent_session_id='parent')
    assert invoke.call_args.args[1]['session_id'] == 'child'
    assert invoke.call_args.args[1]['parent_session_id'] == 'parent'


def test_gateway_root_identity_falls_back_to_session():
    profile = SimpleNamespace(pre_llm_every=1, hypothesis_tournament=None)
    with patch.object(adapter, '_profile', return_value=profile), \
         patch.object(adapter, 'payload_values', return_value={}), \
         patch.object(adapter, '_invoke') as invoke:
        adapter.pre_llm_call(session_id='root', parent_session_id=None)
    assert invoke.call_args.args[1]['parent_session_id'] == 'root'


def test_single_planned_child_uses_goal_context_mode():
    task = {'goal': 'validate', 'context': 'full hypothesis', 'role': 'orchestrator'}
    arguments = {'tasks': [task], 'max_iterations': 7}
    assert adapter._hermes_delegation_args(arguments) == {**task, 'max_iterations': 7}
    assert arguments == {'tasks': [task], 'max_iterations': 7}


def test_multiple_planned_children_keep_batch_mode():
    arguments = {'tasks': [{'goal': 'first'}, {'goal': 'second'}]}
    assert adapter._hermes_delegation_args(arguments) is arguments
