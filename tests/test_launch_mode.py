"""Launch mode: Globus submits an allocation, or the campaign attaches to one already running."""

import pytest

from loam.config import Config
from test_round import hpc_config_values


def deployment(tmp_path, **changes):
    values, _ = hpc_config_values(tmp_path)
    values['execution']['hpc'].update(changes)
    return Config.model_validate(values).execution.hpc


def test_globus_submission_is_the_default(tmp_path):
    environment = deployment(tmp_path).environment()

    assert environment['HPC_LAUNCH_MODE'] == 'globus'
    assert environment['HPC_GLOBUS_ENDPOINT_ID'] == 'endpoint-uuid'


def test_attaching_to_an_existing_allocation_needs_no_endpoint(tmp_path):
    environment = deployment(tmp_path, launch_mode='existing', endpoint_id=None).environment()

    assert environment['HPC_LAUNCH_MODE'] == 'existing'
    assert 'HPC_GLOBUS_ENDPOINT_ID' not in environment


def test_globus_submission_still_needs_an_endpoint(tmp_path):
    with pytest.raises(ValueError, match='endpoint_id'):
        deployment(tmp_path, endpoint_id=None)


def test_an_unknown_launch_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='launch_mode'):
        deployment(tmp_path, launch_mode='submit')
