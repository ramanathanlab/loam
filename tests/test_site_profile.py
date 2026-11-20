"""Site profiles: a campaign config names one, and its values reach the deployment environment.

Every test goes through the config-load -> deployment-environment seam: write a
campaign config and a Site profile to disk, `load` it, read `environment()`.
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from loam.config import load
from test_round import hpc_config_values

PROFILE = {
    'allocation': {'account': 'profile-account', 'queue': 'profile-queue'},
    'software_root': '/soft/loam',
}


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / 'xdg'
    monkeypatch.setenv('XDG_CONFIG_HOME', str(home))
    monkeypatch.delenv('LOAM_SITE_PROFILE', raising=False)
    return home


def write_profile(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values))
    return path


def config_home_profile(config_home, name):
    path = config_home / 'loam' / 'site-profiles' / f'{name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_config(tmp_path, *, site_profile=None, drop=(), endpoint_options=None):
    values, _ = hpc_config_values(tmp_path)
    allocation = values['execution']['hpc']['allocation']
    if endpoint_options is not None:
        allocation['endpoint_options'] = endpoint_options
    for key in drop:
        del allocation[key]
    if site_profile is not None:
        values['site_profile'] = site_profile
    path = tmp_path / 'loam.yaml'
    path.write_text(yaml.safe_dump(values))
    return path


def endpoint_config(environment):
    return json.loads(environment['HPC_ENDPOINT_CONFIG_JSON'])


def test_a_profile_named_in_the_config_dir_fills_the_deployment_environment(tmp_path, config_home):
    write_profile(config_home_profile(config_home, 'aurora-lab'), PROFILE)
    config = write_config(tmp_path, site_profile='aurora-lab', drop=('account', 'queue'))

    environment = load(config).execution.hpc.environment()

    assert environment['LOAM_SOFTWARE_ROOT'] == '/soft/loam'
    assert endpoint_config(environment)['account'] == 'profile-account'
    assert endpoint_config(environment)['queue'] == 'profile-queue'


def test_a_value_set_in_the_campaign_config_overrides_the_profile(tmp_path, config_home):
    write_profile(config_home_profile(config_home, 'aurora-lab'), PROFILE)
    config = write_config(tmp_path, site_profile='aurora-lab', drop=('queue',))

    environment = load(config).execution.hpc.environment()

    assert endpoint_config(environment)['account'] == 'project'
    assert endpoint_config(environment)['queue'] == 'profile-queue'


def test_an_environment_variable_gives_the_profile_path_when_the_config_names_none(tmp_path, config_home, monkeypatch):
    profile = write_profile(tmp_path / 'elsewhere' / 'mine.yaml', PROFILE)
    monkeypatch.setenv('LOAM_SITE_PROFILE', str(profile))
    config = write_config(tmp_path, drop=('account', 'queue'))

    environment = load(config).execution.hpc.environment()

    assert environment['LOAM_SOFTWARE_ROOT'] == '/soft/loam'
    assert endpoint_config(environment)['account'] == 'profile-account'


def test_a_missing_profile_fails_at_load_naming_the_profile(tmp_path, config_home):
    config = write_config(tmp_path, site_profile='nowhere')

    with pytest.raises(ValueError, match=r"site profile 'nowhere'.*not found"):
        load(config)


def test_a_profile_missing_a_required_key_fails_at_load_naming_the_profile_and_key(tmp_path, config_home):
    write_profile(config_home_profile(config_home, 'aurora-lab'), {'allocation': PROFILE['allocation']})
    config = write_config(tmp_path, site_profile='aurora-lab')

    with pytest.raises(ValueError, match=r"site profile 'aurora-lab'.*software_root"):
        load(config)


def test_an_allocation_value_in_neither_config_nor_profile_fails_naming_the_profile_and_key(tmp_path, config_home):
    write_profile(config_home_profile(config_home, 'aurora-lab'),
                  {'allocation': {'queue': 'profile-queue'}, 'software_root': '/soft/loam'})
    config = write_config(tmp_path, site_profile='aurora-lab', drop=('account',))

    with pytest.raises(ValueError, match=r"site profile 'aurora-lab'.*allocation\.account"):
        load(config)


FULL_PROFILE = {
    **PROFILE,
    'data_root': '/data/lab',
    'cache_root': '/cache/lab',
    'model_weights': {'chai': '/weights/chai-1'},
    'inference': {'argo': {'base_url': 'http://127.0.0.1:61612/v1', 'api_key_env': 'ARGO_KEY'}},
    'worker_init': 'module load frameworks',
}
ROOT_EXPORTS = (
    'export LOAM_SOFTWARE_ROOT=/soft/loam\n'
    'export LOAM_DATA_ROOT=/data/lab\n'
    'export LOAM_CACHE_ROOT=/cache/lab\n'
)


def test_every_profile_value_reaches_the_deployment_environment(tmp_path, config_home):
    write_profile(config_home_profile(config_home, 'aurora-lab'), FULL_PROFILE)
    config = write_config(tmp_path, site_profile='aurora-lab', endpoint_options={})

    environment = load(config).execution.hpc.environment()

    assert environment['LOAM_SOFTWARE_ROOT'] == '/soft/loam'
    assert environment['LOAM_DATA_ROOT'] == '/data/lab'
    assert environment['LOAM_CACHE_ROOT'] == '/cache/lab'
    assert json.loads(environment['LOAM_MODEL_WEIGHTS_JSON']) == {'chai': '/weights/chai-1'}
    assert json.loads(environment['LOAM_INFERENCE_JSON']) == {
        'argo': {'base_url': 'http://127.0.0.1:61612/v1', 'api_key_env': 'ARGO_KEY'}}
    assert endpoint_config(environment)['worker_init'] == ROOT_EXPORTS + 'module load frameworks'


def test_a_worker_init_in_the_campaign_config_replaces_the_profile_snippet(tmp_path, config_home):
    write_profile(config_home_profile(config_home, 'aurora-lab'), FULL_PROFILE)
    config = write_config(tmp_path, site_profile='aurora-lab')  # endpoint_options.worker_init: module load tools

    environment = load(config).execution.hpc.environment()

    assert endpoint_config(environment)['worker_init'] == ROOT_EXPORTS + 'module load tools'


def test_profile_paths_must_be_absolute(tmp_path, config_home):
    write_profile(config_home_profile(config_home, 'aurora-lab'), {**PROFILE, 'software_root': 'soft'})
    config = write_config(tmp_path, site_profile='aurora-lab')

    with pytest.raises(ValueError, match=r"site profile 'aurora-lab'.*software_root"):
        load(config)


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


@pytest.mark.parametrize('system', ['aurora', 'polaris'])
def test_the_example_profiles_load(tmp_path, config_home, system):
    shutil.copy(EXAMPLES / 'site-profiles' / f'{system}.yaml', config_home_profile(config_home, 'mine'))
    config = write_config(tmp_path, site_profile='mine', drop=('account',))

    environment = load(config).execution.hpc.environment()

    assert endpoint_config(environment)['account'] == 'YOUR_ALLOCATION'
    assert environment['LOAM_SOFTWARE_ROOT'].endswith('/YOUR_ALLOCATION/YOUR_USER/software')


def test_the_example_campaign_takes_its_account_from_the_site_profile(config_home):
    shutil.copy(EXAMPLES / 'site-profiles' / 'aurora.yaml', config_home_profile(config_home, 'aurora-myproject'))

    environment = load(EXAMPLES / 'loam.yaml').execution.hpc.environment()

    assert endpoint_config(environment)['account'] == 'YOUR_ALLOCATION'
    assert endpoint_config(environment)['queue'] == 'debug-scaling'


def test_committed_examples_hold_no_personal_paths_or_allocations():
    offending = [f'{path.relative_to(EXAMPLES)}: {marker}'
                 for path in EXAMPLES.rglob('*') if path.is_file()
                 for marker in ('/flare/FRAME-IDP', '/Users/', 'FRAME-IDP')
                 if marker in path.read_text(errors='ignore')]
    assert offending == []

