"""Cross-filesystem staging must keep canonical paths and the one real ledger."""
import json

import pytest

from loam.saturation.storage import experiment_control_root, output_root


def control_root(tmp_path, owned=()):
    control = tmp_path / 'control'
    control.mkdir()
    (control / 'experiment.json').write_text(json.dumps({'owned_job_ids': list(owned)}))
    return control


def test_external_output_root_shares_the_ledger_and_resolves_back_to_it(tmp_path):
    control = control_root(tmp_path)
    ledger = control / 'experiment.json'
    assert output_root(control, None) == control.resolve()
    storage = output_root(control, tmp_path / 'other-filesystem')
    assert (storage / 'experiment.json').samefile(ledger)
    assert output_root(control, storage) == storage  # Idempotent once linked.
    job = storage / 'unified-256'
    job.mkdir()
    alias = control / job.name
    alias.symlink_to(job, target_is_directory=True)
    assert experiment_control_root(alias) == control.resolve()
    assert experiment_control_root(job) == control.resolve()
    assert json.loads(ledger.read_text()) == {'owned_job_ids': []}


@pytest.mark.parametrize('use_symlink', [False, True])
def test_other_ledger_is_never_replaced(tmp_path, use_symlink):
    control = control_root(tmp_path)
    storage = tmp_path / 'storage'
    storage.mkdir()
    other = tmp_path / 'other-ledger'
    other.write_text('{"owned_job_ids": ["unowned"]}')
    linked = storage / 'experiment.json'
    if use_symlink:
        linked.symlink_to(other)
    else:
        linked.write_text(other.read_text())
    with pytest.raises(RuntimeError, match='exact existing ownership ledger'):
        output_root(control, storage)
    assert json.loads(linked.read_text())['owned_job_ids'] == ['unowned']
