"""Scientific datasets are supplied outside Git and matched by an explicit manifest."""
import json
from pathlib import Path

import pytest
import stage_single_allocation as staging
from prepare_scaling_inputs import prepare, REPO


def test_external_copy_validates_names_and_ignores_unrelated_source_files(tmp_path):
    source=staging.REFERENCE/'inputs'
    (source/'unrelated-private-file.txt').write_text('Synthetic test-only unrelated file')
    target=tmp_path/'external-bundle'
    prepare(source,target,staging.REFERENCE/'inputs-manifest.json')
    assert not (target/'unrelated-private-file.txt').exists()
    assert len([p for p in target.rglob('*') if p.is_file()])==33


def test_changed_input_is_rejected_before_creating_bundle(tmp_path):
    source=staging.REFERENCE/'inputs'
    (source/'verified-target.fasta').write_text('TEST-ONLY CHANGED INPUT')
    target=tmp_path/'bad-copy'
    with pytest.raises(ValueError,match='changed scientific input'):
        prepare(source,target,staging.REFERENCE/'inputs-manifest.json')
    assert not target.exists()


def test_scientific_data_cannot_be_prepared_inside_the_git_checkout():
    with pytest.raises(ValueError,match='outside the Git checkout'):
        prepare(staging.REFERENCE/'inputs',REPO/'datasets/forbidden',staging.REFERENCE/'inputs-manifest.json')


def test_stager_accepts_a_separate_bundle_and_preserves_its_hashes(tmp_path):
    source=staging.REFERENCE/'inputs';bundle=tmp_path/'science'
    prepare(source,bundle,staging.REFERENCE/'inputs-manifest.json')
    job=staging.stage(tmp_path/'experiment',20,8,'external','campaign','08:00:00','',48,
        inputs=bundle,top_k=144,initial_hypotheses=288,new_per_round=72)
    plan=json.loads((job/'plan.json').read_text())
    assert plan['input_bundle_source']==str(bundle)
    assert plan['seed_hypotheses_count']==24
    assert (job/'.gitignore').read_text()=='*\n'
