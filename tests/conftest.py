"""Synthetic metadata for staging tests; no research inputs are required by Git."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest


@pytest.fixture(autouse=True)
def synthetic_staging_reference(tmp_path,monkeypatch):
    import stage_single_allocation as staging
    packaged=Path(__file__).resolve().parents[1]/'examples/scaling/egfr'
    reference=tmp_path/'synthetic-reference';reference.mkdir()
    for name in ('campaign.yaml','goal.md'):
        shutil.copyfile(packaged/name,reference/name)
    data=reference/'inputs';shutil.copytree(packaged/'inputs',data)
    names=json.loads((packaged/'inputs-manifest.json').read_text())['sha256']
    for name in names:
        path=data/name
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('SYNTHETIC TEST METADATA ONLY; NOT SCIENTIFIC INPUT: '+name+'\n')
    hashes={str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
    (reference/'inputs-manifest.json').write_text(json.dumps({'sha256':hashes}))
    (reference/'seed-hypotheses.json').write_text(json.dumps([
        {'hypothesis_id':str(i),'summary':'Synthetic staging fixture','content':'A test fixture, never scientific evidence.'}
        for i in range(24)]))
    monkeypatch.setattr(staging,'REFERENCE',reference)
    monkeypatch.delenv('LOAM_SCIENCE_INPUTS',raising=False)
    monkeypatch.delenv('LOAM_SEED_HYPOTHESES',raising=False)
