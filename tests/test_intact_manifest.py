"""Follow-up preparation must actually satisfy the executor's row contract."""
import hashlib, importlib, json, runpy, uuid
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_all_conditions_submit_and_long_id_repair_preserves_science(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'skills/hpc-deployment/scripts'))
    executor=importlib.import_module('hpc_agent')
    helper=runpy.run_path(str(ROOT/'skills/diffusion-models/rfd4-proteina/scripts/make_intact_followup_manifest.py'))
    hyp=str(uuid.uuid4());workspace=tmp_path/'work'/hyp;out=workspace/'outputs/r001-02';out.mkdir(parents=True)
    data=workspace/'data';(data/'intact-egfr').mkdir(parents=True);(data/'intact-egfr/P00533.json').write_text(json.dumps({'primaryAccession':'P00533'}));stages=data/'intact-egfr-energy';stages.mkdir()
    for name in ('cofold.py','mmgbsa.py'):(stages/name).write_text('frozen-scientific-recipe')
    batch=tmp_path/'batches/b-123abc';batch.mkdir(parents=True);(batch/'batch.json').write_text(json.dumps({'labels':{'hypothesis_id':hyp}}))
    paths=[]
    for condition in ('dual','one_region','ridge'):
        row=batch/('run_'+condition);row.mkdir();model=row/'model.cif';model.write_text('coordinate-checksum-fixture');summary={'status':'passed','models_total':5,'models':[{'path':str(model),'sha256':hashlib.sha256(model.read_bytes()).hexdigest()}]*5,'target_length':205,'binder_length':60,'binder_sequence':'A'*60,'condition':condition}
        p=row/'summary.json';p.write_text(json.dumps(summary));paths.append(str(p))
    monkeypatch.chdir(out);result=helper['make_manifest'](paths);manifest=Path(result['manifest']);rows=executor._read_rows(manifest);assert len(rows)==3
    original=manifest.read_text();legacy=[json.loads(line) for line in original.splitlines()]
    for row,path in zip(legacy,paths):
        condition=json.loads(Path(path).read_text())['condition'];digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        row['id']=f'{hyp}-r001-02-{condition}-{digest[:10]}'
    legacy_text=''.join(json.dumps(row)+'\n' for row in legacy);manifest.write_text(legacy_text)
    with pytest.raises(ValueError,match='bad id on inputs line 2'):executor._read_rows(manifest)
    repaired=Path(helper['make_manifest'](paths)['manifest']);assert repaired!=manifest
    assert manifest.read_text()==legacy_text
    assert repaired.read_text()==original
    assert executor._read_rows(repaired)==rows  # Same seeds, sequences, sources, hashes and recipes.
