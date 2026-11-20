"""Build twelve real candidate inputs in an owned hypothesis run directory.

Execute with run_code/runpy from .../<hypothesis UUID>/outputs/rNNN-NN.
Only metadata is prepared here; scientific execution remains run_batch(gpus=1).
"""
import hashlib,json,re,uuid
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    out=Path.cwd().resolve();workspace=out.parents[1];hypothesis=str(uuid.UUID(workspace.name));run=out.name
    assert out.parent.name=='outputs' and re.fullmatch(r'r\d{3}-\d{2}',run),'Execute inside the assigned hypothesis run directory'
    data=workspace/'data';helper=data/'run_binder_generation.py';target=data/'verified-target.fasta'
    sequence=''.join(x.strip() for x in target.read_text().splitlines() if not x.startswith('>'));assert len(sequence)==205
    rows=[]
    for condition in ('dual','one_region','ridge'):
        source=data/'verified-egfr-inputs'/f'{condition}.cif';digest=sha(source)
        for index in range(1,5):
            namespace=f'{hypothesis}:{run}:{condition}:{index}'
            seed=int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:4],'big')&0x7fffffff
            rows.append({'id':f'{hypothesis}-{run}-{condition}-{index:02}','condition':condition,'seed':seed,'input_cif':str(source),'input_sha256':digest,'target_fasta':str(target),'generation_script':str(helper),'generation_script_sha256':sha(helper)})
    assert len({r['seed'] for r in rows})==12
    text=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows);dest=out/'candidate-inputs.jsonl';metadata=out/'candidate-inputs.provenance.json'
    proof={'hypothesis_id':hypothesis,'run_id':run,'rows':12,'per_condition':4,'seed_derivation':'first31bits(SHA256(hypothesis_uuid:run_id:condition:index))','manifest_sha256':hashlib.sha256(text.encode()).hexdigest(),'preparation_only':True,'scientific_evidence':False}
    if dest.exists():assert dest.read_text()==text,'An existing manifest has different inputs; do not overwrite it'
    else:dest.write_text(text)
    if metadata.exists():assert json.loads(metadata.read_text())==proof,'Manifest provenance changed'
    else:metadata.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'manifest':str(dest),'rows':12,'per_condition':4,'manifest_sha256':proof['manifest_sha256'],'preparation_only':True}))
if __name__=='__main__':main()
