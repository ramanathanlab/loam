"""Prepare follow-up rows for explicitly selected, real candidate summaries.

Metadata only: run make_manifest([absolute_summary_path, ...]) from the
assigned hypothesis run directory. Scientific selection remains the validator's
decision; this helper does not relax its prespecified confidence gate.
"""
import hashlib
import json
import re
import uuid
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(screen_summaries):
    out = Path.cwd().resolve()
    workspace = out.parents[1]
    hypothesis = str(uuid.UUID(workspace.name))
    assert out.parent.name == 'outputs' and re.fullmatch(r'r\d{3}-\d{2}', out.name)
    assert screen_summaries, 'Select actual candidates before preparing follow-up rows'
    data = workspace / 'data'
    annotation = (data / 'intact-egfr/P00533.json').resolve(strict=True)
    assert json.loads(annotation.read_text())['primaryAccession'] == 'P00533'
    stages = (data / ('intact-egfr-energy-pathsafe' if (data/'intact-egfr-energy-pathsafe').is_dir() else 'intact-egfr-energy')).resolve(strict=True)
    rows = []
    for raw in screen_summaries:
        source = Path(raw).resolve(strict=True)
        assert source.name == 'summary.json', 'Supply per-candidate summary.json, never cohort analysis.json'
        summary = json.loads(source.read_text())
        batch = json.loads((source.parent.parent / 'batch.json').read_text())
        assert batch['labels']['hypothesis_id'] == hypothesis, 'Candidate belongs to another hypothesis'
        assert summary['status'] == 'passed' and summary['models_total'] == len(summary['models']) == 5
        assert summary['target_length'] == 205 and summary['binder_length'] == len(summary['binder_sequence']) == 60
        assert 'C' not in summary['binder_sequence'], 'This energy recipe requires a cysteine-free binder'
        for model in summary['models']:
            assert sha(Path(model['path'])) == model['sha256'], 'Screen coordinate checksum changed'
        digest = sha(source)
        namespace = f'{hypothesis}:{out.name}:intact:{digest}'
        seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:4], 'big') & 0x7fffffff
        # Full UUID/run identity still determines the seed. A short row label
        # also keeps deeply nested Amber filenames within legacy native limits.
        rows.append({'id': f'{summary["condition"]}-{digest[:16]}',
                     'seed': seed, 'screen_summary': str(source), 'screen_summary_sha256': digest,
                     'target_annotation': str(annotation), 'target_annotation_sha256': sha(annotation),
                     'stages_directory': str(stages), 'cofold_script_sha256': sha(stages/'cofold.py'),
                     'mmgbsa_script_sha256': sha(stages/'mmgbsa.py')})
    assert len({row['screen_summary'] for row in rows}) == len(rows), 'Repeated candidate'
    assert len({row['seed'] for row in rows}) == len(rows), 'Repeated seed'
    text = ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
    destination = out / 'intact-followup-inputs.jsonl'
    if destination.exists():
        if destination.read_text() != text:
            previous=[json.loads(line) for line in destination.read_text().splitlines() if line.strip()]
            assert [{k:v for k,v in row.items() if k!='id'} for row in previous]==[{k:v for k,v in row.items() if k!='id'} for row in rows], 'Existing scientific metadata differs; do not overwrite it'
            for prior,current in zip(previous,rows):
                prefix=(hypothesis+'-'+out.name+'-',uuid.UUID(hypothesis).hex+'-'+out.name+'-')
                assert prior['id']==current['id'] or prior['id'].startswith(prefix), 'Existing custom identifiers differ; preserve them'
            # Preserve failed preparation verbatim, including its invalid IDs.
            destination=out/'intact-followup-inputs.short-ids.jsonl'
    if destination.exists():
        assert destination.read_text()==text,'Existing repaired metadata differs; preserve it'
    else:
        destination.write_text(text)
    return {'manifest': str(destination), 'rows': len(rows), 'manifest_sha256': sha(destination),
            'preparation_only': True, 'scientific_evidence': False,
            'selection_policy': 'Explicit validator-selected candidates; no confidence gate changed'}


def make_from_batch(batch_id, candidate_indices):
    """Resolve short condition/index selections against this hypothesis's batch.

    Example selection: {'dual':3,'one_region':2,'ridge':1}. This only resolves
    identifiers; it does not decide which candidate satisfies a scientific gate.
    """
    assert re.fullmatch(r'b-[0-9a-f]+', batch_id), 'Use an actual owned batch ID'
    workspace = Path.cwd().resolve().parents[1]
    root = workspace.parents[1] / 'batches' / batch_id
    batch = json.loads((root/'batch.json').read_text())
    assert batch['labels']['hypothesis_id'] == workspace.name
    paths = []
    for condition, index in candidate_indices.items():
        assert condition in ('dual','one_region','ridge') and type(index) is int and 1 <= index <= 4
        found = []
        for path in root.glob('run_*/summary.json'):
            summary = json.loads(path.read_text())
            if summary['condition'] == condition and summary['candidate_id'].endswith(f'-{condition}-{index:02}'):
                found.append(str(path))
        assert len(found) == 1, f'Expected one real {condition}-{index:02} candidate, found {len(found)}'
        paths.extend(found)
    return make_manifest(paths)


def make_cohort_summary_manifest(batch_id):
    """Prepare the single CPU-summary row without transcribing batch paths."""
    assert re.fullmatch(r'b-[0-9a-f]+', batch_id)
    out=Path.cwd().resolve();workspace=out.parents[1];root=workspace.parents[1]/'batches'/batch_id
    batch=json.loads((root/'batch.json').read_text())
    assert batch['labels']['hypothesis_id']==workspace.name and batch['total']==12
    row={'id':'cohort-summary','batch_dir':str(root),'expected_per_condition':4}
    destination=out/'cohort-summary-inputs.jsonl';text=json.dumps(row,separators=(',',':'))+'\n'
    if destination.exists():assert destination.read_text()==text,'Existing metadata differs; preserve it'
    else:destination.write_text(text)
    return {'manifest':str(destination),'rows':1,'manifest_sha256':sha(destination),'preparation_only':True,'scientific_evidence':False}


if __name__ == '__main__':
    import sys
    print(json.dumps(make_manifest(sys.argv[1:])))
