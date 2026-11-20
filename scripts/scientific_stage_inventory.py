"""Count witnessed stage outputs on distinct, independently audited inputs.

Read-only with respect to campaigns. A numerically valid output does not make
the campaign's narrative claims or completion gates pass. No model is run.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import gemmi

FROZEN = Path(__file__).resolve().parent
sys.path.insert(0, str(FROZEN))
from build_inference_atlas import deduplicate
from report_production_events import finish, production_events

STAGES = {
    'rfd_backbones': 'generated binder backbones',
    'mpnn_sequences': 'sampled inverse-folded sequences (native input excluded)',
    'crop_folding_calls': 'crop cofold input sets',
    'crop_fold_models': 'crop complex predictions',
    'full_folding_calls': 'full-context cofold input sets',
    'full_fold_models': 'full-context complex predictions',
    'minimizations': 'completed energy minimizations',
    'mmgbsa': 'completed single-frame MM/GBSA evaluations',
    'audited_md_trajectories': 'audited production MD trajectories',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fasta_samples(path):
    records = []
    header = None
    sequence = []
    for line in path.read_text().splitlines() + ['>END']:
        if line.startswith('>'):
            if header is not None and re.search(r'\bsample\s*=\s*\d+', header):
                value = ''.join(sequence).replace('/', '')
                assert value and set(value) <= set('ACDEFGHIKLMNPQRSTVWY'), path
                records.append((header, value))
            header, sequence = line[1:], []
        elif line.strip():
            sequence.append(line.strip())
    return records


def numeric_inventory(job, audit_path, output):
    job = job.resolve(strict=True)
    audit_path = audit_path.resolve(strict=True)
    audit = read(audit_path)
    assert audit['model_coordinates_independently_checked'], 'Full coordinate audit required'
    plan = read(job / 'plan.json')
    campaign = Path(plan['campaign']).resolve(strict=True)
    assert campaign == job / 'campaign'
    receipt = read(job / 'submission.json')
    sources = {str(audit_path): sha(audit_path), str(job/'plan.json'): sha(job/'plan.json'),
               str(job/'submission.json'): sha(job/'submission.json'),
               str(Path(__file__)): sha(Path(__file__)),
               str(FROZEN/'build_inference_atlas.py'): sha(FROZEN/'build_inference_atlas.py'),
               str(FROZEN/'report_production_events.py'): sha(FROZEN/'report_production_events.py')}
    timings = {(r['batch_id'], r['row']): r for r in audit['row_timings']}
    screens, full = deduplicate(audit)
    excluded = []
    def owned(row):
        key = (row['batch_id'], Path(row['summary_path']).parent.name)
        valid = timings.get(key, {}).get('recorded_owner') is True
        if not valid:
            excluded.append({'source': row['summary_path'], 'reason': 'No owning run receipt'})
        return valid
    screens = [s for s in screens if owned(s)]
    full = [s for s in full if owned(s)]
    try:
        events, event_resolutions = production_events(audit, screens, full)
        resolved_times = {(x['canonical_batch_id'], x['canonical_row']): x['production_time']
                          for x in event_resolutions}
        event_times = {}
        reconstructed = collections.Counter()
        for kind, values in [('screen', screens), ('expanded', full)]:
            for row in values:
                key = (row['batch_id'], Path(row['summary_path']).parent.name)
                at = finish(timings[key])
                if at is None:
                    at = resolved_times[key]
                event_times[row['summary_path']] = at
                reconstructed[(row['round'], kind, at)] += 1
        assert reconstructed == collections.Counter((e['round'], e['kind'], e['time']) for e in events)
    except (AssertionError, KeyError, FileNotFoundError) as exc:
        event_times, event_resolutions = {}, []
        excluded.append({'scope': 'completion timestamps only', 'reason': repr(exc)})
    verified = collections.defaultdict(list)
    for model in audit['verified_models']:
        verified[(model['kind'], model['round'], model['hypothesis'], model['sha256'])].append(model)
    counts = collections.Counter({key: 0 for key in STAGES})
    durations = collections.defaultdict(list)
    rows, issues = [], []

    def witness(path, folder):
        resolved = path.resolve(strict=True)
        assert resolved.is_file() and resolved.is_relative_to(folder), path
        return {'path': str(path), 'sha256': sha(resolved)}

    def coordinate_witness(models, kind, row, folder):
        seen = set()
        for model in models:
            digest = model.get('sha256')
            candidates = verified.get((kind, row['round'], row['hypothesis'], digest), [])
            candidates = [x for x in candidates if Path(x['path']).resolve(strict=True).is_relative_to(folder)]
            assert candidates, f'Missing independent coordinate witness: {row["summary_path"]}'
            path = Path(candidates[0]['path']).resolve(strict=True)
            assert path not in seen
            seen.add(path)
        assert len(seen) == 5, (kind, len(seen), row['summary_path'])
        return len(seen)

    for kind, candidates in [('crop', screens), ('full', full)]:
        for row in candidates:
            folder = Path(row['summary_path']).parent.resolve(strict=True)
            record = {'kind': kind, 'round': row['round'], 'hypothesis': row['hypothesis'],
                      'batch_id': row['batch_id'], 'summary_path': row['summary_path'],
                      'condition': row['condition'], 'completion_time': event_times.get(row['summary_path']),
                      'counts': {}, 'witnesses': []}
            c = record['counts']
            if kind == 'crop':
                c['crop_folding_calls'] = 1
                c['crop_fold_models'] = coordinate_witness(row['models'], 'crop', row, folder)
                try:
                    path = folder/'generation_stage/generation_manifest.json'
                    manifest = read(path)
                    assert manifest['status'] == 'passed' and manifest['seed'] == row['seed']
                    assert manifest['input_sha256'] == row['input_sha256']
                    backbones = set()
                    for complex_ in manifest['complexes']:
                        artifact = Path(complex_['path'])
                        proof = witness(artifact, folder)
                        assert proof['sha256'] == complex_['sha256']
                        structure = gemmi.read_structure(str(artifact))
                        assert len(structure) == 1
                        binder = structure[0]['B']
                        assert sum(res.name != 'HOH' for res in binder) == 60
                        assert all(math.isfinite(float(v)) for res in binder for atom in res
                                   for v in (atom.pos.x, atom.pos.y, atom.pos.z))
                        backbones.add(artifact.resolve())
                        record['witnesses'].append(proof)
                    assert backbones
                    c['rfd_backbones'] = len(backbones)
                    record['witnesses'].append(witness(path, folder))
                except (AssertionError, OSError, ValueError, KeyError, RuntimeError) as exc:
                    issues.append({'stage': 'rfd_backbones', 'source': row['summary_path'], 'error': repr(exc)})
                try:
                    files = sorted((folder/'mpnn/seqs').glob('*.fa'))
                    assert files
                    samples = [s for path in files for _, s in fasta_samples(path)]
                    assert row['binder_sequence'] in samples and all(len(s) == 60 for s in samples)
                    c['mpnn_sequences'] = len(samples)
                    record['witnesses'].extend(witness(path, folder) for path in files)
                except (AssertionError, OSError, ValueError, KeyError) as exc:
                    issues.append({'stage': 'mpnn_sequences', 'source': row['summary_path'], 'error': repr(exc)})
                for stage, name in [('RFD4-Proteina', 'rfd_backbones'), ('proteinmpnn', 'mpnn_sequences'),
                                    ('chai-lab', 'crop_folding_calls')]:
                    if stage in row.get('stage_seconds', {}):
                        durations[name].append(float(row['stage_seconds'][stage]))
            else:
                proof = read(folder/'validation-report.json')
                models = [{'sha256': sha(Path(x['cif']).resolve(strict=True))} for x in proof['predictions']]
                c['full_folding_calls'] = 1
                c['full_fold_models'] = coordinate_witness(models, 'intact', row, folder)
                energy = read(folder/'mmgbsa/report.json')
                assert energy['minimization_completed'] and energy['mmgbsa_completed']
                assert energy['frames'] == 1
                assert abs(energy['static_delta_total_kcal_mol']-row['static_MMGBSA_kcal_mol']) < 1e-5
                c['minimizations'] = c['mmgbsa'] = 1
                record['witnesses'].append(witness(folder/'mmgbsa/report.json', folder))
                for stage, name in [('cofold', 'full_folding_calls'), ('mmgbsa', 'mmgbsa')]:
                    if stage in row.get('stage_seconds', {}):
                        durations[name].append(float(row['stage_seconds'][stage]))
            counts.update(c)
            rows.append(record)
    energy_script = campaign/'inputs/intact-egfr-energy-pathsafe/mmgbsa.py'
    if not energy_script.exists():
        energy_script = campaign/'inputs/intact-egfr-energy/mmgbsa.py'
    tree = ast.parse(energy_script.read_text())
    steps = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'step']
    minima = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == 'minimizeEnergy']
    assert not steps and minima, 'MD classification requires manual review'
    sources[str(energy_script)] = sha(energy_script)
    result = {'at': time.time(), 'job_id': receipt['job_id'], 'job': str(job),
              'audit': str(audit_path), 'audit_sha256': sha(audit_path),
              'campaign_completion_passed': audit.get('required_campaign_completion_checks_passed', False),
              'numeric_error_count': len(audit.get('errors', [])),
              'unresolved_claim_count': len(audit.get('unowned_or_missing_claim_sources', [])),
              'canonical_crop_input_sets': len(screens), 'canonical_full_input_sets': len(full),
              'stage_counts': dict(counts), 'stage_units': STAGES, 'stage_seconds': dict(durations),
              'issues': issues, 'excluded': excluded, 'sources': sources,
              'event_time_resolutions': event_resolutions,
              'md_classification': {'script': str(energy_script), 'sha256': sha(energy_script),
                                    'integration_step_lines': steps, 'minimization_lines': minima,
                                    'scope': 'Frozen audited scientific route only; no MD trajectories are counted from single minimized frames.'},
              'scope': 'Distinct inputs selected by the unchanged native deduplicator, with recorded run ownership and independent coordinate checks. RFD counts require a hashed generated structure and a 60-residue finite binder; MPNN counts require sampled FASTA records matching the folded binder. Incomplete or unwitnessed stages are not inferred from pipeline success. Outputs are credited at enclosing row completion for timelines. Numerical outputs do not certify narrative claims or campaign completion.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    witness_path = output.with_suffix('.witnesses.jsonl')
    witness_path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    result['witnesses'] = {'path': str(witness_path), 'sha256': sha(witness_path), 'rows': len(rows)}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['job_id','canonical_crop_input_sets','canonical_full_input_sets','stage_counts']}, indent=2))
    print('stage_issues', len(issues), 'excluded', len(excluded))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', type=Path)
    p.add_argument('--audit', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    numeric_inventory(args.job, args.audit or args.job/'campaign/scaling-audit.json', args.output)
