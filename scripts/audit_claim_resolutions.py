#!/usr/bin/env python3
"""Recheck claim ownership after citation-only sidecars, retaining numeric proof.

This never recomputes or changes scientific inventories. The supplied numeric
audit must already pass every required check except citation ownership, and is
retained unchanged with its SHA256. Every original accepted claim is checked
again against owning run/batch receipts and current file hashes.
"""
import argparse
import collections
import copy
import hashlib
import json
import re
import time
from pathlib import Path

from audit_scale20_campaign import claim_key


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, numeric_audit, output):
    assert numeric_audit != output.resolve(), 'Retain the numeric audit at a separate immutable path'
    original = read(numeric_audit)
    assert Path(original['campaign']).resolve() == root
    assert original['model_coordinates_independently_checked']
    assert original['checks']['independent_model_checks_requested_and_passed']
    assert all(value for key, value in original['required_campaign_completion_checks'].items()
               if key != 'all_accepted_artifact_sources_owned_and_present')
    resolutions = read(root / 'claim-source-resolutions.json')
    accepted = collections.Counter()
    unresolved, witnesses, ownership, round_hashes = [], [], {}, {}
    checked = 0
    for path in sorted((root / 'state/rounds').glob('round-*.json')):
        record = read(path)
        n = record['round']
        round_hashes[str(path)] = sha(path)
        assert len(record['threads']) == original['rounds'][str(n)]['selected']
        for thread in record['threads']:
            hyp = thread['hypothesis']['id']
            if hyp not in ownership:
                folders, batches = [], set()
                for receipt in (root / 'state/hypotheses' / hyp / 'runs').glob('r*.json'):
                    run = read(receipt)
                    folders.append((root / 'work' / hyp / 'outputs' / run['run_id']).resolve())
                    batches.update(run.get('batches', []))
                folders.extend((root / 'batches' / batch).resolve() for batch in batches)
                ownership[hyp] = folders
            def owned(p):
                return any(p.resolve().is_relative_to(folder) for folder in ownership[hyp])
            for claim in thread.get('accepted', []):
                source = claim.get('source', '')
                if source.startswith(('http://', 'https://')):
                    continue
                checked += 1
                cleaned = re.sub(r'(?::\d+(?:-\d+)?|#.*)$', '', source.strip(' `"\''))
                target = Path(cleaned)
                if owned(target) and (target.is_file() or target.is_dir() and (target / 'batch.json').is_file()):
                    accepted[n] += 1
                    continue
                resolution = resolutions.get(claim_key(n, hyp, claim))
                files = resolution.get('files', []) if resolution else []
                if files and all((p := Path(entry['path'])).is_file() and owned(p)
                                 and sha(p) == entry['sha256'] for entry in files):
                    accepted[n] += 1
                    witnesses.append({'round': n, 'hypothesis': hyp, 'original_source': source, **resolution})
                else:
                    unresolved.append({'round': n, 'hypothesis': hyp, 'source': source})
    assert set(map(str, accepted)) == set(original['rounds'])
    assert checked == sum(r['accepted_owned_artifact_claims'] for r in original['rounds'].values()) + len(original['unowned_or_missing_claim_sources'])
    report = {'checked_at': time.time(), 'scope': __doc__, 'verifier_sha256': sha(Path(__file__)), 'numeric_audit': str(numeric_audit),
              'numeric_audit_sha256': sha(numeric_audit), 'round_sources_sha256': round_hashes,
              'resolution_sidecar_sha256': sha(root / 'claim-source-resolutions.json'),
              'accepted_local_claims_checked': checked, 'accepted_owned_claims_by_round': dict(accepted),
              'explicit_resolution_witnesses': len(witnesses), 'unresolved': unresolved,
              'all_accepted_local_claims_owned_and_present': not unresolved}
    report_path = root / 'claim-ownership-audit.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    assert not unresolved, report_path
    result = copy.deepcopy(original)
    result['unowned_or_missing_claim_sources'] = unresolved
    result['explicit_claim_source_resolutions'] = witnesses
    for n, count in accepted.items():
        result['rounds'][str(n)]['accepted_owned_artifact_claims'] = count
    key = 'all_accepted_artifact_sources_owned_and_present'
    result['checks'][key] = result['required_campaign_completion_checks'][key] = True
    result['all_machine_checks_passed'] = all(result['checks'].values())
    result['required_campaign_completion_checks_passed'] = all(result['required_campaign_completion_checks'].values())
    result['verification_lineage'] = {'numeric_audit': str(numeric_audit), 'numeric_audit_sha256': sha(numeric_audit),
                                    'claim_audit': str(report_path), 'claim_audit_sha256': sha(report_path),
                                    'note': 'Numerical checks/inventories unchanged; only claim ownership was rechecked after exact hashed citation resolutions. Original claims, sources, files and numeric audit unchanged.'}
    for section in ('checks', 'required_campaign_completion_checks'):
        assert all(result[section][k] == v for k, v in original[section].items() if k != key)
    for n, row in original['rounds'].items():
        assert all(result['rounds'][n][k] == v for k, v in row.items() if k != 'accepted_owned_artifact_claims')
    allowed = {'unowned_or_missing_claim_sources', 'explicit_claim_source_resolutions', 'rounds', 'checks',
               'required_campaign_completion_checks', 'all_machine_checks_passed',
               'required_campaign_completion_checks_passed', 'verification_lineage'}
    assert all(result[k] == original[k] for k in original if k not in allowed)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('scope', 'round_sources_sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign', type=Path)
    parser.add_argument('numeric_audit', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    main(root, args.numeric_audit.resolve(), args.output or root / 'scaling-audit.json')
