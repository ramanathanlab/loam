"""A model-producing thread is not automatically a reviewed hypothesis test, and
blocked work can produce useful data without completing its hypothesis test."""
import json

import pytest

from loam.saturation.metrics import completed_assessment_threads, reviewed_science_threads


def science_thread(h, source, error=None):
    return {'hypothesis': {'id': h}, 'report': {}, 'review': {}, 'error': error, 'accepted': [{'source': source}]}


def test_excludes_failed_unreviewed_and_reused_evidence():
    rows = [{'round': 1, 'hypothesis': h, 'batch_id': 'b-' + b}
            for h, b in [('ok', 'aa'), ('failed', 'bb'), ('unreviewed', 'cc'), ('stale', 'dd'), ('external', 'ee')]]
    audit = {'candidate_rows': rows, 'intact_followups': [], 'batch_states': []}
    ts = [science_thread(h, '/owned/batches/b-' + b + '/summary.json')
          for h, b in [('ok', 'aa'), ('failed', 'bb'), ('unreviewed', 'cc')]]
    ts[1]['error'] = 'review failed'
    ts[2]['review'] = None
    # A different old batch cannot establish a current-round test.
    ts.append(science_thread('stale', '/owned/batches/b-ee/summary.json'))
    ts.append(science_thread('external', 'https://example.org/batches/b-ee/summary.json'))
    assert reviewed_science_threads(audit, [{'round': 1, 'threads': ts}]) == {(1, 'ok')}


def test_summary_lineage_resolution_and_later_recovery(tmp_path):
    folder = tmp_path / 'batches/b-ff'
    folder.mkdir(parents=True)
    (folder / 'inputs.jsonl').write_text(json.dumps({'batch_dir': str(tmp_path / 'batches/b-aa')}) + '\n')
    audit = {'campaign': str(tmp_path),
             'candidate_rows': [{'round': 2, 'hypothesis': 'recovered', 'batch_id': 'b-aa'}],
             'intact_followups': [{'round': 2, 'hypothesis': 'resolved', 'batch_id': 'b-bb'}],
             'batch_states': [{'batch_id': 'b-ff', 'tool': 'scale20_summarize_binder_pipeline'}],
             'explicit_claim_source_resolutions': [
                 {'round': 2, 'hypothesis': 'resolved', 'original_source': 'short-name',
                  'files': [{'path': str(tmp_path / 'batches/b-bb/summary.json')}]}]}
    rounds = {1: {'round': 1, 'threads': [
                  science_thread('recovered', '/owned/batches/b-aa/summary.json', 'failed first attempt')]},
              2: {'round': 2, 'threads': [science_thread('recovered', str(folder)),
                                          science_thread('resolved', 'short-name')]}}
    assert reviewed_science_threads(audit, rounds) == {(2, 'recovered'), (2, 'resolved')}


def assessed_thread(name, outcome='supported', **updates):
    row = {'hypothesis': {'id': name}, 'report': {'outcome': outcome}, 'review': {}, 'error': None}
    row.update(updates)
    return row


@pytest.mark.parametrize('mapping', [False, True])
def test_completed_outcomes_require_fresh_science_and_a_clean_review(mapping):
    rows = [assessed_thread('positive'), assessed_thread('negative', 'contradicted'),
            assessed_thread('uncertain', 'inconclusive'), assessed_thread('blocked', 'blocked'),
            assessed_thread('unknown', 'untested'), assessed_thread('old-only'),
            assessed_thread('errored', error='TimeoutError'), assessed_thread('no-report', report=None),
            assessed_thread('no-review', review=None)]
    records = [{'round': 1, 'threads': rows}]
    fresh = {(1, row['hypothesis']['id']) for row in rows if row['hypothesis']['id'] != 'old-only'}
    actual = completed_assessment_threads({1: records[0]} if mapping else records, fresh)
    assert actual == {(1, 'positive'), (1, 'negative'), (1, 'uncertain')}


def test_round_identity_is_preserved_for_returning_hypotheses():
    records = [{'round': 1, 'threads': [assessed_thread('same')]},
               {'round': 2, 'threads': [assessed_thread('same', 'blocked')]},
               {'round': 3, 'threads': [assessed_thread('same')]}]
    assert completed_assessment_threads(records, {(1, 'same'), (2, 'same')}) == {(1, 'same')}
