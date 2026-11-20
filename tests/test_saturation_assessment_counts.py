"""Blocked work can produce useful data without completing its hypothesis test."""
import pytest

from audit_saturation_campaign import completed_assessment_threads


def thread(name,outcome='supported',**updates):
    row={'hypothesis':{'id':name},'report':{'outcome':outcome},'review':{},'error':None}
    row.update(updates)
    return row


@pytest.mark.parametrize('mapping',[False,True])
def test_completed_outcomes_require_fresh_science_and_a_clean_review(mapping):
    rows=[thread('positive'),thread('negative','contradicted'),thread('uncertain','inconclusive'),
          thread('blocked','blocked'),thread('unknown','untested'),thread('old-only'),
          thread('errored',error='TimeoutError'),thread('no-report',report=None),
          thread('no-review',review=None)]
    records=[{'round':1,'threads':rows}]
    fresh={(1,row['hypothesis']['id']) for row in rows if row['hypothesis']['id']!='old-only'}
    actual=completed_assessment_threads({1:records[0]} if mapping else records,fresh)
    assert actual=={(1,'positive'),(1,'negative'),(1,'uncertain')}


def test_round_identity_is_preserved_for_returning_hypotheses():
    records=[{'round':1,'threads':[thread('same')]},
             {'round':2,'threads':[thread('same','blocked')]},
             {'round':3,'threads':[thread('same')]}]
    assert completed_assessment_threads(records,{(1,'same'),(2,'same')})=={(1,'same')}
