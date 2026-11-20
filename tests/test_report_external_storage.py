"""Secret scans must stay inside receipt-owned jobs, including external scratch."""
import json
from pathlib import Path

import pytest

from verify_saturation_report import owned_infra


def staged(tmp_path, external):
    control=tmp_path/'control';control.mkdir()
    (control/'experiment.json').write_text(json.dumps({'owned_job_ids':['123']}))
    storage=tmp_path/'scratch' if external else control
    if external:
        storage.mkdir();(storage/'experiment.json').symlink_to(control/'experiment.json')
    job=storage/'test';job.mkdir();(job/'infra').mkdir()
    plan={'job_root':str(job),'infra':str(job/'infra'),'job_name':'owned-test'}
    (job/'plan.json').write_text(json.dumps(plan))
    (job/'submission.json').write_text(json.dumps({'job_id':'123','job_name':'owned-test'}))
    return control,job,plan


@pytest.mark.parametrize('external',[False,True])
def test_exact_owned_job_infra_is_allowed(tmp_path,external):
    control,job,_=staged(tmp_path,external)
    assert owned_infra(control,job/'plan.json',{'123'})==job/'infra'


@pytest.mark.parametrize('field',['infra','job_root','job_name'])
def test_plan_cannot_redirect_scan_or_change_identity(tmp_path,field):
    control,job,plan=staged(tmp_path,True)
    plan[field]='other-job' if field=='job_name' else str(tmp_path/'another-job')
    (job/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(AssertionError):owned_infra(control,job/'plan.json',{'123'})


def test_external_job_must_share_the_exact_ledger(tmp_path):
    control,job,_=staged(tmp_path,True)
    link=job.parent/'experiment.json';link.unlink();link.write_text('{}')
    with pytest.raises(AssertionError):owned_infra(control,job/'plan.json',{'123'})


def test_unowned_receipt_is_rejected(tmp_path):
    control,job,_=staged(tmp_path,True)
    with pytest.raises(AssertionError):owned_infra(control,job/'plan.json',{'456'})
