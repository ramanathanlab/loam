"""Simultaneous replicas must not multiply the allocation wall clock."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/audit_scale20_campaign.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_simultaneous_replica_overlap_is_intersection():
    science={'start_epoch':100,'end_epoch':200}
    replicas=[{'JobIDRaw':'a','start_epoch':80,'end_epoch':190},{'JobIDRaw':'b','start_epoch':120,'end_epoch':220}]
    windows=audit.pooled_overlap(science,replicas,2)
    assert windows==[{'start':120,'end':190,'seconds':70,'inference_jobs':['a','b']}]


def test_missing_or_nonoverlapping_replica_is_not_full_split():
    science={'start_epoch':100,'end_epoch':200}
    one={'JobIDRaw':'a','start_epoch':80,'end_epoch':190}
    assert audit.pooled_overlap(science,[one],2)==[]
    assert audit.pooled_overlap(science,[one,{'JobIDRaw':'b','start_epoch':191,'end_epoch':220}],2)==[]


def test_source_review_never_ignores_scientific_parameters(tmp_path):
    canonical=tmp_path/'canonical.py';copied=tmp_path/'copied.py'
    canonical.write_text('"""Documentation"""\nsteps=200\nresult={"models":models,"models_total":5,"steps":steps}\n')
    copied.write_text('steps=200\nresult={"models":models,"models_total":5,"models":models,"steps":steps}\n')
    assert audit.reviewed_source_ast(canonical)==audit.reviewed_source_ast(copied)
    copied.write_text(copied.read_text().replace('steps=200','steps=20'))
    assert audit.reviewed_source_ast(canonical)!=audit.reviewed_source_ast(copied)


def test_profile_small_sample_percentiles_match_inclusive_reference():
    import statistics
    profile_spec=importlib.util.spec_from_file_location('pool_profile',Path(__file__).parents[1]/'scripts/profile_inference_pool.py')
    profile=importlib.util.module_from_spec(profile_spec);profile_spec.loader.exec_module(profile)
    for values in (list(range(24)),list(range(8)),[1,2,3,50,100]):
        expected=statistics.quantiles(values,n=100,method='inclusive')[94]
        assert abs(profile.percentile(list(reversed(values)),.95)-expected)<1e-9


def test_input_audit_rejects_swapped_control_and_changed_delegate(tmp_path):
    import json
    inputs=tmp_path/'inputs';targets=inputs/'verified-egfr-inputs';targets.mkdir(parents=True)
    (targets/'dual.cif').write_text('dual hotspots')
    (targets/'one_region.cif').write_text('one region hotspots')
    (inputs/'run_binder_generation.py').write_text('steps=200')
    folder=tmp_path/'run_example';folder.mkdir()
    row={'condition':'one_region','input_sha256':audit.sha(targets/'dual.cif'),'generation_script_sha256':audit.sha(inputs/'run_binder_generation.py')}
    (folder/'input.json').write_text(json.dumps(row))
    assert [x['field'] for x in audit.input_protocol_issues(tmp_path,folder,'scale20_complete_binder_pipeline')]==['condition_specific_input_sha256']
    row['input_sha256']=audit.sha(targets/'one_region.cif')
    (folder/'input.json').write_text(json.dumps(row))
    assert audit.input_protocol_issues(tmp_path,folder,'scale20_complete_binder_pipeline')==[]
    row['generation_script_sha256']='changed'
    (folder/'input.json').write_text(json.dumps(row))
    assert [x['field'] for x in audit.input_protocol_issues(tmp_path,folder,'scale20_complete_binder_pipeline')]==['generation_script_sha256']


def test_request_concurrency_includes_idle_and_clips_allocation(monkeypatch):
    scripts=Path(__file__).parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    from summarize_inference_balance import concurrency_stats
    requests=[{'started_at':2,'finished_at':6},{'started_at':4,'finished_at':8}]
    result=concurrency_stats(requests,0,10,2)
    assert result['peak']==2 and result['time_weighted_mean']==.8
    assert result['seconds_by_concurrency']=={0:4,1:4,2:2}
    assert result['fraction_of_allocation_at_least_one_attempt_per_replica']==.2
    clipped=concurrency_stats([{'started_at':-5,'finished_at':20}],0,10,1)
    assert clipped['time_weighted_mean']==1 and clipped['seconds_by_concurrency'][1]==10
