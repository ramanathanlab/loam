#!/usr/bin/env python3
"""Check report exports, source closure, tables, slide counts and owned-secret exclusion."""
import argparse
import csv
import hashlib
import json
import re
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owned_infra(experiment, planfile, owned):
    """Resolve credentials only under an exactly receipt-owned job's infra.

    External scratch jobs must link to the same central ownership ledger.
    """
    job=planfile.parent.resolve();plan=read(planfile);receipt=read(job/'submission.json')
    assert receipt['job_id'] in owned and receipt['job_name']==plan['job_name']
    assert Path(plan['job_root']).resolve()==job
    assert (job.parent/'experiment.json').resolve(strict=True)==(experiment/'experiment.json').resolve(strict=True)
    infra=Path(plan['infra']).resolve()
    assert infra==job/'infra'
    return infra


class Document(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.ids=[];self.figures=0;self.options=0

    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        self.links.extend(values[key] for key in ['src','href'] if key in values)
        if 'id' in values:self.ids.append(values['id'])
        self.figures+=tag=='figure';self.options+=tag=='option'


def verify(output, experiment=None):
    manifest=read(output/'build-manifest.json');data=read(output/'data.json')
    for relative,digest in manifest['exports'].items():
        path=(output/relative).resolve()
        assert path.is_relative_to(output) and path.is_file() and sha(path)==digest,relative
    for name,digest in manifest['sources'].items():assert sha(Path(name))==digest,name
    documents={}
    for name in ['index.html','slides.html']:
        doc=Document();doc.feed((output/name).read_text());documents[name]=doc
        assert len(doc.ids)==len(set(doc.ids)),name
        for link in doc.links:
            url=urlsplit(link)
            if url.scheme=='data' or not url.path:continue
            assert not url.scheme and not url.netloc,'External report dependency'
            path=(output/unquote(url.path)).resolve()
            assert path.is_relative_to(output) and path.is_file(),link
    assert documents['index.html'].figures==len(manifest['figures'])
    assert documents['index.html'].options==sum(f['group']=='node' for f in manifest['figures'])
    with zipfile.ZipFile(output/'scaleup.pptx') as archive:
        assert archive.testzip() is None
        count=sum(bool(re.fullmatch(r'ppt/slides/slide\d+\.xml',name)) for name in archive.namelist())
    assert count==manifest['slide_count']
    assert len(re.findall(rb'/Type\s*/Page\b',(output/'scaleup.pdf').read_bytes()))==count
    assert documents['slides.html'].figures==count
    for filename,key in [('throughput','runs'),('node-utilization','nodes'),('hypotheses','hypotheses'),
                         ('screen-trials','screen_trials'),('full-context-trials','full_context_trials'),
                         ('production-events','production_events')]:
        with (output/(filename+'.csv')).open() as stream:
            assert sum(1 for _ in csv.DictReader(stream))==len(data[key]),filename
    assert len(data['nodes'])==sum(r['nodes'] for r in data['runs'])
    for run in data['runs']:
        assert sum(s['run']==run['label'] for s in data['screen_trials'])==run['screen_trials']
        assert sum(s['run']==run['label'] for s in data['full_context_trials'])==run['full_context_trials']
        assert abs(run['allocated_GPU_hours']-4*run['nodes']*run['allocated_seconds']/3600)<1e-8
        if run['qualified_campaign']:
            assert run['scientific_completion_passed'] and run['parent_state']=='COMPLETED'
            assert run['case_status']=='Qualified completed campaign'
        else:
            assert not run['scientific_completion_passed']
            assert run['case_status']=='Diagnostic numerical outputs; campaign not qualified'
            assert (run['numeric_errors_retained'] or run['unresolved_claim_sources_retained']
                    or run['parent_state']!='COMPLETED'
                    or any(value is False for value in run['scientific_completion_checks'].values()))
        tested=[h for h in data['hypotheses'] if h['run']==run['label'] and h['reviewed_fresh_science']]
        assert all(h['reviewed'] and not h['error'] and h['accepted_claims']>0 for h in tested)
        assert len(tested)==len({(h['round'],h['id']) for h in tested})
        if 'completed_scientific_assessment_threads' in run:
            assessed=[h for h in tested if h['outcome'] in {'supported','contradicted','inconclusive'}]
            assessment_ids={(h['round'],h['id']) for h in assessed}
            assert all(h['completed_scientific_assessment']==((h['round'],h['id']) in assessment_ids)
                       for h in data['hypotheses'] if h['run']==run['label'])
            assert len(assessed)==run['completed_scientific_assessment_threads']==run['reviewed_scientific_test_threads']
            assert len({h['id'] for h in assessed})==run['hypotheses_tested']==run['reviewed_scientific_test_hypotheses']
            assert len(tested)==run['reviewed_fresh_science_threads']
            assert len({h['id'] for h in tested})==run['hypotheses_with_reviewed_fresh_science']
        else:
            assert len(tested)==run['reviewed_scientific_test_threads']
            assert len({h['id'] for h in tested})==run['hypotheses_tested']
    assert set(data['gates'])<=set(r['job_id'] for r in data['runs'])
    if 'experiment_costs' in data:
        costs=data['experiment_costs'];ids=[row['job_id'] for row in costs['jobs']]
        assert len(ids)==len(set(ids))
        assert abs(costs['total_allocated_GPU_hours']-sum(row['allocated_GPU_hours'] for row in costs['jobs']))<1e-8
        for row in costs['jobs']:
            assert abs(row['allocated_GPU_hours']-row['accounting']['total_allocated_GPU_seconds']/3600)<1e-8
            assert row['included_as_successful_trial']==(row['job_id'] in {r['job_id'] for r in data['runs'] if r['qualified_campaign']})
        with (output/'experiment-costs.csv').open() as stream:
            assert len(list(csv.DictReader(stream)))==len(ids)
        if experiment is not None:
            assert set(ids)==set(read(experiment/'experiment.json')['owned_job_ids'])
    extra=data['full_goal_report']
    assert {r['job_id'] for r in extra['recipes']}==set(read(experiment/'experiment.json')['owned_job_ids'])
    stage_rows=extra['stage_throughput']
    with (output/'science-stage-throughput.csv').open() as stream:
        assert len(list(csv.DictReader(stream)))==len(stage_rows)
    with (output/'science-node-throughput.csv').open() as stream:
        assert len(list(csv.DictReader(stream)))==extra['stage_node_rows']
    with (output/'science-stage-events.csv').open() as stream:
        event_rows=list(csv.DictReader(stream))
        assert len(event_rows)==extra['stage_event_rows']
    for run in data['runs']:
        selected=[r for r in stage_rows if r['job_id']==run['job_id']]
        assert len(selected)==9
        for row in selected:
            assert row['count']>=0
            assert abs(row['per_allocated_GPU_hour']-row['count']/run['allocated_GPU_hours'])<1e-8
            assert sum(int(float(e.get(row['stage']) or 0)) for e in event_rows if e['job_id']==run['job_id'])==row['count']
        bystage={r['stage']:r for r in selected}
        assert bystage['crop_fold_models']['count']==5*run['screen_trials']
        assert bystage['full_fold_models']['count']==5*run['full_context_trials']
        assert bystage['minimizations']['count']==bystage['mmgbsa']['count']==run['full_context_trials']
        assert bystage['audited_md_trajectories']['count']==0
        inventory=next(x for x in extra['stage_inventories'] if x['job_id']==run['job_id'])
        assert not inventory['md_classification']['integration_step_lines']
        assert inventory['md_classification']['minimization_lines']
    assert 'Recipe book: operating Loam from 20 to 256 nodes' in (output/'index.html').read_text()
    with (output/'science-stage-by-round.csv').open() as stream:
        round_stages=list(csv.DictReader(stream))
    for row in stage_rows:
        assert sum(int(x['count']) for x in round_stages if x['job_id']==row['job_id'] and x['stage']==row['stage'])==row['count']
    with (output/'science-row-budgets.csv').open() as stream:
        budgets=list(csv.DictReader(stream))
    assert len(budgets)==len(extra['science_row_budgets'])
    owned=set(read(experiment/'experiment.json')['owned_job_ids'])
    assert all(x['job_id'] in owned for x in budgets)
    with (output/'all-owned-job-history.csv').open() as stream:
        history=list(csv.DictReader(stream))
    assert {x['job_id'] for x in history}==owned
    assert len(history)==len(owned)
    with (output/'allocation-phase-metrics.csv').open() as stream:
        phases=list(csv.DictReader(stream))
    assert len(phases)==extra['allocation_phases']['rows']
    for run in data['runs']:
        selected=[p for p in phases if p['job_id']==run['job_id']]
        if selected:
            assert abs(sum(float(p['seconds']) for p in selected)-run['final_allocation_seconds'])<.1
    for run in data['runs']:
        sequence=next(x for x in extra['sequence_production'] if x['job_id']==run['job_id'])
        assert sequence['unique_screen_binder_sequences']==len({x['binder_sequence'] for x in data['screen_trials'] if x['run']==run['label']})
        assert sequence['unique_full_binder_sequences']==len({x['binder_sequence'] for x in data['full_context_trials'] if x['run']==run['label']})
    secrets=[]
    if experiment is not None:
        owned=set(read(experiment/'experiment.json')['owned_job_ids'])
        for planfile in experiment.glob('*/plan.json'):
            job=planfile.parent;receipt=job/'submission.json'
            if not receipt.exists() or read(receipt)['job_id'] not in owned:continue
            infra=owned_infra(experiment,planfile,owned)
            paths=[infra/'redis/password',*infra.glob('replicas/*/state/nemotron/*/token')]
            service=infra/'pool-service.json'
            if service.exists():
                token=Path(read(service)['token_file']).resolve();assert token.is_relative_to(infra)
                paths.append(token)
            for path in paths:
                if path.is_file():
                    value=path.read_bytes().strip()
                    if len(value)>=16:secrets.append(value)
    checked=0
    for path in output.rglob('*'):
        if path.is_file() and path.suffix in {'.html','.json','.csv','.svg','.txt','.fasta'}:
            content=path.read_bytes();checked+=1
            assert b'-----BEGIN PRIVATE KEY-----' not in content,path
            assert not any(secret in content for secret in secrets),'Owned service secret in an export'
    result={'verified_at':time.time(),'passed':True,'slides':count,'figures':len(manifest['figures']),
            'node_plots':documents['index.html'].options,'source_hashes_checked':len(manifest['sources']),
            'export_hashes_checked':len(manifest['exports']),'owned_secret_values_checked':len(secrets),
            'text_exports_scanned':checked,'offline_HTML_links_valid':True,'CSV_counts_match':True,
            'GPU_hour_denominators_checked':True,'verifier_sha256':sha(Path(__file__))}
    result['all_owned_experiment_costs_checked']='experiment_costs' in data
    result['reviewed_hypothesis_counts_checked']=True
    result['qualified_vs_diagnostic_status_checked']=True
    result['scientific_stage_counts_and_rates_checked']=True
    result['all_owned_recipe_sources_present']=True
    result['all_owned_history_scientific_row_budgets_and_iteration_counts_checked']=True
    path=output/'verification.json';path.write_text(json.dumps(result,indent=2)+'\n')
    manifest['exports']['verification.json']=sha(path)
    manifest['sources'][str(Path(__file__).resolve())]=sha(Path(__file__))
    (output/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():archive.write(path,Path(output.name)/path.relative_to(output))
    with zipfile.ZipFile(output.with_suffix('.zip')) as archive:assert archive.testzip() is None
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=lambda p:Path(p).resolve())
    parser.add_argument('--experiment',type=lambda p:Path(p).resolve());args=parser.parse_args()
    print(json.dumps(verify(args.output,args.experiment),indent=2))
