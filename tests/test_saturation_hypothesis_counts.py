"""A model-producing thread is not automatically a reviewed hypothesis test."""
import json
from pathlib import Path


def test_excludes_failed_unreviewed_and_reused_evidence(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_saturation_campaign import reviewed_science_threads
    def thread(h,source,**kw):
        return dict(hypothesis={'id':h},report={},review={},error=None,
                    accepted=[{'source':source}],**kw)
    rows=[{'round':1,'hypothesis':h,'batch_id':'b-'+b}
          for h,b in [('ok','aa'),('failed','bb'),('unreviewed','cc'),('stale','dd'),('external','ee')]]
    audit={'candidate_rows':rows,'intact_followups':[], 'batch_states':[]}
    ts=[thread(h,'/owned/batches/b-'+b+'/summary.json') for h,b in [('ok','aa'),('failed','bb'),('unreviewed','cc')]]
    ts[1]['error']='review failed';ts[2]['review']=None
    # A different old batch cannot establish a current-round test.
    ts.append(thread('stale','/owned/batches/b-ee/summary.json'))
    ts.append(thread('external','https://example.org/batches/b-ee/summary.json'))
    assert reviewed_science_threads(audit,[{'round':1,'threads':ts}])=={(1,'ok')}


def test_summary_lineage_resolution_and_later_recovery(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_saturation_campaign import reviewed_science_threads
    folder=tmp_path/'batches/b-ff';folder.mkdir(parents=True)
    (folder/'inputs.jsonl').write_text(json.dumps({'batch_dir':str(tmp_path/'batches/b-aa')})+'\n')
    audit={'campaign':str(tmp_path),'candidate_rows':[{'round':2,'hypothesis':'recovered','batch_id':'b-aa'}],
           'intact_followups':[],'batch_states':[{'batch_id':'b-ff','tool':'scale20_summarize_binder_pipeline'}],
           'explicit_claim_source_resolutions':[{'round':2,'hypothesis':'resolved','original_source':'short-name',
               'files':[{'path':str(tmp_path/'batches/b-bb/summary.json')}]}]}
    audit['intact_followups'].append({'round':2,'hypothesis':'resolved','batch_id':'b-bb'})
    def thread(h,source,error=None):return {'hypothesis':{'id':h},'report':{},'review':{},'error':error,'accepted':[{'source':source}]}
    rounds={1:{'round':1,'threads':[thread('recovered','/owned/batches/b-aa/summary.json','failed first attempt')]},
            2:{'round':2,'threads':[thread('recovered',str(folder)),thread('resolved','short-name')]}}
    assert reviewed_science_threads(audit,rounds)=={(2,'recovered'),(2,'resolved')}
