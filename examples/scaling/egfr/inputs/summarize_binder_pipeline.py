"""CPU-only, failure-aware summary of one owned complete-pipeline batch."""
import hashlib, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

def main():
    row=json.loads(Path(sys.argv[1]).read_text());batch=Path(row['batch_dir']).resolve(strict=True)
    metadata=json.loads((batch/'batch.json').read_text());requested=int(row.get('expected_per_condition',4));groups=defaultdict(list);failures=[];candidates=[]
    for folder in sorted(batch.glob('run_*')):
        status_path=folder/'status.json'
        if not (folder/'done').exists() or not status_path.exists():raise RuntimeError(f'Batch row still unfinished: {folder.name}')
        status=json.loads(status_path.read_text())
        if status.get('returncode')!=0 or status.get('missing_outputs'):
            failures.append({'row':folder.name,'status':status});continue
        p=folder/'summary.json';s=json.loads(p.read_text());assert s['status']=='passed' and s['models_total']==5
        assert len(s['target_sequence'])==205 and len(s['binder_sequence'])==60
        assert len({m['sha256'] for m in s['models']})==5
        for m in s['models']:
            actual=Path(m['path']).resolve(strict=True)
            assert actual.is_relative_to(folder.resolve()),'Model provenance escaped its row'
            assert hashlib.sha256(actual.read_bytes()).hexdigest()==m['sha256']
        c={k:s[k] for k in ('candidate_id','condition','seed','binder_sequence','best_iptm','median_iptm','dual_contact_models','mean_pairwise_target_aligned_binder_CA_RMSD_A','elapsed_seconds')}
        c['summary_path']=str(p);c['summary_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();groups[s['condition']].append(c);candidates.append(c)
    assert len(candidates)+len(failures)==metadata['total']
    assert len({c['seed'] for c in candidates})==len(candidates),'Repeated seed inside comparison'
    mean=lambda v:statistics.mean(v) if v else None
    def describe(cc):
        ip=[c['best_iptm'] for c in cc];rms=[c['mean_pairwise_target_aligned_binder_CA_RMSD_A'] for c in cc]
        return {'successful_candidates':len(cc),'verified_models':5*len(cc),'distinct_sequences':len({c['binder_sequence'] for c in cc}),'mean_best_iptm':mean(ip),'median_best_iptm':statistics.median(ip) if ip else None,'mean_pairwise_pose_RMSD_A':mean(rms),'sample_SD_candidate_pose_RMSD_A':statistics.stdev(rms) if len(rms)>1 else None,'candidates_with_any_dual_contact_model':sum(c['dual_contact_models']>0 for c in cc),'dual_contact_models':sum(c['dual_contact_models'] for c in cc)}
    summaries={k:describe(groups[k]) for k in ('dual','one_region','ridge')};controls=describe(groups['one_region']+groups['ridge'])
    complete=all(len(groups[k])==requested for k in summaries) and not failures
    gate=None
    if complete:
        gate=summaries['dual']['mean_pairwise_pose_RMSD_A']<=controls['mean_pairwise_pose_RMSD_A']-controls['sample_SD_candidate_pose_RMSD_A']
    full={'batch_id':metadata['batch_id'],'batch_dir':str(batch),'expected_candidates_per_condition':requested,'counts':{'requested_rows':metadata['total'],'successful':len(candidates),'failed':len(failures),'distinct_sequences':len({c['binder_sequence'] for c in candidates})},'matched_complete':complete,'groups':summaries,'pooled_controls':controls,'predeclared_baseline_gate':{'metric':'mean pairwise target-aligned binder CA RMSD across five models, averaged per candidate','criterion':'dual mean <= pooled control mean minus control candidate sample SD','passed':gate,'interpretation':'Model-ensemble pose dispersion; no dynamics, affinity or biological validation.'},'candidates':candidates,'failures':failures}
    (Path.cwd()/'full_analysis.json').write_text(json.dumps(full,indent=2)+'\n')
    compact={k:v for k,v in full.items() if k not in ('candidates','failures')};compact['candidate_summary_paths']=[c['summary_path'] for c in candidates];compact['failure_rows']=[f['row'] for f in failures]
    # Compact lines preserve complete metrics without overflowing MCP reads.
    text='{\n'+',\n'.join(' '+json.dumps(k)+': '+json.dumps(v) for k,v in compact.items())+'\n}\n'
    (Path.cwd()/'analysis.json').write_text(text);print(text)
if __name__=='__main__':main()
