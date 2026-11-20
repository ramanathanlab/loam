"""Import only verified, explicitly scoped operator figures into offline reports.
No scientific inventory, throughput, gate or GPU integral is recomputed here.
"""
import hashlib,json,shutil
from pathlib import Path

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def import_verified_control_plane_figures(experiment,output,sources):
 if experiment is None:return [],[]
 root=Path(experiment).resolve();assets=root/'resource-diagnostic-figures';output=Path(output)
 sources[str(Path(__file__).resolve())]=digest(Path(__file__))
 specs=[('registry-index-benchmark','Registry metadata: equivalent API results with fewer file reads',root/'candidates/registry-index/benchmark-result.json',None,None),
        ('retired256-utilization-diagnostic','Retired 256-node diagnostic: 24.9% whole-parent GPU busy',root/'unified-256/diagnostics/retired-final-hardware.json','2976829','unified-256'),
        ('controller-cpu-replay','Controller CPU diagnostic and controlled mock-agent replay',root/'candidates/controller-gc-policy/controller-CPU-figure-data.json','2987415','unified-256b'),
        ('retired256b-pipeline-diagnostic','Retired 256-node pipeline: GPU activity, hypothesis supply and retained errors',assets/'retired256b-pipeline-data.json','2987415','unified-256b')]
 figures=[];records=[]
 for slug,title,expected_source,owned_job,job_label in specs:
  proof_path=assets/(slug+'-provenance.json')
  if not proof_path.exists():continue
  proof=read(proof_path);source=Path(proof['source']).resolve(strict=True)
  assert source==expected_source.resolve(strict=True),'Unexpected operator diagnostic source'
  assert digest(source)==proof['source_sha256'],'Operator diagnostic data changed'
  assert isinstance(proof['scope'],str) and proof['scope']
  if owned_job:
   job=(root/job_label).resolve();receipt=read(job/'submission.json');plan=read(job/'plan.json')
   assert receipt['job_id']==owned_job and owned_job in read(root/'experiment.json')['owned_job_ids']
   assert receipt['job_name']==plan['job_name'] and digest(job/'job.sh')==receipt['script_sha256']
   assert (job.parent/'experiment.json').resolve(strict=True)==(root/'experiment.json').resolve(strict=True)
  if slug=='controller-cpu-replay':
   data=read(source);support=proof['supporting_sources_sha256']
   assert proof['owned_job_id']==data['job_id']==owned_job and proof['owned_job_label']==job_label
   assert support==data['supporting_sources_sha256']
   study=root/'candidates/controller-gc-policy'
   cases=[f'{kind}-{repeat}.json' for kind in ['baseline-default','baseline-gc10000','candidate-default','candidate-gc10000'] for repeat in [1,2]]
   expected_support={p.resolve(strict=True) for p in [study/'replay-matrix.json',job/'diagnostics/controller-native-CPU-summary.json',job/'diagnostics/controller-native-CPU.json']+[study/name for name in cases]}
   resolved={Path(name).resolve(strict=True):value for name,value in support.items()}
   assert len(resolved)==len(support) and set(resolved)==expected_support,'Unexpected controller diagnostic dependency'
   for dependency,value in resolved.items():
    assert digest(dependency)==value,'Controller diagnostic dependency changed'
    sources[str(dependency)]=value
   plot=root/'diagnostics/plot_controller_cpu.py'
   assert digest(plot)==proof['plot_source_sha256']
   sources[str(plot)]=proof['plot_source_sha256']
  if slug=='retired256b-pipeline-diagnostic':
   data=read(source);support=proof['supporting_sources_sha256']
   assert proof['owned_job_id']==data['job_id']==owned_job and proof['owned_job_label']==job_label
   assert support==data['supporting_sources_sha256']
   names=['diagnostics/retired-parent-hardware.json','campaign/state/admission-timings.jsonl',
          'infra/pool-request-timings.jsonl','campaign/state/rounds/round-001.json',
          'diagnostics/retired-scientific-summary.json','campaign/scaling-audit.numeric.json',
          'campaign/scaling-audit.partial.json','campaign/claim-ownership-audit.partial.json',
          'diagnostics/postrun-review-complete.json']
   expected_support={(job/name).resolve(strict=True) for name in names}
   resolved={Path(name).resolve(strict=True):value for name,value in support.items()}
   assert len(resolved)==len(support) and set(resolved)==expected_support,'Unexpected pipeline diagnostic dependency'
   for dependency,value in resolved.items():
    assert digest(dependency)==value,'Pipeline diagnostic dependency changed'
    sources[str(dependency)]=value
   assert data['full_campaign_qualified'] is False and data['recorded_rounds']==1
   assert data['independent_binned_integral_matches_native_whole_parent_integral'] is True
   assert abs(data['cumulative_whole_parent_GPU_busy_percent'][-1]-read(job/'diagnostics/retired-parent-hardware.json')['summary']['overall_GPU_busy_percent'])<1e-8
   assert sum(data['retained_round1_error_counts'])==sum(bool(t.get('error')) for t in read(job/'campaign/state/rounds/round-001.json')['threads'])==244
   plot=root/'diagnostics/plot_retired256b_pipeline.py'
   assert digest(plot)==proof['plot_source_sha256']
   sources[str(plot)]=proof['plot_source_sha256']
  expected={slug+'.png',slug+'.svg'};assert set(proof['outputs'])==expected
  for name,value in proof['outputs'].items():assert digest(assets/name)==value,'Operator figure bytes changed'
  for name in sorted(expected):
   target=output/'figures'/name;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(assets/name,target);sources[str(assets/name)]=proof['outputs'][name]
  relative='diagnostics/'+proof_path.name;target=output/relative;target.parent.mkdir(exist_ok=True);assert not target.exists();shutil.copyfile(proof_path,target)
  sources[str(proof_path)]=digest(proof_path);sources[str(source)]=digest(source)
  figures.append({'id':slug,'title':title,'caption':proof['scope'],'group':'control-plane','provenance_artifact':relative})
  records.append({'figure':slug,'source':str(source),'source_sha256':proof['source_sha256'],'provenance_artifact':relative,'scope':proof['scope']})
 return figures,records
