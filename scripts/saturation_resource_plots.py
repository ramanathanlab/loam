"""Plot closed resource diagnostics separately from completed campaign results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_inference_atlas import write_csv


def resource_plots(experiment, save, output, sources, sha):
    if experiment is None:
        return []
    experiment=Path(experiment).resolve();rows=[]
    sources[str(Path(__file__).resolve())]=sha(Path(__file__))
    def evidence(path):
        path=Path(path);sources[str(path)]=sha(path)
        return json.loads(path.read_text())
    for decision_file in sorted(experiment.glob('*/diagnostics/gpu-sharing/decision.json')):
        decision=evidence(decision_file)
        if not decision['scientific_QC_passed']:continue
        base=decision_file.parent
        for kind,label in [('screen','Cropped screening'),('full_context','Full-context + scoring')]:
            qc=evidence(base/kind/'independent-QC.json')
            result=evidence(base/kind/'result.json')
            evidence(base/kind/'request.json')
            assert qc['passed'] and result['all_completed'] and len(result['phases'])==2
            assert len(result['gpu'].split(','))==1 and len(result['cpus'])==2
            for check in qc['checks']:
                summary=Path(check['summary']);assert sha(summary)==check['sha256'];evidence(summary)
            host=result['host'].split('.')[0]
            telemetry=list((decision_file.parents[2]/'campaign/telemetry').glob('*-'+host+'*.jsonl'))
            assert telemetry, 'Paired GPU busy requires its physical-node telemetry source'
            for path in telemetry:
                sources[str(path)]=sha(path)
            serial,parallel=qc['phases']
            assert serial['phase']=='serial' and parallel['phase']=='parallel'
            rows.append({'job_id':decision['source_job_id'],'kind':kind,'label':label,
                'physical_GPUs':1,'designs_per_phase':2,
                'serial_seconds':serial['elapsed_seconds'],'parallel_seconds':parallel['elapsed_seconds'],
                'throughput_speedup':serial['elapsed_seconds']/parallel['elapsed_seconds'],
                'serial_GPU_busy_percent':serial['diagnostic_GPU_busy_lower_bound_percent'],
                'parallel_GPU_busy_percent':parallel['diagnostic_GPU_busy_lower_bound_percent'],
                'minimum_phase_telemetry_coverage_percent':min(serial['telemetry_coverage_percent'],parallel['telemetry_coverage_percent']),
                'independently_verified_coordinate_files':qc['coordinates_verified'],
                'maximum_best_iptm_absolute_delta':max(pair['best_iptm_absolute_delta'] for pair in qc['pairs']),
                'scope':'Paired diagnostic replays; excluded from novel scientific throughput and hypothesis counts.'})
    if not rows:return []
    write_csv(output/'paired-resource-diagnostics.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained');x=np.arange(len(rows))
    values=[r['throughput_speedup'] for r in rows]
    bars=axes[0].bar(x,values,color='#168b80');axes[0].bar_label(bars,fmt='%.2f×',padding=4)
    axes[0].axhline(1,color='#526777',linestyle='--',lw=1)
    axes[0].set(ylabel='Two-row throughput / sequential throughput',ylim=(0,max(values)*1.25),title='Same physical GPU, unchanged scientific recipe')
    for offset,key,label,color in [(-.18,'serial_GPU_busy_percent','Sequential','#7891a1'),(.18,'parallel_GPU_busy_percent','Two simultaneous rows','#168b80')]:
        bars=axes[1].bar(x+offset,[r[key] for r in rows],.36,label=label,color=color)
        axes[1].bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
    axes[1].set(ylabel='Time-weighted physical GPU busy (%)',ylim=(0,100),title='Productive overlap during the paired test');axes[1].legend(fontsize=9)
    for ax in axes:ax.set_xticks(x,[r['label'] for r in rows]);ax.grid(axis='y',alpha=.2)
    verified=sum(r['independently_verified_coordinate_files'] for r in rows)
    save(fig,'paired-resource-diagnostics','Paired GPU-sharing diagnostics',
         f'Each case replays two completed designs sequentially and then concurrently on the same allocated GPU, with one CPU core per row. All {verified} predicted structures, full-context topology and static energy outputs were independently checked. Small folding/energy differences remain. This small diagnostic is not a completed campaign or a whole-job utilization result. Replays do not add novel designs or tested hypotheses.','diagnostic')
    for path in sorted(experiment.glob('*/diagnostics/cpu-performance/comparison.json')):
        comparison=evidence(path)
        evidence(path.parent/'independent-QC.json')
        for item in comparison['pairs']:
            for cpus in (1,8):
                folder=path.parent/(item['condition']+'-cpu'+str(cpus))
                evidence(folder/'binding.json');evidence(folder/'summary.json')
        fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
        values=[r['speedup'] for r in comparison['pairs']]
        bars=ax.bar([r['condition'] for r in comparison['pairs']],values,color='#bd6b56')
        ax.bar_label(bars,fmt='%.2f×',padding=3);ax.axhline(1,color='#526777',ls='--')
        ax.set(ylim=(0,1.2),ylabel='One-CPU wall time / eight-CPU wall time',title='Extra CPU cores did not improve these paired screens');ax.grid(axis='y',alpha=.2)
        save(fig,'cpu-resource-diagnostic','CPU resource change: measured and rejected',
             'Three paired diagnostic screen replays, across all three conditions. Eight reserved cores produced no speedup in this sample, so the campaign retained one CPU per scientific row. Host/load differences limit this small comparison. All30 structures passed independent numerical QC; duplicate replays are excluded from novel scientific throughput.','diagnostic')
    return rows
