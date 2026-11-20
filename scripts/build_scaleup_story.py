#!/usr/bin/env python3
"""Build the consolidated Jupiter scaling report and presentation from frozen evidence.

Use the RFD4 Python for numpy/matplotlib/gemmi. Presentation dependencies may
be supplied with PYTHONPATH; no campaign environment or jobs are modified.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import re
import statistics as st
import importlib.metadata
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runs/jupiter-binder-reserved-20261112T221509Z'
NEW = ROOT / 'runs/jupiter-scale20-20261113T042053Z'
CET = ZoneInfo('Europe/Berlin')
INK, TEAL, GOLD, GRAY = '#142f43', '#137f78', '#c78032', '#63758a'

ALIASES = {
    '89e54ca6': 'Bridge two mapped EGFR regions to constrain binder pose.',
    '69cc635b': 'Preserve dual-region contacts through independent co-folding.',
    'dc184c7e': 'Preorganize the binding face to bridge both regions.',
    '514b137e': 'Reduce pose dispersion versus matched one-region controls.',
    'a622614f': 'Compare dual-region bridging with a ridge-only control.',
    'b7f0e7e2': 'Improve pose persistence through dual-region topology.',
    '0fbc12d0': 'Test whether the second region stabilizes the interface.',
    'be78b111': 'Improve geometry with two-region rather than one-region conditioning.',
    'ebb43f40': 'Separate spatial conditioning from sampling-budget effects.',
    '263cc822': 'Improve interface geometry through cooperative spatial constraints.',
    '7670cd6e': 'Improve confidence and native geometry with two RFD4 hotspots.',
    '54aa1d5b': 'Increase confidence and pose reproducibility with two regions.',
    '4c7ced73': 'Preserve the dual-region epitope in expanded receptor context.',
    '42ecff23': 'Preserve epitope geometry after expanded-context re-folding.',
}
ROUND_NOTES = {
    'one-r4': 'The first audited design completed sequence design and five Chai models. Comparative superiority remained unresolved; only one distinct screened binder was available.',
    'one-r5': 'Four additional distinct binders reached complex prediction. The stored supported label concerns a small structural cohort; the reviewer flagged statement overreach.',
    'five-r6': 'Five hypotheses ran concurrently. A high tournament rank did not guarantee executable evidence: the first selected thread was blocked. Earlier contact classifications need the later corrected residue mapping.',
    'five-r7': 'Controls began to challenge the dual-region pose-stability claim. Two erroneous contact-analysis claims were subsequently retracted; their zero-contact conclusion is excluded.',
    'five-r8': 'Corrected native-residue mapping restored the contact measurements. In the matched four-dual/four-control test, dual contacts persisted but the prespecified pose-dispersion advantage failed.',
    'twenty-r1': 'The initial 76-GPU science burst generated the largest crop-model volume. Many hypotheses posed overlapping versions of the same mechanism; separate IDs are not independent biological mechanisms.',
    'twenty-r2': 'The loop reused earlier evidence and generated fresh screens and expanded-context follow-ups. One schema failure and one report without a review remain incomplete; two generated numerical comparisons are flagged.',
    'twenty-r3': 'The final round added six complete twelve-candidate screens and fresh follow-ups. Its wall time includes inference renewal and warmup. One selected thread failed the required output schema.',
}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binder_id(sequence):
    return 'binder_' + hashlib.sha256(sequence.encode()).hexdigest()[:12]


def timestamp(text):
    return datetime.fromisoformat(text).replace(tzinfo=CET).timestamp()


def num(x, digits=2):
    return '—' if x is None else f'{x:,.{digits}f}'


def escape(x):
    return html.escape(str(x))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def round_windows(root):
    events = defaultdict(lambda: {'start': [], 'end': []})
    for path in sorted(root.glob('*controller*.log')):
        for line in path.read_text(errors='replace').splitlines():
            m = re.search(r'round (\d+): (tournament|done)', line)
            if m:
                try:
                    when = timestamp(line[:23].replace(',', '.'))
                except ValueError:
                    continue
                events[int(m[1])]['start' if m[2] == 'tournament' else 'end'].append(when)
    return {n: {'start': min(v['start']), 'end': max(v['end']),
                'seconds': max(v['end']) - min(v['start'])}
            for n, v in events.items() if v['start'] and v['end']}


def summarize_binders(models, candidate_rows=None):
    grouped = defaultdict(list)
    rows_by_sequence = defaultdict(list)
    for model in models:
        grouped[model['binder_sequence']].append(model)
    for row in candidate_rows or []:
        rows_by_sequence[row['binder_sequence']].append(row)
    result = []
    for sequence, items in grouped.items():
        items = list({m['sha256']: m for m in items}.values())
        best = max(items, key=lambda m: m['scores']['iptm'])
        source_rows = rows_by_sequence[sequence]
        poses = [c['mean_pairwise_target_aligned_binder_CA_RMSD_A'] for c in source_rows]
        result.append({
            'binder_id': binder_id(sequence), 'sequence': sequence,
            'best_iptm': best['scores']['iptm'], 'median_model_iptm': st.median(m['scores']['iptm'] for m in items),
            'verified_models': len(items), 'models_with_dual_contacts': sum(m['dual_region_contact'] for m in items),
            'best_model_dual_contacts': best['dual_region_contact'],
            'best_model_pairs_below_2A': best.get('close_atom_pairs_lt2A', best.get('close_atom_pairs_2A')),
            'cysteines': sequence.count('C'), 'condition': '; '.join(sorted({c['condition'] for c in source_rows})) or 'not standardized',
            'hypothesis_ids': sorted({m.get('hypothesis', m.get('hypothesis_id')) for m in items}),
            'representative_model': best['path'], 'representative_model_sha256': best['sha256'],
            'representative_hypothesis': best.get('hypothesis', best.get('hypothesis_id')),
            'mean_pairwise_pose_RMSD_A': st.median(poses) if poses else None,
            'source_summaries': sorted({c['summary_path'] for c in source_rows}),
        })
    return sorted(result, key=lambda c: (-c['best_iptm'], c['binder_id']))


def collect():
    baseline = read(ROOT / 'reports/five-node-20261113/analysis.json')
    old_models = read(ROOT / 'reports/five-node-20261113/model-metrics.json')
    audit = read(NEW / 'scaling-audit.json')
    assert audit['required_campaign_completion_checks_passed'] and not audit['errors']
    assert audit['model_coordinates_independently_checked']
    provenance = read(ROOT / 'reports/scale20-20261113/report-provenance.json')
    assert sha(NEW / 'scaling-audit.json') == provenance['audit_source_sha256']
    new_models = [m for m in audit['verified_models'] if m['kind'] == 'crop']
    windows = {'old': round_windows(OLD), 'new': round_windows(NEW)}
    findings = (audit.get('independent_claim_review') or {}).get('findings', [])
    flag_counts = Counter((x['round'], x['hypothesis']) for x in findings)
    rounds, all_hypotheses, binder_export = [], [], []
    seen = set()
    for root, numbers in [(OLD, range(4, 9)), (NEW, range(1, 4))]:
        for n in numbers:
            is_new = root == NEW
            phase = 'twenty' if is_new else ('one' if n < 6 else 'five')
            key = f'{phase}-r{n}'
            record = read(root / f'state/rounds/round-{n:03}.json')
            model_rows = [m for m in (new_models if is_new else old_models) if m['round'] == n]
            candidate_rows = [c for c in audit['candidate_rows'] if c['round'] == n] if is_new else []
            binders = summarize_binders(model_rows, candidate_rows)
            threads = {t['hypothesis']['id']: t for t in record['threads']}
            hypotheses = []
            for position, h in enumerate(record['rankings'], 1):
                thread = threads[h['id']]
                hypothesis = {
                    'round_key': key, 'selection_rank': position, 'hypothesis_id': h['id'],
                    'summary': h['summary'], 'short_proposition': ALIASES.get(h['id'][:8], h['summary']),
                    'elo_at_selection': h['elo'],
                    'same_elo_selected_threads': sum(abs(other['elo'] - h['elo']) < 1e-8 for other in record['rankings']),
                    'stored_outcome': (thread.get('report') or {}).get('outcome', 'missing'),
                    'review_completed': thread.get('review') is not None and thread.get('report') is not None and not thread.get('error'),
                    'accepted_claims_as_recorded': len(thread.get('accepted', [])),
                    'reviewer_overreach_flag': (thread.get('review') or {}).get('statement_overreach'),
                    'independent_claim_findings': flag_counts[(n, h['id'])] if is_new else 0,
                    'recorded_error': thread.get('error'),
                    'source_round_record': str(root / f'state/rounds/round-{n:03}.json'),
                }
                hypotheses.append(hypothesis)
                all_hypotheses.append(hypothesis)
            interval = windows['new' if is_new else 'old'][n]
            followups = [c for c in audit['intact_followups'] if c['round'] == n] if is_new else []
            eligible = [c for c in followups if c['minimized_pose_geometry']['dual_region_contact']
                        and c['minimized_pose_geometry']['close_atom_pairs_lt2A'] == 0]
            lead_followup = max(eligible, key=lambda c: c['intact_iptm_best']) if eligible else None
            row = {
                'key': key, 'campaign': 'continuation' if is_new else 'original', 'round': n,
                'nodes': 20 if is_new else (1 if n < 6 else 5),
                'science_nodes': 19 if is_new else (1 if n < 6 else 5),
                'inference_nodes': 1 if is_new else 'external',
                'selected_hypotheses': len(hypotheses), 'reviewed_threads': sum(bool(h['review_completed']) for h in hypotheses),
                'verified_crop_files': len({m['sha256'] for m in model_rows}),
                'distinct_sequences': len(binders), 'new_sequences': sum(c['sequence'] not in seen for c in binders),
                'round_start': interval['start'], 'round_end': interval['end'],
                'round_wall_seconds': interval['seconds'],
                'crop_files_per_round_hour': len({m['sha256'] for m in model_rows}) / (interval['seconds'] / 3600),
                'median_sequence_best_iptm': st.median(c['best_iptm'] for c in binders),
                'maximum_iptm': binders[0]['best_iptm'],
                'sequences_best_iptm_ge08': sum(c['best_iptm'] >= .8 for c in binders),
                'followup_rows': len(followups),
                'top_hypotheses': hypotheses[:3], 'all_hypotheses': hypotheses,
                'top_binders': binders[:3], 'all_binders': binders,
                'lead_followup': lead_followup, 'note': ROUND_NOTES[key],
                'source_round_record': str(root / f'state/rounds/round-{n:03}.json'),
            }
            rounds.append(row)
            for c in binders:
                binder_export.append({'round_key': key, **c})
            seen.update(c['sequence'] for c in binders)
    phases = []
    for label, keys, allocation, science_nodes, inf_nodes in [
        ('1 node · development', ['one-r4', 'one-r5'], baseline['allocations'][0], 1, 'external'),
        ('5 nodes · tournament', ['five-r6', 'five-r7', 'five-r8'], baseline['allocations'][-1], 5, 'external'),
        ('20 nodes · continuation', ['twenty-r1', 'twenty-r2', 'twenty-r3'],
         {'job': '2876650', 'start': audit['jobs']['2876650']['start_epoch'], 'end': audit['jobs']['2876650']['end_epoch']}, 19, 1),
    ]:
        rr = [r for r in rounds if r['key'] in keys]
        model_count = sum(r['verified_crop_files'] for r in rr)
        seconds = allocation['end'] - allocation['start']
        round_seconds = max(r['round_end'] for r in rr) - min(r['round_start'] for r in rr)
        seqs = {c['sequence'] for r in rr for c in r['all_binders']}
        local_inf_hours = sum(int(j['ElapsedRaw']) for j in audit['inference_jobs']) / 3600 * 4 if science_nodes == 19 else None
        science_gpu_hours = science_nodes * 4 * seconds / 3600
        phases.append({
            'label': label, 'nodes': science_nodes + (inf_nodes if isinstance(inf_nodes, int) else 0),
            'science_nodes': science_nodes, 'local_inference_nodes': inf_nodes, 'science_job': allocation['job'],
            'start': allocation['start'], 'end': allocation['end'], 'allocation_seconds': seconds,
            'round_window_seconds': round_seconds, 'verified_crop_files': model_count,
            'distinct_sequences': len(seqs), 'hypotheses': len({h['hypothesis_id'] for r in rr for h in r['all_hypotheses']}),
            'screened_hypotheses': len({m.get('hypothesis', m.get('hypothesis_id')) for m in (new_models if science_nodes == 19 else old_models) if m['round'] in {r['round'] for r in rr}}),
            'selected_threads': sum(r['selected_hypotheses'] for r in rr), 'reviewed_threads': sum(r['reviewed_threads'] for r in rr),
            'models_per_allocation_hour': model_count / (seconds / 3600),
            'models_per_round_window_hour': model_count / (round_seconds / 3600),
            'science_gpu_hours': science_gpu_hours, 'local_inference_gpu_hours': local_inf_hours,
            'models_per_science_GPU_hour': model_count / science_gpu_hours,
            'models_per_science_plus_local_inference_GPU_hour': model_count / (science_gpu_hours + local_inf_hours) if local_inf_hours is not None else None,
        })
    assert [p['verified_crop_files'] for p in phases] == [25, 215, 2150]
    assert [p['distinct_sequences'] for p in phases] == [5, 43, 358]
    assert len({c['sequence'] for c in binder_export}) == 402
    assert sum(r['verified_crop_files'] for r in rounds) == 2390
    # Recheck every selected representative against the full earlier coordinate audit.
    representative_paths = {}
    for r in rounds:
        for c in r['top_binders']:
            assert sha(c['representative_model']) == c['representative_model_sha256']
            representative_paths[c['representative_model']] = c['representative_model_sha256']
    source_paths = [ROOT / 'reports/five-node-20261113/analysis.json', ROOT / 'reports/five-node-20261113/model-metrics.json',
                    NEW / 'scaling-audit.json', NEW / 'final-metrics.json', NEW / 'independent-claim-review.json',
                    OLD / 'campaign-verification.json', OLD / 'diagnostics/evidence-retraction-20261113T005333Z/audit.json']
    source_paths += [Path(r['source_round_record']) for r in rounds]
    source_paths += list(ROOT.glob('reports/five-node-20261113/*saturation.svg'))
    source_paths += [ROOT/'reports/scale20-20261113'/f'{name}.svg' for name in ['science-slot-leases','node-gpu-utilization','inference-queue-and-gpu-power','aggregate-gpu-activity']]
    telemetry = read(ROOT / 'reports/scale20-20261113/telemetry-summary.json')
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(), 'phases': phases, 'rounds': rounds,
        'hypotheses': all_hypotheses, 'binder_rows': binder_export,
        'followups': audit['intact_followups'], 'independent_claim_findings': findings,
        'telemetry_summary': telemetry, 'baseline': baseline, 'final_metrics': read(NEW / 'final-metrics.json'),
        'owned_allocations': ([{'job':j['job'],'start':j['start'],'end':j['end'],'nodes':j['nodes'],'role':'science','campaign':'original','purpose':j['note']} for j in baseline['allocations']]
                              + [{'job':j['JobIDRaw'],'start':j['start_epoch'],'end':j['end_epoch'],'nodes':int(j['NNodes']),'role':'inference' if j['JobIDRaw'] in {x['JobIDRaw'] for x in audit['inference_jobs']} else 'science','campaign':'continuation','purpose':j['JobName']} for j in audit['jobs'].values()]),
        'pilot': {'science_nodes': 1, 'inference_nodes': 1, 'total_owned_nodes': 2,
                  'crop_models': 20, 'expanded_models': 5, 'sequences': 4,
                  'allocation_seconds': int(audit['jobs']['2876542']['ElapsedRaw']),
                  'crop_models_per_science_allocation_hour': 20 / (int(audit['jobs']['2876542']['ElapsedRaw']) / 3600),
                  'note': 'Operator recipe-validation pilot; no tournament review. Shared inference allocation overlaps other preflight work; excluded from main campaign totals.'},
        'source_sha256': {str(p): sha(p) for p in source_paths},
        'selected_coordinate_sha256': representative_paths,
        'ranking_method': 'Top selected hypotheses retain stored tournament order and selection-time Elo. Equal Elo is explicitly tied; neither Elo nor stored supported labels establish a scientific mechanism. Binder lists deduplicate sequences within a round and sort by the largest verified crop ipTM in that round.',
        'throughput_method': 'Primary rate: verified crop coordinate files divided by the complete science allocation elapsed time. Separate round-window rate spans first scientific round tournament start to last completed round, including interrupted attempts and intervening gaps. External inference is unmetered in the 1/5-node runs; local inference is counted in the 20-node cost boundary.',
    }


def plots(data, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'axes.spines.right': False, 'axes.spines.top': False,
                         'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / (name + '.png'), dpi=190, bbox_inches='tight', facecolor='white')
        fig.savefig(output / (name + '.svg'), bbox_inches='tight', facecolor='white')
        plt.close(fig)
    phases = data['phases']; x = np.arange(3)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for offset, field, color, label in [(-.19, 'models_per_allocation_hour', TEAL, 'Complete science-allocation window'),
                                          (.19, 'models_per_round_window_hour', GOLD, 'Scientific round window (includes gaps/restarts)')]:
        bars = ax.bar(x + offset, [p[field] for p in phases], width=.35, color=color, label=label)
        ax.bar_label(bars, labels=[f'{p[field]:.1f}' for p in phases], padding=4, fontsize=11)
    ax.set(xticks=x, xticklabels=['1 node\nDevelopment + debugging', '5 nodes\nExternal inference', '20 total nodes\n19 science + 1 Nemotron'],
           ylabel='Verified crop model files / hour', ylim=(0, 850), title='Measured throughput across the three main resource configurations')
    ax.legend(loc='upper left', fontsize=9)
    save(fig, 'throughput-by-nodes')
    import matplotlib.dates as mdates
    fig,axes=plt.subplots(2,1,figsize=(11,5.7))
    for ax,campaign,title in zip(axes,['original','continuation'],['Original allocations: development, bootstrap retries and five-node science','Continuation allocations: local inference, preflights and nineteen-node science']):
        jobs=[j for j in data['owned_allocations'] if j['campaign']==campaign]
        points=sorted({j[k] for j in jobs for k in ['start','end']})
        dates=[datetime.fromtimestamp(t,CET) for t in points]
        science=[sum(j['nodes'] for j in jobs if j['role']=='science' and j['start']<=t<j['end']) for t in points]
        inference=[sum(j['nodes'] for j in jobs if j['role']=='inference' and j['start']<=t<j['end']) for t in points]
        ax.stackplot(dates,science,inference,step='post',labels=['Owned scientific allocations','Owned local inference'],colors=[TEAL,GOLD],alpha=.8)
        ax.step(dates,np.array(science)+np.array(inference),where='post',color=INK,lw=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=CET));ax.set(title=title,ylabel='Owned nodes',xlabel='Local CET · 12–13 November 2026',ylim=(0,12 if campaign=='original' else 22));ax.legend(loc='upper left',fontsize=8)
    save(fig,'allocation-timeline')
    rounds = data['rounds']; x = np.arange(len(rounds))
    labels = [f'{r["nodes"]}n / R{r["round"]}' for r in rounds]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    colors = [GRAY if r['nodes'] == 1 else (TEAL if r['nodes'] == 5 else INK) for r in rounds]
    bars = axes[0].bar(x, [r['verified_crop_files'] for r in rounds], color=colors)
    axes[0].bar_label(bars, padding=3, fontsize=10)
    axes[0].set(xticks=x, xticklabels=labels, ylabel='Verified crop model files', ylim=(0, 1100), title='Output by scientific iteration')
    bars = axes[1].bar(x, [r['new_sequences'] for r in rounds], color=colors)
    axes[1].bar_label(bars, padding=3, fontsize=10)
    axes[1].set(xticks=x, xticklabels=labels, ylabel='Sequences first seen across both campaigns', title='Novel sequence throughput, not model multiplicity')
    for ax in axes: ax.tick_params(axis='x', labelrotation=35)
    save(fig, 'output-by-iteration')
    fig, ax = plt.subplots(figsize=(11, 4.1))
    ax.plot(x, [r['maximum_iptm'] for r in rounds], marker='o', color=TEAL, lw=2, label='Highest ipTM in the round')
    ax.plot(x, [r['median_sequence_best_iptm'] for r in rounds], marker='s', color=GOLD, lw=2, label='Median sequence-best ipTM')
    ax.axhline(.8, color=GRAY, ls='--', lw=.8)
    ax.set(xticks=x, xticklabels=labels, ylabel='Chai crop ipTM', ylim=(0, 1), title='High-confidence examples persist; cohort composition changes')
    ax.legend(loc='lower left', fontsize=10)
    save(fig, 'confidence-by-iteration')
    conditions = data['final_metrics']['condition_results']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = ['dual', 'one_region', 'ridge']; pos = np.arange(3)
    bars = axes[0].bar(pos, [conditions[k]['median_sequence_best_iptm'] for k in groups], color=[TEAL, INK, GRAY])
    axes[0].bar_label(bars, fmt='%.3f', padding=4)
    axes[0].set(xticks=pos, xticklabels=['Dual region', 'One region', 'Ridge'], ylim=(0, 1), ylabel='Median sequence-best crop ipTM', title='Twenty-node controlled screen')
    bars = axes[1].bar(pos, [conditions[k]['sequences_contact_both_regions'] / conditions[k]['distinct_sequences'] for k in groups], color=[TEAL, INK, GRAY])
    axes[1].bar_label(bars, labels=[f'{conditions[k]["sequences_contact_both_regions"]}/{conditions[k]["distinct_sequences"]}' for k in groups], padding=4)
    axes[1].set(xticks=pos, xticklabels=['Dual region', 'One region', 'Ridge'], ylim=(0, 1.15), ylabel='Sequence fraction with ≥1 dual-contact model', title='Contact presence alone is insufficient')
    save(fig, 'condition-comparison')
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for condition, color in zip(groups, [TEAL, INK, GRAY]):
        rows = [c for c in data['followups'] if c['condition'] == condition]
        for retained, marker in [(True, 'o'), (False, 'x')]:
            rr = [c for c in rows if bool(c['minimized_pose_geometry']['dual_region_contact']) == retained]
            ax.scatter([c['static_MMGBSA_kcal_mol'] for c in rr], [c['intact_iptm_best'] for c in rr],
                       marker=marker, color=color, alpha=.78, s=45,
                       label=f'{condition}: {"retained" if retained else "lost"} dual contacts')
    ax.set(xlabel='Static single-frame MM/GBSA (kcal/mol; entropy omitted)', ylabel='Best expanded-context Chai ipTM',
           ylim=(0, 1), title='The lowest energy does not identify the best epitope-preserving binder')
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    save(fig, 'energy-versus-confidence')
    # Use the same target-derived projection for each round's leading CA trace.
    import gemmi
    first = gemmi.read_structure(rounds[0]['top_binders'][0]['representative_model'])[0]
    ca = lambda chain: np.array([r['CA'][0].pos.tolist() for r in chain if gemmi.find_tabulated_residue(r.name).is_amino_acid()])
    reference = ca(first['A']); origin = reference.mean(0)
    _, _, axes3 = np.linalg.svd(reference - origin, full_matrices=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.3))
    for ax, r in zip(axes.flat, rounds):
        lead = r['top_binders'][0];model = gemmi.read_structure(lead['representative_model'])[0];target, binder = ca(model['A']), ca(model['B'])
        u, _, v = np.linalg.svd((target - target.mean(0)).T @ (reference - origin))
        rot = u @ np.diag([1, 1, np.linalg.det(u @ v)]) @ v
        t = ((target - target.mean(0)) @ rot) @ axes3.T
        b = ((binder - target.mean(0)) @ rot) @ axes3.T
        ax.plot(t[:, 0], t[:, 1], color='#b3c2ca', lw=.9)
        ax.plot(b[:, 0], b[:, 1], color=TEAL, lw=1.7)
        ax.set_title(f'{r["nodes"]} {"node" if r["nodes"]==1 else "nodes"} · R{r["round"]}\n{lead["binder_id"][7:]} · ipTM {lead["best_iptm"]:.3f}', fontsize=9)
        ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle('Highest-confidence crop pose in each iteration · aligned CA-trace projections', fontsize=13)
    save(fig, 'leading-binder-poses')


def inline_svg(path):
    text = Path(path).read_text()
    return text[text.index('<svg'):]


def table(headers, rows, klass=''):
    return '<div class="table-scroll"><table class="' + klass + '"><thead><tr>' + ''.join('<th>' + escape(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def build_html(data, out):
    phases, rounds = data['phases'], data['rounds']
    comparison = table(['Total nodes', 'Science / inference', 'Allocation minutes', 'Verified crop files', 'Distinct sequences', 'Crop files / allocation hour', 'Crop files / round-window hour', 'Science GPU-hours', 'Local inference GPU-hours'],
                       [[p['nodes'], f'{p["science_nodes"]} / {p["local_inference_nodes"]}', num(p['allocation_seconds']/60), p['verified_crop_files'], p['distinct_sequences'], num(p['models_per_allocation_hour']), num(p['models_per_round_window_hour']), num(p['science_gpu_hours']), num(p['local_inference_gpu_hours']) if p['local_inference_gpu_hours'] is not None else 'external / unmetered'] for p in phases])
    round_table = table(['Iteration', 'Selected / reviewed', 'Crop files', 'Sequences / newly seen', 'Wall minutes incl. restarts', 'Crop files / round hour', 'Best / median-best ipTM', 'Expanded follow-up rows'],
                        [[f'{r["nodes"]} {"node" if r["nodes"]==1 else "nodes"} · R{r["round"]}', f'{r["selected_hypotheses"]} / {r["reviewed_threads"]}', r['verified_crop_files'], f'{r["distinct_sequences"]} / {r["new_sequences"]}', num(r['round_wall_seconds']/60), num(r['crop_files_per_round_hour']), f'{r["maximum_iptm"]:.3f} / {r["median_sequence_best_iptm"]:.3f}', r['followup_rows']] for r in rounds])
    sections = []
    for r in rounds:
        hypothesis_rows = [[h['selection_rank'], h['hypothesis_id'][:8], h['short_proposition'], num(h['elo_at_selection'],1),
                            (str(h['same_elo_selected_threads'])+' selected tied') if h['same_elo_selected_threads']>1 else '—',
                            h['stored_outcome'], h['accepted_claims_as_recorded'], 'yes' if h['reviewer_overreach_flag'] else 'no', h['independent_claim_findings']] for h in r['all_hypotheses']]
        tops = table(['Rank', 'Hypothesis', 'Scientific proposition (editorial shorthand)', 'Selection Elo', 'Elo tie', 'Stored outcome', 'Accepted claims', 'Reviewer overreach', 'Independent flags'], hypothesis_rows[:3])
        all_h = '<details><summary>All selected hypotheses and full stored propositions</summary>' + table(['Rank', 'ID', 'Stored summary', 'Selection Elo', 'Stored outcome', 'Review completed', 'Recorded error'], [[h['selection_rank'],h['hypothesis_id'],h['summary'],h['elo_at_selection'],h['stored_outcome'],h['review_completed'],h['recorded_error'] or '—'] for h in r['all_hypotheses']]) + '</details>'
        binders = table(['Rank', 'Binder ID', 'Best / median ipTM', 'Both-region models', 'Best-model <2Å atom pairs', 'Cys', 'Arm', 'Owning hypothesis'], [[i,c['binder_id'],f'{c["best_iptm"]:.4f} / {c["median_model_iptm"]:.4f}',f'{c["models_with_dual_contacts"]}/{c["verified_models"]}',c['best_model_pairs_below_2A'],c['cysteines'],c['condition'],c['representative_hypothesis'][:8]] for i,c in enumerate(r['top_binders'],1)])
        sequences = ''.join('<details><summary>'+escape(c['binder_id'])+' · full sequence and representative coordinate source</summary><pre>'+escape(c['sequence'])+'</pre><p class="small">'+escape(c['representative_model'])+'</p><p class="small">SHA256 '+escape(c['representative_model_sha256'])+'</p></details>' for c in r['top_binders'])
        extra = ''
        if r['lead_followup']:
            c=r['lead_followup'];p=c['minimized_pose_geometry']
            extra=f'<div class="callout"><b>Expanded-context structural lead:</b> {escape(binder_id(c["binder_sequence"]))} ({escape(c["condition"])}), selected by best expanded-context ipTM among follow-ups retaining both regions after minimization with zero &lt;2Å pairs. ipTM {c["intact_iptm_best"]:.3f}; static MM/GBSA {c["static_MMGBSA_kcal_mol"]:.2f} kcal/mol; crop-aligned binder RMSD {p["target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A"]:.2f} Å. This is a computational selection rule, not an affinity ranking.</div>'
        sections.append(f'<article class="iteration" id="{r["key"]}" data-phase="{r["nodes"]}"><div class="eyebrow">{r["nodes"]} total {"node" if r["nodes"]==1 else "nodes"} · {escape(r["campaign"])} campaign</div><h3>Round {r["round"]}: {r["verified_crop_files"]} crop models, {r["distinct_sequences"]} sequences</h3><p>{escape(r["note"])}</p><h4>Leading tournament selections</h4>{tops}{all_h}<h4>Highest-confidence distinct binders with verified predictions in this iteration</h4>{binders}{sequences}{extra}</article>')
    findings = table(['Campaign / round','Hypothesis','Issue','Independent correction'],[['20 nodes / R'+str(f['round']),f['hypothesis'][:8],f['severity'],f['correction']] for f in data['independent_claim_findings']])
    t=data['telemetry_summary'];b=t['role_observed_node_minutes'];queue=t['inference_observed_requests']['vllm:num_requests_waiting']
    old_lease=data['baseline']['five_node']['mean_slot_occupancy_fraction'];new_lease=t['science_slot_leases']['mean_leased_slot_fraction']
    followups=sorted(data['followups'],key=lambda c:c['intact_iptm_best'],reverse=True)
    energy_table=table(['Round','Binder','Arm','Expanded ipTM','Static MM/GBSA kcal/mol','Minimized both-region contacts','Binder RMSD Å','<2Å pairs'],[[c['round'],binder_id(c['binder_sequence']),c['condition'],num(c['intact_iptm_best'],3),num(c['static_MMGBSA_kcal_mol']),c['minimized_pose_geometry']['dual_region_contact'],num(c['minimized_pose_geometry']['target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A']),c['minimized_pose_geometry']['close_atom_pairs_lt2A']] for c in followups])
    svg=lambda name:inline_svg(out/'figures'/f'{name}.svg')
    old_svg=lambda name:inline_svg(ROOT/'reports/five-node-20261113'/f'{name}.svg')
    new_svg=lambda name:inline_svg(ROOT/'reports/scale20-20261113'/f'{name}.svg')
    report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Loamlite on Jupiter · From 1 to 20 nodes</title><style>
:root{{--ink:{INK};--teal:{TEAL};--gold:{GOLD};--gray:{GRAY};--line:#dce5ea}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#f1f5f7;color:var(--ink);font:16px/1.65 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:28px}}header{{background:var(--ink);color:white;border-radius:16px;padding:38px}}h1{{font-size:clamp(2rem,4vw,3.3rem);line-height:1.08;margin:12px 0}}h2{{font-size:1.7rem;margin-top:0}}h3{{font-size:1.35rem}}h4{{margin-bottom:10px}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:.78rem;font-weight:700}}.lead{{font-size:1.15rem;max-width:900px}}nav{{display:flex;flex-wrap:wrap;gap:16px;padding:20px 0}}a{{color:var(--teal)}}header a{{color:#9ce0d6}}section,.iteration{{border:1px solid var(--line);border-radius:12px;background:white;padding:26px;margin:22px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px;margin:22px 0}}.card{{padding:19px;border:1px solid var(--line);border-radius:10px;background:white}}.card b{{display:block;font-size:2rem}}.callout{{background:#eaf5f2;border-left:4px solid var(--teal);padding:15px;margin:18px 0}}.warning{{background:#fff5e7;border-left-color:var(--gold)}}.small{{font-size:.84rem;color:var(--gray)}}.table-scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.84rem}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#edf3f5;position:sticky;top:0}}td{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;padding:13px;background:#f4f7f9;font-size:.85rem}}details{{padding:9px 0}}summary{{cursor:pointer;color:var(--teal)}}svg{{width:100%;height:auto;max-height:1100px}}.figure{{margin:18px 0}}.filter{{display:flex;gap:10px;flex-wrap:wrap}}button,select,input{{font:inherit;padding:8px 13px;border:1px solid #aebfc9;border-radius:6px;background:white}}button{{cursor:pointer}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}@media(max-width:800px){{main{{padding:15px}}header,section,.iteration{{padding:20px}}.two{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}nav,.filter{{display:none}}section,.iteration{{break-inside:avoid}}header{{color:var(--ink);background:white}}}}
</style></head><body><main><header><div class="eyebrow">Jupiter / Jülich · verified campaign evidence · 13 November 2026</div><h1>Scaling an agentic binder campaign<br>from 1 to 20 nodes</h1><p class="lead">Three resource configurations, eight completed scientific iterations, and a reproducible trail from tournament hypotheses to generated sequences, predicted complexes, and expanded-receptor follow-ups.</p><p><a href="scaleup.pptx">PowerPoint deck</a> · <a href="scaleup.pdf">PDF deck</a> · <a href="slides.html">Present in browser</a> · <a href="data.json">Audited story data</a></p></header>
<nav><a href="#throughput">Throughput</a><a href="#iterations">Every iteration</a><a href="#science">Scientific results</a><a href="#hpc">Node saturation</a><a href="#methods">Methods / sources</a></nav>
<div class="cards"><div class="card"><b>1 → 5 → 20</b>total allocated nodes in main runs</div><div class="card"><b>3.25×</b>20-node / 5-node gross model-file throughput</div><div class="card"><b>402</b>distinct sequences across both campaigns</div><div class="card"><b>2,390</b>verified crop coordinate files</div><div class="card"><b>74</b>expanded-context energy follow-up rows</div></div>
<section><h2>What scaled—and what the results establish</h2><p>The five-node job produced 215 verified crop complex files in 62 minutes. The continuation produced 2,150 in 190.92 science-allocation minutes, reaching <b>675.69 files/hour</b> with <b>19 science nodes plus one local Nemotron node</b>. It completed three tournament/science/review rounds with fresh reviewed evidence in each; 33 of 36 selected hypothesis-round threads completed a report and review. Nineteen hypothesis IDs had completed crop-screen outputs.</p><p>Higher throughput did not establish a general dual-region advantage. In the larger controlled screen, one-region designs had slightly higher median sequence-best confidence than dual-region designs. Only 2/33 complete matched screen batches passed the prespecified dispersion gate. Several promising individual poses survived expanded-context re-folding; the most favorable static energy belonged to a low-confidence off-epitope pose.</p><div class="callout warning">ipTM is structural confidence. Contact persistence, pose RMSD, and static single-frame MM/GBSA are computational readouts. No experimental Kd, affinity, selectivity, or biological activity was measured.</div></section>
<section id="throughput"><h2>Throughput at each node count</h2>{svg('throughput-by-nodes')}{comparison}<h3>Campaign-owned allocation timeline</h3>{svg('allocation-timeline')}<p>Two original five-node bootstrap allocations briefly overlapped, creating a transient ten-node allocation; this was not a completed ten-node scaling benchmark. Two-node preflights and the nineteen-node inference-renewal gap are also shown. These use Slurm accounting start/end times; epilog cleanup can retain nodes after the accounting endpoint. Foreign allocations are excluded. No throughput is invented for transient node counts.</p><p>The main denominator is the full science allocation, including debugging, idle time, repairs, audits, and shutdown preparation. The alternative round window runs from the first scientific tournament start to the final scientific-round checkpoint. It includes intervening gaps and restarts. The one-node development allocation lasted 113.42 minutes, but rounds4–5 occupied only {phases[0]['round_window_seconds']/60:.2f} minutes; its 13.23 gross versus {phases[0]['models_per_round_window_hour']:.2f} round-window files/hour therefore cannot serve as a clean strong-scaling baseline.</p><p>There was also a <b>two-owned-node recipe pilot</b>: one science node alongside the existing one-node Nemotron allocation. It verified four sequences / twenty crop files plus five expanded-context files in an 18.45-minute science allocation (65.04 crop files/hour). It did not run a tournament and is excluded from the three main output totals. A separate two-node integration preflight ended before validator review; bootstrap failures and this preflight remain in the earlier detailed cost inventory.</p><div class="callout">The 3.25× increase from five to twenty total nodes is an operational result. Hypothesis fan-out, software bundling, inference service, controls, candidate budgets, and expanded-context work changed. This was not a matched strong-scaling experiment. External inference was unmetered in the earlier 1/5-node jobs; the continuation used 27.12 allocated GPU-hours for owned local inference.</div></section>
<section><h2>Output by iteration</h2>{svg('output-by-iteration')}{round_table}<p class="small">Original rounds1–3 were preparation/debugging and produced no accepted scientific campaign inventory. The continuation restarted its own counter at round1. New sequences are deduplicated chronologically across both campaigns; four sequences were shared between the original one- and five-node phases. Round-local output is assigned by the owning batch/run, while wall time includes whole-round orchestration. Late follow-ups are charged to their submission round even if a review had already returned.</p></section>
<section id="iterations"><h2>Top hypotheses and binders in every scientific iteration</h2><p>Hypotheses below retain the actual stored tournament selection order and selection-time Elo; equal-Elo selections are tied. These are pre-execution selections, not post hoc scientific winners. Outcome labels and accepted-claim counts are retained as recorded, with reviewer overreach and independent findings visible. Binder ranks use the highest verified crop ipTM per distinct sequence within that iteration; they do not select by affinity.</p><div class="filter"><label>Show iterations <select id="phase"><option value="all">All node counts</option><option value="1">1 node</option><option value="5">5 nodes</option><option value="20">20 total nodes</option></select></label></div></section>{''.join(sections)}
<section id="science"><h2>Scientific progression</h2>{svg('confidence-by-iteration')}{svg('leading-binder-poses')}<p class="small">CA traces are target-aligned two-dimensional projections of each round's single highest-ipTM crop model: receptor gray, binder teal. They are orientation schematics, not all-atom ribbon renderings or clash validation. The coordinate source and checksum for every pose are in the per-round binder details.</p>{svg('condition-comparison')}<p>The twenty-node screen contains 120 dual-region, 119 one-region, and 119 ridge distinct sequences. Their median sequence-best crop ipTM values are 0.750, 0.767, and 0.222. Contacting both regions in at least one predicted model was common even in ridge controls (93/119), so contact presence alone does not establish specificity or reproducibility. These are correlated adaptive cohorts; repeated sequences and model files are not independent experiments.</p><h3>Expanded-receptor structural follow-up</h3><p>The 205-residue screen crop cuts two native EGFR disulfides. Energy follow-ups therefore use canonical EGFR25–646: 622 residues with 25 verified native disulfides, expanded extracellular context including one membrane-boundary residue. This is not the full membrane receptor. Membrane, intracellular domains, glycans, and entropy are omitted. All 74 follow-up rows passed the earlier independent topology/coordinate audit; they represent 66 distinct binder sequences, with 36/74 minimized rows retaining both epitope regions.</p>{svg('energy-versus-confidence')}<p>Static MM/GBSA reached −133.74 kcal/mol for binder_a91ef18fe15e, but its expanded-context ipTM was 0.196, it lost both-region contacts, and the minimized binder shifted 59.44 Å relative to its screening pose. It is not an energy-selected binding hit. The expanded-context table is sorted by confidence and reports pose checks alongside every energy.</p><details><summary>All 74 expanded-context follow-up results</summary>{energy_table}</details></section>
<section id="hpc"><h2>Where the nodes were busy</h2><p>The five-node job averaged <b>{old_lease:.1%}</b> leased science GPU-slot occupancy, peaking at18/20 slots. The twenty-node continuation averaged <b>{new_lease:.1%}</b>, peaking at74/76. A lease includes imports, CPU stages, and filesystem work inside a GPU-assigned row. It is not measured GPU compute utilization.</p><h3>Five-node reconstructed row leases</h3>{old_svg('node-slot-saturation')}<h3>Nineteen science nodes: row leases</h3>{new_svg('science-slot-leases')}<h3>Measured GPU hardware activity</h3>{new_svg('node-gpu-utilization')}<p>Across observed node-minute bins, science GPU busy time averaged <b>{b['science']['mean_GPU_busy_percent']:.2f}%</b> versus <b>{b['inference']['mean_GPU_busy_percent']:.2f}%</b> for Nemotron. GPU busy time is the sampled NVIDIA utilization.gpu counter, not SM occupancy or FLOPS efficiency. The vLLM waiting queue had p95 {queue['p95']:.0f} and maximum {queue['max']:.0f} requests; it was nonzero in only {100*queue['positive_sample_fraction']:.2f}% of observations. In-flight decoding and orchestration can remain expensive even when the waiting queue is empty. Gray intervals are missing observations, never filled with zero.</p>{new_svg('inference-queue-and-gpu-power')}<div class="callout">The 1-inference/19-science split was a measured starting point, not an optimum. The five-node timings and initial local protocol probes supported trying it, but the full run exposed substantial idle science capacity. A controlled next experiment would compare two Nemotron replicas plus eighteen science nodes with fixed workloads and routing between replicas. That alternative was not run and is not included as an achieved node configuration.</div><p>Compute nodes had no internet access. Scientific tools ran locally on Jupiter; the twenty-node campaign used owned local Nemotron throughout. Argo was gated and unused there; any Argo route was fixed to gpt-5.6-luna. The 19-node science job queued for187.27 minutes. Inference renewal after round2 included a gap between allocations and about14 minutes of model shard loading; the third-round timing and total costs retain that maintenance. CPU-only metadata tasks used zero GPUs; dependent side effects were serialized within each agent turn while hypotheses and GPU rows remained concurrent.</p></section>
<section><h2>Failures, corrections, and limits</h2><p>The original contact code briefly mixed author and native residue IDs; two exact round7 claims were retracted, with backups preserved. Corrected native IDs and independently measured contacts are used here. The twenty-node campaign retained two failed output-schema threads and one thread without a claim review. Immutable failed batches, metadata-only incompatible tool registrations, repeated work, and controller recovery costs remain in the earlier complete inventories.</p>{findings}<p>Generated narrative summaries are not used as quantitative truth: one asserted that every screen gate failed, while independent complete-batch aggregation found two passes. Source tables, coordinate checks, and the audited recipe outputs take precedence. Tournament rankings influenced later selections before independent corrections; that history was preserved rather than retroactively rerun.</p></section>
<section id="methods"><h2>Definitions, reproducibility, and downloads</h2><p><b>Hypothesis count:</b> distinct hypothesis IDs with the specified evidence, not distinct biological mechanisms. <b>Binder count:</b> distinct amino-acid sequences. <b>Model-file count:</b> unique audited coordinate-file SHA256 values; not independent sequences or biological replicates. <b>Best ipTM:</b> maximum among that sequence's verified crop predictions in the round. <b>Pose RMSD:</b> target-aligned binder CA displacement, including possible deformation. <b>Dual contacts:</b> at least one target residue from each mapped region within4Å of a binder heavy atom. <b>Clashes:</b> raw heavy-atom pairs below2Å; zero such pairs alone does not validate chemistry.</p><p>All eight scientific-round records and both independent model inventories were read without launching jobs or inference. Every top-three representative coordinate checksum was rechecked. The existing full audit independently checked all sequences, finite coordinates, score files, contact mapping, crop pose dispersion, Amber bond topology, and minimized follow-up pose geometry. See <a href="data.json">story data and source checksums</a>, <a href="iteration-metrics.csv">iteration metrics</a>, <a href="hypotheses.csv">all selected hypotheses</a>, <a href="binder-metrics-by-iteration.csv">round-local binder metrics / sequences / source paths</a>, <a href="leading-binders.fasta">leader sequences</a>, <a href="expanded-followups.csv">expanded follow-up metrics</a>, and <a href="slide-transcript.txt">slide transcript and source notes</a>.</p><p>Detailed prior reports: <a href="../five-node-20261113/index.html">one/five-node audit</a> · <a href="../scale20-20261113/index.html">twenty-node audit and full accounting</a>. All campaign-owned jobs were released after verification; foreign jobs and user-started parent endpoints were preserved.</p><p class="small">Built {escape(data['generated_at'])}. All figures are embedded; the HTML report and browser deck work offline. No new design, minimization, or inference was performed to prepare this report.</p></section></main><script>document.getElementById('phase').addEventListener('change',e=>{{for(const a of document.querySelectorAll('.iteration'))a.hidden=e.target.value!=='all'&&a.dataset.phase!==e.target.value;}});</script></body></html>'''
    for old,new in [('rounds4','rounds 4'),('rounds1','rounds 1'),('round7','round 7'),('round2','round 2'),('peaking at18','peaking at 18'),('peaking at74','peaking at 74'),('for187','for 187'),('about14','about 14'),('within4Å','within 4 Å'),('below2Å','below 2 Å')]:
        report=report.replace(old,new)
    (out/'index.html').write_text(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    data=collect()
    write_csv(out/'iteration-metrics.csv',[{k:v for k,v in r.items() if k not in {'top_hypotheses','all_hypotheses','top_binders','all_binders','lead_followup'}} for r in data['rounds']])
    write_csv(out/'hypotheses.csv',data['hypotheses'])
    write_csv(out/'binder-metrics-by-iteration.csv',data['binder_rows'])
    write_csv(out/'node-count-metrics.csv',data['phases'])
    write_csv(out/'expanded-followups.csv',[{'round':c['round'],'binder_id':binder_id(c['binder_sequence']),'sequence':c['binder_sequence'],'condition':c['condition'],'best_expanded_iptm':c['intact_iptm_best'],'static_MMGBSA_kcal_mol':c['static_MMGBSA_kcal_mol'],'minimized_dual_contacts':c['minimized_pose_geometry']['dual_region_contact'],'minimized_binder_RMSD_A':c['minimized_pose_geometry']['target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A'],'minimized_close_pairs_below2A':c['minimized_pose_geometry']['close_atom_pairs_lt2A'],'source_summary':c['summary_path']} for c in data['followups']])
    leaders={c['binder_id']:c for r in data['rounds'] for c in r['top_binders']}
    (out/'leading-binders.fasta').write_text(''.join(f'>{key}\n{c["sequence"]}\n' for key,c in leaders.items()))
    (out/'data.json').write_text(json.dumps(data,indent=2)+'\n')
    plots(data,out/'figures')
    from scaleup_slides import build_deck
    slides=build_deck(data,out)
    build_html(data,out)
    manifest={'generated_at':data['generated_at'],'builder_sha256':sha(__file__),'slides_builder_sha256':sha(Path(__file__).with_name('scaleup_slides.py')),'source_sha256':data['source_sha256'],'outputs':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in {'build-manifest.json','verification.json'} and not p.name.startswith('pdf-review-')},'slide_count':slides,'new_scientific_execution':False}
    manifest['runtime_versions']={'python':sys.version,**{name:importlib.metadata.version(name) for name in ['python-pptx','reportlab','matplotlib','numpy','Pillow','gemmi']}}
    (out/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'report':str(out/'index.html'),'powerpoint':str(out/'scaleup.pptx'),'pdf':str(out/'scaleup.pdf'),'browser_deck':str(out/'slides.html'),'slides':slides,'rounds':len(data['rounds']),'selected_models_rehashed':len(data['selected_coordinate_sha256'])},indent=2))


if __name__=='__main__':
    main()
