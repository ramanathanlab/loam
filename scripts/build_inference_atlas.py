#!/usr/bin/env python3
"""Build an offline scientific/HPC atlas from completed, independently audited runs.

Read-only with respect to campaigns and Slurm. No inference, simulations or jobs.
Requires numpy/matplotlib (the Jupiter RFD4-Proteina Python provides both).
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import hashlib
import html
import io
import json
import math
import shutil
import statistics
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

COLORS = ['#718096', '#4c78a8', '#d58b37', '#168b80', '#8658a6']
ARMS = {'dual': '#177f86', 'one_region': '#d78a39', 'ridge': '#9270aa'}
GPU_COLORS = ['#3986ac', '#da8b3d', '#168b80', '#a46aaf']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, places=1):
    return 'unavailable' if value is None or not math.isfinite(value) else f'{value:,.{places}f}'


def finite_mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else float('nan')


def quantiles(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {'n': 0, 'median': None, 'p05': None, 'p25': None, 'p75': None, 'p95': None}
    return dict(zip(['n', 'median', 'p05', 'p25', 'p75', 'p95'],
                    [len(values), *map(float, np.percentile(values, [50, 5, 25, 75, 95]))]))


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(json_safe(v), separators=(',', ':')) if isinstance(v, (dict, list)) else
                                  '' if v is None or isinstance(v, float) and not math.isfinite(v) else
                                  "'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v)
                             for k, v in row.items()})


def deduplicate(audit):
    """Same earliest-successful input/seed rule as the completed campaign report."""
    starts = {(r['batch_id'], r['row']): r.get('started_at') or float('inf') for r in audit['row_timings']}
    order = lambda c: (starts.get((c['batch_id'], Path(c['summary_path']).parent.name), float('inf')), c['summary_path'])
    screens, full = {}, {}
    for c in sorted(audit['candidate_rows'], key=order):
        key = (c['condition'], c['input_sha256'], c['seed'], c['target_sequence'])
        screens.setdefault(key, c)
    for c in sorted(audit['intact_followups'], key=order):
        row = read(Path(c['summary_path']).parent / 'input.json')
        key = (c['binder_sequence'], row['seed'], c['target_residues'])
        full.setdefault(key, {**c, 'seed': row['seed']})
    return list(screens.values()), list(full.values())


def load_runs(comparison):
    source = read(comparison)
    runs = []
    for i, trial in enumerate(source['trials']):
        assert trial['completion_checks_passed'] and trial['independent_coordinates_checked']
        root = Path(trial['trial'])
        audit = read(root / 'scaling-audit.json')
        assert audit['required_campaign_completion_checks_passed'] and not audit['errors']
        screens, full = deduplicate(audit)
        assert len(screens) == trial['unique_screen_input_trials']
        assert len(full) == trial['unique_complete_intact_input_trials']
        assert 5 * (len(screens) + len(full)) == trial['verified_files_on_distinct_input_trials']
        progress = read(root / 'scale-progress.json')
        job = audit['jobs'][progress['scientific_job']]
        start, end = job['start_epoch'], job['end_epoch']
        assert abs(end - start - trial['science_allocation_seconds']) < .01
        rows = {(r['batch_id'], r['row']): r for r in audit['row_timings']}
        events = []
        for kind, values in [('crop', screens), ('expanded', full)]:
            for c in values:
                row = rows[(c['batch_id'], Path(c['summary_path']).parent.name)]
                completed = row['started_at'] + row['seconds']
                assert start - 2 <= completed <= end + 2
                events.append({'time': min(end, max(start, completed)), 'kind': kind, 'files': 5,
                               'round': c['round'], 'source': c['summary_path']})
        events.sort(key=lambda e: e['time'])
        bins = []
        for lo in np.arange(start, end, 300):
            hi = min(float(lo) + 300, end)
            rr = [e for e in events if lo <= e['time'] < hi or hi == end and e['time'] == end]
            count = sum(e['files'] for e in rr)
            bins.append({'start_minute': (lo-start)/60, 'seconds': hi-lo, 'files': count,
                         'files_per_hour': count*3600/(hi-lo), 'full_five_minutes': hi-lo >= 299.999})
        assert sum(b['files'] for b in bins) == trial['verified_files_on_distinct_input_trials']
        paired = []
        verified_crop = {v['sha256']: v for v in audit['verified_models'] if v['kind']=='crop'}
        for c in full:
            p = Path(c['source_screen_summary'])
            assert digest(p) == c['source_screen_sha256']
            s = read(p)
            assert s['binder_sequence'] == c['binder_sequence']
            # Historical compact bridge sources can point outside the accepted
            # crop inventory. Preserve the independently audited expanded result,
            # but do not invent an audited crop comparison for those sources.
            source_audited = all(m['sha256'] in verified_crop for m in s['models'])
            if source_audited:
                models = [verified_crop[m['sha256']] for m in s['models']]
                best = max(m['scores']['iptm'] for m in models)
                contacts = sum(bool(m['dual_region_contact']) for m in models)
                assert 'best_iptm' not in s or abs(s['best_iptm']-best)<1e-6
                assert 'dual_contact_models' not in s or s['dual_contact_models']==contacts
            else:
                best, contacts = None, None
            s = {**s, 'source_summary_compact': 'best_iptm' not in s, 'best_iptm': best,
                 'dual_contact_models': contacts, 'source_crop_audited': source_audited}
            paired.append((s, c))
        round_records = {int(p.stem.split('-')[1]): read(p) for p in (root/'state/rounds').glob('round-*.json')}
        label = f"{trial['split']['inference']}:{trial['split']['science']}"
        label += ' repeat' if 'confirm' in root.name else ' initial' if trial['split']['inference'] == 8 else ''
        runs.append({'id': f'run-{i}', 'root': root, 'trial': trial, 'audit': audit, 'progress': progress,
                     'screens': screens, 'full': full, 'pairs': paired, 'start': start, 'end': end,
                     'events': events, 'bins': bins, 'round_records': round_records, 'label': label,
                     'color': COLORS[i], 'inference': trial['split']['inference'], 'science': trial['split']['science']})
    return source, runs


def load_telemetry(run):
    audit, root, progress = run['audit'], run['root'], run['progress']
    science_job = progress['scientific_job']
    inference_jobs = {j['JobIDRaw'] for j in audit['inference_jobs']}
    jobs = {science_job, *inference_jobs}
    directories = {root/'telemetry'}
    if progress.get('inference_pool_experiment'):
        directories.add(Path(progress['inference_pool_experiment'])/'telemetry')
    hosts = sorted(set().union(*(set(audit['jobs'][j]['hostnames']) for j in inference_jobs)))
    science_hosts = sorted(audit['jobs'][science_job]['hostnames'])
    assert not set(hosts) & set(science_hosts)
    hosts += science_hosts
    roles = ['inference'] * (len(hosts)-len(science_hosts)) + ['science'] * len(science_hosts)
    n = int(math.ceil((run['end']-run['start'])/60))
    grouped = collections.defaultdict(list)
    metric_samples = collections.defaultdict(list)
    files = []
    for folder in directories:
        for p in sorted(folder.glob('*.jsonl')):
            if not any(p.name.startswith(j+'-') for j in jobs):
                continue
            files.append(p)
            for line in p.read_text().splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                job, t = d.get('job_id'), d.get('timestamp', 0)
                if job not in jobs or not run['start'] <= t <= run['end']:
                    continue
                allocation = audit['jobs'][job]
                if not allocation['start_epoch'] <= t <= allocation['end_epoch']:
                    continue
                host = d['host'].split('.')[0]
                assert host in hosts
                minute = min(n-1, int((t-run['start'])//60))
                records = [row for row in csv.reader(io.StringIO(d.get('gpu_csv', ''))) if len(row) >= 8]
                if records:
                    gpu = {int(row[0]): float(row[2]) for row in records}
                    assert set(gpu) == {0, 1, 2, 3}
                    values = {'gpu': gpu, 'cpu': d.get('cpu_busy_percent'), 'iowait': d.get('cpu_iowait_percent'),
                              'host_memory': d.get('host_memory_used_percent'),
                              'gpu_memory': finite_mean([100*float(row[4])/float(row[5]) for row in records if float(row[5]) > 0]),
                              'power': sum(float(row[6]) for row in records)}
                    grouped[(host, minute)].append(values)
                if d.get('vllm_metrics'):
                    values = collections.defaultdict(float)
                    for line in d['vllm_metrics']:
                        try:
                            name = line.split('{', 1)[0].split()[0]
                            value = float(line.rsplit(' ', 1)[-1])
                        except (IndexError, ValueError):
                            continue
                        if math.isfinite(value):
                            values[name] += value
                    metric_samples[(job, minute)].append(dict(values))
    arrays = {k: np.full((len(hosts), n), np.nan) for k in ['busy', 'cpu', 'iowait', 'host_memory', 'gpu_memory', 'power']}
    per_gpu = np.full((len(hosts), 4, n), np.nan)
    for (host, minute), samples in grouped.items():
        i = hosts.index(host)
        for gpu in range(4):
            per_gpu[i, gpu, minute] = finite_mean([s['gpu'][gpu] for s in samples])
        arrays['busy'][i, minute] = finite_mean(per_gpu[i, :, minute])
        for key in arrays:
            if key != 'busy':
                arrays[key][i, minute] = finite_mean([s[key] for s in samples])
    # Missing observations remain NaN, including the historical inference renewal gap.
    assert np.isfinite(arrays['busy']).any()
    expected = run['trial']['overall_observed_GPU_busy_percent']
    assert abs(float(np.nanmean(arrays['busy']))-expected) < .05, (run['label'], expected, np.nanmean(arrays['busy']))
    queue = {}
    for key in ['vllm:num_requests_running', 'vllm:num_requests_waiting']:
        values = []
        for minute in range(n):
            replica_means = []
            for job in inference_jobs:
                allocation = audit['jobs'][job]
                lo, hi = run['start']+minute*60, min(run['end'], run['start']+(minute+1)*60)
                if allocation['start_epoch'] < hi and allocation['end_epoch'] > lo:
                    replica_means.append(finite_mean([v.get(key) for v in metric_samples[(job, minute)]]))
            values.append(sum(replica_means) if replica_means and all(math.isfinite(v) for v in replica_means) else float('nan'))
        queue[key] = np.array(values)
    run.update(hosts=hosts, roles=roles, arrays=arrays, per_gpu=per_gpu, queue=queue,
               telemetry_files=files, minutes=(np.arange(n)+.5))


def load_requests(source, runs):
    experiment_root = Path(runs[1]['progress']['inference_pool_experiment'])
    pool = experiment_root/'pool-request-timings.jsonl'
    rows = [json.loads(line) for line in pool.read_text().splitlines()]
    for run in runs:
        if run['inference'] == 1:
            legacy = run['root']/'state/nemotron/request-timings.jsonl'
            run['requests'] = [dict(seconds=r['seconds'], thinking=None, status=r['status'], backend='legacy',
                                    start=r['start'], end=r.get('end', r['start']+r['seconds']))
                               for line in legacy.read_text().splitlines() if (r := json.loads(line))
                               and run['start'] <= r['start'] <= run['end']]
        else:
            phase = [r for r in rows if r.get('phase') == run['root'].name]
            final = {r['request_id']: r for r in sorted(phase, key=lambda r: r['finished_at'])}
            run['requests'] = [dict(seconds=r['total_request_seconds'], thinking=r['thinking_enabled'],
                                    status=r['status'], backend=r.get('backend'), start=r['started_at'],
                                    end=r['finished_at'], prompt_tokens=r.get('prompt_tokens'),
                                    completion_tokens=r.get('completion_tokens'), queue_seconds=r.get('queue_seconds'))
                               for r in final.values()]


class Atlas:
    def __init__(self, output, runs):
        self.output, self.runs = output, runs
        (output/'assets').mkdir(parents=True, exist_ok=True)
        (output/'data').mkdir(exist_ok=True)
        self.figures = {}
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                             'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
                             'axes.labelcolor': '#29404c', 'text.color': '#193542', 'axes.edgecolor': '#bac8cd',
                             'grid.color': '#dbe5e8', 'grid.alpha': .65, 'savefig.facecolor': 'white'})

    def save(self, fig, name, title, caption, section):
        fig.savefig(self.output/'assets'/f'{name}.svg', bbox_inches='tight')
        fig.savefig(self.output/'assets'/f'{name}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        self.figures[name] = {'name': name, 'title': title, 'caption': caption, 'section': section}

    def ecdf(self, ax, values, **kw):
        x = np.sort([v for v in values if v is not None and math.isfinite(v)])
        if len(x):
            ax.step(x, np.arange(1, len(x)+1)/len(x), where='post', **kw)

    def box(self, ax, groups, positions=None, colors=None, width=.55, points=False, showfliers=None):
        positions = list(range(len(groups))) if positions is None else positions
        colors = COLORS if colors is None else colors
        valid = [(i, np.array([v for v in values if v is not None and math.isfinite(v)])) for i, values in enumerate(groups)]
        for i, values in valid:
            if not len(values):
                continue
            bp = ax.boxplot([values], positions=[positions[i]], widths=width, patch_artist=True,
                            showfliers=not points if showfliers is None else showfliers, manage_ticks=False,
                            medianprops={'color': '#102d38', 'linewidth': 1.7},
                            flierprops={'marker': '.', 'markersize': 3, 'alpha': .45})
            bp['boxes'][0].set(facecolor=colors[i % len(colors)], alpha=.3, edgecolor=colors[i % len(colors)])
            if points:
                jitter = np.random.default_rng(710+i).uniform(-width*.38, width*.38, len(values))
                ax.scatter(positions[i]+jitter, values, s=13, color=colors[i % len(colors)], alpha=.45, edgecolors='none')
        ax.grid(axis='y')

    def labels(self, ax, rotation=0):
        ax.set_xticks(range(len(self.runs)), [r['label'] for r in self.runs], rotation=rotation)


def plot_throughput(atlas, source, runs):
    labels = [r['label'] for r in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(10, 4.6), layout='constrained')
    ratios = [1, 2, 4, 8]
    centers = []
    for pos, ratio in enumerate(ratios):
        rr = [r for r in runs if r['inference'] == ratio]
        seconds = sum(r['end']-r['start'] for r in rr)
        center = sum(r['trial']['verified_files_on_distinct_input_trials'] for r in rr)*3600/seconds
        centers.append(center)
        ax.bar(pos, center, width=.6, color=rr[0]['color'], alpha=.26)
        for j, r in enumerate(rr):
            offset = (-.11 if j == 0 else .11) if len(rr) > 1 else 0
            value = r['trial']['distinct_trial_complex_files_per_science_allocation_hour']
            ax.scatter(pos+offset, value, s=75, marker='^' if j else 'o', color=r['color'], zorder=5)
            ax.annotate(f'{value:,.0f}', (pos+offset, value), xytext=(0, 9 if not j else -18),
                        textcoords='offset points', ha='center', fontsize=9)
    ax.set(xticks=range(4), xticklabels=['1:19', '2:18', '4:16', '8:12'],
           ylabel='Verified complex files / allocation hour', xlabel='Inference : science nodes (20 total)',
           title='Observed campaign throughput across node ratios', ylim=(0, max(centers)*1.25))
    ax.grid(axis='y');ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#294a5c', label='Initial campaign'),
                       Line2D([], [], marker='^', ls='', color=COLORS[4], label='Repeat campaign'),
                       plt.Rectangle((0, 0), 1, 1, color='#7ca49e', alpha=.3, label='Allocation-time-weighted rate')], fontsize=8)
    atlas.save(fig, 'ratio-campaign-throughput', 'Campaign throughput by inference:science ratio',
               'Dots are whole campaigns, not individual binders. Bars divide summed distinct-input output by summed science-allocation time. Only 8:12 has a repeat; no replicate confidence intervals are inferred.', 'throughput')

    fig, ax = plt.subplots(figsize=(10, 4.7), layout='constrained')
    groups = [[b['files_per_hour'] for r in runs if r['inference'] == ratio for b in r['bins'] if b['full_five_minutes']] for ratio in ratios]
    atlas.box(ax, groups, colors=[COLORS[i] for i in [0, 1, 2, 3]], width=.6, showfliers=False)
    for pos, ratio in enumerate(ratios):
        for j, r in enumerate([r for r in runs if r['inference'] == ratio]):
            values = [b['files_per_hour'] for b in r['bins'] if b['full_five_minutes']]
            jitter = np.random.default_rng(90+pos+j).uniform(-.18, .18, len(values))
            ax.scatter(pos+jitter, values, s=20, marker='^' if j else 'o', color=r['color'], alpha=.6)
        ax.text(pos, -.12, f'n={len(groups[pos])} blocks', transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
    ax.set(xticks=range(4), xticklabels=['1:19', '2:18', '4:16', '8:12'],
           ylabel='Files/hour within a five-minute completion block', title='Throughput distribution over elapsed campaign time')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#718096', label='Initial-campaign time block'),
                       Line2D([], [], marker='^', ls='', color=COLORS[4], label='8:12 repeat time block')], fontsize=8, loc='upper right')
    atlas.save(fig, 'ratio-block-throughput', 'Within-campaign throughput distributions',
               'Each point is a complete, nonoverlapping five-minute block. Zeros are real periods with no completed output. Partial final blocks are exported but omitted here. These temporal blocks are correlated and are not independent campaign replicates.', 'throughput')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for r in runs:
        t = [0, *[(e['time']-r['start'])/60 for e in r['events']], (r['end']-r['start'])/60]
        y = [0, *np.cumsum([e['files'] for e in r['events']]).tolist(), r['trial']['verified_files_on_distinct_input_trials']]
        axes[0].step(t, y, where='post', label=r['label'], color=r['color'], lw=1.7)
        b = r['bins']
        axes[1].plot([v['start_minute']+v['seconds']/120 for v in b], [v['files_per_hour'] for v in b],
                     color=r['color'], label=r['label'], lw=1.2, alpha=.85)
    axes[0].set(xlabel='Minutes since science allocation started', ylabel='Cumulative verified files', title='Cumulative useful output')
    axes[1].set(xlabel='Minutes since science allocation started', ylabel='Files/hour in completion block', title='Output bursts and idle intervals')
    for ax in axes:ax.grid();ax.legend(fontsize=8)
    atlas.save(fig, 'throughput-timeline', 'Cumulative output and completion-rate timelines',
               'Completion time is reconstructed from the terminal row lease start plus recorded row wall time (rounded by the executor). All allocation bootstrap, repairs, failed attempts, idle time and final row drain remain in the denominator. The historical baseline includes controller repairs and inference renewal.', 'throughput')

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    metrics = [('distinct_sequences_per_science_allocation_hour', 'Distinct binder sequences / hour'),
               ('distinct_intact_trials_per_science_allocation_hour', 'Complete expanded-context trials / hour'),
               ('retained_intact_trials_per_science_allocation_hour', 'Contact-retaining minimized trials / hour'),
               ('high_confidence_sequences_per_science_allocation_hour', 'Crop ipTM ≥0.80 sequence observations / hour')]
    for ax, (key, title) in zip(axes.flat, metrics):
        values = [r['trial'][key] for r in runs]
        ax.bar(x, values, color=[r['color'] for r in runs], alpha=.9)
        for i, value in enumerate(values):ax.text(i, value, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        atlas.labels(ax);ax.set(title=title, ylim=(0, max(values)*1.2));ax.grid(axis='y');ax.set_axisbelow(True)
    atlas.save(fig, 'ratio-scientific-output-types', 'Scientific output rates, separated by endpoint',
               'Expanded-context trials include five cofold models, minimization, topology checks and static MM/GBSA. Contact retention requires both declared target regions in the minimized pose. ipTM 0.80 is a descriptive reporting threshold, not a new workflow gate.', 'throughput')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    for i, r in enumerate(runs):
        t = r['trial'];marker = '^' if i == 4 else 'o'
        axes[0].scatter(t['overall_observed_GPU_busy_percent'], t['distinct_trial_complex_files_per_science_allocation_hour'],
                        s=130, color=r['color'], marker=marker)
        axes[0].annotate(r['label'], (t['overall_observed_GPU_busy_percent'], t['distinct_trial_complex_files_per_science_allocation_hour']), xytext=(5, 6), textcoords='offset points', fontsize=9)
    axes[0].set(xlabel='Mean observed GPU busy (%)', ylabel='Verified files / allocation hour', title='Hardware activity versus scientific throughput')
    values = [r['trial']['trial_window_distinct_trial_files_per_GPU_hour'] for r in runs]
    axes[1].bar(x, values, color=[r['color'] for r in runs]);atlas.labels(axes[1]);axes[1].set(ylabel='Verified files / allocated GPU-hour', title='Efficiency inside each trial window')
    for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True)
    atlas.save(fig, 'throughput-utilization-efficiency', 'Throughput, GPU activity and trial-window efficiency',
               'Trial-window GPU-hours charge scientific GPUs throughout their allocation plus actual overlapping inference allocations. The historical inference-renewal gap is not charged as allocated inference. Pretrial model loading is excluded here; the full experiment ledger charges each shared inference job once, including cold loading and between-trial idle time.', 'throughput')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, kind, title in [(axes[0], 'crop', 'Distinct screen trials completed by round'), (axes[1], 'expanded', 'Distinct expanded-context trials by round')]:
        matrix = np.array([[sum(e['kind']==kind and e['round']==n for e in r['events']) for n in [1, 2, 3]] for r in runs])
        im = ax.imshow(matrix, aspect='auto', cmap='YlGnBu', vmin=0)
        for (i, j), value in np.ndenumerate(matrix):ax.text(j, i, str(value), ha='center', va='center', color='white' if value > matrix.max()*.6 else '#173544')
        ax.set(xticks=[0, 1, 2], xticklabels=['Round 1', 'Round 2', 'Round 3'], yticks=x, yticklabels=labels, title=title)
        fig.colorbar(im, ax=ax, shrink=.75)
    atlas.save(fig, 'round-scientific-output', 'New scientific input trials completed in each round',
               'Duplicate input/seed submissions are assigned only to the earliest successful trial. Later rounds may revisit existing binders or hypotheses; they are not assumed to perform identical fixed work.', 'throughput')


def plot_binders(atlas, runs):
    screen = [r['screens'] for r in runs]
    unmatched = sum(not s['source_crop_audited'] for r in runs for s, c in r['pairs'])
    paired_counts = ', '.join(f"{r['label']}: {sum(s['source_crop_audited'] for s, c in r['pairs'])}" for r in runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    values = [[c['best_iptm'] for c in rr] for rr in screen]
    atlas.box(axes[0], values, points=True);atlas.labels(axes[0]);axes[0].set(ylabel='Best crop ipTM among five models', ylim=(0, 1), title='Screen confidence distribution')
    for r, values in zip(runs, values):atlas.ecdf(axes[1], values, label=f"{r['label']} · n={len(values)}", color=r['color'])
    axes[1].set(xlabel='Best crop ipTM among five models', ylabel='Fraction of distinct screen trials', title='Screen-confidence cumulative distribution', xlim=(0, 1));axes[1].legend(fontsize=8);axes[1].grid()
    atlas.save(fig, 'binder-screen-confidence', 'Crop-interface confidence across all complete screen trials',
               'One point per retained distinct generation input/seed trial, using five cofold predictions. The earliest successful repeat is retained, rather than choosing the best repeated score. Boxes show median/IQR; points show the full observed distribution. ipTM is prediction confidence, not affinity.', 'binders')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for ax, arm in zip(axes, ARMS):
        groups = [[c['best_iptm'] for c in r['screens'] if c['condition']==arm] for r in runs]
        atlas.box(ax, groups, colors=[ARMS[arm]]*5);atlas.labels(ax, 20);ax.set(ylabel='Best crop ipTM', ylim=(0, 1), title=arm.replace('_', ' '))
    atlas.save(fig, 'binder-confidence-by-condition', 'Confidence split by declared conditioning arm',
               'The frozen dual, one-region and ridge inputs are analyzed separately. Incorrectly labelled or altered inputs excluded by the scientific audit are omitted from positive distributions but remain in costs/failure accounting.', 'binders')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    groups = [[c['mean_pairwise_target_aligned_binder_CA_RMSD_A'] for c in r['screens']] for r in runs]
    atlas.box(axes[0], groups, points=True);atlas.labels(axes[0]);axes[0].set(ylabel='Mean pairwise binder Cα RMSD (Å)', title='Pose dispersion across five crop models');axes[0].set_yscale('symlog', linthresh=1)
    for r, values in zip(runs, groups):atlas.ecdf(axes[1], values, color=r['color'], label=r['label'])
    axes[1].set(xlabel='Mean pairwise binder Cα RMSD (Å)', ylabel='Cumulative fraction', title='Pose-consistency distribution');axes[1].set_xscale('symlog', linthresh=1);axes[1].legend(fontsize=8);axes[1].grid()
    atlas.save(fig, 'binder-pose-dispersion', 'Predicted binding-pose consistency',
               'Binder Cα RMSD is measured after target alignment, across the five crop cofold models. Lower dispersion indicates more consistent predicted poses. Symmetric-log axes retain every value and give linear resolution near zero; this plot does not impose a new dispersion gate.', 'binders')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, r in zip(axes.flat, runs):
        for arm, color in ARMS.items():
            cc = [c for c in r['screens'] if c['condition']==arm]
            ax.scatter([c['mean_pairwise_target_aligned_binder_CA_RMSD_A'] for c in cc], [c['best_iptm'] for c in cc], s=17, color=color, alpha=.6, label=arm)
        ax.set(title=r['label'], xlabel='Pairwise binder Cα RMSD (Å)', ylabel='Best crop ipTM', ylim=(0, 1));ax.set_xscale('symlog', linthresh=1);ax.grid()
    axes.flat[-1].axis('off');axes.flat[0].legend(fontsize=8)
    atlas.save(fig, 'binder-confidence-versus-consistency', 'Confidence versus pose consistency, by campaign',
               'All retained complete screen trials are shown; the three conditioning arms have fixed colors. These panels separate confidence from geometric consistency and retain negative/low-confidence results.', 'binders')

    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    bottom = np.zeros(len(runs));palette = plt.get_cmap('YlGnBu')(np.linspace(.15, .9, 6))
    for count, color in zip(range(6), palette):
        fractions = np.array([sum(c['dual_contact_models']==count for c in r['screens'])/len(r['screens'])*100 for r in runs])
        ax.bar(range(5), fractions, bottom=bottom, color=color, label=f'{count}/5 models');bottom += fractions
    atlas.labels(ax);ax.set(ylabel='Share of distinct screen trials (%)', ylim=(0, 100), title='How consistently do models contact both target regions?');ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    atlas.save(fig, 'binder-dual-contact-model-count', 'Dual-region contacts across the five crop models',
               'Each model is checked independently for heavy-atom contacts within 4 Å to both declared target regions. This is a geometrical contact measure; even five of five contacting predictions do not establish experimental binding.', 'binders')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, r in zip(axes.flat, runs):
        for retained, marker, color in [(False, 'x', '#b17867'), (True, 'o', r['color'])]:
            pairs = [(s, c) for s, c in r['pairs'] if s['source_crop_audited'] and bool(c['minimized_pose_geometry']['dual_region_contact'])==retained]
            ax.scatter([s['best_iptm'] for s, c in pairs], [c['intact_iptm_best'] for s, c in pairs], s=28, marker=marker, color=color, alpha=.7, label='Minimized: both contacts' if retained else 'Minimized: contact lost')
        ax.plot([0, 1], [0, 1], '--', color='#a7b6ba', lw=1);ax.set(title=r['label'], xlabel='Source crop best ipTM', ylabel='Expanded-target best ipTM', xlim=(0, 1), ylim=(0, 1));ax.grid()
    axes.flat[-1].axis('off');axes.flat[0].legend(fontsize=7)
    atlas.save(fig, 'binder-crop-to-expanded-confidence', 'Confidence transfer from crop to expanded EGFR',
               f'Every point is a distinct complete expanded-context trial paired to its exact hashed, independently audited crop source. {unmatched} source outside the accepted crop inventory is omitted here; its audited expanded result remains in expanded-only plots. Symbols record minimized contacts. The diagonal denotes equal scores; target context changes ipTM.', 'binders')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for ax, key, title in [(axes[0], 'intact_iptm_best', 'Expanded-target confidence'), (axes[1], 'static_MMGBSA_kcal_mol', 'Single-frame static MM/GBSA')]:
        groups = [[c[key] for c in r['full']] for r in runs]
        atlas.box(ax, groups, points=True);atlas.labels(ax);ax.set(title=title, ylabel='Best expanded ipTM' if key=='intact_iptm_best' else 'Static ΔG proxy (kcal/mol)')
        if key=='intact_iptm_best':ax.set_ylim(0, 1)
    atlas.save(fig, 'binder-expanded-confidence-energy', 'Expanded-context confidence and energy distributions',
               'All complete distinct expanded-context input/seed trials are included. MM/GBSA uses one minimized frame with entropy omitted; it is not an ensemble free energy or measured affinity. EGFR residues 25–646 are modeled without glycans or a membrane.', 'binders')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for r in runs:
        for retained, marker in [(True, 'o'), (False, 'x')]:
            cc = [c for c in r['full'] if bool(c['minimized_pose_geometry']['dual_region_contact'])==retained]
            axes[0].scatter([c['intact_iptm_best'] for c in cc], [c['static_MMGBSA_kcal_mol'] for c in cc], marker=marker, color=r['color'], alpha=.6, s=25)
            axes[1].scatter([c['minimized_pose_geometry']['target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A'] for c in cc], [c['intact_iptm_best'] for c in cc], marker=marker, color=r['color'], alpha=.6, s=25)
    axes[0].set(xlabel='Expanded-target best ipTM', ylabel='Static MM/GBSA (kcal/mol)', title='Energy versus interface confidence', xlim=(0, 1))
    axes[1].set(xlabel='Minimized pose RMSD from source screen (Å)', ylabel='Expanded-target best ipTM', title='Context-induced pose drift', ylim=(0, 1));axes[1].set_xscale('symlog', linthresh=1)
    handles = [Line2D([], [], color=r['color'], marker='o', ls='', label=r['label']) for r in runs]
    handles += [Line2D([], [], color='#444', marker='x', ls='', label='Lost either contact region')]
    axes[0].legend(handles=handles, fontsize=7);axes[0].grid();axes[1].grid()
    atlas.save(fig, 'binder-energy-pose-contact', 'Energy, confidence, pose drift and contact retention',
               'Filled circles retain contacts with both regions after minimization; crosses lose either region. A favorable static energy alone does not demonstrate preserved epitope geometry. No ipTM is recomputed for the minimized structure: its y-value is the preceding expanded cofold confidence.', 'binders')

    fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained')
    for offset, stage, color in [(-.23, 'crop', '#4c78a8'), (0, 'cofold', '#45aa9b'), (.23, 'minimized', '#d58b37')]:
        fractions = []
        for r in runs:
            values = [s['dual_contact_models']>0 if stage=='crop' else c['intact_models_with_dual_region_contact']>0 if stage=='cofold' else bool(c['minimized_pose_geometry']['dual_region_contact']) for s, c in r['pairs'] if s['source_crop_audited']]
            fractions.append(100*sum(values)/len(values))
        ax.bar(np.arange(5)+offset, fractions, width=.22, color=color, label={'crop': 'Source crop: ≥1/5 contacting', 'cofold': 'Expanded cofold: ≥1/5 contacting', 'minimized': 'Minimized pose: both regions'}[stage])
    atlas.labels(ax);ax.set(ylabel='Share of selected follow-up cohort (%)', ylim=(0, 108), title='Contact retention across stages of the same selected cohort');ax.legend(fontsize=8, loc='lower left');ax.grid(axis='y');ax.set_axisbelow(True)
    atlas.save(fig, 'binder-contact-retention-stages', 'Contact retention before and after target expansion',
               'The denominator is the same selected follow-up cohort with audited crop sources at all three stages. Counts: '+paired_counts+'. Unmatched crop sources are omitted only from paired comparisons. These are stage-wise proportions, not a monotonic funnel. Five-model and single-pose endpoints differ.', 'binders')


def role_mean(run, key, role):
    indices = [i for i, value in enumerate(run['roles']) if value == role]
    return np.array([finite_mean(run['arrays'][key][indices, j]) for j in range(len(run['minutes']))])


def lease_array(run):
    hosts = [h for h, role in zip(run['hosts'], run['roles']) if role=='science']
    n = len(run['minutes']);arr = np.zeros((len(hosts), n))
    lo_bins = run['start']+np.arange(n)*60
    hi_bins = np.minimum(lo_bins+60, run['end'])
    for row in run['audit']['row_timings']:
        if not row.get('gpus') or row['host'] not in hosts or row.get('started_at') is None or row.get('seconds') is None:
            continue
        lo, hi = max(run['start'], row['started_at']), min(run['end'], row['started_at']+row['seconds'])
        if hi > lo:
            arr[hosts.index(row['host'])] += row['gpus']*np.maximum(0, np.minimum(hi_bins, hi)-np.maximum(lo_bins, lo))/(hi_bins-lo_bins)
    assert np.max(arr) < 4.15, (run['label'], np.max(arr))
    return hosts, arr


def plot_hardware(atlas, runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, role in zip(axes, ['science', 'inference']):
        for r in runs:
            values = r['arrays']['busy'][[i for i, value in enumerate(r['roles']) if value==role]].ravel()
            atlas.ecdf(ax, values, color=r['color'], label=r['label'])
        ax.set(title=role.capitalize()+' node-minute GPU activity', xlabel='Mean of four GPU busy values (%)', ylabel='Cumulative fraction of observed node-minutes', xlim=(0, 100));ax.grid();ax.legend(fontsize=8)
    atlas.save(fig, 'gpu-activity-distributions', 'GPU utilization distributions by node role',
               'Every value is one observed node-minute, averaging the four GPUs on that node. Missing observations are omitted and coverage is exported. Curves describe hardware activity, not independent performance replicates or SM/FLOPS efficiency.', 'hardware')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    for offset, role, color in [(-.23, 'science', '#4c78a8'), (0, 'inference', '#168b80'), (.23, 'all', '#d58b37')]:
        values = [r['trial']['overall_observed_GPU_busy_percent'] if role=='all' else r['trial']['telemetry']['role_observed_node_minutes'][role]['mean_GPU_busy_percent'] for r in runs]
        axes[0].bar(np.arange(5)+offset, values, width=.22, color=color, label=role.capitalize())
    atlas.labels(axes[0]);axes[0].set(ylabel='Mean observed GPU busy (%)', ylim=(0, 105), title='Utilization by allocated node role');axes[0].legend(fontsize=8)
    for offset, role, color in [(-.15, 'science', '#4c78a8'), (.15, 'inference', '#168b80')]:
        values = [100*r['trial']['telemetry']['role_observed_node_minutes'][role]['observed_node_minutes_at_least_80_percent_busy_fraction'] for r in runs]
        axes[1].bar(np.arange(5)+offset, values, width=.28, color=color, label=role.capitalize())
    atlas.labels(axes[1]);axes[1].set(ylabel='Observed node-minutes at ≥80% mean GPU busy (%)', title='Frequency of highly active node-minutes');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True)
    atlas.save(fig, 'gpu-role-means-high-activity', 'Role averages and highly active intervals',
               'The overall mean weights observed node-minutes; four GPUs per node are expected throughout. The 80% threshold is a descriptive hardware diagnostic. A resident model or allocated lease does not itself imply a busy GPU.', 'hardware')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, role in zip(axes, ['science', 'inference']):
        for i, r in enumerate(runs):
            means = [finite_mean(r['arrays']['busy'][j]) for j, value in enumerate(r['roles']) if value==role]
            jitter = np.linspace(-.16, .16, len(means)) if len(means)>1 else [0]
            ax.scatter(i+np.array(jitter), means, s=35, color=r['color'], alpha=.85)
            ax.hlines(finite_mean(means), i-.23, i+.23, color='#122f3c', lw=2)
        atlas.labels(ax);ax.set(ylabel='Per-node mean GPU busy (%)', ylim=(0, 100), title=role.capitalize()+' nodes: allocation balance');ax.grid(axis='y')
    atlas.save(fig, 'gpu-per-node-average-distribution', 'Distribution of average activity across individual nodes',
               'Each point is one physical node over that campaign; horizontal ticks are equal-node means. This differs slightly from observed-node-minute weighting when coverage varies. Node identities and coverage are included in the CSV and per-run plots below.', 'hardware')

    for r in runs:
        prefix = r['id'];hosts = r['hosts'];n = len(r['minutes']);duration = (r['end']-r['start'])/60
        cmap = plt.get_cmap('YlGnBu').copy();cmap.set_bad('#dfe3e5')
        fig, ax = plt.subplots(figsize=(13, max(6.4, len(hosts)*.29+1.5)), layout='constrained')
        im = ax.imshow(r['arrays']['busy'], origin='lower', aspect='auto', extent=(0, n, -.5, len(hosts)-.5), vmin=0, vmax=100, cmap=cmap)
        ax.set_yticks(range(len(hosts)), [h+(' · inference' if role=='inference' else ' · science') for h, role in zip(hosts, r['roles'])], fontsize=8)
        ax.set(title=r['label']+' · every node, every observed minute', xlabel='Minutes since science allocation started', xlim=(0, duration));fig.colorbar(im, ax=ax, label='Mean of four GPU busy values (%)', shrink=.9)
        atlas.save(fig, prefix+'-node-gpu-heatmap', r['label']+' — per-node GPU utilization heatmap',
                   'Node means use all four measured GPUs. Gray cells have no observation and are not filled with zero. The historical inference renewal gap remains visible. The final partial minute is clipped to the actual allocation end.', prefix)

        columns = 4;rows = int(math.ceil(len(hosts)/columns))
        fig, axes = plt.subplots(rows, columns, figsize=(15, rows*1.9+1), sharex=True, sharey=True, layout='constrained')
        for i, (ax, host, role) in enumerate(zip(axes.flat, hosts, r['roles'])):
            for gpu in range(4):ax.plot(r['minutes'], r['per_gpu'][i, gpu], lw=.65, alpha=.65, color=GPU_COLORS[gpu])
            ax.plot(r['minutes'], r['arrays']['busy'][i], color='#162f39', lw=1.2)
            ax.set(title=host+' · '+role, ylim=(-2, 102), xlim=(0, duration), yticks=[0, 50, 100]);ax.grid(lw=.5);ax.tick_params(labelsize=7)
            ax.title.set_fontsize(8);ax.title.set_color('#116f68' if role=='inference' else '#315a7c')
        for ax in list(axes.flat)[len(hosts):]:ax.axis('off')
        fig.supxlabel('Minutes since science allocation started');fig.supylabel('NVIDIA GPU busy (%)')
        fig.legend(handles=[Line2D([], [], color=GPU_COLORS[g], label=f'GPU {g}') for g in range(4)]+[Line2D([], [], color='#162f39', label='Node mean')], loc='outside upper center', ncols=5, fontsize=9)
        atlas.save(fig, prefix+'-all-gpu-traces', r['label']+' — all four GPU traces on every node',
                   'Small multiples expose GPU imbalance inside a node: one GPU at 100% produces a four-GPU node mean of 25%. Colors identify local GPU indices, with the node mean in dark ink. Missing intervals break the traces.', prefix)

        fig, axes = plt.subplots(1, 3, figsize=(17, max(6.5, len(hosts)*.29+1)), sharey=True, layout='constrained')
        for ax, key, title in zip(axes, ['cpu', 'host_memory', 'gpu_memory'], ['Host CPU busy', 'Host RAM used', 'GPU memory capacity used']):
            im = ax.imshow(r['arrays'][key], origin='lower', aspect='auto', extent=(0, n, -.5, len(hosts)-.5), vmin=0, vmax=100, cmap=cmap)
            ax.set(title=title+' (%)', xlabel='Elapsed minutes', xlim=(0, duration));fig.colorbar(im, ax=ax, shrink=.75)
        axes[0].set_yticks(range(len(hosts)), [h+(' · I' if role=='inference' else ' · S') for h, role in zip(hosts, r['roles'])], fontsize=8)
        atlas.save(fig, prefix+'-cpu-memory', r['label']+' — CPU and memory timelines',
                   'CPU and host memory are node aggregates; GPU memory is the mean fraction used across four GPUs. High memory residency is separate from active computation. Gray means unavailable measurements.', prefix)

        science_hosts, leased = lease_array(r)
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout='constrained')
        for role, color in [('science', '#4c78a8'), ('inference', '#168b80')]:
            axes[0].plot(r['minutes'], role_mean(r, 'busy', role), label=role.capitalize(), color=color)
        axes[0].set(ylabel='Mean GPU busy (%)', ylim=(0, 105), title=r['label']+' · hardware, worker leases and inference activity');axes[0].legend(fontsize=8)
        actual = role_mean(r, 'busy', 'science')/100*4*len(science_hosts)
        axes[1].plot(r['minutes'], leased.sum(0), label='Leased scientific GPU slots', color='#8568a7')
        axes[1].plot(r['minutes'], actual, label='Measured equivalent busy science GPUs', color='#4c78a8')
        axes[1].axhline(4*len(science_hosts), color='#99a8ae', ls='--', lw=1, label='Allocated science GPUs')
        axes[1].set(ylabel='GPU slots / equivalent busy GPUs');axes[1].legend(fontsize=8)
        for key, title, color in [('vllm:num_requests_running', 'Running requests', '#168b80'), ('vllm:num_requests_waiting', 'Waiting requests', '#d58b37')]:
            axes[2].plot(r['minutes'], r['queue'][key], label=title, color=color)
        axes[2].set(ylabel='Inference requests', xlabel='Minutes since science allocation started', xlim=(0, duration));axes[2].legend(fontsize=8)
        for ax in axes:ax.grid()
        atlas.save(fig, prefix+'-work-and-queues', r['label']+' — science leases, actual GPU activity and inference queues',
                   'Worker leases include imports, CPU stages and I/O, so they exceed actual GPU busy time. Equivalent busy GPUs sum the observed node means ×4 and are not a count of continuously occupied devices. Inference curves sum minute means across live replicas; unavailable replica data produces a gap.', prefix)


def plot_inference_and_timing(atlas, source, runs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
    for ax, mode, title in zip(axes, ['all', False, True], ['All observed requests', 'Nonthinking agent calls', 'Thinking-default tournament calls']):
        for r in runs:
            values = [q['seconds'] for q in r['requests'] if q['status']==200 and (mode=='all' or q['thinking'] is mode)]
            if values:atlas.ecdf(ax, values, color=r['color'], label=r['label'])
        ax.set(title=title, xlabel='Request latency (seconds)', ylabel='Cumulative fraction');ax.set_xscale('log');ax.grid();ax.legend(fontsize=7)
    atlas.save(fig, 'inference-latency-distributions', 'Observed inference-request latency distributions',
               'Final successful logical requests are used for pooled runs; retries are not independent samples. Historical server logs lack thinking flags and appear only in the all-request panel. Payload size, task mix and caching differ across campaigns; this is an operational comparison.', 'inference')

    profile = source.get('frozen_request_profiles', {})
    selected = [r for r in profile.get('results', []) if r['label']=='one-replica-warm' or 'affinity-warm' in r['label']]
    selected.sort(key=lambda r: len(r['replicas']))
    if selected:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
        n = [len(r['replicas']) for r in selected]
        for key, title in [('median_request_seconds', 'Median'), ('p95_request_seconds', 'P95')]:axes[0].plot(n, [r[key] for r in selected], 'o-', label=title)
        axes[0].set(xticks=n, xlabel='Ready Nemotron replicas', ylabel='Request seconds', title='Fixed-context benchmark latency');axes[0].legend()
        axes[1].bar(range(len(selected)), [r['successful_reviews_per_second'] for r in selected], color='#168b80')
        for i, r in enumerate(selected):axes[1].text(i, r['successful_reviews_per_second'], f" {r['passed']}/{r['requests']} passed", ha='center', va='bottom', fontsize=8)
        axes[1].set(xticks=range(len(selected)), xticklabels=n, xlabel='Ready Nemotron replicas', ylabel='Strictly validated reviews / second', title='Identical request content; concurrency 12', ylim=(0, max(r['successful_reviews_per_second'] for r in selected)*1.2))
        for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True)
        atlas.save(fig, 'inference-controlled-benchmark', 'Controlled frozen-request inference benchmark',
                   'The same 24 requests and thinking settings are repeated at each replica count. Fifteen of 24 strict checks passed at every count; the same gate/arithmetic failures remain visible. P95 uses linear interpolation. Unlike adaptive campaigns, this comparison fixes request content.', 'inference')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, r in zip(axes, runs[-2:]):
        by = collections.defaultdict(list)
        for q in r['requests']:by[q['backend']].append(q)
        names = sorted(by)
        prompt = [sum(q.get('prompt_tokens') or 0 for q in by[name])/1e6 for name in names]
        output = [sum(q.get('completion_tokens') or 0 for q in by[name])/1e6 for name in names]
        ax.bar(range(len(names)), prompt, label='Prompt tokens', color='#168b80');ax.bar(range(len(names)), output, bottom=prompt, label='Completion tokens', color='#d58b37')
        ax.set(xticks=range(len(names)), xticklabels=[name.replace('nemotron-', '') for name in names], xlabel='Nemotron replica', ylabel='Reported tokens (millions)', title=r['label']+' · request distribution')
        for i, name in enumerate(names):ax.text(i, prompt[i]+output[i], str(len(by[name])), ha='center', va='bottom', fontsize=8)
        ax.legend(fontsize=8);ax.grid(axis='y');ax.set_axisbelow(True)
    atlas.save(fig, 'inference-replica-load', 'How requests and token work were distributed across eight replicas',
               'Labels above bars are final logical request counts; heights are API-reported prompt/completion tokens. Prompt counts include reused/cached prefixes and do not directly measure fresh GPU prefill work or billing. Load-aware cache affinity can produce unequal request counts.', 'inference')

    fig, axes = plt.subplots(1, 5, figsize=(17, 4.8), layout='constrained')
    for ax, stage in zip(axes, ['RFD4-Proteina', 'proteinmpnn', 'chai-lab', 'cofold', 'mmgbsa']):
        groups = [[c['stage_seconds'][stage] for c in (r['screens'] if stage in ('RFD4-Proteina', 'proteinmpnn', 'chai-lab') else r['full']) if stage in c.get('stage_seconds', {})] for r in runs]
        atlas.box(ax, groups, width=.6);atlas.labels(ax, 35);ax.set(title=stage, ylabel='Successful-stage wall seconds');ax.set_yscale('log')
    atlas.save(fig, 'science-stage-duration', 'Measured duration of each scientific stage',
               'Only complete retained input trials contribute stage-duration distributions. Timings include each stage’s process startup/I/O. Failures, discarded duplicate attempts and allocated idle periods remain in campaign rates and costs. These distributions are not summed to infer a parallel critical path.', 'inference')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for offset, metric, color in [(-.16, 'files', '#168b80'), (.16, 'hours', '#d58b37')]:
        values = []
        for r in runs:
            t = r['trial']
            values.append((t['repeated_screen_input_rows']+t['repeated_intact_input_rows'])*5 if metric=='files' else sum(row['seconds']*row['gpus']/3600 for row in r['audit']['row_timings'] if row.get('seconds') and row.get('gpus') and row['returncode']!=0))
        axes[0 if metric=='files' else 1].bar(range(5), values, color=color)
    atlas.labels(axes[0]);atlas.labels(axes[1]);axes[0].set(ylabel='Repeated complete trial files excluded', title='Duplicate-completion overhead');axes[1].set(ylabel='Failed GPU-leased row-hours', title='Failed-attempt lease time')
    for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True)
    atlas.save(fig, 'duplicate-failure-overhead', 'Duplicate output and failed scientific execution overhead',
               'Left: complete repeated identical input/seed submissions ×five complex files, omitted from useful-output counts. Right: receipt wall seconds ×leased GPUs for rows returning a nonzero code, including unapproved sources. This is lease time, not actual GPU busy time; allocated idle time is charged separately in the full ledger.', 'inference')


def binder_id(sequence):
    return 'b_'+hashlib.sha256(sequence.encode()).hexdigest()[:12]


def hypothesis_summary(run, round_number, hypothesis):
    record = run['round_records'].get(round_number, {})
    for thread in record.get('threads', []):
        if thread['hypothesis']['id'] == hypothesis:
            return thread['hypothesis']['summary']
    return hypothesis


def export_data(atlas, source, runs, comparison):
    summary_rows, screens, full, bins, node_rows, minute_rows, round_rows, request_rows = [], [], [], [], [], [], [], []
    for r in runs:
        t = r['trial'];hours = (r['end']-r['start'])/3600
        summary_rows.append({'campaign': r['label'], 'inference_nodes': r['inference'], 'science_nodes': r['science'],
                             'science_allocation_minutes': hours*60, 'screen_input_trials': len(r['screens']), 'expanded_input_trials': len(r['full']),
                             'verified_distinct_trial_files': t['verified_files_on_distinct_input_trials'], 'files_per_hour': t['distinct_trial_complex_files_per_science_allocation_hour'],
                             'distinct_sequences_per_hour': t['distinct_sequences_per_science_allocation_hour'],
                             'expanded_trials_per_hour': t['distinct_intact_trials_per_science_allocation_hour'],
                             'retained_minimized_trials_per_hour': t['retained_intact_trials_per_science_allocation_hour'],
                             'overall_GPU_busy_percent': t['overall_observed_GPU_busy_percent'], 'fully_reviewed_evaluations': t['reviewed_hypothesis_rounds'],
                             'incomplete_threads': len(t['thread_failures']), 'unique_tested_hypotheses': t['unique_tested_hypotheses'],
                             'trial_window_GPU_hours': t['trial_window_allocated_GPU_hours'],
                             'trial_window_files_per_GPU_hour': t['trial_window_distinct_trial_files_per_GPU_hour']})
        for c in r['screens']:
            screens.append({'campaign': r['label'], 'run_id': r['id'], 'round': c['round'], 'hypothesis_id': c['hypothesis'],
                            'hypothesis': hypothesis_summary(r, c['round'], c['hypothesis']), 'binder_id': binder_id(c['binder_sequence']),
                            'sequence': c['binder_sequence'], 'condition': c['condition'], 'seed': c['seed'],
                            'best_crop_iptm': c['best_iptm'], 'median_crop_iptm': c['median_iptm'],
                            'dual_contact_models_of_five': c['dual_contact_models'],
                            'pairwise_binder_CA_RMSD_A': c['mean_pairwise_target_aligned_binder_CA_RMSD_A'],
                            'RFD4_seconds': c['stage_seconds'].get('RFD4-Proteina'), 'proteinMPNN_seconds': c['stage_seconds'].get('proteinmpnn'),
                            'crop_cofold_seconds': c['stage_seconds'].get('chai-lab'), 'source': c['summary_path']})
        for s, c in r['pairs']:
            geometry = c['minimized_pose_geometry']
            full.append({'campaign': r['label'], 'run_id': r['id'], 'round': c['round'], 'hypothesis_id': c['hypothesis'],
                         'hypothesis': hypothesis_summary(r, c['round'], c['hypothesis']), 'binder_id': binder_id(c['binder_sequence']),
                         'sequence': c['binder_sequence'], 'condition': c['condition'], 'seed': c['seed'],
                         'source_crop_iptm': s['best_iptm'], 'expanded_best_iptm': c['intact_iptm_best'],
                         'source_crop_independently_audited': s['source_crop_audited'],
                         'source_crop_summary_was_compact': s['source_summary_compact'],
                         'expanded_median_iptm': c['intact_iptm_median'], 'static_MMGBSA_kcal_mol': c['static_MMGBSA_kcal_mol'],
                         'expanded_dual_contact_models_of_five': c['intact_models_with_dual_region_contact'],
                         'minimized_both_contact_regions': bool(geometry['dual_region_contact']),
                         'minimized_region1_contact_residues': geometry['region1_contact_residues'],
                         'minimized_region2_contact_residues': geometry['region2_contact_residues'],
                         'minimized_close_atom_pairs_lt2A': geometry['close_atom_pairs_lt2A'],
                         'minimized_minimum_interchain_heavy_A': geometry['minimum_interchain_heavy_A'],
                         'minimized_pose_RMSD_to_source_A': geometry['target_crop_aligned_binder_CA_RMSD_to_best_screen_model_A'],
                         'cofold_seconds': c['stage_seconds']['cofold'], 'MMGBSA_seconds': c['stage_seconds']['mmgbsa'],
                         'source': c['summary_path'], 'source_screen': c['source_screen_summary'],
                         'expanded_structure': c['energy_input_pose_geometry']['path'], 'minimized_structure': geometry['path']})
        for b in r['bins']:bins.append({'campaign': r['label'], **b})
        for i, (host, role) in enumerate(zip(r['hosts'], r['roles'])):
            busy = r['arrays']['busy'][i];finite = busy[np.isfinite(busy)]
            node_rows.append({'campaign': r['label'], 'host': host, 'role': role, 'mean_GPU_busy_percent': finite_mean(finite),
                              'p95_node_minute_GPU_busy_percent': float(np.percentile(finite, 95)) if len(finite) else None,
                              'coverage_fraction': len(finite)/len(busy), 'observed_node_minutes': len(finite),
                              'mean_CPU_busy_percent': finite_mean(r['arrays']['cpu'][i]), 'mean_host_memory_percent': finite_mean(r['arrays']['host_memory'][i]),
                              'mean_GPU_memory_percent': finite_mean(r['arrays']['gpu_memory'][i]), 'mean_four_GPU_watts': finite_mean(r['arrays']['power'][i])})
            for j, minute in enumerate(r['minutes']):
                minute_rows.append({'campaign': r['label'], 'host': host, 'role': role, 'minute_bin_start': float(minute-.5),
                                    'mean_GPU_busy_percent': busy[j], **{f'GPU{g}_busy_percent': r['per_gpu'][i, g, j] for g in range(4)},
                                    'CPU_busy_percent': r['arrays']['cpu'][i, j], 'host_memory_percent': r['arrays']['host_memory'][i, j],
                                    'GPU_memory_percent': r['arrays']['gpu_memory'][i, j], 'four_GPU_watts': r['arrays']['power'][i, j]})
        for n, record in r['round_records'].items():
            for priority in sorted(record['steering'], key=lambda p: p['priority'], reverse=True)[:3]:
                round_rows.append({'campaign': r['label'], 'run_id': r['id'], 'round': n, **priority})
        for q in r['requests']:
            request_rows.append({'campaign': r['label'], 'thinking': q['thinking'], 'latency_seconds': q['seconds'],
                                 'status': q['status'], 'backend': q['backend'], 'prompt_tokens': q.get('prompt_tokens'),
                                 'completion_tokens': q.get('completion_tokens'), 'gateway_queue_seconds': q.get('queue_seconds')})
    datasets = {'campaigns.csv': summary_rows, 'screen-binders.csv': screens, 'expanded-binders.csv': full,
                'throughput-blocks.csv': bins, 'node-summary.csv': node_rows, 'node-minute-telemetry.csv': minute_rows,
                'tournament-steering.csv': round_rows, 'inference-request-timings.csv': request_rows}
    for filename, rows in datasets.items():write_csv(atlas.output/'data'/filename, rows)
    sequences = {row['binder_id']: row['sequence'] for row in [*screens, *full]}
    fasta = ''.join(f'>{key}\n{sequence}\n' for key, sequence in sorted(sequences.items()))
    (atlas.output/'data/binders.fasta').write_text(fasta)
    provenance = {'generated_at': time.time(), 'comparison_source': str(comparison.resolve()), 'comparison_sha256': digest(comparison),
                  'campaigns': [{'campaign': r['label'], 'path': str(r['root']), 'scientific_audit_sha256': digest(r['root']/'scaling-audit.json'),
                                 'required_checks_passed': r['audit']['required_campaign_completion_checks_passed'],
                                 'screen_trials': len(r['screens']), 'expanded_trials': len(r['full']),
                                 'telemetry_files': [{'path': str(p), 'sha256': digest(p)} for p in sorted(r['telemetry_files'])],
                                 'observed_node_minutes': int(np.isfinite(r['arrays']['busy']).sum()),
                                 'node_minute_cells': int(r['arrays']['busy'].size)} for r in runs],
                  'scope': 'Report generation reads existing campaign/audit/telemetry files only. No Slurm jobs, inference calls or scientific simulations submitted.',
                  'deduplication': 'Earliest successful complete screen per (condition,input coordinate SHA256,seed,target sequence); expanded follow-up per (binder sequence,seed,target length). All raw repeats/failures stay in original audits and costs.',
                  'completion_time': 'Terminal row lease start plus recorded row seconds; rounded receipt wall time, not filesystem modification time.',
                  'paired_crop_scope': {'compact_source_summaries': sum(s['source_summary_compact'] for r in runs for s, c in r['pairs']),
                                        'unmatched_crop_sources': sum(not s['source_crop_audited'] for r in runs for s, c in r['pairs']),
                                        'method': 'Compact-source aggregates are reconstructed only when every model SHA matches independently audited crop models. Unmatched crop scores remain unavailable and are omitted from paired comparisons; independently audited expanded results remain included.'},
                  'throughput_distribution': 'Only full nonoverlapping five-minute blocks in distribution plots; these are correlated temporal samples, not independent campaign replicates.',
                  'hardware': 'Four GPU values averaged per node and then per one-minute bin. Missing data remains missing. NVIDIA utilization.gpu is not SM occupancy or FLOPS efficiency.',
                  'science': 'EGFR crop205 residues; expanded target 622 residues 25–646 with 25 disulfides. Static single-frame MMGBSA omits entropy, glycans and membrane; no experimental affinity measured.',
                  'cost': source['cost_note'], 'unique_experiment_GPU_hours': source['unique_job_allocated_GPU_hours'],
                  'thinking': 'Prior settings preserved: agent roles nonthinking; tournament generation/ranking thinking-default.',
                  'run_variation': 'Adaptive hypotheses, seeds and output mixes differ; the historical baseline and 2/18 include documented operational repairs. Ratios with one campaign have no estimate of between-campaign repeatability.'}
    write_json(atlas.output/'data/provenance.json', provenance)
    write_json(atlas.output/'data/comparison.json', source)
    distributions = {r['label']: {'screen_iptm': quantiles([c['best_iptm'] for c in r['screens']]),
                                  'screen_pose_RMSD_A': quantiles([c['mean_pairwise_target_aligned_binder_CA_RMSD_A'] for c in r['screens']]),
                                  'expanded_iptm': quantiles([c['intact_iptm_best'] for c in r['full']]),
                                  'MMGBSA_kcal_mol': quantiles([c['static_MMGBSA_kcal_mol'] for c in r['full']]),
                                  'five_minute_files_per_hour': quantiles([b['files_per_hour'] for b in r['bins'] if b['full_five_minutes']])} for r in runs}
    write_json(atlas.output/'data/distributions.json', distributions)
    return {'summary': summary_rows, 'screens': screens, 'full': full, 'nodes': node_rows, 'steering': round_rows,
            'provenance': provenance, 'datasets': list(datasets)+['binders.fasta', 'provenance.json', 'distributions.json', 'comparison.json']}


def table(headers, rows, attributes=''):
    return '<div class="table-wrap"><table '+attributes+'><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def data_url(path, mime):
    return 'data:'+mime+';base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def render_html(atlas, source, runs, data):
    e = html.escape
    wide = {'throughput-timeline', 'ratio-scientific-output-types', 'throughput-utilization-efficiency',
            'round-scientific-output', 'binder-screen-confidence', 'binder-confidence-by-condition', 'binder-pose-dispersion',
            'binder-confidence-versus-consistency', 'binder-crop-to-expanded-confidence', 'binder-expanded-confidence-energy',
            'binder-energy-pose-contact', 'gpu-activity-distributions', 'gpu-role-means-high-activity',
            'gpu-per-node-average-distribution', 'inference-latency-distributions', 'inference-controlled-benchmark',
            'inference-replica-load', 'science-stage-duration', 'duplicate-failure-overhead'}
    def figure(name):
        f = atlas.figures[name];path = atlas.output/'assets'/f'{name}.svg';url = data_url(path, 'image/svg+xml')
        return ('<figure class="plot '+('wide' if name in wide or name.startswith('run-') else '')+'" id="'+name+'">'
                '<div class="plot-title"><h3>'+e(f['title'])+'</h3><div><button class="expand" data-image="img-'+name+'">Expand</button> '
                '<a class="download figure-download" href="#" data-image="img-'+name+'" download="'+name+'.svg">SVG ↓</a></div></div>'
                '<img id="img-'+name+'" loading="lazy" decoding="async" src="'+url+'" alt="'+e(f['title'])+'">'
                '<figcaption>'+e(f['caption'])+'</figcaption></figure>')
    def section(name, title, description):
        figs = ''.join(figure(key) for key, f in atlas.figures.items() if f['section']==name)
        return '<section id="'+name+'"><div class="section-heading"><span>'+name.upper()+'</span><h2>'+title+'</h2><p>'+description+'</p></div><div class="chart-grid">'+figs+'</div></section>'
    recommendation = source['final_recommendation']['aggregate']
    cards = [('8 : 12', 'Best observed inference : science split'),
             (number(recommendation['coordinate_files_per_hour'], 0)+'/h', 'Verified distinct-input complex files'),
             (number(recommendation['overall_observed_GPU_busy_percent'])+'%', 'Combined observed GPU busy'),
             (str(sum(len(r['screens']) for r in runs)), 'Retained generation input trials'),
             (str(sum(len(r['full']) for r in runs)), 'Complete expanded-context trials'),
             (str(len(atlas.figures)), 'Scientific and HPC figures')]
    html_parts = ['''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jupiter · Binder and Inference Atlas</title>
<style>
:root{--ink:#15333f;--muted:#526c78;--teal:#137f79;--gold:#c98636;--paper:#f2f6f7;--line:#dbe5e8}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}a{color:#116f80}button,select,input{font:inherit}button,select,.download{border:1px solid var(--line);background:white;border-radius:7px;padding:.35rem .65rem;color:var(--ink)}button{cursor:pointer}header{background:#102f3b;color:#eaf6f6;padding:3rem max(1.2rem,calc((100vw - 1420px)/2));position:relative;overflow:hidden}header:after{content:'';position:absolute;right:-80px;top:-140px;width:520px;height:520px;border:70px solid #16857c25;border-radius:50%;pointer-events:none}header .eyebrow{color:#84d7ca;letter-spacing:.14em;font-size:.75rem;font-weight:700}h1{font-size:clamp(2.2rem,4.2vw,4rem);line-height:1.12;max-width:960px;margin:.8rem 0 1rem;font-weight:720;letter-spacing:-.04em}header p{max-width:900px;color:#b7d1d8;font-size:1.05rem}header .badge{display:inline-block;background:#244c55;padding:.3rem .7rem;border-radius:20px;font-size:.8rem;margin:.4rem .4rem 0 0}nav{position:sticky;top:0;z-index:5;background:#fffffff2;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);display:flex;gap:1.3rem;padding:.85rem max(1.2rem,calc((100vw - 1420px)/2));overflow:auto;white-space:nowrap}nav a{font-weight:650;text-decoration:none;color:#315663;font-size:.88rem}main{max-width:1460px;padding:1.5rem 1.2rem 4rem;margin:auto}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:1.2rem 1.4rem}.card strong{display:block;font-size:2rem;line-height:1.2;color:var(--teal);font-variant-numeric:tabular-nums}.card span{font-size:.85rem;color:var(--muted)}section{margin-top:3.8rem;scroll-margin-top:75px}.section-heading>span{font-size:.72rem;letter-spacing:.15em;color:var(--teal);font-weight:750}h2{font-size:2rem;letter-spacing:-.025em;margin:.25rem 0 .6rem}h3{font-size:1.05rem;margin:0}.section-heading p,.note{color:var(--muted);max-width:1100px}.callout{border-left:4px solid var(--teal);background:#e3f1ee;padding:1.2rem 1.5rem;border-radius:0 10px 10px 0;margin:1.5rem 0}.callout p{margin:.3rem 0}.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.2rem}.plot{margin:0;background:white;border:1px solid var(--line);border-radius:12px;overflow:hidden;min-width:0}.plot.wide{grid-column:1/-1}.plot-title{padding:1rem 1.1rem .3rem;display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}.plot-title>div{white-space:nowrap}.download{font-size:.8rem;text-decoration:none}.expand{font-size:.8rem}.plot img{display:block;width:100%;height:auto;padding:.7rem .7rem 0}figcaption{padding:.65rem 1.2rem 1.2rem;font-size:.85rem;color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px;background:white;margin:1rem 0}table{border-collapse:collapse;width:100%;font-size:.83rem}th,td{text-align:left;padding:.65rem .75rem;border-bottom:1px solid #e6edef;vertical-align:top}th{background:#e6eff1;white-space:nowrap;color:#254d5c}tbody tr:hover{background:#f0f8f6}td.num{font-variant-numeric:tabular-nums;text-align:right}.controls{display:flex;flex-wrap:wrap;gap:.7rem;align-items:center;margin:1rem 0}input{padding:.45rem .7rem;border:1px solid var(--line);border-radius:7px;min-width:260px}.node-panel{display:none}.node-panel.active{display:block}.pill{padding:.1rem .45rem;border-radius:12px;white-space:nowrap;font-size:.75rem;background:#e0efe7;color:#1c6c50}.pill.lost{background:#f6e9db;color:#885d27}#binder-table th{cursor:pointer;position:sticky;top:0;z-index:1}#binder-table{font-size:.78rem}.binder-scroll{max-height:680px}.sequence{font:11px ui-monospace,SFMono-Regular,monospace;word-break:break-all;max-width:250px;min-width:150px}.hypothesis{max-width:320px;min-width:170px;font-size:.77rem}details{border:1px solid var(--line);background:white;border-radius:10px;margin:.8rem 0;padding:.8rem 1rem}summary{cursor:pointer;font-weight:650}.data-links{display:flex;flex-wrap:wrap;gap:.7rem}.data-links a{border:1px solid var(--line);padding:.6rem .8rem;background:white;border-radius:7px;font-size:.85rem;text-decoration:none}code{font:12px ui-monospace,SFMono-Regular,monospace}footer{color:var(--muted);font-size:.8rem;border-top:1px solid var(--line);padding-top:1.5rem;margin-top:2rem}dialog{border:0;border-radius:10px;padding:1rem;width:96vw;max-width:1900px;max-height:96vh;background:white}dialog::backdrop{background:#102936c9}dialog img{width:100%;height:auto}dialog .close{position:sticky;top:0;float:right;background:#153c49;color:white}dialog h3{margin:0 4rem 1rem 0}#node-select{min-width:250px}.method-list{columns:2;column-gap:3rem}.method-list li{break-inside:avoid;margin:0 0 1rem}.method-list b{color:var(--ink)}@media(max-width:850px){.cards{grid-template-columns:repeat(2,1fr)}.chart-grid{grid-template-columns:1fr}.plot{grid-column:1/-1}.plot-title{display:block}.plot-title>div{margin-top:.5rem}.method-list{columns:1}main{padding:1rem .7rem}header{padding:2rem 1rem}nav{padding:.8rem 1rem;gap:1rem}}@media print{nav,.controls,.expand,.download{display:none}.node-panel{display:block!important}header{background:white;color:#15333f}header p{color:#526c78}.plot{break-inside:avoid}.cards{grid-template-columns:repeat(3,1fr)}.chart-grid{display:block}.plot{margin-bottom:1rem}}
</style></head><body><header><div class="eyebrow">JUPITER / LOAM / AUDITED COMPUTATIONAL CAMPAIGNS</div><h1>Binder quality meets<br>inference–science balance.</h1><p>A visual atlas of five twenty-node campaigns: binder confidence and geometry, scientific throughput, every node’s GPU activity, and the inference bottleneck.</p><span class="badge">20 total nodes · 80 GPUs per campaign</span><span class="badge">Three tournament rounds per campaign</span><span class="badge">Thinking settings preserved</span><span class="badge">Offline HTML · plots and CSVs embedded</span></header><nav><a href="#overview">Overview</a><a href="#throughput">Throughput</a><a href="#binders">Binder metrics</a><a href="#hardware">GPU balance</a><a href="#nodes">Every node</a><a href="#inference">Inference & timing</a><a href="#explorer">Binder explorer</a><a href="#rounds">Tournament rounds</a><a href="#methods">Methods & data</a></nav><main>''']
    html_parts.append('<div class="cards">'+''.join('<div class="card"><strong>'+e(value)+'</strong><span>'+e(label)+'</span></div>' for value, label in cards)+'</div>')
    html_parts.append('<section id="overview"><div class="section-heading"><span>READOUT</span><h2>More inference capacity increased useful scientific output.</h2></div><div class="callout"><p><b>8 inference + 12 science nodes</b> is the best observed operating point in this search. Its two three-round campaigns produced <b>1,623 and 1,452 verified complex files/hour</b>; the allocation-time-weighted rate is <b>1,535/hour</b>, 8.8% above the four-inference-node trial.</p><p>Observed overall GPU busy time was <b>19.6% and 16.4%</b>, with a combined observed-node-minute mean of <b>18.0%</b>. The historical 1:19 campaign measured 6.6%. The repeat-to-repeat variation remains visible below.</p></div><p class="note">A complex file is one predicted protein complex, not an independently validated binder. Each complete crop or expanded-context trial contributes five cofold models; expanded trials additionally include minimization and static MM/GBSA. The same input/seed submitted repeatedly is counted once per campaign. These are adaptive operational campaigns with different seeds and output mixes, not a fixed-work strong-scaling experiment.</p>')
    rows = []
    for r in runs:
        t = r['trial']
        rows.append([e(r['label']), number((r['end']-r['start'])/60), str(t['unique_tested_hypotheses']), number(t['distinct_trial_complex_files_per_science_allocation_hour'], 0),
                     number(t['distinct_sequences_per_science_allocation_hour']), number(t['distinct_intact_trials_per_science_allocation_hour']),
                     number(t['retained_intact_trials_per_science_allocation_hour']), number(t['overall_observed_GPU_busy_percent'])+'%', str(t['reviewed_hypothesis_rounds'])+'/36'])
    html_parts.append(table(['Inference:science', 'Allocation min', 'Hypotheses tested¹', 'Complex files/h', 'Distinct sequences/h', 'Expanded trials/h', 'Contact-retaining trials/h', 'GPU busy', 'Fully reviewed'], rows))
    html_parts.append('<p class="note">¹ Unique hypotheses with a complete independently audited screen; round evaluations can revisit a hypothesis. The initial 8:12 campaign had 36/36 fully reviewed evaluations; the repeat had 33/36. Failed/blocked threads, duplicate submissions and all allocated costs remain in the underlying ledger.</p></section>')
    html_parts.append(section('throughput', 'Throughput across inference:science ratios', 'Campaign-level measurements, time-block distributions and output-type rates answer different questions. The time-block plots describe burstiness; only campaign-level points show repeat-to-repeat variation.'))
    html_parts.append(section('binders', 'What did the binders look like?', 'Confidence, predicted contact geometry, pose consistency and static energy are complementary computational endpoints. All distributions below use retained complete input trials; they include low-confidence and lost-contact outcomes.'))
    html_parts.append(section('hardware', 'Where did the GPUs spend their time?', 'Compare science and inference roles, utilization distributions, and balance between physical nodes. GPU busy time is sampled NVIDIA utilization.gpu, separate from memory occupancy, worker leases or FLOPS efficiency.'))
    html_parts.append('<section id="nodes"><div class="section-heading"><span>NODE TELEMETRY</span><h2>Every node. Every GPU. Every observed minute.</h2><p>Select a campaign to inspect its 20 physical nodes. Expand a figure to read individual node names and GPU traces. Missing telemetry stays gray or breaks a trace.</p></div><div class="controls"><label for="node-select">Campaign </label><select id="node-select">'+''.join('<option value="'+r['id']+'"'+(' selected' if r==runs[-1] else '')+'>'+e(r['label'])+'</option>' for r in runs)+'</select></div>')
    for r in runs:
        t = r['trial'];coverage = t['telemetry_node_minute_coverage']
        html_parts.append('<div id="nodes-'+r['id']+'" class="node-panel '+('active' if r==runs[-1] else '')+'"><p class="note"><b>'+e(r['label'])+'</b> · '+str(r['inference'])+' inference / '+str(r['science'])+' science nodes · '+number(100*coverage['fraction'], 2)+'% observed node-minute coverage · '+number((r['end']-r['start'])/60)+' minutes.</p><div class="chart-grid">'+''.join(figure(key) for key, f in atlas.figures.items() if f['section']==r['id'])+'</div>')
        node_rows = [[e(n['host']), e(n['role']), number(n['mean_GPU_busy_percent']), number(n['p95_node_minute_GPU_busy_percent']), number(100*n['coverage_fraction'], 2)+'%', number(n['mean_CPU_busy_percent']), number(n['mean_host_memory_percent']), number(n['mean_GPU_memory_percent'])] for n in data['nodes'] if n['campaign']==r['label']]
        html_parts.append('<details><summary>Exact per-node statistics</summary>'+table(['Node', 'Role', 'Mean GPU busy %', 'P95 node-minute busy %', 'Coverage', 'CPU busy %', 'Host RAM %', 'GPU memory %'], node_rows)+'</details></div>')
    html_parts.append('</section>')
    html_parts.append(section('inference', 'Inference, scientific timing and overhead', 'Observed campaign latency mixes changing contexts and tasks. The frozen-request benchmark separately fixes request content. Scientific stage wall times, repeated submissions and failed row leases show where further overhead remains.'))
    html_parts.append('<section id="explorer"><div class="section-heading"><span>SCIENTIFIC RESULTS</span><h2>Expanded-context binder explorer</h2><p>All '+str(len(data['full']))+' complete distinct expanded-context trials. Click a column heading to sort, or filter by campaign, round, condition or minimized contacts. Default ordering is expanded cofold ipTM, a confidence score rather than measured affinity. Sequences, coordinates and source paths are preserved in the downloadable CSV.</p></div><div class="controls"><select id="binder-run"><option value="">All campaigns</option>'+''.join('<option value="'+r['id']+'">'+e(r['label'])+'</option>' for r in runs)+'</select><select id="binder-round"><option value="">All rounds</option><option>1</option><option>2</option><option>3</option></select><select id="binder-arm"><option value="">All conditioning arms</option>'+''.join('<option value="'+arm+'">'+arm.replace('_', ' ')+'</option>' for arm in ARMS)+'</select><label><input id="binder-retained" type="checkbox" style="min-width:0"> Both minimized contact regions</label><input id="binder-search" placeholder="Search sequence, binder or hypothesis" aria-label="Search binders"><span id="binder-count"></span></div><div class="table-wrap binder-scroll"><table id="binder-table"><thead><tr>')
    headers = ['Campaign', 'Round', 'Binder', 'Condition', 'Crop ipTM', 'Expanded ipTM', 'MM/GBSA kcal/mol', 'Min. contact residues R1/R2', 'Pose drift Å', 'Pairs <2 Å', 'Hypothesis', 'Sequence']
    html_parts.append(''.join('<th data-column="'+str(i)+'">'+e(header)+'</th>' for i, header in enumerate(headers))+'</tr></thead><tbody>')
    for c in sorted(data['full'], key=lambda c: c['expanded_best_iptm'], reverse=True):
        retained = c['minimized_both_contact_regions'];r1, r2 = c['minimized_region1_contact_residues'], c['minimized_region2_contact_residues']
        cells = [e(c['campaign']), str(c['round']), '<code>'+c['binder_id']+'</code>', e(c['condition']), number(c['source_crop_iptm'], 3), number(c['expanded_best_iptm'], 3), number(c['static_MMGBSA_kcal_mol'], 2),
                 '<span class="pill '+('' if retained else 'lost')+'">'+str(r1)+' / '+str(r2)+'</span>', number(c['minimized_pose_RMSD_to_source_A'], 2), str(c['minimized_close_atom_pairs_lt2A']),
                 '<div class="hypothesis" title="'+e(c['hypothesis_id'])+'">'+e(c['hypothesis'])+'</div>', '<div class="sequence">'+e(c['sequence'])+'</div>']
        numeric = {1: c['round'], 4: c['source_crop_iptm'] if c['source_crop_iptm'] is not None else -1, 5: c['expanded_best_iptm'], 6: c['static_MMGBSA_kcal_mol'], 7: min(r1, r2), 8: c['minimized_pose_RMSD_to_source_A'], 9: c['minimized_close_atom_pairs_lt2A']}
        html_parts.append('<tr data-run="'+c['run_id']+'" data-round="'+str(c['round'])+'" data-arm="'+c['condition']+'" data-retained="'+str(int(retained))+'">'+''.join('<td'+(' data-value="'+str(numeric[i])+'"' if i in numeric else '')+'>'+cell+'</td>' for i, cell in enumerate(cells))+'</tr>')
    html_parts.append('</tbody></table></div><p class="note">Contact residues use a 4 Å heavy-atom cutoff. “Pairs &lt;2 Å” reports close interchain heavy-atom pairs after minimization; interpretation remains geometry-dependent. Pose drift aligns the target crop and compares binder Cα positions with the exact source screen. No new ipTM is computed after minimization.</p></section>')
    html_parts.append('<section id="rounds"><div class="section-heading"><span>TOURNAMENTS</span><h2>Hypothesis steering over three rounds</h2><p>Top three steering-priority hypotheses per campaign/round. Priority combines evidence, Elo and uncertainty, so a high priority can mean an unresolved or contradictory hypothesis worth testing. These are exploration priorities, not validated biological winners.</p></div>')
    for r in runs:
        html_parts.append('<details><summary>'+e(r['label'])+' · three tournament/science/review rounds</summary>')
        for n in [1, 2, 3]:
            a = r['audit']['rounds'][str(n)]
            rr = [p for p in data['steering'] if p['run_id']==r['id'] and p['round']==n]
            html_parts.append('<h3>Round '+str(n)+' · '+str(a['reviewed'])+'/12 fully reviewed · '+str(a['accepted_owned_artifact_claims'])+' accepted owned-artifact claims</h3>'+table(['Hypothesis summary', 'Priority', 'Elo', 'Evidence rating', 'Supports / contradictions', 'Next action'], [[e(p['summary']), number(p['priority'], 3), number(p['elo'], 0), number(p['evidence_rating'], 0), str(p['independent_supports'])+' / '+str(p['independent_contradictions']), e(p['recommended_action'].replace('_', ' '))] for p in rr]))
        html_parts.append('</details>')
    html_parts.append('</section><section id="methods"><div class="section-heading"><span>METHODS & EXPORTS</span><h2>Reproducible measurements, explicit scope</h2></div><ul class="method-list"><li><b>Scientific provenance.</b> All five campaigns completed three rounds and passed their required independent scientific audits. Coordinates, sequences, scores, target topology, geometry and energy-file values were checked against immutable receipts. Noncanonical successful inputs are excluded from positive output; attempts and costs remain.</li><li><b>Repeat handling.</b> Retain the earliest complete successful screen per condition/input SHA/seed/target sequence, and follow-up per binder sequence/seed/target length. Confidence distributions and leaderboards use that same retained set. Independent seeds remain distinct trials. Cross-campaign totals do not imply globally novel sequences.</li><li><b>Rates and distributions.</b> Rates divide output by full science-allocation elapsed time, including bootstrap, repairs and final draining. The block distribution uses only complete five-minute intervals and has no claim of independent replicates. The exact terminal time is reconstructed from lease start plus rounded receipt seconds.</li><li><b>Controlled versus operational.</b> The 1:19 historical reference includes controller repairs/inference renewal; 2:18 includes compatibility repairs. Adaptive selections, seeds and output mixes vary. Only 8:12 has two campaigns. The frozen inference benchmark fixes request content and concurrency.</li><li><b>GPU telemetry.</b> One-minute means are derived from roughly ten-second NVIDIA samples, four GPUs per node. Gray or missing values stay missing. Role averages use observed node-minutes; individual-node summaries describe minute means. Memory residency, GPU leases, GPU busy time and FLOPS efficiency are different quantities.</li><li><b>Computational binder metrics.</b> Five-model Chai cofolding supplies ipTM and pose consistency. Target expansion uses EGFR residues 25–646 with 25 disulfides. Single minimized-frame MM/GBSA omits entropy, glycans and membrane. Confidence/contact/energy outputs are computational screening evidence, not experimental binding affinity.</li><li><b>Model/execution controls.</b> Prior nonthinking agent roles and thinking-default tournament generation/ranking were preserved, along with scientific recipes, nominal candidate budgets, role concurrency and the 24-row science pool. Native inference continuously batches up to 16 sequences. No Argo/Tara requests are needed for this local Nemotron route.</li><li><b>Costs and cleanup.</b> The four new campaigns and their shared inference allocations consumed '+number(source['unique_job_allocated_GPU_hours'], 1)+' allocated GPU-hours in the unique-job ledger; the historical baseline is separate. Each shared model allocation is charged once, including cold loading/idle time. All experiment-owned jobs and gateway were released. Other jobs, endpoints and tunnels were left untouched.</li></ul><h3>Download the plotted measurements</h3><p class="note">This HTML embeds every plot and data download. The companion archive additionally contains standalone SVG/PNG figures, CSVs, provenance and the report-builder source. No new jobs or inference requests were submitted to make this report.</p><div class="data-links">')
    for name in data['datasets']:
        p = atlas.output/'data'/name;mime = 'text/csv' if p.suffix=='.csv' else 'application/json' if p.suffix=='.json' else 'text/plain'
        html_parts.append('<a href="'+data_url(p, mime)+'" download="'+name+'">'+e(name)+' ↓</a>')
    html_parts.append('</div></section><footer>Generated '+time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())+' from completed campaign artifacts. '+str(len(atlas.figures))+' figures · '+str(sum(len(r['screens']) for r in runs))+' retained generation trials · '+str(sum(len(r['full']) for r in runs))+' retained expanded-context trials. Source audit and telemetry hashes are in provenance.json.</footer></main><dialog id="figure-dialog"><button class="close" autofocus>Close ✕</button><h3></h3><img alt="Expanded scientific figure"></dialog><script>\n')
    html_parts.append('''
const dialog=document.getElementById('figure-dialog');
document.querySelectorAll('.expand').forEach(button=>button.addEventListener('click',()=>{const img=document.getElementById(button.dataset.image);dialog.querySelector('img').src=img.src;dialog.querySelector('h3').textContent=img.alt;dialog.showModal()}));
document.querySelectorAll('.figure-download').forEach(link=>link.addEventListener('click',()=>{link.href=document.getElementById(link.dataset.image).src}));
dialog.querySelector('.close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close()});
document.getElementById('node-select').addEventListener('change',event=>{document.querySelectorAll('.node-panel').forEach(panel=>panel.classList.toggle('active',panel.id==='nodes-'+event.target.value))});
const binderTable=document.getElementById('binder-table'),binderRows=Array.from(binderTable.tBodies[0].rows);
function filterBinders(){const run=document.getElementById('binder-run').value,round=document.getElementById('binder-round').value,arm=document.getElementById('binder-arm').value,retained=document.getElementById('binder-retained').checked,search=document.getElementById('binder-search').value.toLowerCase();let count=0;binderRows.forEach(row=>{const show=(!run||row.dataset.run===run)&&(!round||row.dataset.round===round)&&(!arm||row.dataset.arm===arm)&&(!retained||row.dataset.retained==='1')&&(!search||row.textContent.toLowerCase().includes(search));row.hidden=!show;if(show)count++});document.getElementById('binder-count').textContent=count+' / '+binderRows.length+' trials'}
['binder-run','binder-round','binder-arm','binder-retained'].forEach(id=>document.getElementById(id).addEventListener('change',filterBinders));document.getElementById('binder-search').addEventListener('input',filterBinders);filterBinders();
let sortColumn=5,sortDirection=-1;
binderTable.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{const column=Number(th.dataset.column);sortDirection=sortColumn===column?-sortDirection:-1;sortColumn=column;const rows=Array.from(binderTable.tBodies[0].rows);rows.sort((a,b)=>{const ac=a.cells[column],bc=b.cells[column];return sortDirection*(ac.dataset.value!==undefined?Number(ac.dataset.value)-Number(bc.dataset.value):ac.textContent.localeCompare(bc.textContent))});rows.forEach(row=>binderTable.tBodies[0].appendChild(row))}));
</script></body></html>''')
    (atlas.output/'index.html').write_text(''.join(html_parts))


def verify_and_package(atlas, source, runs, comparison):
    assert all(r['audit']['required_campaign_completion_checks_passed'] for r in runs)
    total = sum(r['trial']['verified_files_on_distinct_input_trials'] for r in runs)
    assert sum(sum(e['files'] for e in r['events']) for r in runs) == total
    assert all(abs(np.nanmean(r['arrays']['busy'])-r['trial']['overall_observed_GPU_busy_percent'])<.05 for r in runs)
    report = (atlas.output/'index.html').read_text()
    assert report.count('<figure ') == len(atlas.figures)
    assert report.count('<tr data-run=') == sum(len(r['full']) for r in runs)
    # Check actual private pool credentials without writing or displaying values.
    experiment = Path(runs[1]['progress']['inference_pool_experiment'])
    secret_files = [experiment/'pool-token', *experiment.glob('replicas/*/state/nemotron/*/token')]
    secrets = [p.read_text().strip() for p in secret_files if p.is_file()]
    for p in atlas.output.rglob('*'):
        if p.suffix in ('.html', '.json', '.csv', '.svg', '.py', '.txt'):
            text = p.read_text()
            assert not any(s in text for s in secrets if len(s)>=16), 'Private credential found in report artifact'
    result = {'generated_at': time.time(), 'figures': len(atlas.figures), 'campaigns': len(runs), 'rounds': sum(len(r['round_records']) for r in runs),
              'screen_input_trials': sum(len(r['screens']) for r in runs), 'expanded_input_trials': sum(len(r['full']) for r in runs),
              'verified_complex_files_on_distinct_input_trials': total, 'all_campaign_audits_passed': True,
              'telemetry_reproduces_source_means_within_percent_points': .05, 'five_minute_blocks_conserve_output': True,
              'known_private_credentials_checked': len(secrets), 'credential_values_found': False,
              'html_self_contained_images': True, 'html_sha256': digest(atlas.output/'index.html'),
              'source_comparison_sha256': digest(comparison), 'read_only_report_generation': True}
    write_json(atlas.output/'verification.json', result)
    write_json(atlas.output/'figure-manifest.json', list(atlas.figures.values()))
    shutil.copyfile(Path(__file__), atlas.output/'build_inference_atlas.py')
    (atlas.output/'README.txt').write_text('Jupiter binder and inference atlas\n\nOpen index.html in a browser. All scientific plots and CSV downloads are embedded in the HTML.\nStandalone SVG/PNG figures are under assets/; underlying CSV/JSON/FASTA data are under data/.\nverification.json records accounting/data-integrity checks; data/provenance.json records original source hashes.\n\nRebuild on Jupiter from the repository:\n/e/project1/deepwing/meigooni1/agent/software/RFD4-Proteina/.venv/bin/python scripts/build_inference_atlas.py --comparison reports/inference-balance-20261113/comparison.json --output reports/inference-atlas-20261114\n\nThe builder reads completed artifacts and submits no jobs or inference calls. Original campaign data are required to rebuild.\n')
    archive = atlas.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for p in sorted(atlas.output.rglob('*')):
            if p.is_file():zipped.write(p, str(Path(atlas.output.name)/p.relative_to(atlas.output)))
    with zipfile.ZipFile(archive) as zipped:assert zipped.testzip() is None
    print(json.dumps({**result, 'html': str(atlas.output/'index.html'), 'archive': str(archive),
                      'HTML_MiB': round((atlas.output/'index.html').stat().st_size/2**20, 2),
                      'ZIP_MiB': round(archive.stat().st_size/2**20, 2)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=Path('reports/inference-balance-20261113/comparison.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/inference-atlas-20261114'))
    args = parser.parse_args();source, runs = load_runs(args.comparison)
    print('Loaded five independently audited campaigns; preparing telemetry', flush=True)
    for r in runs:
        load_telemetry(r);print('Telemetry verified: '+r['label'], flush=True)
    load_requests(source, runs)
    atlas = Atlas(args.output.resolve(), runs)
    for function in [plot_throughput, plot_binders, plot_hardware, plot_inference_and_timing]:
        print('Rendering '+function.__name__, flush=True)
        function(atlas, source, runs) if function in [plot_throughput, plot_inference_and_timing] else function(atlas, runs)
    data = export_data(atlas, source, runs, args.comparison)
    render_html(atlas, source, runs, data)
    verify_and_package(atlas, source, runs, args.comparison)


if __name__ == '__main__':
    main()
