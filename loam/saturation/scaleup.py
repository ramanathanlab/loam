"""Stage a larger successor to a single-allocation trial, only once the trial has passed its saturation gate.

    loam-scaleup TRIAL --nodes 256 --label unified-256 [--output-root PATH] [--check-only | --submit]

TRIAL is a completed, owned trial job whose `saturation-audit.json` (`loam-saturation`) passed
the principle gate. The successor keeps the trial's per-science-node density: admission,
and tournament width scale with science nodes; the inference : science
ratio, GPU sharing and generation batch size stay the same, and its inputs must hash identically. Walltime is the
trial's measured allocation with headroom. Nothing is submitted without --submit. Ported from
the moeen branch's stage_saturation_scaleup.py; the trial size is no longer fixed at 20 nodes,
nor the successor at 256. Moeen's checks of `tournament-audit.json`, `scientific-protocol.json`
and per-role model settings stay there with the tools that write them; develop writes none.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from ..config import load
from ..staging import check, read, require, sha, stage, submit
from .storage import LEDGER, experiment_control_root, output_root


def qualify(trial: Path, nodes: int) -> tuple[dict, dict, dict, Path]:
    """The trial's plan, receipt and audit summary, if it qualifies a successor of `nodes` nodes."""
    root = experiment_control_root(trial)
    receipt, plan = read(trial / 'submission.json'), read(trial / 'plan.json')
    require(receipt['job_id'] in read(root / LEDGER)['owned_job_ids'],
            f'Trial job {receipt["job_id"]} is not owned by the experiment at {root}')
    if nodes <= plan['nodes']:
        raise ValueError(f'A successor must be larger than the {plan["nodes"]}-node trial')
    audit_path = trial / 'saturation-audit.json'
    require(audit_path.exists(), 'The saturation audit of the trial is not available; no larger job staged or submitted')
    audit = read(audit_path)
    require(audit['principle_gate_passed'] and all(audit['principle_gate'].values()),
            'The trial\'s principle gate failed; no larger job staged or submitted')
    check(trial)
    summary = audit['current']['summary']
    require(summary['job_id'] == receipt['job_id'], 'The saturation audit is of another job')
    require(summary['scientific_completion_passed'], 'The trial campaign did not complete its required checks')
    require(summary['recorded_rounds'] == summary['clean_synthesis_rounds'] == 3,
            'The trial did not complete three clean synthesis rounds')
    science = read(trial / 'campaign/scaling-audit.json')
    require(science['model_coordinates_independently_checked'] and science['required_campaign_completion_checks_passed'],
            'The trial\'s scientific audit did not pass')
    require(not science['errors'] and not science['unowned_or_missing_claim_sources'],
            'The trial\'s scientific audit has errors or unowned claim sources')
    lineage = science.get('verification_lineage', {})
    for name in ('numeric_audit', 'claim_audit'):
        if name in lineage:
            require(sha(lineage[name]) == lineage[name + '_sha256'], f'The {name} changed since the audit')
    for n in range(1, 4):
        record = read(trial / f'campaign/state/rounds/round-{n:03d}.json')
        manifest_path = Path(record['brief']['synthesis_receipt'])
        manifest = read(manifest_path)
        require(manifest['status'] == 'complete' and manifest['thread_count'] == len(record['threads']),
                f'Round {n} synthesis did not cover every thread')
        require(sha(manifest_path.parent / 'input.json') == manifest['input_sha256'], f'Round {n} synthesis input changed')
        for call in manifest['calls']:
            require(call['status'] == 'complete' and call['prompt_bytes'] <= manifest['prompt_budget_bytes'] <= 64000,
                    f'Round {n} synthesis call {call["node"]} did not complete within its budget')
            prompt = read(manifest_path.parent / f'{call["node"]}-request.json')['prompt']
            require(len(prompt.encode()) == call['prompt_bytes'], f'Round {n} synthesis request {call["node"]} changed')
            require(sha(manifest_path.parent / f'{call["node"]}-response.json') == call['response_sha256'],
                    f'Round {n} synthesis response {call["node"]} changed')
    return plan, receipt, summary, audit_path


def successor_settings(trial: Path, plan: dict, summary: dict, nodes: int) -> dict:
    """Staging settings for `nodes` nodes at the trial's per-science-node density."""
    cfg = load(trial / 'campaign/campaign.yaml')
    inference = round(nodes * plan['inference_count'] / plan['nodes'])
    factor = (nodes - inference) / plan['science_count']

    def scaled(value):
        return math.ceil(value * factor)
    # Headroom for larger service startup and dispatch: a requested upper bound, not a prediction.
    hours = max(6, math.ceil((summary['final_allocation_seconds'] * 1.5 + 1800) / 3600))
    if hours > 99:
        raise ValueError(f'The trial ran too long to size a successor walltime from ({hours} hours with headroom)')
    return dict(nodes=nodes, inference=inference, concurrency=scaled(cfg.concurrency),
                top_k=scaled(cfg.tournament.top_k), initial_hypotheses=scaled(cfg.tournament.initial_hypotheses),
                new_per_round=scaled(cfg.tournament.new_per_round),
                tournament_workers=math.ceil(cfg.tournament.workers * inference / plan['inference_count']),
                science_workers_per_gpu=cfg.execution.hpc.science_workers_per_gpu,
                generation_batch_size=cfg.tournament.gen_batch_size, walltime=f'{hours:02d}:00:00',
                controller_gc_threshold0=plan.get('controller_gc_threshold0'))


def prepare(trial: Path, nodes: int, label: str, *, check_only: bool = False, submit_job: bool = False,
            storage_root: Path | None = None):
    """Stage (and with `submit_job`, submit) the successor of a qualifying trial. Returns its job directory,
    or with `check_only` just the settings it would be staged with."""
    trial = Path(trial).resolve()
    root = experiment_control_root(trial)
    plan, receipt, summary, audit_path = qualify(trial, nodes)
    settings = successor_settings(trial, plan, summary, nodes)
    destination = Path(storage_root).resolve() if storage_root else trial.parent
    print(json.dumps({'qualified_trial_job': receipt['job_id'], 'successor_settings': settings,
                      'check_only': check_only, 'output_root': str(destination),
                      'ownership_ledger': str(root / LEDGER)}, indent=2), flush=True)
    if check_only:
        return settings
    alias = root / label
    if destination != root and (alias.exists() or alias.is_symlink()):
        raise RuntimeError(f'Refuse to replace an existing job or output alias: {alias}')
    destination = output_root(root, destination)
    job = stage(trial / 'campaign/campaign.yaml', destination, label=label,
                inference_profile=trial / 'inference-profile.json', inputs=trial / 'campaign/inputs',
                source=Path(plan['runtime_repo']), allow_unmeasured=not plan.get('recipe_measured', True),
                **settings)
    if destination != root:
        alias.symlink_to(job, target_is_directory=True)
    require(read(job / 'inputs-integrity-preflight.json')['input_sha256']
            == read(trial / 'inputs-integrity-preflight.json')['input_sha256'],
            'The successor\'s inputs differ from the qualified trial\'s')
    successor = read(job / 'plan.json')
    successor.update(qualified_trial_job=receipt['job_id'], qualification_audit=str(audit_path),
                     qualification_audit_sha256=sha(audit_path),
                     qualification_scientific_audit_sha256=sha(trial / 'campaign/scaling-audit.json'),
                     measured_trial_allocation_seconds=summary['final_allocation_seconds'], qualified_at=time.time(),
                     ownership_ledger=str(root / LEDGER), control_alias=str(alias))
    (job / 'plan.json').write_text(json.dumps(successor, indent=2) + '\n')
    (job / 'preflight.json').write_text(json.dumps(check(job), indent=2) + '\n')
    if submit_job:
        submit(destination, job)
    return job


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='loam-scaleup', description=__doc__.split('\n\n')[0])
    parser.add_argument('trial', type=Path, help='the qualifying trial job directory')
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output-root', type=Path,
                        help='another filesystem for the job; it shares the experiment\'s ownership ledger')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='print the successor settings; stage nothing')
    mode.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    prepare(args.trial, args.nodes, args.label, check_only=args.check_only, submit_job=args.submit,
            storage_root=args.output_root)


if __name__ == '__main__':
    main()
