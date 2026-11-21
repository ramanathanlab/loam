"""Single-allocation staging: one job directory that owns a whole campaign, and its one submission.

    loam-stage REFERENCE ROOT --label trial-20 --nodes 20 --inference-profile vllm.json [--submit]

stages a fresh job under ROOT/LABEL from REFERENCE, an ordinary campaign config, and submits it
only with --submit. The job runs `loam.coordinator` inside one allocation of NODES nodes: the
scaling recipe (loam.scaling) splits them into inference and science roles and sizes the
campaign. Account and partition are the campaign config's, which the Site profile fills in
where the config leaves them out. Ported from the moeen branch's stage_single_allocation.py and
check_single_allocation.py.

A staged job directory holds:

- `campaign/`: the campaign config, goal, inputs, state and work, every path bound to the job
  (`loam.allocation.validate_campaign_paths`), so it never resumes another campaign's state.
- `infra/control-runtime/`: a frozen copy of loam and its skills, so edits to the checkout
  while the job waits in the queue cannot change what it runs. `prepared-control-source.json`
  seals it, and `inputs-integrity-preflight.json` seals the inputs.
- `inference-profile.json`: how each inference node serves the model (see `InferenceProfile`).
- `plan.json`, `job.sh`, `preflight.json`; after submission `submission.json`.

ROOT holds `experiment.json`, the ledger of job ids this experiment owns (loam.saturation.storage).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import Field

from .allocation import validate_campaign_paths
from .config import ConfigModel, NonEmptyStr, PositiveInt, load
from .saturation.storage import LEDGER
from .scaling import recipe

REPO = Path(__file__).resolve().parents[1]
RUNTIME_FOLDERS = ('loam', 'skills')
RUNTIME_FILES = ('pyproject.toml', 'uv.lock')
SERVER_SCRIPT = 'skills/hpc-deployment/scripts/mcp_server.py'
LABEL = re.compile(r'[a-z0-9-]+')
WALLTIME = re.compile(r'\d{2}:[0-5]\d:[0-5]\d')


class InferenceProfile(ConfigModel):
    """How each inference node serves one model over the OpenAI-compatible API.

    `command` runs once on each inference node, with `{host}` and `{port}` replaced by that node's
    name and `port`; it must serve `<model>` at http://{host}:{port}/v1. It runs without a shell;
    `${LOAM_SOFTWARE_ROOT}` and the Site profile's other roots are filled in at staging. The coordinator puts
    every replica behind one `loam.inference_pool`, and points each campaign model named `model`
    at the pool.
    """

    model: NonEmptyStr
    command: list[NonEmptyStr] = Field(min_length=1)
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    checkpoint: NonEmptyStr | None = None
    startup_timeout: PositiveInt = 5400


def write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + '\n')


def read(path: Path):
    return json.loads(Path(path).read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def python_executable() -> Path:
    """The interpreter the job runs loam with: LOAM_PYTHON, else the checkout's virtualenv."""
    return Path(os.environ.get('LOAM_PYTHON') or REPO / '.venv/bin/python').absolute()


def freeze_runtime(infra: Path) -> Path:
    """Copy loam's source (never environments, weights or results) into the job."""
    target = infra / 'control-runtime'
    target.mkdir()
    for folder in RUNTIME_FOLDERS:
        shutil.copytree(REPO / folder, target / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in RUNTIME_FILES:
        if (REPO / name).is_file():
            shutil.copyfile(REPO / name, target / name)
    return target


def source_hashes(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def expand_roots(text: str, roots: dict[str, str]) -> str:
    """`text` with each `${NAME}` of a Site profile root replaced; nothing else is expanded or run."""
    for name, value in roots.items():
        text = text.replace('${' + name + '}', value)
    return text


def rebase(path: Path, source: Path, runtime: Path) -> str:
    """`path` inside the frozen runtime if it lies in the `source` checkout, else unchanged."""
    path = Path(path).resolve()
    return str(runtime / path.relative_to(source)) if path.is_relative_to(source) else str(path)


def stage(reference: Path, root: Path, *, label: str, nodes: int, inference_profile: Path,
          inference: int | None = None, walltime: str | None = None, reservation: str = '',
          inputs: Path | None = None, seed_hypotheses: Path | None = None,
          science_workers_per_gpu: int | None = None, concurrency: int | None = None,
          top_k: int | None = None, initial_hypotheses: int | None = None, new_per_round: int | None = None,
          tournament_workers: int | None = None, generation_batch_size: int | None = None,
          tournament_max_seconds: float | None = None, controller_gc_threshold0: int | None = None,
          after: str | None = None, allow_unmeasured: bool = False, source: Path = REPO) -> Path:
    """Stage ROOT/LABEL from the campaign config `reference`; never submit. Returns the job directory.

    Settings left as None come from the scaling recipe for `nodes` nodes of the reference's System.
    `source` is the checkout the reference's rules and skills live in, rebased into the frozen
    runtime. `after` is an owned job this one waits for (afterany).
    """
    reference = Path(reference).absolute()
    cfg = load(reference)
    hpc = cfg.execution.hpc
    if cfg.execution.mode != 'hpc' or hpc is None:
        raise ValueError(f'{reference} is not an HPC campaign config; there is no allocation to stage')
    if not LABEL.fullmatch(label):
        raise ValueError('label must contain only lowercase letters, digits and hyphens')
    walltime = walltime or hpc.allocation.walltime
    if not WALLTIME.fullmatch(walltime):
        raise ValueError('walltime must have format HH:MM:SS')
    if controller_gc_threshold0 is not None and (type(controller_gc_threshold0) is not int
                                                 or not 700 <= controller_gc_threshold0 <= 100000):
        raise ValueError('controller_gc_threshold0 must be an integer in [700, 100000]')
    profile = InferenceProfile.model_validate(read(inference_profile))
    roots = hpc.site_profile.roots() if hpc.site_profile else {}
    profile.command = [expand_roots(part, roots) for part in profile.command]
    served = [role for role, spec in cfg.models.items() if spec.model == profile.model]
    if not served:
        raise ValueError(f'No campaign model is {profile.model!r}, the model the inference nodes serve')
    system = hpc.system_definition()
    sized = recipe(system, nodes, inference=inference, science_workers_per_gpu=science_workers_per_gpu,
                   allow_unmeasured=allow_unmeasured)
    concurrency = concurrency or sized.concurrency
    top_k = top_k or sized.top_k
    initial_hypotheses = initial_hypotheses or sized.initial_hypotheses
    new_per_round = new_per_round or sized.new_per_round
    tournament_workers = tournament_workers or sized.tournament_workers
    generation_batch_size = generation_batch_size or sized.generation_batch_size
    if not 0 < concurrency <= top_k <= initial_hypotheses:
        raise ValueError('Require 0 < concurrency <= top_k <= initial_hypotheses')

    root = Path(root).absolute()
    job = root / label
    if job.exists() or job.is_symlink():
        raise RuntimeError(f'Refuse to overwrite an existing staged job: {job}')
    root.mkdir(parents=True, exist_ok=True)
    if not (root / LEDGER).exists():
        write(root / LEDGER, {'created_at': time.time(), 'purpose': 'Single-allocation campaigns',
                              'owned_job_ids': [], 'reference': str(reference)})
    job.mkdir(mode=0o700)
    (job / '.gitignore').write_text('*\n')
    campaign = job / 'campaign'
    campaign.mkdir()
    infra = job / 'infra'
    infra.mkdir(mode=0o700)
    runtime = freeze_runtime(infra)
    source = Path(source).resolve()
    if inputs:
        inputs = Path(inputs).resolve(strict=True)
        shutil.copytree(inputs, campaign / 'inputs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    else:
        (campaign / 'inputs').mkdir()
    (campaign / 'goal.md').write_text(cfg.goal_text())
    for folder in ('state', 'work'):
        (campaign / folder).mkdir()

    raw = yaml.safe_load(reference.read_text())
    raw.update(goal=str(campaign / 'goal.md'), state_dir=str(campaign / 'state'),
               rules=[rebase(p, source, runtime) for p in cfg.rules],
               skills_dirs=[rebase(p, source, runtime) for p in cfg.skills_dirs],
               concurrency=concurrency, synthesis_concurrency=sized.synthesis_concurrency, rounds=sized.rounds)
    raw.pop('workspace_inputs', None)
    raw['tournament'].update(jnana_root=str(cfg.tournament.jnana_root), top_k=top_k,
                             initial_hypotheses=initial_hypotheses, new_per_round=new_per_round,
                             gen_batch_size=generation_batch_size, workers=tournament_workers,
                             scheme='swiss', swiss_rounds=sized.swiss_rounds)
    if tournament_max_seconds:
        raw['tournament']['max_seconds'] = tournament_max_seconds
    execution = raw['execution']
    execution['workdir'] = str(campaign / 'work')
    deployment = execution['hpc']
    deployment.pop('endpoint_id', None)
    deployment.update(launch_mode='existing', base_dir=str(campaign), worker_code_dir=str(runtime),
                      server_script=str(runtime / SERVER_SCRIPT),
                      inspect_roots=list(dict.fromkeys([str(campaign), *hpc.inspect_roots])),
                      science_workers_per_gpu=sized.science_workers_per_gpu,
                      campaign_id=f'single-{label}-{uuid.uuid4().hex[:10]}')
    deployment['allocation'].update(nodes=sized.science, walltime=walltime)
    (campaign / 'campaign.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    staged = load(campaign / 'campaign.yaml')
    allocation = staged.execution.hpc.allocation

    seeds = seed_state(campaign, seed_hypotheses, staged.goal_text())
    hashes = source_hashes(campaign / 'inputs')
    write(job / 'inputs-integrity-preflight.json', {'checked_at': time.time(), 'input_bundle_source': str(inputs),
                                                     'input_sha256': hashes})
    write(job / 'inference-profile.json', profile.model_dump())
    python = python_executable()
    plan = {
        'mode': 'single-allocation', 'system': system.name, 'nodes': nodes, 'inference_count': sized.inference,
        'science_count': sized.science, 'recipe_measured': sized.measured,
        'campaign': str(campaign), 'infra': str(infra), 'job_root': str(job),
        'job_name': f'loam-{label}-{uuid.uuid4().hex[:10]}', 'walltime': walltime, 'reservation': reservation,
        'account': allocation.account, 'partition': allocation.queue, 'created_at': time.time(),
        'gpus_per_node': system.gpus_per_node, 'cpus_per_node': len(system.cpu_ids),
        'runtime_repo': str(runtime), 'python': str(python), 'reference': str(reference),
        'inference_profile': str(job / 'inference-profile.json'), 'served_roles': served,
        'input_bundle_source': str(inputs) if inputs else None, 'seed_hypotheses_count': seeds,
        'science_workers_per_gpu': sized.science_workers_per_gpu, 'tournament_workers': tournament_workers,
        'synthesis_concurrency': sized.synthesis_concurrency, 'generation_batch_size': generation_batch_size,
        'admission': {'concurrency': concurrency, 'top_k': top_k, 'initial_hypotheses': initial_hypotheses,
                      'new_per_round': new_per_round},
    }
    if controller_gc_threshold0 is not None:
        plan['controller_gc_threshold0'] = controller_gc_threshold0
    if after:
        plan['afterany_owned_job_id'] = str(after)
    write(job / 'plan.json', plan)
    write(job / 'prepared-control-source.json', {
        'at': time.time(), 'runtime_repo': str(runtime), 'source_sha256': source_hashes(runtime),
        'inference_profile_sha256': sha(job / 'inference-profile.json')})
    write_job_script(job, runtime, python, staged.execution.hpc)
    write(job / 'preflight.json', check(job))
    return job


def seed_state(campaign: Path, seed_hypotheses: Path | None, goal: str) -> int:
    """Start Jnana's state from private seed hypotheses, if given, with ids scoped to this campaign."""
    if not seed_hypotheses:
        return 0
    seeds = []
    for i, seed in enumerate(read(seed_hypotheses)):
        seed = copy.deepcopy(seed)
        seed['hypothesis_id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{campaign}:{seed["hypothesis_id"]}'))
        seed.update(elo_rating=1200, reviews=[], tournament_matches=[], tournament_wins=0, tournament_losses=0,
                    last_tournament_time=None, created_at=time.time() + i * .001)
        seeds.append(seed)
    state = {key: {} for key in ('experiments', 'analyses', 'papers', 'agent_states', 'datasets',
                                 'training_plans', 'evaluations')}
    state.update(hypotheses=seeds, metadata={'research_goal': goal}, tournament_state={'matches': []})
    write(campaign / 'state' / 'jnana.json', state)
    return len(seeds)


def write_job_script(job: Path, runtime: Path, python: Path, deployment) -> None:
    """The batch script: the Site profile's environment, then the coordinator from the frozen runtime."""
    quote = shlex.quote
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail']
    if os.environ.get('LOAM_SITE_PROFILE'):
        lines.append(f'export LOAM_SITE_PROFILE={quote(str(Path(os.environ["LOAM_SITE_PROFILE"]).absolute()))}')
    if deployment.site_profile is not None:
        lines.append(deployment.site_profile.full_worker_init(deployment.allocation.endpoint_options.get('worker_init')))
    lines += [f'export LOAM_PYTHON={quote(str(python))}',
              f'export PYTHONPATH={quote(str(runtime))}:${{PYTHONPATH:-}}',
              f'cd {quote(str(runtime))}',
              f'exec {quote(str(python))} -u -m loam.coordinator {quote(str(job))}']
    script = job / 'job.sh'
    script.write_text('\n'.join(lines) + '\n')
    script.chmod(0o700)


def _require(condition: bool, problem: str) -> None:
    if not condition:
        raise RuntimeError(problem)


def verify_seals(job: Path, plan: dict, *, runtime: Path | None = None) -> None:
    """Fail if the staged inputs, the frozen runtime or the inference profile changed since staging."""
    campaign = Path(plan['campaign'])
    integrity = job / 'inputs-integrity-preflight.json'
    if integrity.exists():
        for name, digest in read(integrity)['input_sha256'].items():
            path = campaign / 'inputs' / name
            _require(path.is_file() and sha(path) == digest, f'Staged input changed since staging: {name}')
    seal = read(job / 'prepared-control-source.json')
    frozen = Path(seal['runtime_repo'])
    _require(frozen.resolve() == Path(plan['runtime_repo']).resolve(), 'The plan names another runtime than the seal')
    if runtime is not None:
        _require(Path(runtime).resolve() == frozen.resolve(), f'Running from {runtime}, not the frozen runtime {frozen}')
    for name, digest in seal['source_sha256'].items():
        path = frozen / name
        _require(path.is_file() and sha(path) == digest, f'Frozen runtime source changed since staging: {name}')
    _require(sha(plan['inference_profile']) == seal['inference_profile_sha256'],
             'The inference profile changed since staging')


def check(job: Path) -> dict:
    """Read-only preflight of a staged job: its seals, paths and sizing agree with its plan. Returns the proof."""
    job = Path(job).resolve()
    plan = read(job / 'plan.json')
    campaign = Path(plan['campaign'])
    cfg = load(campaign / 'campaign.yaml')
    bindings = validate_campaign_paths(cfg, campaign)
    verify_seals(job, plan)
    hpc = cfg.execution.hpc
    system = hpc.system_definition()
    _require(plan['mode'] == 'single-allocation' and plan['inference_count'] + plan['science_count'] == plan['nodes'],
             'The plan does not split its nodes into inference and science')
    _require(hpc.launch_mode == 'existing', 'A single-allocation campaign must attach to its allocation')
    _require(hpc.allocation.nodes == plan['science_count'], 'The campaign is not sized to the science nodes')
    _require(hpc.science_workers_per_gpu == plan['science_workers_per_gpu'], 'Science workers per GPU differ')
    _require(system.gpus_per_node == plan['gpus_per_node'], 'GPUs per node differ from the System')
    _require((hpc.allocation.account, hpc.allocation.queue) == (plan['account'], plan['partition']),
             'Account or partition differ from the plan')
    _require(cfg.tournament.workers == plan['tournament_workers'], 'Tournament workers differ')
    _require(cfg.synthesis_concurrency == plan['synthesis_concurrency'], 'Synthesis concurrency differs')
    admission = {'concurrency': cfg.concurrency, 'top_k': cfg.tournament.top_k,
                 'initial_hypotheses': cfg.tournament.initial_hypotheses,
                 'new_per_round': cfg.tournament.new_per_round}
    _require(admission == plan['admission'], 'Admission settings differ from the plan')
    profile = InferenceProfile.model_validate(read(plan['inference_profile']))
    _require(all(cfg.models[role].model == profile.model for role in plan['served_roles']),
             'A served role no longer names the served model')
    result = {'job_root': str(job), 'system': plan['system'], 'nodes': plan['nodes'],
              'inference': plan['inference_count'], 'science': plan['science_count'],
              'concurrent_hypotheses': cfg.concurrency, 'selected_hypotheses_per_round': cfg.tournament.top_k,
              'science_GPU_slots': system.gpus_per_node * plan['science_count'],
              'science_worker_slots': system.gpus_per_node * plan['science_count'] * hpc.science_workers_per_gpu,
              'control_submitter_agents': hpc.pool_size, 'recipe_measured': plan['recipe_measured'],
              'reservation': plan['reservation'], 'campaign_path_bindings': bindings}
    receipt = job / 'submission.json'
    if receipt.exists():
        submitted = read(receipt)
        _require(submitted['job_name'] == plan['job_name'], 'The submission receipt names another job')
        _require(submitted['script_sha256'] == sha(job / 'job.sh'), 'job.sh changed since submission')
        result['job_id'] = submitted['job_id']
    return result


def submit(root: Path, job: Path) -> str:
    """Submit this staged job exactly once, and record it in the experiment's ledger. Returns the job id."""
    root, job = Path(root).resolve(), Path(job).resolve()
    plan = read(job / 'plan.json')
    if (job / 'submission.json').exists() or (job / 'submission-intent.json').exists():
        raise RuntimeError('A submission was already attempted; reconcile its receipt before retrying')
    check(job)
    profile = InferenceProfile.model_validate(read(plan['inference_profile']))
    # Fail before the scheduler when this checkout's runtime has not been installed where the job runs.
    command = shutil.which(profile.command[0]) or profile.command[0]
    for name in (command, plan['python']):
        if not Path(name).is_file() or not os.access(name, os.X_OK):
            raise RuntimeError('Required runtime executable is missing: ' + name)
    dependencies = ()
    if plan.get('afterany_owned_job_id'):
        previous = plan['afterany_owned_job_id']
        if previous not in read(root / LEDGER)['owned_job_ids']:
            raise RuntimeError(f'Dependency {previous} must belong to this experiment')
        dependencies = (('afterany', previous),)
    cfg = load(Path(plan['campaign']) / 'campaign.yaml')
    system = cfg.execution.hpc.system_definition()
    args = system.batch_submission(script=str(job / 'job.sh'), name=plan['job_name'], workdir=str(job),
                                   account=plan['account'], queue=plan['partition'], nodes=plan['nodes'],
                                   walltime=plan['walltime'], reservation=plan['reservation'],
                                   dependencies=dependencies)
    intent = {'job_name': plan['job_name'], 'job_root': str(job), 'job_script_sha256': sha(job / 'job.sh'),
              'command': args, 'requested_at': time.time()}
    write(job / 'submission-intent.json', intent)
    result = subprocess.run(args, capture_output=True, text=True, timeout=120, check=True)
    job_id = system.submitted_job_id(result.stdout)
    write(job / 'submission.json', {'job_id': job_id, 'submitted_at': time.time(), 'job_name': plan['job_name'],
                                    'script_sha256': intent['job_script_sha256']})
    ledger = read(root / LEDGER)
    ledger['owned_job_ids'] = list(dict.fromkeys([*ledger['owned_job_ids'], job_id]))
    write(root / LEDGER, ledger)
    print(json.dumps({'job_id': job_id, 'nodes': plan['nodes'], 'inference': plan['inference_count'],
                      'science': plan['science_count'], 'job_root': str(job)}), flush=True)
    return job_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='loam-stage', description=__doc__.split('\n\n')[0])
    parser.add_argument('reference', type=Path, help='the campaign config to stage from')
    parser.add_argument('root', type=Path, help='experiment directory holding the ownership ledger and jobs')
    parser.add_argument('--label', required=True, help='job directory name: lowercase letters, digits, hyphens')
    parser.add_argument('--nodes', type=int, required=True, help='whole nodes in the one allocation')
    parser.add_argument('--inference-profile', type=Path, required=True,
                        help='JSON: how each inference node serves the model (loam.staging.InferenceProfile)')
    parser.add_argument('--inference', type=int, help='inference nodes; default: the measured split')
    parser.add_argument('--walltime', help="HH:MM:SS; default: the campaign config's")
    parser.add_argument('--reservation', default='')
    parser.add_argument('--inputs', type=Path, help='scientific input bundle, copied into the job; never from Git')
    parser.add_argument('--seed-hypotheses', type=Path, help='optional JSON list of starting hypotheses')
    parser.add_argument('--science-workers-per-gpu', type=int)
    for name in ('concurrency', 'top-k', 'initial-hypotheses', 'new-per-round', 'tournament-workers',
                 'generation-batch-size', 'controller-gc-threshold0'):
        parser.add_argument('--' + name, type=int, help='default: the scaling recipe' if 'gc' not in name else None)
    parser.add_argument('--tournament-max-seconds', type=float)
    parser.add_argument('--after', help='an owned job id this job waits for')
    parser.add_argument('--allow-unmeasured', action='store_true',
                        help='size an unmeasured System or node shape anyway, with a warning')
    parser.add_argument('--submit', action='store_true', help='submit the staged job to the scheduler')
    args = vars(parser.parse_args(argv))
    send, reference, root = args.pop('submit'), args.pop('reference'), args.pop('root')
    try:
        job = stage(reference, root, **args)
    except (ValueError, RuntimeError) as error:
        print(f'loam-stage: {error}', file=sys.stderr)
        sys.exit(1)
    print(f'Staged {job}', flush=True)
    if send:
        submit(root, job)


if __name__ == '__main__':
    main()
