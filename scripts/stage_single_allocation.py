#!/usr/bin/env python3
"""Stage a fresh, self-contained Slurm job. Submission requires --submit.

The checked-in EGFR example and native vLLM launcher replace dependencies on
private campaign directories. Site paths, GPU/CPU counts and account are explicit.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import uuid

import yaml

from loam.config import expand_environment, load
from loam.allocation import validate_campaign_paths
from loam.scaling import dense_recipe

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / 'examples/scaling/egfr'
RUNTIME = REPO  # native launcher sources are now shipped in scripts/


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_runtime(infra):
    """Copy source, not environments/weights/evidence, before queue waiting."""
    target = infra / 'control-runtime'
    target.mkdir()
    for folder in ('loam', 'scripts', 'skills'):
        shutil.copytree(REPO / folder, target / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(REPO / 'examples/rules', target / 'examples/rules')
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copyfile(REPO / name, target / name)
    (target / '.venv').symlink_to(REPO / '.venv', target_is_directory=True)
    return target


def stage(root, nodes, inference, label, mode, walltime, reservation, concurrency, *,
          top_k=None, initial_hypotheses=None, new_per_round=None, tournament_workers=None,
          science_workers_per_gpu=1, generation_batch_size=4, gpus_per_node=4,
          cpus_per_node=288, account='deepwing', partition='booster', software_root=None,
          vllm=None, model_dir=None, reference=None, inference_profile=None,
          controller_gc_threshold0=None, tournament_max_seconds=None,
          inputs=None, seed_hypotheses=None):
    """Write inputs, configuration, ownership ledger and job script; never submit."""
    root = Path(root).resolve()
    reference = Path(reference or REFERENCE).resolve(strict=True)
    data = Path(inputs or os.environ.get('LOAM_SCIENCE_INPUTS') or reference / 'inputs').resolve(strict=True)
    manifest_path = reference / 'inputs-manifest.json'
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())['sha256']
        missing = [name for name in expected if not (data / name).is_file()]
        if missing:
            raise RuntimeError('Scientific data are external to Git. Set LOAM_SCIENCE_INPUTS or --inputs to the complete input bundle; missing: ' + ', '.join(missing))
        for name,digest in expected.items():
            if sha(data / name) != digest:
                raise RuntimeError('Scientific input differs from the reference manifest: ' + name)
    if not re.fullmatch(r'[a-z0-9-]+', label):
        raise ValueError('label must contain only lowercase letters, digits and hyphens')
    if mode not in ('smoke', 'campaign'):
        raise ValueError('mode must be smoke or campaign')
    recipe = dense_recipe(nodes, gpus_per_node, inference=inference,
                          science_workers_per_gpu=science_workers_per_gpu)
    top_k = concurrency if top_k is None else top_k
    initial_hypotheses = initial_hypotheses if initial_hypotheses is not None else 2 * top_k
    new_per_round = new_per_round if new_per_round is not None else max(1, top_k // 2)
    tournament_workers = tournament_workers or 2 * inference
    if not 0 < concurrency <= top_k <= initial_hypotheses:
        raise ValueError('Require 0 < concurrency <= top_k <= initial_hypotheses')
    if new_per_round < 1 or generation_batch_size < 1 or cpus_per_node < 1:
        raise ValueError('Batch size, new hypotheses and CPU count must be positive')
    if not re.fullmatch(r'\d{2}:[0-5]\d:[0-5]\d', walltime):
        raise ValueError('walltime must have format HH:MM:SS')
    if controller_gc_threshold0 is not None and (type(controller_gc_threshold0) is not int
                                               or not 700 <= controller_gc_threshold0 <= 100000):
        raise ValueError('controller_gc_threshold0 must be an integer in [700, 100000]')
    job = root / label
    if job.exists() or job.is_symlink():
        raise RuntimeError('Refuse to overwrite an existing staged job')
    software = Path(software_root or os.environ.get('LOAM_SOFTWARE_ROOT', REPO.parent / 'software')).expanduser().resolve()
    campaign = job / 'campaign'
    variables = {'LOAM_REPO_ROOT': str(REPO), 'LOAM_SOFTWARE_ROOT': str(software),
                 'LOAM_CAMPAIGN_DIR': str(campaign)}
    raw = expand_environment(yaml.safe_load((reference / 'campaign.yaml').read_text()), variables)
    # Resolve reference-relative resources before relocating the campaign.
    for field in ('rules', 'skills_dirs'):
        raw[field] = [str((reference / p).resolve()) for p in raw.get(field, [])]
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'experiment.json').exists():
        write(root / 'experiment.json', {'created_at': time.time(),
              'purpose': 'Single-allocation Nemotron/science campaign', 'owned_job_ids': [],
              'reference': str(reference)})
    job.mkdir(mode=0o700)
    (job / '.gitignore').write_text('*\n')
    campaign.mkdir()
    infra = job / 'infra'
    infra.mkdir(mode=0o700)
    runtime = freeze_runtime(infra)
    native = infra / 'inference-runtime/scripts'
    native.mkdir(parents=True)
    for name in ('jupiter_nemotron.py', 'jupiter_nemotron_api.py'):
        shutil.copyfile(RUNTIME / 'scripts' / name, native / name)
    shutil.copytree(data, campaign / 'inputs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (campaign / 'data').symlink_to('inputs', target_is_directory=True)
    goal = expand_environment((reference / 'goal.md').read_text(), variables)
    goal = goal.replace(str(REPO), str(runtime))
    # Allocation size is operational context, never a change to the scientific assay.
    goal = re.sub(r'at (?:twenty|\d+) total owned Jupiter nodes',
                  f'at {nodes} total owned Jupiter nodes', goal)
    (campaign / 'goal.md').write_text(goal)
    cfg = expand_environment(raw)
    # Keep queued jobs independent of edits made in the developer checkout.
    for field in ('rules', 'skills_dirs'):
        cfg[field] = [p.replace(str(REPO), str(runtime)) for p in cfg[field]]
    cfg.update(goal=str(campaign / 'goal.md'), state_dir=str(campaign / 'state'),
               workspace_inputs=str(campaign / 'inputs'), concurrency=concurrency,
               synthesis_concurrency=inference, rounds=3)
    cfg['execution']['workdir'] = str(campaign / 'work')
    hpc = cfg['execution']['hpc']
    hpc.update(base_dir=str(campaign), worker_code_dir=str(runtime),
               server_script=str(runtime / 'skills/hpc-deployment/scripts/mcp_server.py'),
               inspect_roots=[str(campaign), str(software)],
               science_workers_per_gpu=science_workers_per_gpu,
               gpus_per_node=gpus_per_node, cpus_per_node=cpus_per_node,
               frozen_tool_library=True, pool_size=min(64, max(12, concurrency * 2)),
               campaign_id='single-' + label + '-' + uuid.uuid4().hex[:10], startup_timeout=7200,
               endpoint_id='existing-allocation-no-globus')
    hpc['allocation'].update(nodes=recipe.science, walltime=walltime, account=account, queue=partition)
    hpc['allocation']['endpoint_options'] = {'partition': partition,
                                          'scheduler_options': f'#SBATCH --gres=gpu:{gpus_per_node}'}
    cfg['execution']['setup'] = re.sub(
        r'one Slurm allocation with \d+ Nemotron nodes and \d+ scientific\s+nodes, \d+ scientific GPU slots',
        f'one Slurm allocation with {inference} Nemotron nodes and {recipe.science} scientific nodes, '
        f'{gpus_per_node * recipe.science} scientific GPU slots', cfg['execution']['setup']).replace(str(REPO), str(runtime))
    cfg['models']['reasoner'] = copy.deepcopy(cfg['models']['reasoner'])
    cfg['models']['reasoner'].setdefault('settings', {})['max_tokens'] = 32768
    cfg['tournament'].update(top_k=top_k, initial_hypotheses=initial_hypotheses,
         new_per_round=new_per_round, gen_batch_size=generation_batch_size,
         workers=tournament_workers, scheme='swiss', swiss_rounds=1,
         max_seconds=tournament_max_seconds or (2400 if nodes <= 20 else 10800))
    cfg['timeouts']['mcp_init'] = 60
    for spec in [*cfg['models'].values(), cfg['tournament']['generation'], cfg['tournament']['ranking']]:
        spec['base_url'] = 'http://127.0.0.1:1/v1'  # replaced after this allocation's gateway is ready
        spec['api_key_env'] = 'LOAM_NEMOTRON_TOKEN'
    (campaign / 'campaign.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (campaign / 'state').mkdir()
    (campaign / 'work').mkdir()
    seeds = []
    seed_file = Path(seed_hypotheses or os.environ.get('LOAM_SEED_HYPOTHESES') or reference / 'seed-hypotheses.json')
    if seed_file.exists():
        for i, seed in enumerate(json.loads(seed_file.read_text())):
            seed = copy.deepcopy(seed)
            seed['hypothesis_id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(campaign) + ':' + seed['hypothesis_id']))
            seed.update(elo_rating=1200, reviews=[], tournament_matches=[], tournament_wins=0,
                        tournament_losses=0, last_tournament_time=None, created_at=time.time() + i * .001)
            seeds.append(seed)
    state = {k: {} for k in ['experiments', 'analyses', 'papers', 'agent_states', 'datasets',
                              'training_plans', 'evaluations']}
    state.update(hypotheses=seeds, metadata={'research_goal': goal}, tournament_state={'matches': []})
    write(campaign / 'state/jnana.json', state)
    scripts = [p for p in (campaign / 'inputs').rglob('*.py') if '__pycache__' not in p.parts]
    write(campaign / 'scientific-protocol.json', {
        'files': [{'file': str(p.relative_to(campaign / 'inputs')), 'sha256': sha(p)} for p in sorted(scripts)],
        'thinking': 'Agent roles false; tournament generation/ranking unchanged default true',
        'native_guard_patch': 'Attached-parent role validation; no separate inference allocation'})
    input_hashes = {str(p.relative_to(campaign / 'inputs')): sha(p)
                    for p in sorted((campaign / 'inputs').rglob('*')) if p.is_file()}
    write(job / 'inputs-integrity-preflight.json', {'checked_at': time.time(),
          'source_campaign': str(reference), 'input_bundle_source': str(data),
          'all_input_files_identical': True, 'input_sha256': input_hashes})
    profile = Path(inference_profile or REPO / 'examples/scaling/nemotron.json').resolve(strict=True)
    model = model_dir or software / 'nemotron-vllm/models/fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe'
    executable = vllm or software / 'nemotron-vllm/.venv/bin/vllm'
    request = expand_environment(json.loads(profile.read_text()),
                                 {**variables, 'LOAM_MODEL_DIR': str(Path(model).resolve()),
                                  'LOAM_VLLM': str(Path(executable).absolute())})
    request.update(warmup_seconds=5400, account=account, reservation=reservation,
                   walltime=walltime, tensor_parallel_size=gpus_per_node)
    write(job / 'inference-request-template.json', request)
    plan = {'mode': 'single-allocation', 'nodes': nodes, 'inference_count': inference,
            'science_count': recipe.science, 'campaign': str(campaign), 'infra': str(infra),
            'job_root': str(job), 'job_name': 'loam-' + label + '-' + uuid.uuid4().hex[:10],
            'walltime': walltime, 'reservation': reservation, 'account': account, 'partition': partition,
            'run_mode': mode, 'created_at': time.time(), 'model_startup_timeout': 5400,
            'gpus_per_node': gpus_per_node, 'cpus_per_node': cpus_per_node,
            'software_root': str(software), 'redis_executable': str(software / 'redis/bin/redis-server'),
            'runtime_repo': str(runtime),
            'input_bundle_source': str(data), 'seed_hypotheses_count': len(seeds),
            'inference_request_template': str(job / 'inference-request-template.json'),
            'science_workers_per_gpu': science_workers_per_gpu,
            'tournament_workers': tournament_workers, 'synthesis_concurrency': inference,
            'generation_batch_size': generation_batch_size, 'mcp_init_timeout_seconds': 60,
            'admission': {'concurrency': concurrency, 'top_k': top_k,
                          'initial_hypotheses': initial_hypotheses, 'new_per_round': new_per_round}}
    if controller_gc_threshold0 is not None:
        plan['controller_gc_threshold0'] = controller_gc_threshold0
    write(job / 'plan.json', plan)
    write(job / 'prepared-control-source.json', {'at': time.time(), 'runtime_repo': str(runtime),
          'source_sha256': {str(p.relative_to(runtime)): sha(p) for p in sorted(runtime.rglob('*'))
                           if p.is_file() and '.venv' not in p.parts},
          'native_source_sha256': {str(p.relative_to(infra)): sha(p) for p in sorted(native.glob('*.py'))},
          'inference_request_template_sha256': sha(job / 'inference-request-template.json')})
    script = job / 'job.sh'
    python = Path(os.environ.get('LOAM_PYTHON', REPO / '.venv/bin/python')).absolute()
    quote = shlex.quote
    site_exports=''.join('export '+name+'='+quote(os.environ[name])+'\n' for name in
        ('LOAM_CC','LOAM_CXX','LOAM_LD','LOAM_REDIS_LIBRARY_PATH') if os.environ.get(name))
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+site_exports
        + 'export LOAM_SOFTWARE_ROOT=' + quote(str(software)) + '\n'
        + 'source ' + quote(str(runtime / 'scripts/jupiter_env.sh')) + '\n'
        + 'export LOAM_PYTHON=' + quote(str(python)) + '\n'
        + 'export SLURM_EXPORT_ENV=ALL\nexport PYTHONPATH=' + quote(str(runtime)) + ':${PYTHONPATH:-}\n'
        + 'exec ' + quote(str(python)) + ' -u ' + quote(str(runtime / 'scripts/run_single_allocation.py'))
        + ' ' + quote(str(job)) + '\n')
    script.chmod(0o700)
    validate_campaign_paths(load(campaign / 'campaign.yaml'), campaign)
    from check_single_allocation import check
    write(job / 'preflight.json', check(job))
    return job


def submit(root, job):
    """Submit this exact fresh staged job once and record its ownership."""
    root, job = Path(root).resolve(), Path(job).resolve()
    plan = json.loads((job / 'plan.json').read_text())
    if (job / 'submission.json').exists() or (job / 'submission-intent.json').exists():
        raise RuntimeError('A submission was already attempted; reconcile its receipt before retrying')
    from check_single_allocation import check
    check(job)
    # Fail before sbatch when a clean checkout has not been installed at the site.
    request = json.loads(Path(plan['inference_request_template']).read_text())
    for name in [request['command'][0], plan['redis_executable'],
                 os.environ.get('LOAM_PYTHON', str(REPO / '.venv/bin/python'))]:
        if not Path(name).is_file() or not os.access(name, os.X_OK):
            raise RuntimeError('Required runtime executable is missing: ' + name)
    from jupiter_nemotron import model_snapshot
    model_snapshot(Path(request['model_dir']), request['checkpoint_revision'])
    intent = {'job_name': plan['job_name'], 'job_root': str(job),
              'job_script_sha256': sha(job / 'job.sh'), 'requested_at': time.time()}
    write(job / 'submission-intent.json', intent)
    args = ['sbatch', '--parsable', '--export=NIL', '--account=' + plan['account'],
            '--partition=' + plan['partition'], '--nodes=' + str(plan['nodes']),
            '--ntasks-per-node=1', '--cpus-per-task=' + str(plan['cpus_per_node']),
            '--gres=gpu:' + str(plan['gpus_per_node']), '--exclusive', '--time=' + plan['walltime'],
            '--job-name=' + plan['job_name'], '--chdir=' + str(job),
            '--output=' + str(job / 'slurm-%j.log')]
    if plan['reservation']:
        args.append('--reservation=' + plan['reservation'])
    for field, kind in [('afterany_owned_job_id', 'afterany'), ('validation_job_id', 'afterok')]:
        if plan.get(field):
            previous = str(plan[field])
            ledger = json.loads((root / 'experiment.json').read_text())
            if previous not in ledger['owned_job_ids']:
                raise RuntimeError('Dependency must belong to this experiment')
            args.append('--dependency=' + kind + ':' + previous)
    args.append(str(job / 'job.sh'))
    result = subprocess.run(args, capture_output=True, text=True, timeout=120, check=True)
    job_id = result.stdout.strip().split(';')[0]
    if not job_id.isdigit():
        raise RuntimeError('Ambiguous sbatch acknowledgement; inspect submission intent before retrying')
    write(job / 'submission.json', {'job_id': job_id, 'submitted_at': time.time(),
          'job_name': plan['job_name'], 'script_sha256': intent['job_script_sha256']})
    experiment = json.loads((root / 'experiment.json').read_text())
    experiment['owned_job_ids'] = list(dict.fromkeys(experiment['owned_job_ids'] + [job_id]))
    write(root / 'experiment.json', experiment)
    print(json.dumps({'job_id': job_id, 'nodes': plan['nodes'], 'inference': plan['inference_count'],
                      'science': plan['science_count'], 'job_root': str(job)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--nodes', type=int, required=True)
    p.add_argument('--inference', type=int)
    p.add_argument('--label', required=True)
    p.add_argument('--gpus-per-node', type=int, default=4)
    p.add_argument('--cpus-per-node', type=int, default=288)
    p.add_argument('--science-workers-per-gpu', type=int, default=2)
    p.add_argument('--concurrency', type=int)
    p.add_argument('--top-k', type=int)
    p.add_argument('--initial-hypotheses', type=int)
    p.add_argument('--new-per-round', type=int)
    p.add_argument('--tournament-workers', type=int)
    p.add_argument('--generation-batch-size', type=int, default=4)
    p.add_argument('--tournament-max-seconds', type=int)
    p.add_argument('--controller-gc-threshold0', type=int)
    p.add_argument('--account', default='deepwing')
    p.add_argument('--partition', default='booster')
    p.add_argument('--walltime', default='08:00:00')
    p.add_argument('--reservation', default='')
    p.add_argument('--mode', choices=['smoke', 'campaign'], default='campaign')
    p.add_argument('--software-root', type=Path)
    p.add_argument('--vllm', type=Path)
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--inputs', type=Path, help='External scientific input bundle; never supplied by Git')
    p.add_argument('--seed-hypotheses', type=Path, help='Optional private starting proposal JSON; omit to generate a fresh pool')
    p.add_argument('--inference-profile', type=Path)
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    recipe = dense_recipe(a.nodes, a.gpus_per_node, inference=a.inference,
                          science_workers_per_gpu=a.science_workers_per_gpu)
    keywords = vars(a).copy()
    send = keywords.pop('submit')
    root = keywords.pop('root')
    for name in ('inference', 'concurrency', 'top_k', 'initial_hypotheses', 'new_per_round', 'tournament_workers'):
        if keywords[name] is None:
            keywords[name] = getattr(recipe, name)
    job = stage(root, **keywords)
    print('Staged ' + str(job), flush=True)
    if send:
        submit(root, job)


if __name__ == '__main__':
    main()
