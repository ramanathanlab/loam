"""Configuration-aware launch gates. --check performs only offline validation."""
import json
import os
import sys
from pathlib import Path

import httpx

from loam.config import load
from loam.skills import discover


def main():
    cfg = load(sys.argv[1])
    check = '--check' in sys.argv[2:]
    for path in [Path(cfg.goal), *cfg.rules, cfg.tournament.jnana_root / 'jnana']:
        if not path.exists():
            raise ValueError(f'required campaign input is missing: {path}')
    skills = discover(cfg.skills_dirs)
    if not skills:
        raise ValueError('no skills found')
    if cfg.execution.hpc:
        hpc = cfg.execution.hpc
        endpoint = json.loads(hpc.environment()['HPC_ENDPOINT_CONFIG_JSON'])
        if hpc.system == 'jupiter' and not endpoint.get('worker_init'):
            raise ValueError('Jupiter requires endpoint_options.worker_init')
        if not hpc.worker_code_dir or not Path(hpc.worker_code_dir, 'skills/hpc-deployment/scripts/hpc_agent.py').is_file():
            raise ValueError('worker_code_dir does not contain hpc_agent.py')
    print(f'Configuration valid: {len(skills)} skills, {cfg.execution.mode} execution', flush=True)
    if check:
        return
    specs = [*cfg.models.values(), cfg.tournament.generation, cfg.tournament.ranking]
    seen = set()
    for spec in specs:
        if spec.api_key_env and not os.environ.get(spec.api_key_env):
            raise ValueError(f'missing required model credential: {spec.api_key_env}')
        if not spec.base_url or (spec.base_url, spec.model) in seen:
            continue
        seen.add((spec.base_url, spec.model))
        # Check the actual configured model at its actual endpoint; never print tokens.
        with httpx.Client(timeout=120, headers={'Authorization': f'Bearer {spec.api_key(custom_endpoint=True)}'}) as client:
            response = client.post(spec.base_url.rstrip('/') + '/chat/completions', json={
                'model': spec.model, 'messages': [{'role': 'user', 'content': 'Reply with exactly LOAM_PREFLIGHT_OK'}],
                'max_tokens': 256,
            })
            response.raise_for_status()
            if 'LOAM_PREFLIGHT_OK' not in json.dumps(response.json().get('choices', [])):
                raise ValueError(f'model preflight failed: {spec.model} at {spec.base_url}')
    print('Configured models answered; starting campaign', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.exit(f'launch.sh: {exc}')
