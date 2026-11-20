#!/usr/bin/env python3
"""Render site paths into a new endpoint config directory; never start it."""
import argparse
from pathlib import Path
import shutil

import yaml
from loam.config import expand_environment


def render(destination, software, repo, name):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Invalid endpoint display name')
    destination=Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Refuse to replace an existing endpoint configuration')
    destination.mkdir(parents=True,exist_ok=True)
    templates=Path(__file__).resolve().parents[1]/'globus_compute_endpoint_configs/jupiter'
    for item in templates.iterdir():
        if item.suffix in ('.yaml','.json','.j2'):
            shutil.copyfile(item,destination/item.name)
    path=destination/'user_environment.yaml'
    env=expand_environment(yaml.safe_load(path.read_text()), {
        'LOAM_SOFTWARE_ROOT':str(Path(software).resolve()),
        'LOAM_REPO_ROOT':str(Path(repo).resolve()),
        'GLOBUS_COMPUTE_USER_DIR':str(Path.home()/'.globus_compute')})
    path.write_text(yaml.safe_dump(env,sort_keys=False))
    for filename in ('config.yaml','user_config_template.yaml.j2'):
        p=destination/filename;p.write_text(p.read_text().replace('display_name: loam','display_name: '+name))
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path)
    p.add_argument('--software-root',type=Path,required=True)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--name',default='loam')
    a=p.parse_args();print(render(a.destination,a.software_root,a.repo_root,a.name))
