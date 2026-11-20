#!/usr/bin/env python3
"""Validate/copy an operator-supplied scientific bundle outside the Git checkout.

This command does not download data, disclose it, start inference or submit jobs.
Only names listed in the reference manifest are copied; extra source files are
ignored. Optional private proposal texts remain external as well.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path(__file__).resolve().parents[1]


def prepare(source,output,manifest,seed_hypotheses=None):
    source=Path(source).resolve(strict=True);output=Path(output).resolve()
    if output.is_relative_to(REPO):
        raise ValueError('Scientific data must be prepared outside the Git checkout')
    if output.exists():raise RuntimeError('Refuse to overwrite an existing input bundle')
    expected=json.loads(Path(manifest).read_text())['sha256']
    for name,digest in expected.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid manifest path')
        path=source/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Missing or changed scientific input: '+name)
    output.mkdir(parents=True,mode=0o700)
    for name in expected:
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
    if seed_hypotheses:
        # Kept next to, rather than inside, the immutable 33-file assay bundle.
        seed=Path(seed_hypotheses).resolve(strict=True)
        target=output.with_name(output.name+'-seed-hypotheses.json')
        if target.exists():raise RuntimeError('Private proposal destination exists')
        shutil.copyfile(seed,target)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--manifest',type=Path,default=REPO/'examples/scaling/egfr/inputs-manifest.json')
    p.add_argument('--seed-hypotheses',type=Path)
    a=p.parse_args();print(prepare(a.source,a.output,a.manifest,a.seed_hypotheses))
