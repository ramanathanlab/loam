#!/usr/bin/env python3
"""Clone the frozen energy stage with cwd-relative native filenames only."""
import argparse, hashlib, json, shutil, time
from pathlib import Path


def prepare(trial):
    source=trial/'inputs/intact-egfr-energy';target=trial/'inputs/intact-egfr-energy-pathsafe'
    if target.exists():raise RuntimeError('Path-safe stage exists; inspect its receipt instead of overwriting')
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
    script=target/'mmgbsa.py';old=script.read_text()
    needle="        result = subprocess.run(list(map(str, command)), cwd=out, stdout=stream,"
    assert old.count(needle)==1
    replacement="        # Amber native programs have bounded filename buffers. Keep arguments\n        # beneath their existing cwd relative; chemistry/settings are unchanged.\n        command = [str(part.relative_to(out)) if isinstance(part, Path) and part.is_relative_to(out) else str(part) for part in command]\n        result = subprocess.run(command, cwd=out, stdout=stream,"
    script.write_text(old.replace(needle,replacement))
    compile(script.read_text(),str(script),'exec')
    receipt={'created_at':time.time(),'change':'Only command-line filenames beneath the unchanged subprocess cwd become relative. Force field, solvent, salt, disulfides, coordinates, minimization, frames and numerical settings unchanged. Original source retained.','source':str(source),'target':str(target),'old_mmgbsa_sha256':hashlib.sha256((source/'mmgbsa.py').read_bytes()).hexdigest(),'new_mmgbsa_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'cofold_sha256':hashlib.sha256((target/'cofold.py').read_bytes()).hexdigest()}
    (trial/'pathsafe-energy-stage.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trial',type=Path);a=p.parse_args();print(json.dumps(prepare(a.trial.resolve())))
