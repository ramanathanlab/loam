"""One real RFD4 binder-generation row; input paths and seed come from JSON."""
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import gemmi


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision_metadata(software):
    """Read checkout identity without requiring git on compute-node PATH."""
    try:
        directory = software / '.git'
        if directory.is_file():
            directory = (software / directory.read_text().strip().removeprefix('gitdir: ')).resolve()
        head = (directory / 'HEAD').read_text().strip()
        revision = head
        if head.startswith('ref: '):
            ref = head[5:]
            loose = directory / ref
            revision = loose.read_text().strip() if loose.exists() else None
            packed = directory / 'packed-refs'
            if revision is None and packed.exists():
                for line in packed.read_text().splitlines():
                    fields = line.split()
                    if len(fields) == 2 and fields[1] == ref:
                        revision = fields[0]
                        break
        return {'revision': revision, 'head': head, 'method': 'checkout metadata files'}
    except OSError as error:
        return {'revision': None, 'error': str(error), 'method': 'checkout metadata files'}


def main():
    row = json.loads(Path(sys.argv[1]).read_text())
    source = Path(row['input_cif']).resolve(strict=True)
    output = Path.cwd()
    generated = output / 'generation'
    if generated.exists():
        raise RuntimeError('Generation output directory must be fresh')
    target_chain = row.get('target_chain', 'A')
    binder_chain = row.get('binder_chain', 'B')
    expected_length = int(row.get('binder_length', 60))
    seed = int(row['seed'])
    target_length = len(gemmi.read_structure(str(source))[0][target_chain])
    software = Path(os.environ['LOAM_SOFTWARE_ROOT']) / 'RFD4-Proteina'
    home = Path(os.environ['HPC_LOAM_HOME'])
    command = [str(software / '.venv/bin/rfd4-generate'),
               f'inputs={source}', f'out_dir={generated}',
               'recycle_strategy=medium/front_loaded', 'num_replicates=1',
               'batch_size=1', 'compile_lengths=null', 'num_workers=0',
               'save_chain_of_interest=true', f'seed={seed}']
    started = time.monotonic()
    with (output / 'rfd4.log').open('w') as log:
        subprocess.run(['bash', '-lc',
                        'source "$1"; source "$2"; '
                        'export CLUSTER=local WANDB_MODE=disabled OMP_NUM_THREADS=4; '
                        'cd "$3"; shift 3; exec "$@"', 'rfd4',
                        str(home / 'scripts/jupiter_env.sh'),
                        str(software / '.venv/bin/activate'), str(software), *command],
                       check=True, stdout=log, stderr=subprocess.STDOUT, timeout=5900)
    complexes = []
    for path in sorted((generated / 'inference_samples').rglob('*generated.cif')):
        structure = gemmi.read_structure(str(path))
        chains = {chain.name: chain for chain in structure[0]}
        if target_chain not in chains or binder_chain not in chains:
            continue
        assert len(chains[target_chain]) == target_length, 'Target length changed'
        assert len(chains[binder_chain]) == expected_length, 'Binder length mismatch'
        atoms = [a for chain in chains.values() for residue in chain for a in residue]
        assert atoms and all(math.isfinite(x) for a in atoms for x in a.pos.tolist())
        complexes.append({'path': str(path), 'sha256': sha(path),
                          'chains': [{'chain': name, 'residues': len(chain),
                                      'sequence': ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code
                                                          for r in chain)}
                                     for name, chain in chains.items()],
                          'finite_coordinates': True})
    if not complexes:
        raise RuntimeError('No generated binder-target complex passed structural checks')
    metadata = revision_metadata(software)
    result = {'status': 'passed', 'seed': seed, 'input': str(source),
              'input_sha256': sha(source), 'command': command,
              'software_revision': metadata['revision'], 'software_git_metadata': metadata,
              'complexes': complexes,
              'host': socket.gethostname(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'elapsed_seconds': time.monotonic() - started}
    (output / 'generation_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
