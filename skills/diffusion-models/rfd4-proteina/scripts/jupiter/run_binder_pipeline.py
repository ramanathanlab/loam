"""A complete, independently seeded RFD4 -> ProteinMPNN -> Chai GPU row.

Input: JSON filename. Output: summary.json plus full generation/sequence/fold
artifacts, with unchanged production sampling settings. No scientific inference
is performed on the login node. Use RFD4's Python environment and gpus=1.
"""
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import gemmi
import numpy as np
from scipy.spatial.distance import cdist
REGION1 = {61, 63, 68, 70, 71, 72, 73, 97, 99, 123, 124, 126, 127, 130, 132, 133, 153, 155}
REGION2 = {182, 183}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def tool_root(tool):
    """A tool's install, found through the Site profile: one directory per tool under its software root.

    The profile's `worker_init` has already exported the tool caches and runtime library paths this
    row needs, so no stage sources a System-specific environment script.
    """
    software = os.environ.get('LOAM_SOFTWARE_ROOT')
    if not software:
        raise RuntimeError('LOAM_SOFTWARE_ROOT is unset: the Site profile must name the software root')
    return Path(software) / tool

def residues(chain):
    return [r for r in chain if gemmi.find_tabulated_residue(r.name).is_amino_acid()]

def sequence(chain):
    return ''.join((gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues(chain)))

def ca(chain):
    return np.array([[r['CA'][0].pos.x, r['CA'][0].pos.y, r['CA'][0].pos.z] for r in residues(chain)])

def pose_rmsd(target, binder, target_ref, binder_ref):
    x, y = (target - target.mean(0), target_ref - target_ref.mean(0))
    u, _, vt = np.linalg.svd(x.T @ y)
    fix = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ fix @ vt
    aligned = (binder - target.mean(0)) @ rotation + target_ref.mean(0)
    return float(np.sqrt(np.mean(np.sum((aligned - binder_ref) ** 2, axis=1))))

def main():
    row = json.loads(Path(sys.argv[1]).read_text())
    out = Path.cwd()
    started = time.monotonic()
    source = Path(row['input_cif']).resolve(strict=True)
    seed = int(row['seed'])
    assert 0 <= seed < 2 ** 31
    assert sha(source) == row['input_sha256'], 'Annotated input checksum mismatch'
    assert row['condition'] in ('dual', 'one_region', 'ridge')
    target_fasta = Path(row['target_fasta']).resolve(strict=True)
    expected = ''.join((x.strip() for x in target_fasta.read_text().splitlines() if not x.startswith('>')))
    assert len(expected) == 205
    times = {}
    commands = []

    def progress(stage):
        (out / 'progress.json').write_text(json.dumps({'stage': stage, 'elapsed_seconds': time.monotonic() - started, 'stage_seconds': times, 'seed': seed, 'condition': row['condition']}, indent=2) + '\n')

    def command(tool, args, timeout=2400, cwd=None, log_name=None):
        argv = [str(x) for x in args]
        commands.append({'tool': tool, 'argv': argv})
        t = time.monotonic()
        with (out / (log_name or tool + '.log')).open('w') as log:
            subprocess.run(['bash', '-lc', 'source "$1"; export OMP_NUM_THREADS=4; shift 1; exec "$@"', 'pipeline', str(tool_root(tool) / '.venv/bin/activate'), *argv], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd)
        times[tool] = time.monotonic() - t
    progress('generation')
    gen = out / 'generation_stage'
    gen.mkdir()
    gi = gen / 'generation_input.json'
    gi.write_text(json.dumps({'input_cif': str(source), 'seed': seed, 'binder_length': 60}))
    helper = Path(row['generation_script']).resolve(strict=True)
    assert sha(helper) == row['generation_script_sha256']
    command('RFD4-Proteina', [tool_root('RFD4-Proteina') / '.venv/bin/python', helper, gi], cwd=gen, log_name='generation.log')
    manifest = json.loads((gen / 'generation_manifest.json').read_text())
    assert len(manifest['complexes']) == 1
    generated = Path(manifest['complexes'][0]['path'])
    s = gemmi.read_structure(str(generated))
    chains = {c.name: c for c in s[0]}
    assert set(chains) == {'A', 'B'}
    target, reference = (sequence(chains['A']), sequence(chains['B']))
    assert target == expected and len(reference) == 60
    pdb = out / 'generated.pdb'
    s.write_pdb(str(pdb))
    progress('sequence_design')
    command('proteinmpnn', [tool_root('proteinmpnn') / '.venv/bin/python', tool_root('proteinmpnn') / 'protein_mpnn_run.py', '--pdb_path', pdb, '--pdb_path_chains', 'B', '--out_folder', out / 'mpnn', '--num_seq_per_target', '1', '--batch_size', '1', '--sampling_temp', '0.1', '--seed', seed], timeout=600)
    fastas = list((out / 'mpnn/seqs').glob('*.fa'))
    assert len(fastas) == 1
    records = [r.splitlines() for r in fastas[0].read_text().split('>') if r.strip()]
    assert len(records) == 2 and "designed_chains=['B']" in records[0][0] and ("fixed_chains=['A']" in records[0][0])
    binder = ''.join(records[1][1:])
    assert len(binder) == 60 and set(binder) <= set('ACDEFGHIKLMNPQRSTVWY')
    fasta = out / 'complex.fasta'
    fasta.write_text(f'>protein|name=target\n{target}\n>protein|name=binder\n{binder}\n')
    progress('complex_folding')
    command('chai-lab', [tool_root('chai-lab') / '.venv/bin/chai-lab', 'fold', fasta, out / 'fold', '--num-trunk-recycles', '3', '--num-diffn-timesteps', '200', '--num-diffn-samples', '5', '--seed', seed, '--no-use-msa-server'])
    progress('analysis')
    models = []
    poses = []
    for p in sorted((out / 'fold').glob('pred.model_idx_*.cif')):
        ss = gemmi.read_structure(str(p))
        cc = {c.name: c for c in ss[0]}
        assert set(cc) == {'A', 'B'}
        assert sequence(cc['A']) == target and sequence(cc['B']) == binder
        ids = {r.seqid.num for r in residues(cc['A'])}
        assert REGION1 | REGION2 <= ids, 'Native hotspot IDs absent'
        xyz = []
        rid = []
        bxyz = []
        for r in residues(cc['A']):
            for at in r:
                if at.element.name not in ('H', 'D'):
                    xyz.append(at.pos.tolist())
                    rid.append(r.seqid.num)
        for r in residues(cc['B']):
            for at in r:
                if at.element.name not in ('H', 'D'):
                    bxyz.append(at.pos.tolist())
        assert np.isfinite(xyz).all() and np.isfinite(bxyz).all()
        distances = cdist(xyz, bxyz)
        contact = {i for i, d in zip(rid, distances.min(1)) if d <= 4}
        sp = p.with_name(p.name.replace('pred.', 'scores.')).with_suffix('.npz')
        z = np.load(sp)
        scores = {k: float(np.asarray(z[k]).item()) for k in ('iptm', 'ptm', 'aggregate_score')}
        assert all((math.isfinite(x) for x in scores.values()))
        one, two = (len(contact & REGION1), len(contact & REGION2))
        models.append({'path': str(p), 'sha256': sha(p), 'score_path': str(sp), 'scores': scores, 'region1_contact_residues': one, 'region2_contact_residues': two, 'dual_region_contact': bool(one and two), 'minimum_interchain_heavy_A': float(distances.min()), 'close_atom_pairs_lt2A': int((distances < 2).sum())})
        poses.append((ca(cc['A']), ca(cc['B'])))
    assert len(models) == 5
    pairwise = [pose_rmsd(*poses[i], *poses[j]) for i in range(5) for j in range(i)]
    zero = [pose_rmsd(*poses[i], *poses[0]) for i in range(1, 5)]
    result = {'status': 'passed', 'candidate_id': row['id'], 'condition': row['condition'], 'seed': seed, 'input_cif': str(source), 'input_sha256': sha(source), 'target_sequence': target, 'binder_sequence': binder, 'binder_length': len(binder), 'target_length': len(target), 'models': models, 'models_total': 5, 'best_iptm': max((x['scores']['iptm'] for x in models)), 'median_iptm': float(np.median([x['scores']['iptm'] for x in models])), 'dual_contact_models': sum((x['dual_region_contact'] for x in models)), 'mean_pairwise_target_aligned_binder_CA_RMSD_A': float(np.mean(pairwise)), 'mean_target_aligned_binder_CA_RMSD_to_model0_excluding_reference_A': float(np.mean(zero)), 'stage_seconds': times, 'elapsed_seconds': time.monotonic() - started, 'host': socket.gethostname(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'commands': commands, 'limitation': 'Computational confidence/geometry only; no affinity or free-energy measurement.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    progress('complete')
    print(json.dumps({k: result[k] for k in ('status', 'candidate_id', 'condition', 'seed', 'models_total', 'best_iptm', 'dual_contact_models', 'elapsed_seconds')}))
if __name__ == '__main__':
    main()
