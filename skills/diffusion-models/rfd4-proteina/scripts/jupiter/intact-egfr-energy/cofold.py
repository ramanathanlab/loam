"""Native Chai validation of a genuinely generated and designed binder."""
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

import gemmi
import numpy as np

row = json.loads(Path(sys.argv[1]).read_text())
root = Path(row['root'])
software = Path(row['software'])
designed = json.loads((root / 'sequence-report.json').read_text())
command = [str(software / 'chai-lab/.venv/bin/chai-lab'), 'fold', str(root / 'cofold.fasta'),
           str(root / 'cofold'), '--num-trunk-recycles', '3', '--num-diffn-timesteps', '200',
           '--num-diffn-samples', '5', '--seed', str(row['seed']), '--no-use-msa-server']
shell = ('source ' + shlex.quote(str(software / 'chai-lab/.venv/bin/activate'))
         + '; export OMP_NUM_THREADS=4; exec ' + shlex.join(command))
if not row.get('harvest_only', False):
    log = root / 'chai.log'
    log.touch(mode=0o600)
    with log.open('w') as stream:
        result = subprocess.run(['bash', '-c', shell], stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
    assert result.returncode == 0, result.returncode
expected = [''.join(record.splitlines()[1:]) for record in (root / 'cofold.fasta').read_text().split('>') if record.strip()]
predictions = []
for path in sorted((root / 'cofold').glob('pred.model_idx_*.cif')):
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    sequences = [''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in c.get_polymer()).upper()
                 for c in structure[0]]
    assert sorted(sequences) == sorted(expected), (path.name, list(map(len, sequences)))
    assert all(math.isfinite(x) for c in structure[0] for r in c for a in r for x in (a.pos.x, a.pos.y, a.pos.z))
    score_path = path.with_name(path.name.replace('pred.', 'scores.').replace('.cif', '.npz'))
    data = np.load(score_path)
    metrics = {}
    for key in data.files:
        value = data[key]
        if value.size <= 16:
            metrics[key] = value.tolist()
    assert math.isfinite(float(data['ptm'].reshape(-1)[0]))
    assert math.isfinite(float(data['iptm'].reshape(-1)[0]))
    predictions.append({'cif': str(path), 'scores': str(score_path), 'metrics': metrics,
                        'target_sequence_verified': True, 'binder_sequence_verified': True})
assert len(predictions) == 5, len(predictions)
report = {'purpose': row.get('purpose', 'hypothesis-scoped EGFR experiment'),
          'cofolding_completed': True, 'measured_binding': False, 'command': command,
          'scope': 'RFD4 generation, ProteinMPNN sequence design and Chai co-folding; minimization/MMGBSA separate',
          'harvest_only': row.get('harvest_only', False),
          'sequence_report': str(root / 'sequence-report.json'), 'predictions': predictions}
(root / 'validation-report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'cofolding_completed': True, 'predictions': len(predictions),
                  'iptm': [x['metrics'].get('iptm') for x in predictions]}))
