#!/usr/bin/env python3
"""Record installed package versions and bounded import checks, without credentials.

Run with the Loam environment. GPU inference is a separate acceptance gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CHECKS = {
    'loam': 'import academy, fastmcp, globus_compute_sdk, ensemble_launcher',
    'fetch-literature': 'import requests',
    'rag-build-index': 'import faiss, datasets, sentence_transformers',
    'rag-query': 'import faiss, datasets, sentence_transformers',
    'openmm': 'import openmm',
    'molecular-simulations': 'import molecular_simulations',
    'chai-lab': 'import chai_lab, torch',
    'boltz': 'import boltz, torch',
    'proteinmpnn': 'import torch, numpy',
    'foundry': 'import rfd3, atomworks, torch',
    'RFD4-Proteina': 'import rfproteina, atomworks, torch',
    'protenix': 'import os; os.environ["LAYERNORM_TYPE"]="torch"; from protenix.model.protenix import Protenix',
    'esmfold2': 'from esm.models.esmfold2 import EsmFold2Model',
    'fairchem': 'import fairchem.core',
    'pyrosetta': 'import pyrosetta',
    'genslm-esm': 'import genslm_esm',
    'reactra': 'import reactra',
    'ambertools': 'import parmed',
    'jnana-engine': 'from jnana.protognosis.periodic_tournament import run_incremental_tournament',
    'hermes-agent': 'import hermes_cli.main',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--software-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'software')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'ports/jupiter/status.json')
    args = parser.parse_args()
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'software': {}}
    for name, code in CHECKS.items():
        directory = args.software_root / name
        python = directory / '.venv/bin/python'
        entry = {'environment': str(python), 'gpu_inference_verified': False}
        if not python.is_file():
            entry.update(import_ok=False, error='Environment not installed')
        else:
            try:
                result = subprocess.run([str(python), '-c', code], text=True,
                                        capture_output=True, timeout=120)
                entry.update(import_ok=result.returncode == 0,
                             detail=(result.stdout + result.stderr)[-2500:])
                versions = subprocess.run([str(python), '-c',
                    'import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'],
                    capture_output=True, text=True, check=True, timeout=30)
                entry['packages'] = json.loads(versions.stdout)
            except (subprocess.SubprocessError, ValueError) as exc:
                entry.update(import_ok=False, error=str(exc))
        if (directory / '.git').is_dir():
            entry['source_commit'] = subprocess.check_output(
                ['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        report['software'][name] = entry
        print(f'{name}: {"import OK" if entry["import_ok"] else "BLOCKED"}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    main()
