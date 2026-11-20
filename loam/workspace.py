"""Persistent per-hypothesis experiment records and code-generator runs.

Records:
    state/hypotheses/<id>/experiment.json   statement + one entry per validated round
    state/hypotheses/<id>/runs/<run>.json   instruction, state, result and batch receipts

Files (agents read all of it; a code-generator writes into its own run dir):
    <execution.workdir>/<id>/outputs/<run>/ what that run wrote
    <execution.workdir>/<id>/data/          campaign inputs copied at creation (local/shared HPC)
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .schemas import CodeResult, Hypothesis


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data) -> None:
    """Write via a temp file and rename, so a crash mid-write never leaves a truncated file behind."""
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)


class Workspace:
    def __init__(self, cfg: Config, hypothesis: Hypothesis):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', hypothesis.id):
            raise ValueError(f'unsafe hypothesis id: {hypothesis.id!r}')
        name = hypothesis.id
        self.hypothesis_id = hypothesis.id
        self.records = cfg.state_dir / 'hypotheses' / name
        self.remote = cfg.execution.mode in {'mcp', 'hpc'}
        self.root = posixpath.join(cfg.execution.workdir, name)
        if not self.remote or cfg.workspace_inputs:
            Path(self.root).mkdir(parents=True, exist_ok=True)
        (self.records / 'runs').mkdir(parents=True, exist_ok=True)
        if not (self.records / 'experiment.json').is_file():
            if cfg.workspace_inputs:
                data = Path(self.root) / 'data'
                staged = Path(self.root) / '.data.tmp'
                if staged.exists():
                    shutil.rmtree(staged)
                shutil.copytree(cfg.workspace_inputs, staged)
                if data.exists():
                    shutil.rmtree(data)
                os.replace(staged, data)
            write_json(self.records / 'experiment.json', {
                'id': hypothesis.id, 'summary': hypothesis.summary, 'content': hypothesis.content,
                'created': _now(), 'rounds': []})
        elif self.experiment().get('id') != hypothesis.id:
            raise ValueError(f'workspace identity mismatch for {hypothesis.id!r}')

    # --- records ------------------------------------------------------------

    def experiment(self) -> dict:
        return json.loads((self.records / 'experiment.json').read_text())

    def add_round(self, entry: dict) -> None:
        """Record this round's outcome, replacing any entry a crashed attempt at the same round left."""
        experiment = self.experiment()
        experiment['rounds'] = [r for r in experiment['rounds'] if r['round'] != entry['round']] + [entry]
        write_json(self.records / 'experiment.json', experiment)


    # --- runs ---------------------------------------------------------------

    def open_run(self, round_no: int, instruction: str) -> tuple[str, str]:
        """Reserve the next run id; return it and the directory that run may write to."""
        numbers = [int(path.stem.rsplit('-', 1)[1])
                   for path in (self.records / 'runs').glob(f'r{round_no:03d}-*.json')
                   if path.stem.rsplit('-', 1)[1].isdigit()]
        run_id = f'r{round_no:03d}-{max(numbers, default=0) + 1:02d}'
        run_dir = posixpath.join(self.root, 'outputs', run_id)
        if not self.remote:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
        write_json(self.records / 'runs' / f'{run_id}.json',
                   {'run_id': run_id, 'state': 'open', 'opened': _now(), 'instruction': instruction})
        return run_id, run_dir

    def close_run(self, run_id: str, result: CodeResult | None, error: str = '') -> None:
        path = self.records / 'runs' / f'{run_id}.json'
        run = json.loads(path.read_text())
        run.update(state='crashed' if error else result.status, closed=_now(), error=error,
                   result=result and result.model_dump())
        write_json(path, run)

    def add_batch(self, run_id: str, batch_id: str) -> None:
        """Persist an HPC submission receipt before returning it to the model."""
        path = self.records / 'runs' / f'{run_id}.json'
        run = json.loads(path.read_text())
        batches = run.setdefault('batches', [])
        if batch_id not in batches:
            batches.append(batch_id)
            write_json(path, run)

    def batch_ids(self) -> list[str]:
        """Known batches for this hypothesis, including receipts from interrupted runs."""
        found = []
        for path in sorted((self.records / 'runs').glob('r*.json')):
            found.extend(json.loads(path.read_text()).get('batches', []))
        return list(dict.fromkeys(found))
