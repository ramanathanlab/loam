"""Audit a completed single-allocation job against the saturation gates.

    loam-saturation JOB --baseline BASELINE_JOB [--output PATH]

Reads the scheduler's accounting the way the job's System says, so run it
where that scheduler is reachable. Writes the full audit to JOB/saturation-audit.json (or --output)
and prints the gains and gate verdicts. It never submits or changes jobs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import metrics

PRINTED = ('gains', 'principle_gate', 'principle_gate_passed', 'final_gate', 'final_gate_passed')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='loam-saturation', description=__doc__.split('\n\n')[0])
    parser.add_argument('job', type=Path, help='staged job directory of the trial to audit')
    parser.add_argument('--baseline', type=Path, required=True,
                        help='staged job directory of the earlier trial the gains are measured against')
    parser.add_argument('--output', type=Path, help='default: JOB/saturation-audit.json')
    args = parser.parse_args(argv)
    result = metrics.evaluate(args.job.resolve(), args.baseline.resolve())
    output = args.output or args.job / 'saturation-audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in PRINTED}, indent=2))


if __name__ == '__main__':
    main()
