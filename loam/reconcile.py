"""The acceptance boundary. Deterministic: only claims an independent reviewer accepted,
with a named verification and source_ref, ever reach the ledger. Each gets its steering
weight here, from how it was verified."""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urldefrag, urlparse

from .schemas import Review, ValidationReport
from .steering import claim_delta, read_ledger

log = logging.getLogger('loam')


def _is_source_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {'http', 'https'} and bool(parsed.netloc)


def _source_url(value: str) -> str:
    parsed = urlparse(urldefrag(value.strip()).url)
    return parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()).geturl()


#: A locator a reviewer appends to say WHERE in a file it checked: `lines 2, 5-6`,
#: `line 9`, `:12`, `:12-18`. Stripped before comparing, because a claim cites the
#: artifact while its verification cites the same artifact plus a position, and the
#: two must still compare equal.
#:
#: Measured 2026-11-11: 24 of 129 review source_refs carried such a suffix (e.g.
#: `manifest.json lines 62-66,95-99,138-142`). `split('#')` alone left it inside the
#: filename, so every one of those claims was dropped as a source mismatch -- the
#: reviewers were citing their evidence MORE precisely than the parser allowed.
_LINE_LOCATOR = re.compile(
    r"""(?: \s*[,;]?\s* (?:lines?|ll?\.) \s* [\d,\s–—-]+   # " lines 2, 5-6"
        |   \s*:\s* \d+ (?:\s*[-–—]\s*\d+)?                # ":12", ":12-18"
        )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _strip_line_locator(path: str) -> str:
    """Remove trailing line locators. Looped: `.json lines 2-3:12` needs two passes."""
    while True:
        trimmed = _LINE_LOCATOR.sub('', path).strip()
        if trimmed == path:
            return path
        path = trimmed


def _clean_path(value: str) -> PurePosixPath | None:
    """Normalize a cited path; None if it escapes with `..`."""
    path = _strip_line_locator(value.strip().split('#', 1)[0])
    if not path:
        return None
    normalized = PurePosixPath(posixpath.normpath(path))
    return None if '..' in normalized.parts else normalized


def _run_records(workspace) -> list[dict]:
    """Every recorded run of this hypothesis, newest id last. A run only has a
    record because `open_run` reserved it, which is what makes it citable."""
    runs = []
    for path in sorted((workspace.records / 'runs').glob('r*.json')):
        try:
            runs.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue  # a torn record must not invalidate the rest
    return runs


def _owned_artifact(value: str, workspace) -> tuple[str, str] | None:
    """Return (run_id, normalized path) for an artifact a RECORDED run owns.

    Two kinds of ownership, both recorded by the harness rather than inferred
    from the path's shape:
      * `<workspace.root>/outputs/<run_id>/...` -- the dir `open_run` reserved;
      * `.../batches/<batch_id>/...`            -- a batch the run submitted, as
        persisted by `add_batch`. These live under base_dir, OUTSIDE the
        workspace, so they are matched by their `batches/<id>` segment.

    A path an agent invented (e.g. a sibling `outputs/r011-01/` it created
    itself) resolves to nothing: no record reserved it, so nothing owns it.
    """
    normalized = _clean_path(value)
    if normalized is None:
        return None
    text = str(normalized)
    # Match on the unique `outputs/<run_id>` and `batches/<batch_id>` SEGMENTS
    # rather than on a workspace-root prefix. Aurora exposes the same tree as
    # both /flare/... and /lus/flare/projects/... (/flare is a symlink, verified
    # by inode 2026-11-11): the config carries one spelling and tools report the
    # resolved other, so any root-prefix comparison silently fails. A segment
    # match is spelling-independent, and ownership is still proven by the run
    # record rather than by the path.
    for run in _run_records(workspace):
        run_id = run.get('run_id')
        if not run_id:
            continue
        for marker in [f'outputs/{run_id}'] + [f'batches/{b}' for b in (run.get('batches') or [])]:
            if text == marker or text.endswith(f'/{marker}') or f'{marker}/' in text:
                return run_id, text[text.index(marker):]
    return None


def citable_runs(workspace) -> list[dict]:
    """The runs a claim of this hypothesis may cite, newest last.

    A run appears only because `open_run` recorded it, so this is the exact set
    `_owned_artifact` will accept -- the orchestrator shows it to a validator
    rather than letting it guess.
    """
    runs = []
    for run in _run_records(workspace):
        run_id = run.get('run_id')
        if not run_id:
            continue
        runs.append({'run_id': run_id,
                     'run_dir': posixpath.join(workspace.root, 'outputs', run_id),
                     'batches': list(run.get('batches') or [])})
    return runs


def unowned_sources(report: ValidationReport, workspace) -> list[tuple[int, str]]:
    """(index, source) for each artifact claim no recorded run owns.

    URL-sourced claims are skipped: those are checked against the review's
    source_ref by `_source_url`, and ownership does not apply to them.
    """
    bad = []
    for index, claim in enumerate(report.claims):
        if _is_source_url(claim.source):
            continue
        if _owned_artifact(claim.source, workspace) is None:
            bad.append((index, claim.source))
    return bad


def owned_claim_count(report: ValidationReport, workspace) -> int:
    """How many claims would survive the provenance check."""
    return len(report.claims) - len(unowned_sources(report, workspace))


def accepted(report: ValidationReport, review: Review | None, unit: str, workspace=None) -> list[dict]:
    """Return claims that passed a complete review and deterministic provenance checks."""
    if review is None:
        return []
    expected = list(range(len(report.claims)))
    if [verdict.index for verdict in review.verdicts] != expected:
        raise ValueError('review must contain one ordered verdict for every claim')
    out = []
    for v in review.verdicts:
        if v.verdict != 'accepted':
            continue
        claim = report.claims[v.index]
        if (v.verification == 'source_confirmed') != _is_source_url(claim.source):
            log.warning('dropping claim %d (%s): %s does not fit source %r',
                        v.index, unit, v.verification, claim.source)
            continue
        if v.verification == 'source_confirmed':
            if _source_url(v.source_ref) != _source_url(claim.source):
                log.warning('dropping claim %d (%s): source_ref does not match claim source', v.index, unit)
                continue
            group = f'source:{claim.source}'
        else:
            if workspace is None:
                log.warning('dropping claim %d (%s): no workspace supplied for artifact ownership', v.index, unit)
                continue
            source = _owned_artifact(claim.source, workspace)
            source_ref = _owned_artifact(v.source_ref, workspace)
            # Distinct reasons: conflating them cost real debugging time, because
            # "does not match" was logged for paths that matched each other
            # exactly and were simply not owned by any recorded run.
            if source is None:
                log.warning('dropping claim %d (%s): claim source %r is not owned by any recorded '
                            'run of this hypothesis (its run dir or a batch it submitted)',
                            v.index, unit, claim.source)
                continue
            if source_ref is None:
                log.warning('dropping claim %d (%s): review source_ref %r is not owned by any '
                            'recorded run of this hypothesis', v.index, unit, v.source_ref)
                continue
            if source != source_ref:
                log.warning('dropping claim %d (%s): review checked %s but the claim cites %s',
                            v.index, unit, source_ref[1], source[1])
                continue
            run_id, _ = source
            group = f'run:{run_id}'
        out.append({**claim.model_dump(), 'verification': v.verification, 'source_ref': v.source_ref,
                    'checked': v.checked, 'policy': v.verification, 'group': group,
                    'delta': round(claim_delta(claim.stance, v.verification), 4)})
    return out


def record(ledger: Path, round_no: int, hypothesis_id: str, claims: list[dict]) -> None:
    with ledger.open('a') as f:
        for claim in claims:
            f.write(json.dumps({'round': round_no, 'hypothesis_id': hypothesis_id, **claim}) + '\n')


def drop_round(ledger: Path, round_no: int) -> None:
    """Atomically canonicalize the ledger, removing malformed and retried-round rows."""
    if not ledger.is_file():
        return
    rows = [row for row in read_ledger(ledger) if row.get('round') != round_no]
    tmp = ledger.with_name(ledger.name + '.tmp')
    tmp.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    os.replace(tmp, ledger)
