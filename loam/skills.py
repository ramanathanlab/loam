"""Loam skills (directories holding a SKILL.md) as a catalog plus one reader tool."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .systems import SYSTEMS


def discover(dirs: list[Path]) -> dict[str, Path]:
    """name -> skill directory, from every SKILL.md under the given dirs."""
    found = {}
    for root in dirs:
        for md in sorted(root.rglob('SKILL.md')):
            meta = _frontmatter(md)
            found[meta.get('name') or md.parent.name] = md.parent
    return found


def catalog(skills: dict[str, Path]) -> str:
    lines = [f'- {name}: {_frontmatter(d / "SKILL.md").get("description", "")}' for name, d in skills.items()]
    return 'AVAILABLE SKILLS (read one with read_skill before using its tool):\n' + '\n'.join(lines)


def read(skills: dict[str, Path], name: str, file: str | None = None) -> str:
    if name not in skills:
        return f'No skill named {name!r}. Available: {", ".join(skills)}'
    root = skills[name].resolve()
    target = (root / (file or 'SKILL.md')).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        files = [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]
        return f'No file {file!r} in skill {name}. Files: {files}'
    return target.read_text(errors='replace')


# Paths that belong to one System's filesystem or one person's machine. A SKILL.md reaches software
# only through Site profile variables such as $LOAM_SOFTWARE_ROOT; System paths go in a reference file.
FORBIDDEN_PATHS = ('/flare/', '/lus/', '/e/project', '/Users/', '/opt/aurora')
# An allocation account written out (`-A NAME`, `--account NAME`, `account: NAME`) rather than taken from
# a variable; accounts come from the Site profile.
HARD_CODED_ACCOUNT = re.compile(r'(?:\s-A\s+|--account[=\s]+|\baccount:\s*)(?![\s"\'$]*\$)[\w"\'-]')


def layout_problems(skill: Path) -> list[str]:
    """What is wrong with a skill's layout: an empty list when it is System-neutral and complete.

    The layout: SKILL.md lists the Systems it supports in its frontmatter (`systems: [aurora, jupiter]`)
    and holds no System-specific or personal paths and no accounts. Each claimed System has
    references/<system>.md, the agent's instructions for that System, and no other System does.
    A skill that does not claim every System says it is unsupported elsewhere.
    """
    text = (skill / 'SKILL.md').read_text(errors='replace')
    problems = []
    claimed = _frontmatter(skill / 'SKILL.md').get('systems')
    if not isinstance(claimed, list) or not claimed:
        problems.append('SKILL.md: frontmatter must list the Systems it supports, e.g. `systems: [aurora]`')
        claimed = []
    claimed = [str(system) for system in claimed]
    for system in claimed:
        if system not in SYSTEMS:
            problems.append(f'SKILL.md: {system!r} is not a defined System ({", ".join(SYSTEMS)})')
        elif not (skill / 'references' / f'{system}.md').is_file():
            problems.append(f'SKILL.md claims {system!r} but references/{system}.md is missing')
    problems += [f'references/{system}.md exists but SKILL.md does not claim {system!r}'
                 for system in SYSTEMS if system not in claimed and (skill / 'references' / f'{system}.md').is_file()]
    if set(SYSTEMS) - set(claimed) and 'unsupported' not in text.lower():
        problems.append('SKILL.md must say the skill is unsupported on the Systems it does not claim')
    for line_no, line in enumerate(text.splitlines(), 1):
        problems += [f'SKILL.md:{line_no}: System-specific or personal path {path!r}'
                     for path in FORBIDDEN_PATHS if path in line]
        if HARD_CODED_ACCOUNT.search(line):
            problems.append(f'SKILL.md:{line_no}: hard-coded allocation account')
    return problems


def _frontmatter(md: Path) -> dict:
    text = md.read_text(errors='replace')
    if not text.startswith('---'):
        return {}
    try:
        return yaml.safe_load(text.split('---', 2)[1]) or {}
    except yaml.YAMLError:
        return {}
