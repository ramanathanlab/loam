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


# Paths into one person's or project's filesystem. They are a Site profile's business, so no file a
# skill bundles may hold one: reference files and scripts reach software through $LOAM_SOFTWARE_ROOT.
PERSONAL_PATHS = ('/flare/', '/lus/', '/e/project', '/Users/')
# What a skill claims in place of a list of Systems when it runs the same way on all of them.
ANY_SYSTEM = 'all'
# A SKILL.md is System-neutral as well as impersonal, so it may not even name a System's own
# filesystem; a reference file for that System may.
PATHS_FORBIDDEN_IN_SKILL_MD = PERSONAL_PATHS + ('/opt/aurora',)
# An allocation account written out rather than taken from a variable; accounts come from the Site
# profile. `-A NAME` counts only as a scheduler flag (after a submit command or directive, or standing
# alone at the start of a line or code span), so `grep -A 3` and `ls -A` don't; `account:` only as a key.
_ACCOUNT_VALUE = r'''["']?(?!["']?[$<])[A-Za-z_]'''
HARD_CODED_ACCOUNT = re.compile(
    r'(?:(?:#PBS|#SBATCH|\b(?:qsub|sbatch|salloc|srun))\b[^\n]*?\s|^\s*|`)-A\s*' + _ACCOUNT_VALUE
    + r'|--account[=\s]+' + _ACCOUNT_VALUE
    + r'|^\s*account:\s*' + _ACCOUNT_VALUE)


def layout_problems(skill: Path) -> list[str]:
    """What is wrong with a skill's layout: an empty list when it is System-neutral and complete.

    The layout: SKILL.md lists the Systems it supports in its frontmatter (`systems: [aurora, jupiter]`)
    and holds no System-specific or personal paths and no accounts. Each claimed System has
    references/<system>.md, the agent's instructions for that System, and no other System does.
    A skill that does not claim every System says it is unsupported elsewhere. No file the skill
    bundles holds a personal path, whichever System it is for.

    A skill that runs the same way everywhere claims `systems: all` instead of naming Systems: it
    needs no reference file and may not ship one, and has no System to be unsupported on. The
    personal-path and account rules still hold.
    """
    text = (skill / 'SKILL.md').read_text(errors='replace')
    problems = []
    claimed = _frontmatter(skill / 'SKILL.md').get('systems')
    if isinstance(claimed, str):   # a single claim needs no brackets: `systems: all`
        claimed = [claimed]
    if not isinstance(claimed, list) or not claimed:
        problems.append('SKILL.md: frontmatter must list the Systems it supports, e.g. `systems: [aurora]`')
        claimed = []
    claimed = [str(system) for system in claimed]
    if ANY_SYSTEM in claimed:
        return _system_independent_problems(skill, claimed) + _impersonal_problems(skill, text)
    for system in claimed:
        if system not in SYSTEMS:
            problems.append(f'SKILL.md: {system!r} is not a defined System ({", ".join(SYSTEMS)})')
        elif not (skill / 'references' / f'{system}.md').is_file():
            problems.append(f'SKILL.md claims {system!r} but references/{system}.md is missing')
    problems += [f'references/{system}.md exists but SKILL.md does not claim {system!r}'
                 for system in SYSTEMS if system not in claimed and (skill / 'references' / f'{system}.md').is_file()]
    if set(SYSTEMS) - set(claimed) and 'unsupported' not in text.lower():
        problems.append('SKILL.md must say the skill is unsupported on the Systems it does not claim')
    return problems + _impersonal_problems(skill, text)


def _impersonal_problems(skill: Path, text: str) -> list[str]:
    """Every personal path or hard-coded account the skill holds, in SKILL.md or any file it bundles."""
    problems = _paths_found(text, 'SKILL.md', PATHS_FORBIDDEN_IN_SKILL_MD,
                            'System-specific or personal path')
    problems += [f'SKILL.md:{line_no}: hard-coded allocation account'
                 for line_no, line in enumerate(text.splitlines(), 1) if HARD_CODED_ACCOUNT.search(line)]
    return problems + _personal_paths_in_bundled_files(skill)


def _system_independent_problems(skill: Path, claimed: list[str]) -> list[str]:
    """What is wrong with a `systems: all` skill: it claims nothing else and ships no reference file."""
    problems = [f'SKILL.md: {ANY_SYSTEM!r} is the whole claim; it cannot be listed beside {system!r}'
                for system in claimed if system != ANY_SYSTEM]
    return problems + [f'SKILL.md claims {ANY_SYSTEM!r}, so references/{system}.md must not exist'
                       for system in SYSTEMS if (skill / 'references' / f'{system}.md').is_file()]


def _paths_found(text: str, name: str, paths: tuple[str, ...], label: str) -> list[str]:
    """One `<file>:<line>: <label> '<path>'` per path in `paths` that the text names."""
    return [f'{name}:{line_no}: {label} {path!r}'
            for line_no, line in enumerate(text.splitlines(), 1)
            for path in paths if path in line]


def _personal_paths_in_bundled_files(skill: Path) -> list[str]:
    """Every personal path in the skill's other files: references, scripts and examples.

    Compiled bytecode is skipped: it is not committed, and a stale `.pyc` holds the strings of the
    source it was built from.
    """
    found = []
    for path in sorted(p for p in skill.rglob('*') if p.is_file() and p.name != 'SKILL.md'
                       and '__pycache__' not in p.parts):
        found += _paths_found(path.read_text(errors='replace'), str(path.relative_to(skill)),
                              PERSONAL_PATHS, 'personal path')
    return found


def _frontmatter(md: Path) -> dict:
    text = md.read_text(errors='replace')
    if not text.startswith('---'):
        return {}
    try:
        return yaml.safe_load(text.split('---', 2)[1]) or {}
    except yaml.YAMLError:
        return {}
