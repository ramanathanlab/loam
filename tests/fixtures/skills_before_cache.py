"""Loam skills (directories holding a SKILL.md) as a catalog plus one reader tool."""

from __future__ import annotations

from pathlib import Path

import yaml


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


def _frontmatter(md: Path) -> dict:
    text = md.read_text(errors='replace')
    if not text.startswith('---'):
        return {}
    try:
        return yaml.safe_load(text.split('---', 2)[1]) or {}
    except yaml.YAMLError:
        return {}
