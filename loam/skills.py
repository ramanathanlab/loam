"""Loam skills (directories holding a SKILL.md) as a catalog plus one reader tool."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
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
    # Preserve insertion order and caller-provided names, and stat every file
    # on every lookup so edits/deletion/retargeting remain visible. Strings
    # are immutable: sharing a rendered catalog cannot share caller mutations.
    versions = []
    for name, directory in skills.items():
        path = (directory / 'SKILL.md').absolute()
        versions.append((name, path, _file_version(path)))
    return _cached_catalog(tuple(versions))


@lru_cache(maxsize=128)
def _cached_catalog(versions) -> str:
    lines = [f'- {name}: {_frontmatter(path).get("description", "")}' for name, path, _ in versions]
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


def _file_version(md: Path) -> tuple[int, ...]:
    stat = md.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=512)
def _cached_frontmatter(md: Path, version: tuple[int, ...]) -> dict:
    # The version is intentionally part of the key: replacement, edits,
    # permissions and symlink retargeting invalidate old metadata naturally.
    return _read_frontmatter(md)


def _frontmatter(md: Path) -> dict:
    md = md.absolute()
    for _ in range(3):
        version = _file_version(md)
        result = _cached_frontmatter(md, version)
        if _file_version(md) == version:
            # Parsing used to return a new object for each caller. In
            # particular, YAML anchors/nested mutable values must not become
            # shared state through this metadata cache.
            return deepcopy(result)
    # An actively rewritten file must not make this reader spin indefinitely.
    return _read_frontmatter(md)


def _read_frontmatter(md: Path) -> dict:
    text = md.read_text(errors='replace')
    if not text.startswith('---'):
        return {}
    try:
        return yaml.safe_load(text.split('---', 2)[1]) or {}
    except yaml.YAMLError:
        return {}
