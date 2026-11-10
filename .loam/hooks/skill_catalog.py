"""Resolve a campaign's `skills:` list into the concrete skills agents can read.

The skill library has two levels: a *task* directory (`structure-prediction`)
holding one *tool* skill per subdirectory (`chai/SKILL.md`, `boltz/SKILL.md`),
plus a few standalone skills at the top level (`hpc-deployment/SKILL.md`).
Hermes resolves `skill_view` only on a tool's own name -- a task name has no
SKILL.md and answers 404, which is how a campaign naming five task directories
once sent its children through 108 failed `skill_view` calls.

So a campaign may name either level, and this module turns that into what the
agents need:

- a task name expands to every tool under it, so the validator can choose the
  task first and then the tool that fits the step;
- a tool name pins that single tool (use it when results must stay comparable
  across validators, e.g. one folding model for the whole campaign);
- a standalone skill is itself.

Every resolved entry carries the bare tool name, which is what `skill_view`
accepts and what the workspace and metrics code match reads against.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - Hermes normally supplies PyYAML
    yaml = None

SKILL_FILE = 'SKILL.md'
# Folders inside a skill that hold its supporting files, never further skills.
_SUPPORT_DIRS = frozenset({'references', 'templates', 'assets', 'scripts'})
_DESCRIPTION_LIMIT = 220


@dataclass(frozen=True)
class Skill:
    name: str          # what skill_view takes
    task: str          # the task directory it sits in; '' for a standalone skill
    description: str


@dataclass(frozen=True)
class Resolution:
    skills: tuple[Skill, ...] = ()
    missing: tuple[str, ...] = ()      # configured names nothing matched
    ambiguous: tuple[str, ...] = ()    # configured names more than one skill matched

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(skill.name for skill in self.skills)

    def errors(self) -> list[str]:
        problems = [f'campaign skill {name!r} matches no task or tool in the skill library' for name in self.missing]
        problems += [f'campaign skill {name!r} matches more than one skill; name it as task/tool' for name in self.ambiguous]
        return problems


def library_roots(repo_root: str | os.PathLike[str] | None = None) -> list[Path]:
    """Where Hermes will look: the running home's library, else this checkout's tree.

    `run-campaign` links the checkout's `skills/` into `$HERMES_HOME/skills`, so
    the home is the truth at run time. The checkout is the fallback for a shell
    or test with no home set.
    """

    home = (os.environ.get('HERMES_HOME') or '').strip()
    if home:
        library = Path(home).expanduser() / 'skills'
        if library.is_dir():
            return [library]
    if repo_root:
        tree = Path(repo_root) / 'skills'
        if tree.is_dir():
            return [tree]
    return []


def _front_matter(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return {}
    match = re.match(r'---\r?\n(.*?)\r?\n---', text, re.DOTALL)
    if not match:
        return {}
    block = match.group(1)
    if yaml is not None:
        try:
            loaded = yaml.safe_load(block)
        except yaml.YAMLError:
            loaded = None
        if isinstance(loaded, dict):
            return loaded
    # Minimal fallback: `key: value`, with indented continuation lines folded in
    # (covers the `description: >-` blocks several skills use).
    fields: dict[str, Any] = {}
    key = ''
    for line in block.splitlines():
        if line[:1].isspace() and key:
            fields[key] = f"{fields[key]} {line.strip()}".strip()
            continue
        name, separator, value = line.partition(':')
        if separator:
            key = name.strip()
            value = value.strip()
            fields[key] = '' if value in ('>', '>-', '|', '|-') else value
    return fields


def _summary(text: Any) -> str:
    """The first sentence of a description, bounded, on one line."""

    flat = ' '.join(str(text or '').split())
    if len(flat) <= _DESCRIPTION_LIMIT:
        return flat
    cut = flat.find('. ', 0, _DESCRIPTION_LIMIT)
    return flat[:cut + 1] if cut > 40 else flat[:_DESCRIPTION_LIMIT].rstrip() + '…'


@dataclass(frozen=True)
class _Entry:
    directory: Path
    task: str


def _index(roots: Iterable[Path]) -> tuple[list[_Entry], dict[str, list[_Entry]]]:
    """Every skill under the roots, plus each task directory's tools."""

    entries: list[_Entry] = []
    tasks: dict[str, list[_Entry]] = {}
    for root in roots:
        for current, dirs, files in os.walk(root, followlinks=True):
            here = Path(current)
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in _SUPPORT_DIRS)
            if here == root or SKILL_FILE not in files:
                continue
            parts = here.relative_to(root).parts
            entry = _Entry(here, parts[0] if len(parts) >= 2 else '')
            entries.append(entry)
            if entry.task:
                tasks.setdefault(entry.task, []).append(entry)
            dirs[:] = []  # a skill's own subfolders are its files, not more skills
    return entries, tasks


def resolve(names: Iterable[str], roots: Iterable[Path]) -> Resolution:
    """Resolve configured task/tool names, in order, without duplicates."""

    entries, tasks = _index(list(roots))
    chosen: list[_Entry] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    front: dict[Path, dict[str, Any]] = {}

    def fields(entry: _Entry) -> dict[str, Any]:
        if entry.directory not in front:
            front[entry.directory] = _front_matter(entry.directory / SKILL_FILE)
        return front[entry.directory]

    for raw in names:
        name = str(raw).strip().strip('/')
        if not name:
            continue
        if name in tasks:
            matches = tasks[name]
        elif '/' in name:
            task, _, tool = name.rpartition('/')
            matches = [e for e in entries if e.task == task and e.directory.name == tool]
        else:
            matches = [e for e in entries if e.directory.name == name] or [
                e for e in entries if fields(e).get('name') == name
            ]
            if len(matches) > 1:
                ambiguous.append(name)
                continue
        if not matches:
            missing.append(name)
            continue
        chosen += [m for m in matches if m not in chosen]

    skills = []
    for entry in chosen:
        meta = fields(entry)
        skills.append(Skill(
            # The directory name is what Hermes's recursive lookup matches first.
            name=entry.directory.name,
            task=entry.task,
            description=_summary(meta.get('description')),
        ))
    return Resolution(tuple(skills), tuple(missing), tuple(ambiguous))


def resolve_for(payload: Mapping[str, Any]) -> Resolution:
    """Resolve the payload's `loam_campaign.skills` against the running library."""

    campaign = payload.get('loam_campaign') or {}
    names = (campaign.get('skills') or ()) if isinstance(campaign, Mapping) else ()
    if not names:
        return Resolution()
    return resolve(names, library_roots(payload.get('repo_root')))


def catalog_lines(skills: Iterable[Skill]) -> list[str]:
    """Skills grouped by task, one tool per line: `- task:` then `  - `tool` -- what it does`."""

    lines: list[str] = []
    current = None
    for skill in skills:
        if skill.task != current:
            current = skill.task
            if skill.task:
                lines.append(f'- {skill.task}:')
        indent = '  ' if skill.task else ''
        lines.append(f"{indent}- `{skill.name}`" + (f' -- {skill.description}' if skill.description else ''))
    return lines


def catalog_mapping(skills: Iterable[Skill]) -> dict[str, dict[str, str]]:
    """The same catalog as a JSON-able {task: {tool: description}}; standalone skills under ''."""

    out: dict[str, dict[str, str]] = {}
    for skill in skills:
        out.setdefault(skill.task, {})[skill.name] = skill.description
    return out
