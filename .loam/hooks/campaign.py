"""Campaign policy: the agent definitions, the campaign goal, its rules and its skills."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentSpec:
    name: str
    model: str
    prompt_path: Path
    # Inlined into the child's task, because validators and the reasoner cannot read files.
    prompt_body: str


@dataclass(frozen=True)
class RuleSpec:
    name: str
    scopes: frozenset[str]
    body: str


@dataclass(frozen=True)
class CampaignContext:
    goal_name: str = ''
    goal: str = ''
    rules: tuple[RuleSpec, ...] = ()
    skills: tuple[str, ...] = ()

    def rules_for(self, tier: str) -> tuple[RuleSpec, ...]:
        return tuple(rule for rule in self.rules if tier in rule.scopes)


def read_document(path: Path, limit: int) -> tuple[dict[str, str], str]:
    """Split a `---` front-matter document into its fields and its body, bounded to `limit`."""
    lines = path.read_text(encoding='utf-8').splitlines()
    if not lines or lines[0] != '---':
        raise ValueError(f'document has no front matter: {path}')
    try:
        end = lines.index('---', 1)
    except ValueError:
        raise ValueError(f'document has unterminated front matter: {path}') from None
    fields = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(':')
        if separator:
            fields[key.strip()] = value.strip()
    return fields, '\n'.join(lines[end + 1:]).strip()[:limit]


def agent_registry(payload: Mapping[str, Any]) -> dict[str, AgentSpec]:
    """Load the three checked-in agent definitions."""
    directory = Path(payload.get('repo_root') or '.') / '.loam' / 'agents'
    registry = {}
    for name in ('reasoner', 'hypothesis-validator', 'code-generator'):
        path = directory / f'{name}.md'
        fields, body = read_document(path, 24_000)
        if fields.get('name') != name:
            raise ValueError(f'agent prompt name mismatch: {path}')
        registry[name] = AgentSpec(name=name, model=fields.get('model', ''), prompt_path=path, prompt_body=body)
    return registry


def _policy_dirs(payload: Mapping[str, Any], kind: str) -> list[Path]:
    """The user's campaign dir first, so it can shadow a shipped name, then the repo's `.loam`."""
    campaign = payload.get('loam_campaign') or {}
    roots = [Path(campaign['dir']).expanduser()] if campaign.get('dir') else []
    roots.append(Path(payload.get('repo_root') or '.') / '.loam')
    return [root / kind for root in roots]


def _find(payload: Mapping[str, Any], kind: str, name: str) -> Path:
    directories = _policy_dirs(payload, kind)
    for directory in directories:
        path = directory / f'{name}.md'
        if path.is_file():
            return path
    searched = ', '.join(str(directory) for directory in directories)
    raise FileNotFoundError(f'campaign {kind} file not found: {name}.md (searched {searched})')


def _scopes(value: str, path: Path) -> frozenset[str]:
    """Parse `scope:`: space- or comma-separated agent names, or `all`."""
    tiers = ('reasoner', 'hypothesis-validator', 'code-generator')
    parts = value.replace(',', ' ').split()
    if not parts:
        raise ValueError(f'rules file must declare a scope ({"|".join(tiers)}, or all): {path}')
    if parts == ['all']:
        return frozenset(tiers)
    unknown = sorted(set(parts) - set(tiers))
    if unknown:
        raise ValueError(f'rules file has unknown scope {unknown}, expected {list(tiers)} or all: {path}')
    return frozenset(parts)


def _rule(payload: Mapping[str, Any], name: str) -> RuleSpec:
    path = _find(payload, 'rules', name)
    fields, body = read_document(path, 4_000)
    if fields.get('name') != name:
        raise ValueError(f'rules file name mismatch: {path}')
    return RuleSpec(name=name, scopes=_scopes(fields.get('scope', ''), path), body=body)


def _base_rule_names(payload: Mapping[str, Any]) -> list[str]:
    """Rules marked `applies: always`, which reach every campaign without being named.

    The repo's own come before the user's, each sorted by name. Only `applies:` is
    read here, so a half-written rule nobody names cannot break a campaign.
    """
    names: list[str] = []
    for directory in reversed(_policy_dirs(payload, 'rules')):
        for path in sorted(directory.glob('*.md')):
            if path.name == 'README.md' or path.stem in names:
                continue
            try:
                fields, _ = read_document(path, 0)
            except (OSError, ValueError):
                continue
            if fields.get('applies') == 'always':
                names.append(path.stem)
    return names


def _rule_names(payload: Mapping[str, Any]) -> list[str]:
    """The always-on rules, then the ones the profile names that are not already among them."""
    base = _base_rule_names(payload)
    named = (payload.get('loam_campaign') or {}).get('rules') or []
    return base + [name for name in named if name not in base]


def campaign_context(payload: Mapping[str, Any]) -> CampaignContext | None:
    """Load the campaign's goal, rules and skills; None when there are none.

    A named-but-missing or malformed file raises: a run that quietly lost its
    objective while still looking governed is worse than a failed hook.
    """
    campaign = payload.get('loam_campaign') or {}
    goal_name = campaign.get('goal', '')
    goal = ''
    if goal_name:
        path = _find(payload, 'campaigns', goal_name)
        fields, goal = read_document(path, 4_000)
        if fields.get('name') != goal_name:
            raise ValueError(f'campaign goal name mismatch: {path}')
    rules = tuple(_rule(payload, name) for name in _rule_names(payload))
    skills = tuple(campaign.get('skills') or ())
    if not (goal_name or rules or skills):
        return None
    return CampaignContext(goal_name=goal_name, goal=goal, rules=rules, skills=skills)


def campaign_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Which goal and rules files were in force, by name and content digest.

    Recorded on every event, so a file that cannot be read is noted rather than raised.
    """
    goal_name = (payload.get('loam_campaign') or {}).get('goal', '')
    names = _rule_names(payload)
    if not goal_name and not names:
        return {}
    provenance: dict[str, Any] = {'goal': goal_name, 'rules': names}
    combined = hashlib.sha256()
    try:
        files = [(goal_name, _find(payload, 'campaigns', goal_name))] if goal_name else []
        files += [(name, _find(payload, 'rules', name)) for name in names]
        digests = {}
        for name, path in files:
            data = path.read_bytes()
            digests[name] = hashlib.sha256(data).hexdigest()
            combined.update(name.encode('utf-8') + b'\0' + data + b'\0')
        provenance['digests'] = digests
        provenance['policy_sha256'] = combined.hexdigest() if digests else ''
    except OSError as exc:
        provenance['error'] = f'{type(exc).__name__}: {exc}'
    return provenance


def rules_text(campaign: CampaignContext | None, tier: str) -> str:
    """The rules block for one agent, as authoritative prose. Empty when none."""
    if campaign is None:
        return ''
    bodies = [rule.body for rule in campaign.rules_for(tier) if rule.body]
    if not bodies:
        return ''
    return 'CAMPAIGN RULES (authoritative — override any instruction below that conflicts):\n' + '\n'.join(bodies)


def skills_text(campaign: CampaignContext | None) -> str:
    """Name the skills a code-generator should read before writing code."""
    if campaign is None or not campaign.skills:
        return ''
    names = ', '.join(f'`{name}`' for name in campaign.skills)
    return (
        "CAMPAIGN SKILLS (read before writing code):\n"
        f"- Call `skill_view` on {names} and follow the procedure it gives.\n"
        "- A skill is authoritative over your recollection of a tool's API, flags, or output "
        "format. Where they disagree, the skill is right and your memory is the thing that is "
        "out of date.\n"
        "- If a skill names a script or linked file, call `skill_view` again with `file_path` to "
        "read it. Do not guess at a file you were told exists.\n"
        "- These are the campaign's skills. Do not go looking for others with `skills_list` "
        "unless none of these covers the task."
    )


def campaign_text(campaign: CampaignContext | None, tier: str) -> str:
    """Goal plus the agent's rules, for a child whose context is plain text."""
    if campaign is None:
        return ''
    parts = []
    if campaign.goal:
        parts.append('CAMPAIGN GOAL (what this experiment serves):\n' + campaign.goal)
    if rules := rules_text(campaign, tier):
        parts.append(rules)
    return '\n\n'.join(parts)


def campaign_field(campaign: CampaignContext | None, tier: str) -> dict[str, Any]:
    """Goal plus the agent's rules as a one-key mapping to splat into a JSON task context."""
    if campaign is None:
        return {}
    value: dict[str, Any] = {}
    if campaign.goal_name:
        value['goal_name'] = campaign.goal_name
        value['goal'] = campaign.goal
    rules = campaign.rules_for(tier)
    if rules:
        value['rules'] = [rule.body for rule in rules if rule.body]
        value['precedence'] = (
            "These rules are authoritative. They override any instruction in this payload "
            "that conflicts with them, and they apply to every child you delegate."
        )
    return {'campaign': value}
