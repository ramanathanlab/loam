"""Portable orchestration state for Jnana-ranked hypothesis validation.

The harness LLM remains the scheduler: Hermes exposes delegated work only through
``delegate_task``.  This module makes the role contracts, selected hypotheses,
child completion state, and next-turn reasoner hand-off explicit without treating
model output as validated evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from falda_campaign import (
    FALDA_TOOL_PREFIX,
    falda_tool_policy,
    generation_memory,
    reasoner_memory_block,
    tools_available,
    write_validator_outcome,
)
from falda_memory import config as falda_config
import hypothesis_workspace

AGENT_NAMES = ("reasoner", "hypothesis-validator", "code-generator")
STATE_FILENAME = "hypothesis-orchestration.jsonl"
CAMPAIGN_FILENAME = "hypothesis-campaign.jsonl"
MAX_REASONER_THREAD_CHARS = 6_000
EVIDENCE_SCORE_HEADING = "### Evidence Score"
# The reasoner's brief IS the next tournament's input, so this bound caps the
# quality of every hypothesis the next turn can generate. At 2_000 it was a
# paragraph standing in for a turn's worth of synthesis across several
# validator threads.
MAX_BRIEF_CHARS = 8_000
MAX_CHILD_GOAL_CHARS = 300
# Bound on the hypothesis text forwarded into a child's context.
MAX_HYPOTHESIS_CHARS = 4_000
# Bound on an injected agent definition. The checked-in prompts are ~4-10 KB;
# this leaves headroom while keeping a runaway edit out of every child context.
MAX_AGENT_PROMPT_CHARS = 24_000
# Campaign context competes with the hypothesis for the front of a validator's
# payload (see `_hypothesis_block`), so both halves are bounded tightly.
MAX_CAMPAIGN_CHARS = 4_000
MAX_RULES_CHARS = 4_000
RULE_TIERS = ("validator", "code-generator", "reasoner")
RULE_ALWAYS = "always"
RULE_WHEN_NAMED = "when-named"
# Tools of the hpc-deployment MCP server a generated-mode code-generator may use.
HPC_TOOLS = ("run_code", "run_batch", "batch_status", "list_batches", "register_tool", "list_tools", "inspect_filesystem")
# The subset of HPC_TOOLS that only looks: no compute, no registration, no
# submission. A hypothesis-validator may call these itself (see
# `_observation_tools`); everything executable still goes through a child.
VALIDATOR_OBSERVATION_TOOLS = ("batch_status", "inspect_filesystem", "list_batches")
# Read-only skill access for the tier that writes the code. A code-generator was
# confined to the HPC tools, so it wrote GPU code from memory while a reviewed
# recipe sat in the campaign's skill library unreadable. These two only read:
# `skill_manage` is deliberately absent, because the child consults skills and
# does not author them.
SKILL_TOOLS = ("skills_list", "skill_view")
# The tool the adapter registers so the coordinator can launch a round without
# transcribing its arguments. Kept in step with DISPATCH_TOOL_NAME in
# .loam/adapters/hermes/plugin/__init__.py, which tests/test_round_dispatch.py
# asserts, because a drift here silently tells the model to call a tool that
# does not exist.
DISPATCH_TOOL = "loam_dispatch_round"
_LOAM_TAG = re.compile(r"\[loam-[a-z-]+=[^\]]*\]")


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    model: str
    prompt_path: Path
    allowed_children: tuple[str, ...]
    direct_execution: bool
    # The prompt body, for injection into the child's task context. It cannot be
    # delivered as a path to read: `tool_policy` allows a hypothesis-validator
    # `delegate_task` plus the read-only HPC observation tools, none of which
    # reads a local file, so a validator told to read its own definition gets
    # back "hypothesis-validator must delegate executable work to
    # code-generator" and then runs with no role instructions at all. The
    # reasoner is likewise restricted. Only a code-generator could read a path,
    # and it already receives its contract inline.
    prompt_body: str = ""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(payload: Mapping[str, Any]) -> Path:
    state_dir = _text(payload.get("loam_state_dir")) or os.environ.get("LOAM_STATE_DIR", "").strip()
    if state_dir:
        return Path(state_dir).expanduser() / STATE_FILENAME
    root = Path(_text(payload.get("repo_root")) or ".")
    return root / ".loam" / "state" / STATE_FILENAME



def _campaign_path(payload: Mapping[str, Any]) -> Path:
    return _state_path(payload).with_name(CAMPAIGN_FILENAME)

def _agent_dir(payload: Mapping[str, Any]) -> Path:
    root = Path(_text(payload.get("repo_root")) or ".")
    return root / ".loam" / "agents"


def _front_matter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"agent prompt has no front matter: {path}")
    values: dict[str, str] = {}
    for line in lines[1:]:
        if line == "---":
            return values
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    raise ValueError(f"agent prompt has unterminated front matter: {path}")


def _prompt_body(path: Path, limit: int = MAX_AGENT_PROMPT_CHARS) -> str:
    """Return a front-matter document's body, bounded in size.

    Shared by the agent prompts and the campaign/rules files, which use the same
    `---` front matter followed by prose.
    """

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0] == "---":
        for index in range(1, len(lines)):
            if lines[index] == "---":
                lines = lines[index + 1 :]
                break
        else:
            raise ValueError(f"agent prompt has unterminated front matter: {path}")
    return "\n".join(lines).strip()[:limit]


def agent_registry(payload: Mapping[str, Any]) -> dict[str, AgentSpec]:
    """Load the checked-in prompt roles and their non-negotiable capabilities."""

    policies = {
        "reasoner": ((), False),
        "hypothesis-validator": (("code-generator",), False),
        "code-generator": ((), True),
    }
    registry: dict[str, AgentSpec] = {}
    for name in AGENT_NAMES:
        path = _agent_dir(payload) / f"{name}.md"
        front_matter = _front_matter(path)
        if front_matter.get("name") != name:
            raise ValueError(f"agent prompt name mismatch: {path}")
        allowed_children, direct_execution = policies[name]
        registry[name] = AgentSpec(
            name=name,
            description=_text(front_matter.get("description")),
            model=_text(front_matter.get("model")),
            prompt_path=path,
            allowed_children=allowed_children,
            direct_execution=direct_execution,
            prompt_body=_prompt_body(path),
        )
    return registry


@dataclass(frozen=True)
class RuleSpec:
    """One campaign rules file: policy text plus the tiers it applies to."""

    name: str
    scopes: frozenset[str]
    body: str
    # True when the rule applies to every campaign without being named, because
    # it states something invariant (what makes a datum trustworthy; how an
    # experiment is run and reported) rather than a domain's assay. This text was
    # unconditional while it lived inline in `_hypothesis_block`; moving it into
    # files must not quietly turn invariant policy into an opt-in.
    always: bool = False
    # Advisory note naming the `tool_policy` clause that actually enforces this
    # rule, when one exists. Injected text cannot enforce anything on its own
    # (see `_SCRUBBERS`, which exists because it could not), so a rule that must
    # hold needs a veto in `pre-tool.py`; recording which keeps the distinction
    # between "stated" and "enforced" visible instead of implied.
    enforced_by: str = ""


@dataclass(frozen=True)
class CampaignContext:
    """The invariant half of a campaign's context: its goal and its rules.

    The evolving half -- current state -- is not here. It already reaches the
    model as the reasoner brief and the reconciled findings the recall hooks
    inject.
    """

    goal_name: str = ""
    goal: str = ""
    rules: tuple[RuleSpec, ...] = ()
    skills: tuple[str, ...] = ()

    def rules_for(self, tier: str) -> tuple[RuleSpec, ...]:
        return tuple(rule for rule in self.rules if tier in rule.scopes)


def campaign_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Which campaign policy was in force, by name and content digest.

    Moving the objective and the rules out of this hook and into editable files
    makes a result harder to attribute than hardcoding did, unless the policy
    that produced it is recorded beside it. A digest is enough: it says whether
    two runs were governed identically, and names the file to go read.

    Best effort by design. A missing or malformed policy file must fail loudly
    where it is injected (`campaign_context`), but the manifest and ledger are
    written on every event and must not fall over with it, so a read failure is
    recorded rather than raised.
    """

    campaign = payload.get("loam_campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    goal_name = _text(campaign.get("goal"))
    raw_rules = campaign.get("rules")
    named = [name for item in (raw_rules if isinstance(raw_rules, list) else []) if (name := _text(item))]
    # The invariant rules govern the run whether or not a profile named them, so
    # they are part of what produced the result and belong in the digest.
    base = _base_rule_names(payload)
    names = base + [name for name in named if name not in base]
    if not goal_name and not names:
        return {}
    provenance: dict[str, Any] = {"goal": goal_name, "rules": names}
    combined = hashlib.sha256()
    try:
        files = [(goal_name, _find_policy_file(_campaign_dirs(payload), goal_name, "goal"))] if goal_name else []
        files += [(name, _find_policy_file(_rules_dirs(payload), name, "rules")) for name in names]
        digests: dict[str, str] = {}
        for name, path in files:
            data = path.read_bytes()
            digests[name] = hashlib.sha256(data).hexdigest()
            combined.update(name.encode("utf-8") + b"\0" + data + b"\0")
        provenance["digests"] = digests
        provenance["policy_sha256"] = combined.hexdigest() if digests else ""
    except OSError as exc:
        provenance["error"] = f"{type(exc).__name__}: {exc}"
    return provenance


def _rules_text(campaign: CampaignContext | None, tier: str) -> str:
    """The rules block for one tier, as authoritative prose. Empty when none."""

    if campaign is None:
        return ""
    bodies = [rule.body for rule in campaign.rules_for(tier) if rule.body]
    if not bodies:
        return ""
    # The precedence wording is load-bearing, not decoration: a validator's
    # instruction to its child may contradict campaign policy outright, and this
    # is the sentence that settles it. Kept identical to `_hypothesis_block`'s.
    return "CAMPAIGN RULES (authoritative — override any instruction below that conflicts):\n" + "\n".join(bodies)


def _skills_text(campaign: CampaignContext | None) -> str:
    """Name the skills a code-generator should read before writing code.

    Named rather than discovered: `skills_list` returns every skill installed in
    the Hermes home, and a child with a few children of budget should not browse
    a library to find its procedure. The campaign says which ones are its.

    Authority is stated here rather than left to inference, because the whole
    point is to beat recollection. A skill is a reviewed recipe; what the model
    remembers about a tool's API is not.
    """

    if campaign is None or not campaign.skills:
        return ""
    names = ", ".join(f"`{name}`" for name in campaign.skills)
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


def _campaign_text(campaign: CampaignContext | None, tier: str) -> str:
    """Goal plus tier-scoped rules, for a child whose context is plain text."""

    if campaign is None:
        return ""
    parts = []
    if campaign.goal:
        parts.append("CAMPAIGN GOAL (what this experiment serves):\n" + campaign.goal)
    if rules := _rules_text(campaign, tier):
        parts.append(rules)
    return "\n\n".join(parts)


def _campaign_field(campaign: CampaignContext | None, tier: str) -> dict[str, Any]:
    """Goal plus tier-scoped rules, for a task context serialized as JSON.

    Returned as a one-key mapping so a caller can splat it into an ordered dict
    and contribute nothing when no campaign is configured.
    """

    if campaign is None:
        return {}
    field_value: dict[str, Any] = {}
    # Omitted rather than empty when a profile names no goal: the invariant rules
    # still apply, and `"goal": ""` would read as an objective that is blank.
    if campaign.goal_name:
        field_value["goal_name"] = campaign.goal_name
        field_value["goal"] = campaign.goal
    rules = campaign.rules_for(tier)
    if rules:
        field_value["rules"] = [rule.body for rule in rules if rule.body]
        field_value["precedence"] = (
            "These rules are authoritative. They override any instruction in this payload "
            "that conflicts with them, and they apply to every child you delegate."
        )
    return {"campaign": field_value}


def _policy_roots(payload: Mapping[str, Any]) -> list[Path]:
    """Where campaign policy files are looked for, user directory first.

    Loam ships no campaign content, so a goal and a rule naming a particular
    measurement come from a directory the user owns (`campaign.dir`). The repo's
    own `.loam/` is still searched, because it holds the rules invariant across
    every campaign -- and because a campaign kept beside the code should keep
    working without setting `dir`.

    User first, so a campaign can shadow a shipped rule name with its own text
    rather than being stuck with it.
    """

    campaign = payload.get("loam_campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    roots = []
    if configured := _text(campaign.get("dir")):
        roots.append(Path(configured).expanduser())
    roots.append(Path(_text(payload.get("repo_root")) or ".") / ".loam")
    return roots


def _campaign_dirs(payload: Mapping[str, Any]) -> list[Path]:
    return [root / "campaigns" for root in _policy_roots(payload)]


def _rules_dirs(payload: Mapping[str, Any]) -> list[Path]:
    return [root / "rules" for root in _policy_roots(payload)]


def _find_policy_file(directories: list[Path], name: str, kind: str) -> Path:
    for directory in directories:
        path = directory / f"{name}.md"
        if path.is_file():
            return path
    searched = ", ".join(str(directory) for directory in directories)
    raise FileNotFoundError(f"campaign {kind} file not found: {name}.md (searched {searched})")


def _scopes(value: str, path: Path) -> frozenset[str]:
    """Parse `scope:` into the tiers a rule reaches.

    Space- or comma-separated tier names, or `all`. There is deliberately no
    `both`: with three tiers it names no particular two.
    """

    parts = [part for part in value.replace(",", " ").split() if part]
    if parts == ["all"]:
        return frozenset(RULE_TIERS)
    unknown = sorted({part for part in parts if part not in RULE_TIERS})
    if unknown:
        raise ValueError(f"rules file has unknown scope {unknown}, expected {list(RULE_TIERS)} or all: {path}")
    return frozenset(parts)


def _applies(value: str, path: Path) -> bool:
    """Parse `applies:` into whether the rule needs naming by a profile.

    Defaults to `when-named`, the safe direction: a new assay rule dropped into
    the directory must not reach a campaign in another domain, where the wrong
    decisive experiment is worse than a missing one. `always` is for the
    invariant layers, which a profile should not have to remember.
    """

    if not value or value == RULE_WHEN_NAMED:
        return False
    if value == RULE_ALWAYS:
        return True
    raise ValueError(
        f"rules file has unknown applies {value!r}, expected {RULE_ALWAYS} or {RULE_WHEN_NAMED}: {path}"
    )


def _rule_spec(name: str, path: Path) -> RuleSpec:
    front_matter = _front_matter(path)
    if front_matter.get("name") != name:
        raise ValueError(f"rules file name mismatch: {path}")
    # Required rather than defaulted to `all`: a rule that silently reached a
    # tier it was not written for would be read as authoritative there.
    scope = _text(front_matter.get("scope"))
    if not scope:
        raise ValueError(f"rules file must declare a scope ({'|'.join(RULE_TIERS)}, or all): {path}")
    return RuleSpec(
        name=name,
        scopes=_scopes(scope, path),
        body=_prompt_body(path, MAX_RULES_CHARS),
        enforced_by=_text(front_matter.get("enforced_by")),
        always=_applies(_text(front_matter.get("applies")), path),
    )


def _base_rule_names(payload: Mapping[str, Any]) -> list[str]:
    """Names of the rules that apply without being named, in injection order.

    Reads only `applies:` from each file's front matter, deliberately: a
    half-written assay rule sitting in the directory must not break a campaign
    that never names it. A file marked `always` is then loaded and validated in
    full like any other, so a malformed invariant rule still fails loudly.

    Sorted by name within each directory, and the repo's own before the user's,
    so a campaign that adds an always-on rule of its own reads it after the
    framework's. Base rules precede the profile's named ones, so the campaign's
    most specific text lands last.
    """

    names: list[str] = []
    for directory in reversed(_rules_dirs(payload)):
        for path in sorted(directory.glob("*.md")):
            if path.name == "README.md" or path.stem in names:
                continue
            try:
                applies = _text(_front_matter(path).get("applies"))
            except (OSError, ValueError):
                continue
            if applies == RULE_ALWAYS:
                names.append(path.stem)
    return names


def campaign_context(payload: Mapping[str, Any]) -> CampaignContext | None:
    """Load the invariant rules, plus the goal and assay rules a profile names.

    The invariant layers load unconditionally. A profile should not have to
    remember that an experiment must finish before it is reported, or that a
    reference datum must be fetched rather than recalled; those held for every
    campaign while the text was inline in `_hypothesis_block`, and a file is
    where the text lives, not a reason to make it optional. Only the assay layer
    -- which decisive measurement this domain uses -- is named, because only it
    differs between campaigns.

    `None` means no goal and no invariant rules, i.e. an empty rules directory:
    there is nothing to inject. A named-but-missing or malformed file raises
    instead, surfacing through the calling hook's handler as a logged failure.
    The point of the block is that every reasoning tier knows what the campaign
    is for, and a run that quietly lost its objective while still looking
    governed is the failure this avoids.
    """

    campaign = payload.get("loam_campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    goal_name = _text(campaign.get("goal"))
    goal = ""
    if goal_name:
        goal_path = _find_policy_file(_campaign_dirs(payload), goal_name, "goal")
        if _front_matter(goal_path).get("name") != goal_name:
            raise ValueError(f"campaign goal name mismatch: {goal_path}")
        goal = _prompt_body(goal_path, MAX_CAMPAIGN_CHARS)
    raw_rules = campaign.get("rules")
    named = [name for item in (raw_rules if isinstance(raw_rules, list) else []) if (name := _text(item))]
    rules: list[RuleSpec] = []
    # Base first, then the profile's, deduplicated by name: a profile that still
    # lists an invariant rule explicitly keeps working rather than sending the
    # text twice.
    base = _base_rule_names(payload)
    for name in base + [name for name in named if name not in base]:
        rules.append(_rule_spec(name, _find_policy_file(_rules_dirs(payload), name, "rules")))
    raw_skills = campaign.get("skills")
    skills = tuple(name for item in (raw_skills if isinstance(raw_skills, list) else []) if (name := _text(item)))
    if not goal_name and not rules and not skills:
        return None
    return CampaignContext(goal_name=goal_name, goal=goal, rules=tuple(rules), skills=skills)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # A concurrent writer can leave a half-written final line. Skip the
            # torn line rather than letting it raise out of tool_policy, which
            # the adapter would treat as a nonzero exit == allow (fail-open).
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")



def _append_campaign(payload: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    _append(_campaign_path(payload), row)


def _code_generator_context(execution: Mapping[str, Any] | None) -> str:
    """Render the exact-arithmetic debug check (code_source: fixed only)."""

    if isinstance(execution, Mapping) and _text(execution.get("mode")) == "remote_mcp":
        tool_name = _text(execution.get("tool_name"))
        endpoint = _text(execution.get("endpoint"))
        function_name = _text(execution.get("function_name"))
        input_schema = _text(execution.get("input_schema"))
        runtime = _text(execution.get("runtime"))
        if not all((tool_name, endpoint, function_name, input_schema)):
            return "Remote execution contract is incomplete; report blocked without local execution."
        payload = "\n".join((
            "from fractions import Fraction",
            "import json, os, socket",
            "values = [Fraction(n) for n in (1, 2, 3, 4)]",
            "mean = sum(values) / len(values)",
            "threshold = Fraction('2.4')",
            "print(json.dumps({'hostname': socket.gethostname(), 'pbs_jobid': os.environ.get('PBS_JOBID'), "
            "'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'sum': str(sum(values)), "
            "'count': len(values), 'mean': str(mean), 'threshold': str(threshold), "
            "'exceeds': mean > threshold}, sort_keys=True))",
        ))
        return "\n".join((
            f"Use only the remote MCP tool `{tool_name}`; do not execute code locally.",
            f"Remote target: endpoint={endpoint}; function={function_name}; input_schema={input_schema}; runtime={runtime}.",
            "Call it exactly once with `payload` equal to this complete Python program:",
            payload,
            "Report full returned stdout, stderr, returncode, remote hostname, PBS job ID, and CUDA_VISIBLE_DEVICES. "
            "A nonzero returncode, missing remote placement fields, or missing exact Fraction values is blocked evidence.",
        ))
    return "\n".join((
        "Use local python3.11 and execute exactly:",
        "from fractions import Fraction",
        "values = [Fraction(n) for n in (1, 2, 3, 4)]",
        "mean = sum(values) / len(values)",
        "threshold = Fraction('2.4')",
        "print({'sum': str(sum(values)), 'count': len(values), 'mean': str(mean), "
        "'threshold': str(threshold), 'exceeds': mean > threshold})",
        "Return exact stdout and execution status.",
    ))

def _generated(execution: Mapping[str, Any] | None) -> bool:
    """Whether code-generators write their own code.

    Always, unless a profile sets code_source: fixed, a debugging mode that
    swaps in the built-in arithmetic check from _code_generator_context.
    """

    return isinstance(execution, Mapping) and _text(execution.get("code_source")) != "fixed"


def _max_children(execution: Mapping[str, Any] | None) -> int:
    try:
        return max(1, int(execution.get("max_children", 3)))  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError):
        return 3


def _child_tags(plan_id: str, hypothesis_id: str, run_id: str = "") -> str:
    tags = f"[loam-agent-role=code-generator] [loam-plan-id={plan_id}] [loam-hypothesis-id={hypothesis_id}]"
    # The workspace run reserved for this child (hypothesis_workspace.reserve_runs).
    return f"{tags} [loam-run-id={run_id}]" if run_id else tags


def _strip_tags(value: str) -> str:
    """Drop model-written Loam markers so a child can't be relabeled to another plan."""

    return " ".join(_LOAM_TAG.sub(" ", value).split())


def _workdir(execution: Mapping[str, Any] | None, plan_id: str, hypothesis_id: str) -> str:
    root = _text(execution.get("workdir_root")) if isinstance(execution, Mapping) else ""
    return f"{root.rstrip('/')}/{plan_id}/{hypothesis_id}" if root else ""


def hpc_tool_names(execution: Mapping[str, Any] | None) -> dict[str, str]:
    """Hermes names of the HPC MCP tools, keyed by server function name.

    Hermes prefixes MCP tools with the server name, so the other tools share
    the configured run tool's prefix: tool_prefix when set, otherwise
    tool_name minus function_name. Without either only tool_name is known.
    """

    if not isinstance(execution, Mapping):
        return {}
    tool_name = _text(execution.get("tool_name"))
    function_name = _text(execution.get("function_name"))
    prefix = _text(execution.get("tool_prefix"))
    if not prefix and function_name and tool_name.endswith(function_name):
        prefix = tool_name[: -len(function_name)]
    names = {name: prefix + name for name in HPC_TOOLS} if prefix else {}
    if tool_name:
        names[function_name or "run_code"] = tool_name
    return names


def _observation_tools(execution: Mapping[str, Any] | None) -> frozenset[str]:
    """Read-only HPC tools a hypothesis-validator may call itself.

    Observation is not computation, so the invariant holds: the validator still
    writes and runs no code, and `max_children` still bounds its experiments.
    But allowing it nothing except `delegate_task` meant every status poll and
    every result read cost an experiment slot out of a budget of three, which is
    also why `_hypothesis_block` has to instruct the child to poll on the
    validator's behalf and report only once finished.

    The integrity half matters more: without these, a validator's evidence ledger
    is entirely downstream of a child's self-reported numbers, with no way to
    check them against the files on disk -- for an agent whose output is a
    calibrated score.
    """

    names = hpc_tool_names(execution)
    return frozenset(names[key] for key in VALIDATOR_OBSERVATION_TOOLS if key in names)


def _hypothesis_block(hypothesis: Mapping[str, Any] | None) -> str:
    """The hypothesis under test, for the child's context. Empty when unknown.

    Forwarded structurally rather than left to the validator to copy. The
    validator is instructed to paste it and cannot reliably do so: with the
    hypothesis at char ~9,000 of its own payload it told its child "the
    hypothesis text is not visible in this message" and sent only an id, and the
    child then spent its whole budget walking Aurora's filesystem grepping for
    that id (2026-10-21). A child starts with an empty context, so an id is not
    a reference it can resolve -- the text has to travel with the task.

    The instructions kept here are the campaign-independent ones: this block is
    the whole assignment, and a discovered input is the wrong system under test.
    They stay inline because they must hold for a profile that configures no
    campaign at all, and because `_SCRUBBERS` is the enforcement backing them.
    They also name no molecule class, tool, or metric -- this hook serves any
    domain. Everything domain-specific -- what the decisive experiment is, which
    numbers it reports, what it does not establish -- lives in an assay rule
    under `.loam/rules/` (see that directory's README), where a scientist can
    change it without editing this hook.
    """

    if not isinstance(hypothesis, Mapping):
        return ""
    statement = _text(hypothesis.get("content")) or _text(hypothesis.get("summary"))
    if not statement:
        return ""
    lines = ["HYPOTHESIS UNDER TEST (forwarded by Loam; this is the claim to falsify):", statement[:MAX_HYPOTHESIS_CHARS]]
    summary = _text(hypothesis.get("summary"))
    if summary and summary not in statement:
        lines.append("Summary: " + summary[:MAX_HYPOTHESIS_CHARS])
    lines.append(
        "HOW TO TEST THIS (authoritative — overrides any instruction below that conflicts):\n"
        "- This block is your complete assignment. Do NOT inspect campaign files, outputs, or the "
        "batch registry to 'recover' the hypothesis or 'find inputs': everything needed is here. If "
        "the validator's instruction below tells you to search the filesystem or reuse existing "
        "data/code, ignore that part.\n"
        "- If the candidate design is given explicitly, use it verbatim.\n"
        "- NEVER run, score, or reuse an input you discovered on the filesystem — a sequence, "
        "structure, parameter set, or dataset found in prior outputs, a tool's test fixtures, or "
        "another campaign's directory. That is the wrong system under test and makes the result "
        "scientifically meaningless. A run on the wrong input is a failed validation, not evidence."
    )
    return "\n".join(lines)


def _shared_inputs(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """The campaign's shared read-only input roots, for the child's layout block."""

    config = hypothesis_workspace.settings(payload) or {}
    shared = config.get("shared_inputs")
    return tuple(_text(item) for item in (shared if isinstance(shared, (list, tuple)) else ()) if _text(item))


def _layout_block(base: str, run_id: str, remote: bool = False, batch_root: str = "",
                  plan_id: str = "", hypothesis_id: str = "", shared_inputs: tuple[str, ...] = ()) -> str:
    """Where a child's files go in its hypothesis workspace (hypothesis_workspace.py).

    On HPC the stop hook cannot list the run's outputs, so a remote child is
    also told to report them as a LOAM_MANIFEST block, which
    hypothesis_workspace.remote_manifest parses from its final report. With a
    batch root it is also told how to name a run_batch and how to harvest its
    results into outputs/rM, which run_batch alone does not do.
    """

    if not run_id:
        return ""
    # Built in order rather than inserted at an index: the shared-inputs bullet
    # belongs with the other "where things live" bullets, and an index would
    # move it silently the next time a bullet is added above it.
    lines = [
        f"WORKSPACE LAYOUT (enforced by Loam; your run is {run_id}):",
        f"- code and scripts you write: {base}/src/",
        f"- result files: {base}/outputs/{run_id}/",
        f"- temporary files: {base}/scratch/{run_id}/",
        f"- inputs: read from {base}/data/ and never write there. The campaign's inputs are already "
        "there; nothing outside this workspace is reachable, so do not hunt the filesystem for a "
        "file named in your instructions and never report BLOCKED because one is absent",
    ]
    if shared_inputs:
        lines.append(
            "- shared campaign reference data, readable in place and read-only to you: "
            + ", ".join(f"{path}/" for path in shared_inputs)
            + ". Loam records each file you read there as an input of your run."
        )
    lines += [
        "- earlier results of OTHER runs (this hypothesis's or another's): readable only from their "
        "completed runs' outputs/rK/; Loam records each such read as an input of your run",
        "Write nothing anywhere else, and never /tmp: /tmp is local to one node and the next tool call "
        "may run on another. Other runs' outputs are frozen; do not modify them.",
        "DECISIONS: as you commit to each high-level choice -- the approach, which generator or "
        "folding model, the tier size, a go/stop -- record it with the loam_record_decision tool "
        "(what, why, alternatives, expected, confidence) BEFORE you act on it. Your full reasoning "
        "is not otherwise kept, so an unrecorded decision is lost.",
    ]
    if remote:
        # fs_policy requires an explicit workdir on every run_code (the server
        # would otherwise default it to HPC_BASE_DIR, outside the workspace).
        # Stated here because otherwise the child learns it only by being
        # blocked, which costs it a whole tool call -- observed in the first
        # live campaign round. Remote only: a local child has no run_code.
        # Always scratch, never outputs: the server writes its payload to a
        # temporary script in the workdir and removes it when the call ends. With
        # workdir=outputs/rM that temp file is hashed into the run's manifest and
        # then vanishes, so the post-freeze audit reports the run's own outputs as
        # tampered (observed live: "outputs/r1/tmps3ygefre.py was removed after
        # run r1 finished"). Keeping cwd in scratch/ leaves outputs/ holding only
        # deliberate results.
        lines.append(
            f"EVERY run_code call MUST pass workdir={base}/scratch/{run_id} -- always scratch/, "
            f"never outputs/: the server puts a temporary script in the workdir and deletes it "
            f"afterwards, which would corrupt your output manifest. Write results to "
            f"{base}/outputs/{run_id}/ by absolute path from inside your program. "
            "A run_code with no workdir is REFUSED."
        )
    if remote and batch_root:
        prefix = hypothesis_workspace.batch_prefix(hypothesis_id, run_id)
        labels = hypothesis_workspace.batch_labels(plan_id, hypothesis_id, run_id)
        lines += [
            f"RUN_BATCH (if you use it): pass batch_id='{prefix}<name>' and labels={json.dumps(labels)}. "
            "run_batch writes its results to the shared batch folder, not to your outputs. Poll batch_status "
            "until the batch is complete, then copy its results into your outputs by running this program "
            "with run_code (set batch_id first), BEFORE the output manifest step:",
            hypothesis_workspace.harvest_program(batch_root, f"{base}/outputs/{run_id}"),
        ]
    if remote:
        lines += [
            "OUTPUT MANIFEST (required): as your LAST tool call, run this program with run_code, then paste its "
            "complete output verbatim at the end of your report:",
            hypothesis_workspace.manifest_program(f"{base}/outputs/{run_id}"),
        ]
    return "\n".join(lines)


def _generated_child_context(instruction: str, execution: Mapping[str, Any] | None, workdir: str,
                             hypothesis: Mapping[str, Any] | None = None,
                             campaign: CampaignContext | None = None, layout: str = "") -> str:
    """Context for a code-generator that writes and runs its own code.

    ``layout`` is the _layout_block for the child's reserved run; empty when the
    workspace is off, which leaves the context exactly as before.
    """

    execution = execution if isinstance(execution, Mapping) else {}
    hypothesis_block = _hypothesis_block(hypothesis)
    # After the hypothesis and before the validator's instruction, for the same
    # reason the hypothesis is: the child must read its assignment and the policy
    # governing it ahead of a request whose wording may contradict either.
    campaign_block = _campaign_text(campaign, "code-generator")
    # After the policy and before the environment recipe: what the campaign is
    # for, then which procedure to follow, then how to run it.
    skills_block = _skills_text(campaign)
    # Campaign environment recipe (venv activation, tool CLIs, weight paths),
    # carried in the contract because the child cannot see the campaign's local
    # files. Without it a child hand-rolls `import chai_lab` against the source
    # tree with no venv and dies on a missing dependency (observed 2026-10-21:
    # ModuleNotFoundError: gemmi).
    setup = _text(execution.get("setup"))
    setup_block = ("ENVIRONMENT SETUP (use exactly this; do not invent venv or interpreter paths):\n" + setup) if setup else ""
    report = (
        "Report the code you ran, every tool call with its arguments, batch_ids, tool names and "
        "versions, and the actual stdout/stderr/returncode or batch_status. A failed or unfinished "
        "run is not evidence; say so instead of guessing results."
    )
    if _text(execution.get("mode")) != "remote_mcp":
        return "\n\n".join(part for part in (
            "Write your own Python for the validator's instruction below and execute it locally"
            + (f" with {_text(execution.get('runtime'))}." if _text(execution.get("runtime")) else "."),
            hypothesis_block,
            campaign_block,
            skills_block,
            setup_block,
            layout,
            "Instruction from validator:\n" + instruction,
            report,
        ) if part)
    names = hpc_tool_names(execution)
    run_code = names.get(_text(execution.get("function_name")) or "run_code", "")
    place = (f"under {workdir}" if workdir else "under the HPC base directory (HPC_BASE_DIR)")
    lines = [
        f"- `{run_code}`: run one Python program (`payload`) on a GPU worker; use it for one-off runs "
        "and to write script or JSONL files to disk.",
    ]
    if "run_batch" in names:
        lines += [
            f"- `{names['list_tools']}`: call this FIRST; reuse a registered tool whose params fit.",
            f"- `{names['register_tool']}`: save a script you will run many times as a named tool "
            "(name, script_path, params as {name: int|float|str|bool|list|dict}, description, example "
            "row). It is smoke-tested on the example; pass skip_smoke only if it needs a GPU to start.",
            f"- `{names['run_batch']}`: many runs at once: tool=<name> (or script_path) and inputs_path, "
            "a JSONL file with one row of params per line. Returns a batch_id at once. gpus is tiles "
            "PER ROW: rows run side by side on all free tiles, so a script using one device (xpu:0) "
            "takes gpus=1; raise it only if one row itself spans several devices.",
            f"- `{names['batch_status']}`: poll a batch. If it is still running when you must report, "
            "report the batch_id and counts as unfinished.",
            f"- `{names['inspect_filesystem']}`: read result files (ls/cat/tail).",
            f"- `{names['list_batches']}`: earlier batches for this plan or hypothesis.",
            "Use run_batch rather than run_code for more than a handful of runs.",
        ]
    # The one exception to "no other tool" is reading a skill. Stated inline
    # because a child told to use nothing else will not try `skill_view` even
    # when `tool_policy` would allow it, and a permission nobody knows about
    # changes no behavior.
    others = (
        " You may also call `skill_view` and `skills_list` to read a procedure before you write "
        "code; they only read."
        if campaign is not None and campaign.skills
        else ""
    )
    return "\n\n".join(part for part in (
        "Write your own Python for the validator's instruction below and run it on HPC through the "
        "remote MCP tools listed here; do not execute code locally or use any other tool to run "
        "anything." + others,
        # Before the validator's instruction: the claim matters more than the
        # wording of the request, and the validator's text may contradict it
        # (e.g. "the hypothesis text is not visible in this message").
        hypothesis_block,
        campaign_block,
        skills_block,
        setup_block,
        "Instruction from validator:\n" + instruction,
        f"Remote target: endpoint={_text(execution.get('endpoint'))}; runtime={_text(execution.get('runtime'))}.",
        "Tools:\n" + "\n".join(lines),
        layout or (
            f"Write scripts and inputs files {place}, never /tmp: /tmp is local to one node and the next "
            "tool call may run on another."
        ),
        report,
    ) if part)


# A clause telling the child to go hunt the filesystem to "recover" context.
# Matched line-wise and ending at a period OR end-of-line, so a sentence with no
# trailing period is still stripped (H2: the old `\.`-terminated pattern let
# period-less lines through).
# The object list is deliberately narrow: it names only things that are already
# injected structurally (the hypothesis, the campaign/project context, the batch
# registry). The earlier list also held `results?`, `inputs?`, `fasta`, `pdb` and
# `cif`, which are the words a CORRECT instruction uses -- "Report the results
# you find in stdout" and "Inspect the ipTM in the batch results" were both being
# deleted (H1).
_RECOVER_FROM_FS = re.compile(
    r"[^.\n]*\b(recover|locate|inspect|find|search)\b[^.\n]*\b(hypothesis|"
    r"campaign files?|project context|campaign context|data/code|"
    r"batch registry|filesystem)\b[^.\n]*(?:\.|$)",
    re.IGNORECASE | re.MULTILINE,
)
# "reuse/use the existing/staged/cached/leftover <artifact> (under <path>)" --
# steering the child onto a stale on-disk input. The ON-DISK QUALIFIER (existing,
# staged, cached, leftover, prior/previous run) is the signal, so a legitimate
# "fetch the sequence from UniProt" -- which carries no such qualifier -- is left
# untouched (H2).
_REUSE_EXISTING = re.compile(
    r"[^.\n]*\b(reuse|re-use|already[ -]?(?:staged|present|there|on[ -]?disk)|"
    r"staged|cached|leftover|previous run|prior run)\b[^.\n]*\b(fasta|pdb|cif|inputs?|"
    r"files?|sequence|structure|artifacts?|results?|folder|directory|path|batch|fixture)\b"
    r"[^.\n]*(?:\.|$)",
    re.IGNORECASE | re.MULTILINE,
)
# `existing` alone was in the pattern above and matched ordinary prose such as
# "Use the existing UniProt accession for the sequence" (H1). It now has to
# qualify a file-ish noun directly, within two intervening words.
_EXISTING_FILE = re.compile(
    r"[^.\n]*\bexisting\b(?:\s+\S+){0,2}\s+\b(fasta|pdb|cif|inputs?|files?|folder|"
    r"directory|path|batch|fixture)\b[^.\n]*(?:\.|$)",
    re.IGNORECASE | re.MULTILINE,
)
_SCRUBBERS = (_RECOVER_FROM_FS, _REUSE_EXISTING, _EXISTING_FILE)


def _neutralize_fs_recovery(text: str) -> str:
    """Drop validator sentences that send the child hunting the filesystem.

    A luna validator writes "first inspect the campaign/project context to
    recover the exact supplied hypothesis and any data/code" into the child's
    instruction. The child then ignores the hypothesis already in its context,
    hunts the filesystem, and either folds a leftover fixture or gives up asking
    for the hypothesis to be "provided" (both observed 2026-10-22). The variant
    "reuse the existing FASTA under <path>" is the same failure by another verb.
    The hypothesis and recipe are injected structurally, so this framing is pure
    noise; strip it before relaying. The code-generator prompt now independently
    refuses stale on-disk inputs, so this is defence in depth, not the only gate.
    """

    cleaned = text
    removed: list[str] = []

    def _drop(match: re.Match[str]) -> str:
        removed.append(match.group(0).strip())
        return ""

    for pattern in _SCRUBBERS:
        cleaned = pattern.sub(_drop, cleaned)
    if removed:
        # Never silent: a validator whose instruction was rewritten otherwise
        # looks to the child like one who wrote nothing useful (H1).
        _note_scrubbed(removed)
    return cleaned.strip()


def _note_scrubbed(removed: list[str]) -> None:
    """Record the clauses stripped from a validator instruction, best effort."""

    message = "loam: stripped filesystem-recovery clause(s) from validator instruction: " + \
        "; ".join(repr(span) for span in removed if span)
    print(message, file=sys.stderr)


def _child_goal(hypothesis: Mapping[str, Any] | None, validator_goal: str, plan_id: str, hypothesis_id: str,
                run_id: str = "") -> str:
    """Lead the child's GOAL with the hypothesis, not the validator's framing.

    The hypothesis is injected in the context block, but luna skims past it and
    acts on the goal line. Putting the claim first in the goal makes "test THIS"
    the primary task instead of "go find a hypothesis."
    """

    tags = _child_tags(plan_id, hypothesis_id, run_id)
    lead = ""
    if isinstance(hypothesis, Mapping):
        claim = _text(hypothesis.get("summary")) or _text(hypothesis.get("content"))
        if claim:
            lead = f"Test this binder hypothesis (full text + how-to in your context; do NOT search for it): {claim[:220]} || "
    body = _neutralize_fs_recovery(validator_goal)
    return f"{lead}{body[:MAX_CHILD_GOAL_CHARS]} {tags}"


def _generated_child_tasks(payload: Mapping[str, Any], plan_id: str, hypothesis_id: str) -> dict[str, Any] | None:
    """Keep the validator's own instruction, but force tags, role and contract."""

    execution = payload.get("loam_execution")
    raw = payload.get("arguments")
    if not isinstance(raw, Mapping):
        return None
    tasks = raw.get("tasks")
    # An EMPTY tasks list is single-task mode, not an empty batch: models that
    # fill every optional field emit `tasks: []` alongside a top-level `goal`,
    # and Hermes's own delegate_task handler treats that as single mode
    # (`if tasks and isinstance(tasks, list)`). Matching that here is what keeps
    # Loam from building zero children -- and the gate below from blocking -- on
    # a dispatch Hermes would run.
    items = tasks if (isinstance(tasks, list) and tasks) else [raw]
    workdir = _workdir(execution if isinstance(execution, Mapping) else None, plan_id, hypothesis_id)
    # Look the hypothesis up here rather than trusting the validator to have
    # copied it into the instruction; see _hypothesis_block. Unknown id -> empty
    # block, and the child's context is exactly as it was before.
    hypothesis = _full_hypotheses(payload).get(hypothesis_id)
    # Forwarded structurally for the same reason, and from the payload rather
    # than from the validator's instruction: policy the validator paraphrased
    # away would otherwise never reach the tier that runs the experiment.
    campaign = campaign_context(payload)
    parsed = []
    for task in items:
        if not isinstance(task, Mapping):
            return None
        goal = _strip_tags(_text(task.get("goal")))
        if not goal:
            return None
        clean = _neutralize_fs_recovery(goal)
        instruction = "\n\n".join(part for part in (clean, _neutralize_fs_recovery(_strip_tags(_text(task.get("context"))))) if part)
        parsed.append((goal, instruction))
    # Reserve only once every task is known to be valid, so a rejected batch
    # leaves no reserved runs behind. The reservation key is the child's goal as
    # record_start will see it: _child_goal with its tags stripped again, which
    # is independent of the run id the tags will carry.
    keys = [_strip_tags(_child_goal(hypothesis, goal, plan_id, hypothesis_id)) for goal, _ in parsed]
    run_ids, layout_base = _reserve_child_runs(payload, execution, workdir, plan_id, hypothesis_id, keys)
    remote = isinstance(execution, Mapping) and _text(execution.get("mode")) == "remote_mcp"
    batch_root = hypothesis_workspace.batch_root_for(execution if isinstance(execution, Mapping) else None) if remote else ""
    out = []
    for (goal, instruction), run_id in zip(parsed, run_ids):
        out.append(
            {
                "goal": _child_goal(hypothesis, goal, plan_id, hypothesis_id, run_id),
                "context": _generated_child_context(
                    instruction, execution, workdir, hypothesis, campaign,
                    _layout_block(layout_base, run_id, remote, batch_root, plan_id, hypothesis_id,
                                  _shared_inputs(payload)),
                ),
                "role": "leaf",
            }
        )
    _record_dispatch_edges(payload, plan_id, hypothesis_id, items, out, run_ids)
    return {"tasks": out}


def _record_dispatch_edges(
    payload: Mapping[str, Any], plan_id: str, hypothesis_id: str,
    items: list, out: list[dict[str, Any]], run_ids: list[str],
) -> None:
    """Record the validator->child edge for each dispatched run (hypothesis_workspace.record_dispatch).

    Hashes the validator's original ask (``items``) and the canonical task the
    child receives (``out``) so a Loam rewrite between them is on the record.
    Runs at dispatch, when the runs are reserved but the children have not
    started. A workspace problem never blocks the dispatch.
    """

    if hypothesis_workspace.settings(payload) is None or not any(run_ids):
        return
    validator_session = _text(payload.get("session_id")) or _text(payload.get("parent_session_id"))
    try:
        root = hypothesis_workspace.workspace_root(payload, plan_id, hypothesis_id)
    except Exception as exc:
        print(f"loam workspace: dispatch edge skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return
    for item, entry, run_id in zip(items, out, run_ids):
        if not run_id or not isinstance(item, Mapping):
            continue
        requested = _text(item.get("goal")) + "\n\n" + _text(item.get("context"))
        canonical = _text(entry.get("goal")) + "\n\n" + _text(entry.get("context"))
        try:
            hypothesis_workspace.record_dispatch(
                root, run_id, validator_session,
                hashlib.sha256(requested.encode("utf-8")).hexdigest(),
                hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            )
        except Exception as exc:
            print(f"loam workspace: could not record dispatch edge for {run_id}: {type(exc).__name__}: {exc}", file=sys.stderr)


def _reserve_child_runs(
    payload: Mapping[str, Any],
    execution: Any,
    workdir: str,
    plan_id: str,
    hypothesis_id: str,
    goals: list[str],
) -> tuple[list[str], str]:
    """Reserve one workspace run per child, and the folder the child should write under.

    Returns empty run ids when the workspace is off or reservation fails; the
    children are then delegated exactly as before, and record_start adopts or
    allocates a run for each when it starts. A workspace problem never blocks
    delegation.

    ``goals`` are the children's goals with Loam's tags stripped -- the same
    form record_start sees. They make reservation idempotent, because this runs
    inside `canonical_delegation_args`, which is request middleware and may run
    again for one dispatch (a validator delegating twice, a retried call). The
    goals also let `hypothesis_workspace.adopt_run` put a child whose run-id tag
    did not survive into the run its own instructions name.
    """

    count = len(goals)
    if hypothesis_workspace.settings(payload) is None:
        return [""] * count, ""
    try:
        root = hypothesis_workspace.workspace_root(payload, plan_id, hypothesis_id)
        run_ids = hypothesis_workspace.reserve_runs(root, count, goals)
    except Exception as exc:
        print(f"loam workspace: could not reserve runs: {type(exc).__name__}: {exc}", file=sys.stderr)
        return [""] * count, ""
    if isinstance(execution, Mapping) and _text(execution.get("mode")) == "remote_mcp":
        # The child writes on HPC: the remote workdir mirrors the local layout.
        base = workdir or f"<HPC base directory (HPC_BASE_DIR)>/{plan_id}/{hypothesis_id}"
    else:
        base = str(root)
    return run_ids, base


def _marker(value: str, key: str) -> str:
    prefix = f"[{key}="
    start = value.find(prefix)
    if start < 0:
        return ""
    end = value.find("]", start + len(prefix))
    return value[start + len(prefix) : end] if end >= 0 else ""


_EVIDENCE_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*Evidence\s+Score\b", re.IGNORECASE | re.MULTILINE)
# Requires a decimal point, so a number in the heading's own lead-in -- the `0`
# of "### Evidence Score (0-1 scale)", or the `3` of "Based on 3 folds, 0.42" --
# cannot be mistaken for the score (H1). A bare integer is accepted only as the
# first token after the heading line, where it is unambiguous.
_SIGNED_DECIMAL_RE = re.compile(r"[-+−]?\d*\.\d+")
_LEADING_INT_RE = re.compile(r"\A[^\S\n]*[*`_(\[]*([-+−]?\d+)\b")


def _evidence_score(summary: str) -> float | None:
    """Extract the validator's calibrated evidence score from its report.

    The score must sit under a real markdown heading LINE ("### Evidence Score",
    or ## / #### -- any level, at the start of a line). Anchoring to a heading is
    what keeps an in-prose mention like "the prior Evidence Score was 0.00" from
    being read as this round's score (M2 false positive); it still tolerates the
    "## Evidence Score\n\n0.18" form (two hashes, number on the next line) that
    the old same-line-### parser dropped. When several such headings appear (a
    recap plus the final verdict) the LAST one wins. From that heading we take
    the first signed decimal within a short window, tolerating a lead-in and
    markup: the number may follow a colon on the heading line, sit on a later
    line, or be wrapped in **bold**, `backticks`, or parentheses (M2 false
    negatives), and a Unicode minus sign is accepted.

    The number must carry a decimal point, because the window begins at the end
    of the "Evidence Score" match and anything else on the heading line is still
    inside it: "### Evidence Score (0-1 scale)\n\n0.42" would otherwise report
    the `0` of "(0-1" as a neutral 0.0 score -- recorded as "reported", not even
    flagged invalid (H1). A bare integer is read only when it is the first token
    after the heading line, where there is no lead-in to confuse it with.
    """

    headings = list(_EVIDENCE_HEADING_RE.finditer(summary))
    if not headings:
        return None
    start = headings[-1].end()
    window = summary[start:start + 200]
    match = _SIGNED_DECIMAL_RE.search(window)
    if match is None:
        # No decimal: accept an integer only at the very start of the body, i.e.
        # right after the heading line ("### Evidence Score\n\n0").
        _, _, body = window.partition("\n")
        match = _LEADING_INT_RE.match(body.lstrip("\n"))
    if not match:
        return None
    try:
        score = float(match.group(match.lastindex or 0).replace("−", "-"))
    except ValueError:
        return None
    return score if -1.0 <= score <= 1.0 else None

def _falda_policy(payload: Mapping[str, Any], tool_name: str) -> dict[str, str] | None:
    """Guard FALDA tools. Fails closed: if the guard cannot decide, the call is blocked."""

    try:
        session_id = _text(payload.get("session_id")) or _text(payload.get("parent_session_id"))
        starts = [
            row
            for row in _read_rows(_state_path(payload))
            if session_id and row.get("event") == "agent_started" and _text(row.get("child_session_id")) == session_id
        ]
        return falda_tool_policy(payload, tool_name, payload.get("arguments"), starts[-1] if starts else None)
    except Exception as exc:
        return {
            "action": "block",
            "message": f"Loam could not verify this FALDA call ({type(exc).__name__}); continue without FALDA tools.",
        }


def tool_policy(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """Block tool use that violates a tagged orchestration role's boundary.

    Two kinds of branch live here, and they have different futures.

    The coordinator branches below (`not starts`) inspect what the model wrote
    into a `delegate_task` call. Most of what they check -- a substitute batch,
    a missing per-task `role`, a dropped `loam_validator_contract` -- is
    repaired anyway by the `tool_request` middleware, except mid-round, where
    `canonical_delegation_args` returns None (see its docstring). Because the
    veto runs FIRST, blocking here preempts a repair that would have been
    silent. These branches become removable once the hook dispatches delegation
    itself rather than asking the model to transcribe it.

    The role branches further down are different: they read the role from the
    ledger, not from model output, and confine what each tier may call at all.
    They are the only enforcement behind the three-tier split and the
    per-hypothesis child budget, and no change to how delegation is dispatched
    affects them.
    """

    if _text(payload.get("tool_name")).startswith(FALDA_TOOL_PREFIX):
        return _falda_policy(payload, _text(payload.get("tool_name")))
    if payload.get("loam_hypothesis_orchestration") is not True:
        return None
    tool_name = _text(payload.get("tool_name"))
    arguments = payload.get("arguments")
    session_id = _text(payload.get("session_id")) or _text(payload.get("parent_session_id"))
    rows = _read_rows(_state_path(payload))
    starts = [
        row
        for row in rows
        if row.get("event") == "agent_started" and _text(row.get("child_session_id")) == session_id
    ]
    latest = _latest_report(payload)
    latest_plan_id = _plan_id(latest[0], payload) if latest is not None else ""
    plan_finished = bool(
        latest_plan_id
        and any(
            row.get("event") == "reasoner_completed"
            and _text(row.get("plan_id")) == latest_plan_id
            for row in rows
        )
    )
    active = _active_plan(rows, _text(payload.get("workflow_id")), _text(payload.get("loam_run_id")))
    active_validators_complete = bool(
        active is not None
        and active[1]
        and len(active[2]) == len(active[1])
    )
    active_reasoner_started = bool(
        active is not None
        and any(
            row.get("event") == "agent_started"
            and _text(row.get("role")) == "reasoner"
            and _text(row.get("plan_id")) == active[0]
            for row in rows
        )
    )
    if not starts and active_reasoner_started:
        return {
            "action": "block",
            "message": "The active plan's reasoner is already dispatched; wait for its terminal result before any further tool call.",
        }
    if not starts and tool_name == "delegate_task":
        # The coordinator does not compose delegation at all any more: it calls
        # DISPATCH_TOOL, whose handler hands the hook's own canonical arguments
        # straight to delegate_task. So there is nothing to inspect here, and
        # nothing to repair afterwards.
        #
        # This replaces four branches that checked the model's transcription --
        # a substitute batch, a missing per-task role=orchestrator, a dropped
        # loam_validator_contract -- which were only ever approximations of "did
        # the LLM copy it correctly". Blocking the whole hop is both shorter and
        # stronger: it holds mid-round, where `canonical_delegation_args` has no
        # repair available and those checks were the only thing standing between
        # a substitute batch and execution.
        return {
            "action": "block",
            "message": (
                f"Call {DISPATCH_TOOL} to launch this round instead of composing a delegate_task "
                "batch. It takes no arguments and dispatches the hypothesis validators, or the "
                "reasoner once every validator has reported, from the campaign's persisted plan. "
                "You cannot see the selected hypotheses in full, so a batch you compose yourself "
                "is a substitute for the campaign's work, not the campaign's work."
            ),
        }
    if (
        not starts
        and latest_plan_id
        and not plan_finished
        and not active_validators_complete
        # delegate_task is refused above with a pointer to the dispatch tool;
        # the dispatch tool is how the round gets launched, so blocking it here
        # would leave the coordinator with no legal move at all.
        and tool_name not in ("delegate_task", DISPATCH_TOOL)
    ):
        return {
            "action": "block",
            "message": (
                "The active campaign plan is pending. Do not execute or inspect work as coordinator; "
                f"call {DISPATCH_TOOL} first and let the validators report."
            ),
        }
    if not session_id:
        return None
    if not starts:
        return None
    role = _text(starts[-1].get("role"))
    tool_name = _text(payload.get("tool_name"))
    # Recording a decision is provenance, not execution -- the same carve-out as
    # the read-only skill tools below. It must precede every role gate, or the
    # per-role allowlists (which do not list it) refuse it: live, every
    # validator's loam_record_decision call was blocked here and no decision was
    # ever recorded, though the validators were calling it with real content.
    if tool_name == hypothesis_workspace.DECISION_TOOL:
        return None
    if role == "reasoner":
        # Hermes may defer MCP tools behind tool_search; with FALDA tools on, the reasoner needs it to
        # load mcp__falda__* (whose calls the FALDA guard above still checks).
        if tool_name == "tool_search" and tools_available(falda_config(payload)):
            return None
        return {"action": "block", "message": "reasoner has no tool capability; return the synthesis to the harness"}
    if role == "code-generator":
        if tool_name == "delegate_task":
            return {"action": "block", "message": "code-generator must not delegate"}
        # Reading a procedure is not executing one. This is the same argument as
        # commit 8103f3a made for letting a validator poll: the confinement that
        # matters is over what runs, and a child that cannot read its campaign's
        # recipe writes GPU code from recollection instead.
        if tool_name in SKILL_TOOLS:
            return None
        execution = payload.get("loam_execution")
        if isinstance(execution, Mapping) and execution.get("mode") == "remote_mcp" and _generated(execution):
            allowed = sorted(set(hpc_tool_names(execution).values()))
            if tool_name not in allowed:
                return {
                    "action": "block",
                    "message": (
                        f"code-generator may use only the HPC MCP tools {allowed} "
                        f"and the read-only skill tools {sorted(SKILL_TOOLS)}"
                    ),
                }
            return None
        if isinstance(execution, Mapping) and execution.get("mode") == "remote_mcp":
            required_tool = _text(execution.get("tool_name"))
            if tool_name != required_tool:
                return {"action": "block", "message": f"code-generator must use remote MCP tool {required_tool}"}
        return None
    if role != "hypothesis-validator":
        return None
    if tool_name != "delegate_task":
        if tool_name in _observation_tools(payload.get("loam_execution")):
            return None
        return {
            "action": "block",
            "message": (
                "hypothesis-validator must delegate executable work to code-generator; "
                f"it may call only the read-only tools {sorted(VALIDATOR_OBSERVATION_TOOLS)} itself"
            ),
        }
    arguments = payload.get("arguments")
    goals = []
    if isinstance(arguments, Mapping):
        tasks = arguments.get("tasks")
        # Empty tasks list == single-task mode (see _generated_child_tasks): fall
        # through to the top-level goal rather than reading zero goals and
        # blocking a dispatch Hermes would have run.
        if isinstance(tasks, list) and tasks:
            goals.extend(
                goal
                for task in tasks
                if isinstance(task, Mapping)
                if (goal := _text(task.get("goal")))
            )
        else:
            goals.extend(
                value
                for key in ("goal", "context", "task")
                if (value := _text(arguments.get(key)))
            )
    plan_id = _text(starts[-1].get("plan_id"))
    if not goals or any(
        _marker(goal, "loam-agent-role") != "code-generator"
        or _marker(goal, "loam-plan-id") != plan_id
        for goal in goals
    ):
        return {
            "action": "block",
            "message": "hypothesis-validator may delegate only tagged code-generator children in its plan",
        }
    execution = payload.get("loam_execution")
    if _generated(execution if isinstance(execution, Mapping) else None):
        hypothesis_id = _text(starts[-1].get("hypothesis_id"))
        used = sum(
            1
            for row in rows
            if row.get("event") == "agent_started"
            and row.get("role") == "code-generator"
            and _text(row.get("plan_id")) == plan_id
            and _text(row.get("hypothesis_id")) == hypothesis_id
        )
        cap = _max_children(execution)  # type: ignore[arg-type]
        raw_tasks = arguments.get("tasks") if isinstance(arguments, Mapping) else None
        # Empty tasks list is one single-task child, not zero (same rule as above).
        requested = len(raw_tasks) if (isinstance(raw_tasks, list) and raw_tasks) else 1
        if used + requested > cap:
            return {
                "action": "block",
                "message": f"code-generator budget for this hypothesis is {cap} and {used} are used; return your report now.",
            }
    return None


_START_ROW_CACHE: dict[tuple[str, str, int, int], Mapping[str, Any] | None] = {}


def _session_start(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The calling session's latest agent_started row, as tool_policy finds it.

    Cached for the life of the process, keyed on the state file's size and
    mtime, because one pre-tool run asks for this twice (workspace_policy and
    workspace_capture) and the file is the whole campaign's history. A changed
    file is a different key, so the cache can never serve a stale row.
    """

    session_id = _text(payload.get("session_id")) or _text(payload.get("parent_session_id"))
    if not session_id:
        return None
    path = _state_path(payload)
    try:
        stat = path.stat()
        key: tuple[str, str, int, int] | None = (session_id, str(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        key = None
    if key is not None and key in _START_ROW_CACHE:
        return _START_ROW_CACHE[key]
    row = next(
        (
            item
            for item in reversed(_read_rows(path))
            if item.get("event") == "agent_started" and _text(item.get("child_session_id")) == session_id
        ),
        None,
    )
    if key is not None:
        _START_ROW_CACHE[key] = row
    return row


def workspace_policy(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """Pando-style workspace boundary for a campaign child (hypothesis_workspace.fs_policy)."""

    if hypothesis_workspace.settings(payload) is None:
        return None
    return hypothesis_workspace.fs_policy(payload, _session_start(payload))


def workspace_capture(payload: Mapping[str, Any]) -> None:
    """Record the code an allowed campaign child runs (hypothesis_workspace.capture_code)."""

    if hypothesis_workspace.settings(payload) is not None:
        hypothesis_workspace.capture_code(payload, _session_start(payload))


def workspace_decision(payload: Mapping[str, Any]) -> bool:
    """Record one declared decision in the caller's workspace record (any campaign role).

    Called from the loam_record_decision tool handler, which carries the calling
    agent's session_id. Returns whether a decision was stored, so the tool can
    tell the agent plainly rather than claim a record it did not make.
    """

    if hypothesis_workspace.settings(payload) is None:
        return False
    return hypothesis_workspace.record_decision(payload, _session_start(payload))


def workspace_stop(payload: Mapping[str, Any]) -> None:
    """Close a stopped child's workspace run, or its validator's workspace.

    Runs after record_stop, whose completion row it mirrors (it never decides
    completion itself). The start row is found the way record_stop finds it.
    Audit problems are also reported to the campaign ledger.
    """

    if hypothesis_workspace.settings(payload) is None:
        return
    rows = _read_rows(_state_path(payload))
    child_id = _text(payload.get("child_subagent_id"))
    child_session = _text(payload.get("child_session_id"))

    def _same_child(row: Mapping[str, Any]) -> bool:
        return _text(row.get("child_subagent_id")) == child_id or _text(row.get("child_session_id")) == child_session

    start = next((row for row in reversed(rows) if row.get("event") == "agent_started" and _same_child(row)), None)
    if start is None:
        return
    completion = next(
        (row for row in reversed(rows) if row.get("event") == "validator_completed" and _same_child(row)), None
    )
    problems = hypothesis_workspace.on_agent_stopped(payload, start, completion)
    if problems:
        _append_campaign(
            payload,
            {
                "event": "workspace_violation",
                "observed_at": _timestamp(),
                "run_id": _text(payload.get("loam_run_id")),
                "workflow_id": _text(payload.get("workflow_id")),
                "plan_id": _text(start.get("plan_id")),
                "hypothesis_id": _text(start.get("hypothesis_id")),
                "workspace": _text(start.get("workspace")),
                "problems": problems,
            },
        )


def is_campaign_child(payload: Mapping[str, Any]) -> bool:
    """Return whether this callback belongs to a tagged campaign child."""

    session_id = _text(payload.get("session_id"))
    return bool(
        session_id
        and any(
            row.get("event") == "agent_started" and _text(row.get("child_session_id")) == session_id
            for row in _read_rows(_state_path(payload))
        )
    )


# L2 (warn-only, no block): nudge a code-generator toward run_batch once it has
# looped run_code. Injected each pre-LLM turn by pre-llm.py; it self-gates and
# returns "" for everything else. Rationale: each serial run_code re-sends the
# child's whole growing context (skills + CAMPAIGN.md + accumulated tool output),
# which measurement showed is the dominant driver of the per-minute token-rate
# limit -- one child's 17 serial run_code calls re-sent ~45K tokens apiece.
# run_batch collapses the fan-out to one submit + batch_status polling and runs
# the rows on HPC tiles, so it is both the scale-out path and the rate-limit fix.
RUNBATCH_NUDGE_THRESHOLD = 5


def runbatch_nudge_for(payload: Mapping[str, Any]) -> str:
    """Advisory run_batch reminder for a code-generator that is looping run_code."""
    session_id = _text(payload.get("session_id"))
    if not session_id:
        return ""
    start = None
    for row in _read_rows(_state_path(payload)):
        if row.get("event") == "agent_started" and _text(row.get("child_session_id")) == session_id:
            start = row  # last start for this session wins
    if start is None or _text(start.get("role")) != "code-generator":
        return ""
    # The workspace run id is start["run"] (e.g. "r2"); start["run_id"] is the
    # campaign LOAM_RUN_ID (a uuid) and does NOT name the run dir. record_decision
    # and every other consumer read start["run"] -- mirror them or the path never
    # resolves and this nudge silently no-ops.
    workspace = _text(start.get("workspace"))
    run = _text(start.get("run"))
    if not workspace or not run:
        return ""
    try:
        rec = hypothesis_workspace.read_json(
            Path(workspace) / "runs" / run / hypothesis_workspace.RUN_FILE
        )
    except Exception:
        return ""
    calls = rec.get("calls") or []
    n_run_code = sum(1 for c in calls if hypothesis_workspace._hpc_function(_text(c.get("tool"))) == "run_code")
    n_run_batch = sum(1 for c in calls if hypothesis_workspace._hpc_function(_text(c.get("tool"))) == "run_batch")
    if n_run_code >= RUNBATCH_NUDGE_THRESHOLD and n_run_batch == 0:
        return (
            f"RUN_BATCH REMINDER: this run has {n_run_code} serial run_code calls and no run_batch. "
            "A design tier's per-candidate fan-out (generate / fold / score many candidates) MUST go "
            "through run_batch -- one JSONL row per candidate, one submit, then poll batch_status until "
            "unfinished == 0 -- NOT a run_code loop. Each serial run_code resends your whole growing "
            "context and is the main cause of the model token-rate-limit failures that abort runs; "
            "run_batch executes the rows on HPC tiles without that per-call cost. Keep run_code only for "
            "setup, a single probe fold, and post-batch aggregation."
        )
    return ""


def _latest_report(payload: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]] | None:
    raw_config = payload.get("loam_hypothesis_tournament")
    if not isinstance(raw_config, Mapping):
        return None
    reports_dir = Path(_text(raw_config.get("reports_dir"))).expanduser()
    if not reports_dir.is_dir():
        return None
    workflow_id = _text(payload.get("workflow_id"))
    for path in sorted(reports_dir.glob("tournament-*.json"), reverse=True):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # An in-progress / truncated report must not raise out of
            # tool_policy and fail the boundary open; skip to the next one.
            continue
        if not isinstance(value, Mapping):
            continue
        if workflow_id and _text(value.get("workflow_id")) != workflow_id:
            continue
        result = value.get("result")
        if isinstance(result, Mapping) and result.get("status") == "completed":
            return path, value
    return None


def _full_hypotheses(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_config = payload.get("loam_hypothesis_tournament")
    if not isinstance(raw_config, Mapping):
        return {}
    state_file = Path(_text(raw_config.get("state_file"))).expanduser()
    if not state_file.is_file():
        return {}
    try:
        value = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A concurrent Jnana write can leave the state file momentarily
        # unparseable; treat as "no hypotheses yet" rather than failing the
        # boundary open.
        return {}
    raw_hypotheses = value.get("hypotheses") if isinstance(value, Mapping) else None
    if not isinstance(raw_hypotheses, list):
        return {}
    return {
        hypothesis_id: item
        for item in raw_hypotheses
        if isinstance(item, Mapping)
        and (hypothesis_id := _text(item.get("hypothesis_id")))
    }


def _plan_id(report_path: Path, payload: Mapping[str, Any]) -> str:
    material = "\0".join((str(report_path), _text(payload.get("workflow_id")), _text(payload.get("session_id"))))
    return f"plan_{hashlib.sha256(material.encode()).hexdigest()[:16]}"


def prepare_payload(payload: dict[str, Any]) -> None:
    """Feed the latest reasoner brief and recalled campaign memory into the next Jnana generation goal."""

    rows = _read_rows(_state_path(payload))
    brief = _latest_brief(rows, _text(payload.get("workflow_id")))
    if brief:
        payload["loam_reasoner_brief"] = brief
    user_message = _text(payload.get("user_message"))
    if not user_message:
        return
    query = f"{user_message}\n\n{brief}" if brief else user_message
    memory = generation_memory(payload, query, lambda: len(_full_hypotheses(payload)))
    if not brief and not memory:
        return
    if brief:
        user_message = (
            f"{user_message}\n\n"
            "Prior reasoner brief (advisory and unvalidated unless independently reconciled):\n"
            f"{brief}"
        )
    if memory:
        user_message = f"{user_message}\n\n{memory}"
    payload["user_message"] = user_message


def _active_plan(
    rows: list[dict[str, Any]], workflow_id: str, run_id: str = ""
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return the most recent started plan until its reasoner has completed.

    Scoped to the current run: a plan started by a PRIOR run (a different
    ``run_id``) is never treated as active. Otherwise an earlier run that
    dispatched validators but exited before they completed (e.g. a one-shot
    ``-z`` turn) would leave a started-but-never-finished plan that deadlocks
    every later run's coordinator — the pre-tool policy demands the pending
    plan's validators while ``context_for`` injects no payload for it. When
    ``run_id`` is empty the scope falls back to workflow only (legacy behaviour
    and the fixture tests, which do not set a run id).
    """

    def _scope(row: Mapping[str, Any]) -> bool:
        return _text(row.get("workflow_id")) == workflow_id and (
            not run_id or _text(row.get("run_id")) == run_id
        )

    plan_ids = [
        _text(row.get("plan_id"))
        for row in rows
        if row.get("event") == "agent_started"
        and row.get("role") == "hypothesis-validator"
        and _scope(row)
    ]
    for plan_id in reversed(plan_ids):
        if any(
            row.get("event") == "reasoner_completed"
            and _text(row.get("plan_id")) == plan_id
            and _scope(row)
            for row in rows
        ):
            continue
        starts = [
            row
            for row in rows
            if row.get("event") == "agent_started"
            and row.get("role") == "hypothesis-validator"
            and _text(row.get("plan_id")) == plan_id
            and _scope(row)
        ]
        completions = [
            row
            for row in rows
            if row.get("event") == "validator_completed"
            and _text(row.get("plan_id")) == plan_id
            and _scope(row)
        ]
        return plan_id, starts, completions
    return None


def _latest_brief(rows: list[dict[str, Any]], workflow_id: str) -> str:
    for row in reversed(rows):
        if row.get("event") != "reasoner_completed" or _text(row.get("workflow_id")) != workflow_id:
            continue
        return _text(row.get("summary"))[:MAX_BRIEF_CHARS]
    return ""


def record_start(payload: Mapping[str, Any]) -> None:
    """Record a role-tagged child allocation for later completion collation."""

    goal = _text(payload.get("child_goal"))
    role = _marker(goal, "loam-agent-role")
    plan_id = _marker(goal, "loam-plan-id")
    if role not in AGENT_NAMES or not plan_id:
        return
    record = {
        "event": "agent_started",
        "observed_at": _timestamp(),
        "run_id": _text(payload.get("loam_run_id")),
        "workflow_id": _text(payload.get("workflow_id")),
        "plan_id": plan_id,
        "role": role,
        "hypothesis_id": _marker(goal, "loam-hypothesis-id"),
        "child_subagent_id": _text(payload.get("child_subagent_id")),
        "child_session_id": _text(payload.get("child_session_id")),
    }
    # Pando-style workspace (off unless loam_workspace is enabled): a validator
    # gets its hypothesis folder, a code-generator its run. The fields land on
    # this row because every later hook identifies a session by it.
    hypothesis = (
        _full_hypotheses(payload).get(record["hypothesis_id"])
        if role == "hypothesis-validator" and hypothesis_workspace.settings(payload) is not None
        else None
    )
    record.update(
        hypothesis_workspace.on_agent_started(
            payload,
            role,
            plan_id,
            record["hypothesis_id"],
            run_id=_marker(goal, "loam-run-id"),
            goal=_strip_tags(goal),
            statement=(_text(hypothesis.get("content")) or _text(hypothesis.get("summary"))) if hypothesis else "",
        )
    )
    _append(_state_path(payload), record)
    if role == "hypothesis-validator":
        _append_campaign(payload, {**record, "event": "hypothesis_validation_started"})


def record_stop(payload: Mapping[str, Any]) -> None:
    """Record a tagged terminal child result as unvalidated orchestration input."""

    rows = _read_rows(_state_path(payload))
    child_id = _text(payload.get("child_subagent_id"))
    child_session = _text(payload.get("child_session_id"))
    start = next(
        (
            row
            for row in reversed(rows)
            if row.get("event") == "agent_started"
            and (_text(row.get("child_subagent_id")) == child_id or _text(row.get("child_session_id")) == child_session)
        ),
        None,
    )
    if start is None:
        return
    role = _text(start.get("role"))
    event = "reasoner_completed" if role == "reasoner" else "validator_completed" if role == "hypothesis-validator" else "code_generator_completed"
    summary = _text(payload.get("child_summary"))
    record = {
        "event": event,
        "observed_at": _timestamp(),
        "run_id": _text(payload.get("loam_run_id")),
        "workflow_id": _text(payload.get("workflow_id")),
        "plan_id": _text(start.get("plan_id")),
        "role": role,
        "hypothesis_id": _text(start.get("hypothesis_id")),
        "child_subagent_id": child_id,
        "child_session_id": child_session,
        "status": _text(payload.get("child_status")) or "error",
        "summary": summary,
    }
    if role == "hypothesis-validator":
        score = _evidence_score(summary)
        plan = _text(start.get("plan_id"))
        hyp = _text(start.get("hypothesis_id"))
        # C2: a reported score is only trustworthy if a code-generator child
        # actually completed an execution for this plan+hypothesis. The check is
        # domain-blind on purpose -- it asks only whether an execution child
        # reached the "completed" terminal state (the single success status, see
        # ledger._observation_status), never what the code computed -- so it
        # holds for folds, docking, small-molecule work, arithmetic, or any
        # future tool. An unbacked score becomes None/unbacked_no_exec, and the
        # claimed value is retained under reported_evidence_score for audit so
        # the reasoner can see the validator asserted a number without running.
        backed = any(
            row.get("event") == "code_generator_completed"
            and _text(row.get("plan_id")) == plan
            and _text(row.get("hypothesis_id")) == hyp
            and _text(row.get("status")) == "completed"
            for row in rows
        )
        if score is not None and not backed:
            record["reported_evidence_score"] = score
            record["evidence_score"] = None
            record["evidence_score_status"] = "unbacked_no_exec"
        else:
            record["evidence_score"] = score
            record["evidence_score_status"] = (
                "reported" if score is not None else "missing_or_invalid"
            )
    _append(_state_path(payload), record)
    if role == "hypothesis-validator":
        _append_campaign(payload, {**record, "event": "hypothesis_validation_completed"})
        write_validator_outcome(payload, record)
    elif role == "reasoner":
        plan_id = _text(start.get("plan_id"))
        validations = [
            row
            for row in rows
            if row.get("event") == "validator_completed" and _text(row.get("plan_id")) == plan_id
        ]
        _append_campaign(
            payload,
            {
                **record,
                "event": "hypothesis_campaign_round_completed",
                "validator_results": validations,
                "synthesis": summary,
            },
        )


def _validator_tasks(plan_id: str, selected: list[dict[str, Any]], registry: Mapping[str, AgentSpec], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The round's validator task objects, as delegate_task arguments.

    Separated from the directive that announces them because both the directive
    and `canonical_delegation_args` need these objects, and the latter used to
    recover them by string-parsing the directive's own text for a marker line.
    Removing the printed array broke that parse, which is the clearest possible
    argument against a function reading its own prose output.
    """

    validator = registry["hypothesis-validator"]
    execution = payload.get("loam_execution")
    campaign = campaign_context(payload)
    tasks = []
    for item in selected:
        hypothesis = item.get("hypothesis")
        hypothesis_id = _text(hypothesis.get("hypothesis_id")) if isinstance(hypothesis, Mapping) else ""
        if _generated(execution if isinstance(execution, Mapping) else None):
            tasks.append(_generated_validator_task(plan_id, hypothesis_id, item, execution, validator.prompt_body, campaign))  # type: ignore[arg-type]
            continue
        tasks.append(
            {
                "goal": (
                    "Validate the supplied hypothesis by delegating exactly one leaf child with "
                    "the required_child goal and context from your supplied JSON context. Do not "
                    "calculate directly or use any tool other than delegate_task. After the child "
                    "returns, assess its actual stdout and return the required report headings, "
                    "including a bare decimal Evidence Score. "
                    f"[loam-agent-role=hypothesis-validator] [loam-plan-id={plan_id}] "
                    f"[loam-hypothesis-id={hypothesis_id}]"
                ),
                "context": json.dumps(
                    {
                        "loam_validator_contract": "v1",
                        "selected_hypothesis": item,
                        **_campaign_field(campaign, "validator"),
                        "execution_contract": execution if isinstance(execution, Mapping) else {},
                        "required_child": {
                            "role": "leaf",
                            "goal": (
                                "Run the required exact arithmetic check "
                                f"[loam-agent-role=code-generator] [loam-plan-id={plan_id}] "
                                f"[loam-hypothesis-id={hypothesis_id}]"
                            ),
                            "context": _code_generator_context(execution if isinstance(execution, Mapping) else None),
                        },
                        "validator_instructions": (
                            "Delegate exactly one child using required_child as a leaf task. "
                            "Do not calculate directly. After its terminal result, return exactly these headings: "
                            "### Hypothesis; ### Outcome; ### Evidence Ledger; ### Credit Assignment; "
                            "### Evidence Score; ### Next Step. The Evidence Score must be a bare decimal "
                            "in [-1.00, 1.00], for example +0.75, immediately after its heading."
                        ),
                        # Last on purpose; see the generated-mode task for why.
                        "agent_definition": validator.prompt_body,
                    },
                    ensure_ascii=False,
                ),
                "role": "orchestrator",
            }
        )
    return tasks


def _validator_directive(plan_id: str, selected: list[dict[str, Any]], registry: Mapping[str, AgentSpec], payload: Mapping[str, Any]) -> str:
    """Announce what this round will launch; the arguments travel in the dispatch."""

    validator = registry["hypothesis-validator"]
    execution = payload.get("loam_execution")
    tasks = _validator_tasks(plan_id, selected, registry, payload)
    # The task array is NOT printed. It used to be, with instructions to copy it
    # verbatim, and the copying is what failed: a validator at char ~9,000 of an
    # 11 KB payload told its child the hypothesis "is not visible in this
    # message" while holding it (2026-10-21). `DISPATCH_TOOL` hands these exact
    # task objects to delegate_task from inside the adapter, so the coordinator
    # never transcribes them and this directive only has to say what is pending
    # and what the budget is. Everything the array carried -- the hypothesis, the
    # campaign goal and rules, the agent definition, the execution contract --
    # still reaches each validator; it travels in the dispatch instead of
    # through the coordinator's context.
    count = len(tasks)
    lines = [
        f"Loam orchestration directive: {count} hypothesis validator"
        f"{'s are' if count != 1 else ' is'} planned for this round. "
        f"Call {DISPATCH_TOOL} to launch {'them' if count != 1 else 'it'} now. It takes no arguments.",
        (
            "Do not compose a delegate_task batch yourself: the selected hypotheses are not in your "
            "context in full, and a batch you write is a substitute for the campaign's work rather "
            "than the campaign's work. Each validator receives its hypothesis, the campaign goal and "
            f"rules, and its role definition ({validator.prompt_path.name}) inlined by the dispatch. "
            "Do not perform experimental code or tool calls yourself."
        ),
        (
            f"Every executable action MUST be delegated by that validator to at most {_max_children(execution)} leaf code-generator children; no other child may execute code."  # type: ignore[arg-type]
            if _generated(execution if isinstance(execution, Mapping) else None)
            else "Every executable action MUST be delegated by that validator to one leaf code-generator child; no other child may execute code."
        ),
    ]
    return "\n".join(lines)


def _generated_validator_task(plan_id: str, hypothesis_id: str, item: Mapping[str, Any], execution: Mapping[str, Any], agent_definition: str = "", campaign: CampaignContext | None = None) -> dict[str, Any]:
    """Validator task for generated mode: it designs the experiment itself."""

    cap = _max_children(execution)
    return {
        "goal": (
            f"Validate the supplied hypothesis by designing the smallest decisive experiment and delegating "
            f"it to at most {cap} code-generator leaf children, one at a time, iterating on their actual "
            "results. Do not compute anything yourself; delegate_task is your only executable tool, though "
            f"you may call {', '.join(VALIDATOR_OBSERVATION_TOOLS)} to check on and read a child's run. "
            "As you commit to each high-level decision -- the experiment design, what to delegate, "
            "whether the evidence is sufficient -- record it with the loam_record_decision tool before "
            "you act on it; your reasoning is not otherwise kept. "
            "Then return the required report headings, including a bare decimal Evidence Score. "
            f"[loam-agent-role=hypothesis-validator] [loam-plan-id={plan_id}] "
            f"[loam-hypothesis-id={hypothesis_id}]"
        ),
        # Deliberately ordered, so sort_keys must stay off. Alphabetical order
        # put `agent_definition` first, and at ~8.6 KB it pushed the actual
        # assignment to roughly char 9,000 of an 11 KB payload -- a validator
        # then told its child "the hypothesis text is not visible in this
        # message" while holding it (2026-10-21). The assignment and the
        # instruction for passing it down come first; the role boilerplate goes
        # last, where its length costs nothing.
        "context": json.dumps(
            {
                "loam_validator_contract": "v1",
                "selected_hypothesis": item,
                # Immediately after the assignment and well inside the first
                # ~2 KB. This is the second field competing for the front of the
                # payload, so both halves are bounded (MAX_CAMPAIGN_CHARS,
                # MAX_RULES_CHARS) rather than left to grow.
                **_campaign_field(campaign, "validator"),
                "execution_contract": execution,
                "child_template": {
                    "role": "leaf",
                    "goal_suffix": _child_tags(plan_id, hypothesis_id),
                    "max_children": cap,
                    "workdir": _workdir(execution, plan_id, hypothesis_id),
                },
                "validator_instructions": (
                    "Follow agent_definition above; it is your role specification and is provided in full "
                    "because you cannot read its file. "
                    "Every code-generator child starts with an empty context and cannot see this payload, "
                    "the tournament state or the campaign files, so write selected_hypothesis into each "
                    "child's goal/context in full -- statement, candidate sequence and the falsifying "
                    "prediction. A hypothesis id is a provenance tag, not a reference a child can resolve. "
                    "Restate the hypothesis as a falsifiable claim and design the smallest decisive experiment. "
                    "Delegate each executable step to a code-generator child: role=leaf, goal = what to compute "
                    "and the acceptance condition, ending with child_template.goal_suffix; context = exact "
                    "inputs, parameter ranges and expected outputs. Children write their own code, can reuse "
                    "registered HPC tools, and run large sweeps as batches. At most max_children children in "
                    "total. Never compute or run code yourself. Do poll and read a child's run yourself with "
                    f"{', '.join(VALIDATOR_OBSERVATION_TOOLS)} rather than spending a child on it, and verify a "
                    "child's reported numbers against the files it actually wrote. Then return exactly these headings: "
                    "### Hypothesis; ### Outcome; ### Evidence Ledger; ### Credit Assignment; "
                    "### Evidence Score; ### Next Step. The Evidence Score must be a bare decimal "
                    "in [-1.00, 1.00], for example +0.75, immediately after its heading."
                ),
                # Last on purpose: it is the longest field and the least urgent.
                # Inlined rather than referenced because the validator cannot
                # read the file (see AgentSpec.prompt_body).
                "agent_definition": agent_definition,
            },
            ensure_ascii=False,
        ),
        "role": "orchestrator",
    }


def _elo_ratings(payload: Mapping[str, Any]) -> dict[str, float]:
    """Jnana Elo per hypothesis id, from the latest tournament report."""

    report = _latest_report(payload)
    if report is None:
        return {}
    result = report[1].get("result")
    rankings = result.get("top_hypotheses") if isinstance(result, Mapping) else None
    ratings: dict[str, float] = {}
    for ranking in rankings if isinstance(rankings, list) else []:
        if not isinstance(ranking, Mapping):
            continue
        hypothesis_id = _text(ranking.get("hypothesis_id"))
        value = ranking.get("elo_rating")
        if hypothesis_id and isinstance(value, (int, float)) and not isinstance(value, bool):
            ratings[hypothesis_id] = float(value)
    return ratings


def _reasoner_directive(plan_id: str, completions: list[dict[str, Any]], registry: Mapping[str, AgentSpec],
                        campaign: CampaignContext | None = None,
                        elo_ratings: Mapping[str, float] | None = None,
                        payload: Mapping[str, Any] | None = None,
                        for_reasoner: bool = False) -> str:
    """The coordinator's dispatch directive, or with `for_reasoner` the reasoner's own task context.

    The task drops the coordinator's dispatch lines: a reasoner that reads "call
    loam_dispatch_round now" tries it and returns that instead of a synthesis.
    """
    reasoner = registry["reasoner"]
    # Elo was reaching the validators (via `_validator_directive`) but never the
    # reasoner, which is asked to "assess their evidence and ELO signals
    # together" and told that a thread missing either score is incomplete
    # metadata. Every thread it has ever seen was therefore incomplete by its own
    # rule. The rating is per hypothesis, so it is looked up from the report
    # rather than carried on the completion rows.
    elo_ratings = elo_ratings or {}
    threads = [
        {
            "hypothesis_id": row.get("hypothesis_id", ""),
            "validator_status": row.get("status"),
            "evidence_score": row.get("evidence_score"),
            "evidence_score_status": row.get("evidence_score_status"),
            "elo_rating": elo_ratings.get(_text(row.get("hypothesis_id"))),
            "validator_report": _text(row.get("summary"))[:MAX_REASONER_THREAD_CHARS],
        }
        for row in completions
    ]
    memory = reasoner_memory_block(payload or {}, plan_id, completions)
    coordinator = not for_reasoner
    return "\n".join(
        part
        for part in (
            "Loam reasoner task: the validator threads below are terminal. Synthesize them as your definition "
            "(\"agent_definition\") describes and return your report; you dispatch nothing."
            if for_reasoner
            else "",
            coordinator and f"Loam orchestration directive: all selected validator threads are terminal. Delegate exactly one reasoner before responding: call {DISPATCH_TOOL} now (no arguments).",
            # The reasoner goes out through the round's dispatch tool too (the
            # coordinator's own delegate_task is blocked), and this directive
            # carries the whole definition, so it is still big enough for
            # Hermes to spill it to a file. Demanding a verbatim copy of text
            # the model may not be able to see is what deadlocked the
            # coordinator three times live; the middleware supplies the
            # definition either way, so say that instead.
            coordinator and f"The reasoner's role definition ({reasoner.prompt_path.name}) is below and Loam itself puts it into the dispatched task as \"agent_definition\"; the reasoner must not execute experimental code. Call {DISPATCH_TOOL} once (no arguments) to dispatch the reasoner -- Loam builds the canonical task, definition included; a delegate_task you compose yourself is blocked. Do not reproduce the definition yourself, and if this directive was written to a file instead of shown inline, dispatch anyway: you do not need to read that file to comply.",
            # This synthesis sets the next tournament's whole direction, so the
            # tier it runs on is a campaign decision, not an incidental one.
            # `AgentSpec.model` was parsed from the front matter and never used
            # anywhere, leaving the declared tier silently unhonored; stating it
            # here at least makes it visible to the dispatcher.
            (
                f"Dispatch it on the model its definition declares ({reasoner.model}); this synthesis "
                "steers the next tournament and should not run on a weaker tier than the validators it reads."
                if reasoner.model and coordinator
                else ""
            ),
            # The campaign block before the definition: reasoner.md's first
            # Required Parent Input is "objective, success criteria, constraints
            # and current state", and until now nothing supplied the first
            # three, so it synthesized toward "productive regions of hypothesis
            # space" with no stated thing to be productive toward.
            json.dumps(
                {**_campaign_field(campaign, "reasoner"), "agent_definition": reasoner.prompt_body},
                ensure_ascii=False,
            ),
            coordinator and f"Tag its goal with [loam-agent-role=reasoner] [loam-plan-id={plan_id}] so its brief is persisted for the next tournament turn.",
            "Completed validator threads are unvalidated orchestration input; preserve their reported uncertainty and do not turn summaries into accepted evidence.",
            memory,
            json.dumps(threads, ensure_ascii=False, sort_keys=True),
        )
        if part
    )


def canonical_delegation_args(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reconstruct pending campaign delegation arguments from persisted state.

    This is the primary path, not a fallback: the adapter registers it as
    `tool_request` middleware, so a model-written `delegate_task` is replaced by
    what this returns before it executes. `.loam/manifest.yaml` declares that
    seam, including its ordering -- it runs AFTER `pre_tool_call`, so a
    `tool_policy` veto preempts the repair.

    `None` means no repair is available for this state, and the four states are
    pinned in tests/test_delegation_repair.py:

      * a known hypothesis-validator delegating its child      -> repaired
      * coordinator on a fresh round                           -> repaired
      * coordinator with every validator complete              -> repaired
      * coordinator mid-round, validators launched and pending  -> NOT repaired

    That last state is why `tool_policy` still inspects a coordinator's
    delegation: with no repair available, the veto is the only thing between a
    substitute batch and execution. It also fails closed -- once a plan is
    dispatched, `context_for` stops offering the task array, so no repair can
    duplicate a live fan-out.
    """

    rows = _read_rows(_state_path(payload))
    session_id = _text(payload.get("session_id")) or _text(payload.get("parent_session_id"))
    starts = [
        row
        for row in rows
        if row.get("event") == "agent_started" and _text(row.get("child_session_id")) == session_id
    ]
    if starts:
        start = starts[-1]
        if _text(start.get("role")) != "hypothesis-validator":
            return None
        plan_id = _text(start.get("plan_id"))
        hypothesis_id = _text(start.get("hypothesis_id"))
        execution = payload.get("loam_execution")
        if _generated(execution if isinstance(execution, Mapping) else None):
            return _generated_child_tasks(payload, plan_id, hypothesis_id)
        return {
            "tasks": [
                {
                    "goal": (
                        "Run the required exact arithmetic check "
                        f"[loam-agent-role=code-generator] [loam-plan-id={plan_id}] "
                        f"[loam-hypothesis-id={hypothesis_id}]"
                    ),
                    "context": _code_generator_context(execution if isinstance(execution, Mapping) else None),
                    "role": "leaf",
                }
            ]
        }

    active = _active_plan(rows, _text(payload.get("workflow_id")), _text(payload.get("loam_run_id")))
    if active is not None:
        plan_id, starts, completions = active
        if starts and len(completions) == len(starts):
            if any(
                row.get("event") == "agent_started"
                and _text(row.get("role")) == "reasoner"
                and _text(row.get("plan_id")) == plan_id
                for row in rows
            ):
                return None
            return {
                "goal": (
                    "Synthesize validator results. As you commit to each synthesis decision -- which "
                    "hypotheses to promote or retire, how to steer the next round -- record it with the "
                    "loam_record_decision tool before you finalize your brief; your reasoning is not "
                    f"otherwise kept. [loam-agent-role=reasoner] [loam-plan-id={plan_id}]"
                ),
                "context": _reasoner_directive(plan_id, completions, agent_registry(payload), campaign_context(payload), _elo_ratings(payload), payload, for_reasoner=True),
                "role": "leaf",
            }
        return None

    # Build the round's arguments directly. This used to re-run `context_for`
    # and parse a marker line out of its prose, which tied the hook's ability to
    # dispatch to the exact wording of a message meant for a model.
    report = _latest_report(payload)
    if report is None:
        return None
    report_path, report_value = report
    plan_id = _plan_id(report_path, payload)
    workflow_id = _text(payload.get("workflow_id"))
    if any(
        row.get("event") == "reasoner_completed"
        and _text(row.get("plan_id")) == plan_id
        and _text(row.get("workflow_id")) == workflow_id
        for row in rows
    ):
        return None
    selected = _selected_hypotheses(payload, report_value)
    if not selected:
        return None
    return {"tasks": _validator_tasks(plan_id, selected, agent_registry(payload), payload)}


def _selected_hypotheses(payload: Mapping[str, Any], report_value: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The ranked hypotheses this round should validate, with their full text.

    `None` means there is nothing to dispatch: an empty ranking, or a ranked
    hypothesis whose full text is missing from the tournament state. The second
    case returns None rather than a shorter list on purpose -- dispatching a
    partial fan-out would silently drop a hypothesis the tournament selected.

    Shared by `context_for`, which announces the round, and
    `canonical_delegation_args`, which builds the arguments that launch it. One
    implementation, because two would eventually disagree about what the round is.
    """

    result = report_value.get("result")
    rankings = result.get("top_hypotheses") if isinstance(result, Mapping) else None
    if not isinstance(rankings, list) or not rankings:
        return None
    full_hypotheses = _full_hypotheses(payload)
    selected: list[dict[str, Any]] = []
    for rank, ranking in enumerate(rankings, start=1):
        if not isinstance(ranking, Mapping):
            continue
        hypothesis = full_hypotheses.get(_text(ranking.get("hypothesis_id")))
        if not hypothesis:
            return None
        selected.append({"rank": rank, "elo_rating": ranking.get("elo_rating"), "hypothesis": hypothesis})
    return selected


def context_for(payload: dict[str, Any]) -> str:
    """Inject the next scheduler action after the current tournament iteration."""

    registry = agent_registry(payload)
    rows = _read_rows(_state_path(payload))
    workflow_id = _text(payload.get("workflow_id"))
    brief = _latest_brief(rows, workflow_id)
    active = _active_plan(rows, workflow_id, _text(payload.get("loam_run_id")))
    if active is not None:
        plan_id, starts, completions = active
        if len(completions) == len(starts) and all(_text(row.get("status")) for row in completions):
            return _reasoner_directive(plan_id, completions, registry, campaign_context(payload), _elo_ratings(payload), payload)
        return ""
    report = _latest_report(payload)
    if report is None:
        return f"Latest reasoner brief for the next tournament turn:\n{brief}" if brief else ""
    report_path, report_value = report
    plan_id = _plan_id(report_path, payload)
    if any(
        row.get("event") == "reasoner_completed"
        and _text(row.get("plan_id")) == plan_id
        and _text(row.get("workflow_id")) == workflow_id
        for row in rows
    ):
        return f"Latest reasoner brief for the next tournament turn:\n{brief}" if brief else ""
    selected = _selected_hypotheses(payload, report_value)
    if selected is None:
        return ""
    return _validator_directive(plan_id, selected, registry, payload)
