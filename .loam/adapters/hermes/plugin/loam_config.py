"""Load a named Loam profile from global configuration.

The global plugin stays inert when no usable configuration exists. Environment
variables remain explicit emergency overrides, not the normal operational API.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - Hermes normally supplies PyYAML
    yaml = None

DEFAULT_CONFIG_PATH = Path.home() / ".hermes" / "loam.yaml"
DEFAULT_INTERVAL = 5
_ALLOWED_POLICIES = {"allow_and_audit", "block_exact_duplicate"}


@dataclass(frozen=True)
class HypothesisTournamentConfig:
    """Bounded Jnana tournament settings for an opt-in Loam profile."""

    jnana_root: Path
    state_file: Path
    reports_dir: Path
    initial_hypotheses: int
    matches_per_run: int
    top_k: int
    timeout_seconds: int
    provider: str
    inference_endpoint: str
    endpoint_is_chat_completions: bool
    model: str
    api_key_env: str | None
    omit_temperature: bool
    json_max_tokens: int
    # Optional knowledge-graph pre-pass before the first generation
    # (.loam/hooks/knowledge_brief.py); None when switched off.
    knowledge: Mapping[str, Any] | None = None
    # Optional per-role model overrides, keyed by role name ("generation" for
    # the hypothesis generators, "ranking" for the pairwise judge). Each value
    # carries only the keys the profile actually set; the hook falls back to the
    # fields above for everything else. Empty when no role block is configured.
    roles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)



@dataclass(frozen=True)
class CampaignConfig:
    """Invariant campaign context: what the campaign is for, and its rules.

    This is campaign policy rather than compute configuration, so it sits beside
    `execution` instead of inside it. `goal` and `rules` name files that the
    hooks read, the way agent prompts are already loaded: the hook holds the
    paths and this module does not, so a name is validated here and read there.

    `dir` is where those files live -- a directory the user owns, outside this
    repo. Loam ships no campaign content: a goal, and a rule naming a particular
    measurement, belong to whoever is running the campaign, and checking one in
    would make this framework quietly about protein binders. The repo's own
    `.loam/rules/` holds only rules invariant across every campaign, and those
    load whether or not `dir` is set.

    Names stay bare and validated even though `dir` is a free path: the user
    chooses the directory once, in their own profile, and nothing inside the
    campaign can then reach a file outside it.

    Only the invariant half of a campaign's context lives here. The evolving
    half -- current state -- already exists as the reasoner brief and the
    reconciled findings that the pre-LLM recall hooks inject.
    """

    goal: str
    rules: tuple[str, ...] = ()
    dir: str = ""
    # Skills a code-generator should consult before writing code. Names only:
    # these are resolved by the harness's own skill system, not by Loam, so
    # unlike `rules` they name nothing this repo or `dir` contains. Named rather
    # than left open because `skills_list` returns every skill installed in the
    # Hermes home, and a child with three children of budget should not browse.
    skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution contract supplied to hypothesis-validation children."""

    mode: str
    tool_name: str = ""
    endpoint: str = ""
    function_name: str = ""
    input_schema: str = ""
    runtime: str = ""
    # Campaign-specific environment setup handed verbatim to every
    # code-generator child (how to activate venvs, which tool CLIs exist, where
    # weights live). The child runs on the remote host and cannot see the
    # campaign's local files, so a recipe it must follow has to travel in the
    # contract. Empty by default; generic Loam carries no host paths.
    setup: str = ""
    # generated (default): the code-generator writes its own code from the
    # validator's instruction. fixed is for debugging the orchestration only:
    # it runs Loam's built-in arithmetic check instead. Never a default.
    code_source: str = "generated"
    max_children: int = 3
    workdir_root: str = ""
    # Prefix of the HPC server's other MCP tool names (run_batch, list_tools,
    # ...). Empty: derived from tool_name when it ends with function_name.
    tool_prefix: str = ""
    # Other spellings of a workdir_root prefix, as (alias, canonical) pairs:
    # on Aurora /lus/flare/projects/X and /flare/X are one directory. The
    # workspace checks compare HPC paths as text, so they need to be told.
    workdir_aliases: tuple[tuple[str, str], ...] = ()
    # Where the HPC server keeps run_batch results (its HPC_BATCH_ROOT).
    # Empty: <workdir_root>/../batches (hypothesis_workspace.batch_root_for).
    batch_root: str = ""


_EXECUTION_MODES = {"remote_mcp", "local"}
_CODE_SOURCES = {"fixed", "generated"}


@dataclass(frozen=True)
class WorkspaceConfig:
    """Pando-style per-hypothesis workspace (see .loam/hooks/hypothesis_workspace.py)."""

    # Parent of the <plan_id>/<hypothesis_id> folders. Empty: <state_dir>/hypotheses,
    # resolved in payload_values so a LOAM_STATE_DIR override moves it too.
    root: str = ""
    # true: pre-tool blocks layout violations. false: they are only logged.
    strict: bool = True
    # Background verifier for HPC runs (.loam/reconcile/verify_workspace.py):
    # an interpreter that has fastmcp, and the HPC MCP server's URL. Both empty:
    # no verifier is launched and a missing remote manifest stays missing.
    verifier_python: str = ""
    verifier_mcp_url: str = ""
    # Campaign inputs every hypothesis starts with, COPIED into its own data/
    # so the record is self-contained: `inputs_dir` is a local directory copied
    # by the hook, `remote_inputs_dir` an HPC directory the operator staged once
    # and the verifier copies (a hook cannot reach HPC). Either may be set.
    inputs_dir: str = ""
    remote_inputs_dir: str = ""
    # Read-only reference data too large to copy per hypothesis. Children may
    # READ these absolute paths in place; each read is recorded as an input.
    # Nothing is ever written there, and it is opt-in: unset, the only readable
    # inputs are the hypothesis's own data/.
    shared_inputs: tuple[str, ...] = ()
    # Refuse a seed bigger than this rather than silently copying a dataset
    # into every hypothesis. 0 keeps the hook's default.
    max_input_bytes: int = 0
    # Keep the FULL per-agent trace (every tool result verbatim) when capturing
    # it from state.db at stop. Default false strips the bodies of skill reads,
    # which are already hashed under the run's `skills`, so the trace is not a
    # second copy of the whole skill library.
    trace_full: bool = False


@dataclass(frozen=True)
class LoamProfile:
    name: str
    workflow_id: str
    state_dir: Path
    pre_llm_every: int
    duplicate_policy: str
    hypothesis_orchestration: bool = False
    campaign: CampaignConfig | None = None
    execution: ExecutionConfig | None = None
    hypothesis_tournament: HypothesisTournamentConfig | None = None
    workspace: WorkspaceConfig | None = None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _interval(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default



# A campaign or rules name becomes a file name under `.loam/`, so it may not
# carry a path at all: `../../../etc/passwd`, or an absolute path, would
# otherwise let a profile pull arbitrary text into every child's context.
# Requiring a leading alphanumeric also rejects `.` and `..` outright.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _campaign_name(value: Any, field_name: str) -> str:
    name = _clean_text(value)
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"campaign.{field_name} must be a bare file name: {value!r}")
    return name


def _campaign_config(value: Any) -> CampaignConfig | None:
    """Parse the optional `campaign:` block.

    `None` means a profile with no campaign context, which is the pre-existing
    behavior: nothing is injected. A malformed block raises instead, for the same
    reason `_execution_config` rejects a bad mode -- a typo that silently dropped
    the objective would leave every validator aiming at nothing while the run
    still looked governed.

    `goal` is required whenever the block exists. Rules alone would be the older
    "rules context" framing, in which policy says what not to do and nothing says
    what the campaign is for.
    """

    raw = _as_mapping(value)
    if raw is None:
        return None
    goal = _campaign_name(raw.get("goal"), "goal")
    rules = raw.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError(f"campaign.rules must be a list of names: {rules!r}")
    names = tuple(_campaign_name(item, "rules") for item in rules)
    if len(set(names)) != len(names):
        raise ValueError(f"campaign.rules must not repeat a name: {list(names)!r}")
    raw_skills = raw.get("skills", [])
    if not isinstance(raw_skills, list):
        raise ValueError(f"campaign.skills must be a list of names: {raw_skills!r}")
    skills = tuple(_campaign_name(item, "skills") for item in raw_skills)
    if len(set(skills)) != len(skills):
        raise ValueError(f"campaign.skills must not repeat a name: {list(skills)!r}")
    return CampaignConfig(
        goal=goal, rules=names, dir=_campaign_dir(raw.get("dir")), skills=skills
    )


def _campaign_dir(value: Any) -> str:
    """Parse the optional `campaign.dir`, the user's own campaign directory.

    Absolute after `~` expansion, because a relative path would resolve against
    whatever directory the harness happened to spawn the hook in. Empty means
    the campaign's files are expected in the repo, which is the older behavior
    and still works for someone keeping their campaign beside the code.
    """

    raw = _clean_text(value)
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"campaign.dir must be an absolute path: {value!r}")
    return str(path)


def _execution_config(value: Any) -> ExecutionConfig | None:
    raw = _as_mapping(value)
    if raw is None:
        return None
    # A bad value in either field raises rather than returning None. `None`
    # means "no execution block", which silently turns a configured HPC
    # campaign into the local arithmetic self-test (no generated mode, an empty
    # HPC tool allowlist, no max_children budget, no setup/workdir context) --
    # with orchestration enabled it instead makes load_profile reject the whole
    # profile, so every Loam hook goes inert. Neither belongs to a typo.
    mode = _clean_text(raw.get("mode"))
    if mode not in _EXECUTION_MODES:
        raise ValueError(
            f"execution.mode must be one of {sorted(_EXECUTION_MODES)}: {mode!r}"
        )
    code_source = _clean_text(raw.get("code_source")) or "generated"
    if code_source not in _CODE_SOURCES:
        raise ValueError(
            f"execution.code_source must be one of {sorted(_CODE_SOURCES)}: {code_source!r}"
        )
    extras = dict(
        code_source=code_source,
        max_children=_positive_int(raw.get("max_children"), 3 if code_source == "generated" else 1),
        workdir_root=_clean_text(raw.get("workdir_root")),
        tool_prefix=_clean_text(raw.get("tool_prefix")),
        setup=_clean_text(raw.get("setup")),
        workdir_aliases=_aliases(raw.get("workdir_aliases")),
        batch_root=_clean_text(raw.get("batch_root")),
    )
    if mode == "local":
        return ExecutionConfig(mode=mode, runtime=_clean_text(raw.get("runtime")), **extras)
    tool_name = _clean_text(raw.get("tool_name"))
    endpoint = _clean_text(raw.get("endpoint"))
    function_name = _clean_text(raw.get("function_name"))
    input_schema = _clean_text(raw.get("input_schema"))
    if not all((tool_name, endpoint, function_name, input_schema)):
        return None
    return ExecutionConfig(
        mode=mode,
        tool_name=tool_name,
        endpoint=endpoint,
        function_name=function_name,
        input_schema=input_schema,
        runtime=_clean_text(raw.get("runtime")),
        **extras,
    )

def _workspace_config(value: Any) -> WorkspaceConfig | None:
    """`workspace: {enabled: true, ...}` turns the workspace on; anything else leaves it off."""

    raw = _as_mapping(value)
    if raw is None or raw.get("enabled") is not True:
        return None
    return WorkspaceConfig(
        root=_clean_text(raw.get("root")),
        strict=raw.get("strict") is not False,
        verifier_python=_clean_text(raw.get("verifier_python")),
        verifier_mcp_url=_clean_text(raw.get("verifier_mcp_url")),
        inputs_dir=_clean_text(raw.get("inputs_dir")),
        remote_inputs_dir=_clean_text(raw.get("remote_inputs_dir")),
        shared_inputs=_shared_inputs(raw.get("shared_inputs")),
        max_input_bytes=_positive_int(raw.get("max_input_bytes"), 0),
        trace_full=raw.get("trace_full") is True,
    )


def _shared_inputs(value: Any) -> tuple[str, ...]:
    """`shared_inputs: [/abs/path, ...]` -- absolute paths only, deduplicated.

    A relative path is dropped rather than resolved: these are matched against
    paths a child names, and the hook's working directory is not the child's.
    """

    items = value if isinstance(value, (list, tuple)) else [value] if value else []
    paths = []
    for item in items:
        text = _clean_text(item).rstrip("/")
        if text.startswith("~"):
            text = str(Path(text).expanduser()).rstrip("/")
        if text.startswith("/") and text not in paths:
            paths.append(text)
    return tuple(paths)


def _aliases(value: Any) -> tuple[tuple[str, str], ...]:
    """`workdir_aliases: {alias: canonical}` as normalized absolute-path pairs."""

    raw = _as_mapping(value) or {}
    pairs = []
    for alias, canonical in raw.items():
        alias, canonical = _clean_text(alias).rstrip("/"), _clean_text(canonical).rstrip("/")
        if alias.startswith("/") and canonical.startswith("/") and alias != canonical:
            pairs.append((alias, canonical))
    return tuple(sorted(pairs))


def _tournament_config(value: Any, state_dir: Path | None = None) -> HypothesisTournamentConfig | None:
    raw = _as_mapping(value)
    if raw is None or raw.get("enabled") is not True:
        return None
    required = {
        "jnana_root": _clean_text(raw.get("jnana_root")),
        "provider": _clean_text(raw.get("provider")) or "openai",
        "inference_endpoint": _clean_text(raw.get("inference_endpoint")) or _clean_text(raw.get("base_url")),
        "model": _clean_text(raw.get("model")),
        "api_key_env": _clean_text(raw.get("api_key_env")),
    }
    configured_state_file = _clean_text(raw.get("state_file"))
    configured_reports_dir = _clean_text(raw.get("reports_dir"))
    if any(not required[key] for key in ("jnana_root", "provider", "inference_endpoint", "model")):
        return None
    if state_dir is None and (not configured_state_file or not configured_reports_dir):
        return None
    return HypothesisTournamentConfig(
        jnana_root=Path(required["jnana_root"]).expanduser(),
        state_file=state_dir / "jnana-memory.json" if state_dir else Path(configured_state_file).expanduser(),
        reports_dir=state_dir / "tournament-reports" if state_dir else Path(configured_reports_dir).expanduser(),
        initial_hypotheses=_positive_int(raw.get("initial_hypotheses"), 3),
        matches_per_run=_positive_int(raw.get("matches_per_run"), 1),
        top_k=_positive_int(raw.get("top_k"), 3),
        timeout_seconds=_positive_int(raw.get("timeout_seconds"), 30),
        provider=required["provider"],
        inference_endpoint=required["inference_endpoint"],
        endpoint_is_chat_completions=raw.get("endpoint_is_chat_completions") is True,
        model=required["model"],
        api_key_env=required["api_key_env"] or None,
        omit_temperature=raw.get("omit_temperature") is True,
        json_max_tokens=_positive_int(raw.get("json_max_tokens"), 1024),
        knowledge=_knowledge_config(raw.get("knowledge")),
        roles=_role_configs(raw),
    )


# Tournament roles a profile may point at their own provider/model/endpoint, and
# the keys such a block may carry. Everything else (jnana_root, state paths,
# match counts, the knowledge pre-pass) stays shared across roles.
_TOURNAMENT_ROLES = ("generation", "ranking")
_ROLE_TEXT_KEYS = ("provider", "model", "inference_endpoint", "api_key_env")
_ROLE_FLAG_KEYS = ("endpoint_is_chat_completions", "omit_temperature")


def _role_configs(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Collect `generation:` / `ranking:` model overrides, dropping empty blocks.

    Only keys the profile actually set are carried through, so the hook can tell
    "unset" from "set to the same value as the default" and fall back per key.
    """

    roles: dict[str, dict[str, Any]] = {}
    for role in _TOURNAMENT_ROLES:
        block = _as_mapping(raw.get(role))
        if block is None:
            continue
        override: dict[str, Any] = {}
        for key in _ROLE_TEXT_KEYS:
            if text := _clean_text(block.get(key)):
                override[key] = text
        for key in _ROLE_FLAG_KEYS:
            if key in block:
                override[key] = block.get(key) is True
        if "json_max_tokens" in block:
            override["json_max_tokens"] = _positive_int(block.get("json_max_tokens"), 1024)
        if override:
            roles[role] = override
    return roles


def _knowledge_config(value: Any) -> dict[str, Any] | None:
    """`knowledge: {enabled: true, ...}` turns the pre-pass on; anything else leaves it off."""

    raw = _as_mapping(value)
    if raw is None or raw.get("enabled") is not True:
        return None
    tools = raw.get("tools")
    return {
        "enabled": True,
        "name": _clean_text(raw.get("name")) or "OKN",
        "mcp_url": _clean_text(raw.get("mcp_url")) or "https://apps.okn.us/okn-mcp-dev/mcp",
        "max_tool_calls": _positive_int(raw.get("max_tool_calls"), 12),
        **({"model": model} if (model := _clean_text(raw.get("model"))) else {}),
        "timeout_seconds": _positive_int(raw.get("timeout_seconds"), 900),
        "max_brief_chars": _positive_int(raw.get("max_brief_chars"), 4000),
        **({"tools": [str(t) for t in tools]} if isinstance(tools, list) else {}),
        **({"seed_terms": [str(s) for s in seeds]} if isinstance(seeds := raw.get("seed_terms"), list) else {}),
    }


def config_path() -> Path:
    override = _clean_text(os.environ.get("LOAM_CONFIG_PATH"))
    return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH


def state_dir_override() -> Path | None:
    """Return the per-run output directory, when explicitly supplied."""

    value = _clean_text(os.environ.get("LOAM_STATE_DIR"))
    return Path(value).expanduser() if value else None


def load_profile() -> LoamProfile | None:
    """Load the selected profile, returning None when Loam is intentionally inert."""

    path = config_path()
    if yaml is None or not path.is_file():
        return None
    try:
        root = _as_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError):
        return None
    if not root or root.get("enabled") is not True:
        return None
    requested = _clean_text(os.environ.get("LOAM_PROFILE")) or _clean_text(root.get("active_profile"))
    profiles = _as_mapping(root.get("profiles"))
    if not requested or not profiles:
        return None
    raw = _as_mapping(profiles.get(requested))
    if not raw or raw.get("enabled") is False:
        return None
    workflow_id = _clean_text(raw.get("workflow_id"))
    state_dir = _clean_text(raw.get("state_dir"))
    override = state_dir_override()
    if not workflow_id or (override is None and not state_dir):
        return None
    policy = _clean_text(raw.get("duplicate_policy")) or "allow_and_audit"
    if policy not in _ALLOWED_POLICIES:
        return None
    orchestration = _as_mapping(raw.get("hypothesis_orchestration")) is not None and raw["hypothesis_orchestration"].get("enabled") is True
    # Deliberately not required by orchestration: an existing profile has no
    # `campaign:` block, and rejecting it here would take every hook inert
    # rather than merely leaving the campaign context uninjected.
    campaign = _campaign_config(raw.get("campaign"))
    execution = _execution_config(raw.get("execution"))
    if orchestration and execution is None:
        return None
    effective_state_dir = override or Path(state_dir).expanduser()
    return LoamProfile(
        name=requested,
        workflow_id=workflow_id,
        state_dir=effective_state_dir,
        pre_llm_every=_interval(raw.get("pre_llm_every", DEFAULT_INTERVAL)),
        duplicate_policy=policy,
        hypothesis_orchestration=orchestration,
        campaign=campaign,
        execution=execution,
        hypothesis_tournament=_tournament_config(raw.get("hypothesis_tournament"), override),
        workspace=_workspace_config(raw.get("workspace")),
    )


def emergency_enabled() -> bool:
    return _clean_text(os.environ.get("LOAM_ENABLED")).casefold() in {"1", "true", "yes", "on"} or any(
        _clean_text(os.environ.get(name))
        for name in ("LOAM_STATE_DIR", "LOAM_LEDGER_PATH", "LOAM_PARENT_MEMORY_PATH", "LOAM_WORKFLOW_ID")
    )


def effective_profile() -> LoamProfile | None:
    """Config is normal activation; legacy env activation remains a temporary fallback."""

    profile = load_profile()
    if profile is not None:
        return profile
    if not emergency_enabled():
        return None
    override = state_dir_override()
    state_dir = override or Path(_clean_text(os.environ.get("LOAM_LEDGER_PATH")) or ".loam/state").expanduser().parent
    return LoamProfile(
        name="environment-override",
        workflow_id=_clean_text(os.environ.get("LOAM_WORKFLOW_ID")) or "environment-workflow",
        state_dir=state_dir,
        pre_llm_every=_interval(os.environ.get("LOAM_PRE_LLM_EVERY", DEFAULT_INTERVAL)),
        duplicate_policy="allow_and_audit",
    )


def payload_values(profile: LoamProfile) -> dict[str, Any]:
    """Values passed to each hook, avoiding hook-local global-config reads."""

    values: dict[str, Any] = {
        "loam_profile": profile.name,
        "workflow_id": profile.workflow_id,
        "loam_state_dir": str(profile.state_dir),
        "loam_duplicate_policy": profile.duplicate_policy,
        "loam_hypothesis_orchestration": profile.hypothesis_orchestration,
    }
    campaign = profile.campaign
    if campaign is not None:
        # Names plus the directory holding them. The hooks do the reading; this
        # module never opens a campaign file.
        values["loam_campaign"] = {
            "goal": campaign.goal,
            "rules": list(campaign.rules),
            "dir": campaign.dir,
            "skills": list(campaign.skills),
        }
    execution = profile.execution
    if execution is not None:
        values["loam_execution"] = {
            "mode": execution.mode,
            "tool_name": execution.tool_name,
            "endpoint": execution.endpoint,
            "function_name": execution.function_name,
            "input_schema": execution.input_schema,
            "runtime": execution.runtime,
            "setup": execution.setup,
            "code_source": execution.code_source,
            "max_children": execution.max_children,
            "workdir_root": execution.workdir_root,
            "tool_prefix": execution.tool_prefix,
        }
        if execution.workdir_aliases:
            values["loam_execution"]["workdir_aliases"] = dict(execution.workdir_aliases)
        if execution.batch_root:
            values["loam_execution"]["batch_root"] = execution.batch_root
    workspace = profile.workspace
    if workspace is not None:
        values["loam_workspace"] = {
            "enabled": True,
            "root": str(Path(workspace.root).expanduser() if workspace.root else profile.state_dir / "hypotheses"),
            "strict": workspace.strict,
        }
        if workspace.trace_full:
            values["loam_workspace"]["trace_full"] = True
        for key in ("verifier_python", "verifier_mcp_url", "remote_inputs_dir", "max_input_bytes"):
            if getattr(workspace, key):
                values["loam_workspace"][key] = getattr(workspace, key)
        if workspace.inputs_dir:
            values["loam_workspace"]["inputs_dir"] = str(Path(workspace.inputs_dir).expanduser())
        if workspace.shared_inputs:
            values["loam_workspace"]["shared_inputs"] = list(workspace.shared_inputs)
    run_id = _clean_text(os.environ.get("LOAM_RUN_ID"))
    if run_id:
        values["loam_run_id"] = run_id
    tournament = profile.hypothesis_tournament
    if tournament is not None:
        # Deliberately omit any secret value: hooks receive only an environment
        # variable name from which they may resolve the API credential.
        values["loam_hypothesis_tournament"] = {
            "enabled": True,
            "jnana_root": str(tournament.jnana_root),
            "state_file": str(tournament.state_file),
            "reports_dir": str(tournament.reports_dir),
            "initial_hypotheses": tournament.initial_hypotheses,
            "matches_per_run": tournament.matches_per_run,
            "top_k": tournament.top_k,
            "timeout_seconds": tournament.timeout_seconds,
            "provider": tournament.provider,
            "inference_endpoint": tournament.inference_endpoint,
            "endpoint_is_chat_completions": tournament.endpoint_is_chat_completions,
            "model": tournament.model,
            "api_key_env": tournament.api_key_env or "",
            "omit_temperature": tournament.omit_temperature,
            "json_max_tokens": tournament.json_max_tokens,
        }
        if tournament.knowledge is not None:
            values["loam_hypothesis_tournament"]["knowledge"] = dict(tournament.knowledge)
        for role, override in tournament.roles.items():
            values["loam_hypothesis_tournament"][role] = dict(override)
    return values
