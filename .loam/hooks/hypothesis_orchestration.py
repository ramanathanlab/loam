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

import hypothesis_workspace

AGENT_NAMES = ("reasoner", "hypothesis-validator", "code-generator")
STATE_FILENAME = "hypothesis-orchestration.jsonl"
CAMPAIGN_FILENAME = "hypothesis-campaign.jsonl"
MAX_REASONER_THREAD_CHARS = 6_000
EVIDENCE_SCORE_HEADING = "### Evidence Score"
MAX_BRIEF_CHARS = 2_000
MAX_CHILD_GOAL_CHARS = 300
# Bound on the hypothesis text forwarded into a child's context.
MAX_HYPOTHESIS_CHARS = 4_000
# Bound on an injected agent definition. The checked-in prompts are ~4-10 KB;
# this leaves headroom while keeping a runaway edit out of every child context.
MAX_AGENT_PROMPT_CHARS = 24_000
# Tools of the hpc-deployment MCP server a generated-mode code-generator may use.
HPC_TOOLS = ("run_code", "run_batch", "batch_status", "list_batches", "register_tool", "list_tools", "inspect_filesystem")
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
    # only `delegate_task`, so a validator told to read its own definition gets
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


def _prompt_body(path: Path) -> str:
    """Return the agent prompt with its front matter removed, bounded in size."""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0] == "---":
        for index in range(1, len(lines)):
            if lines[index] == "---":
                lines = lines[index + 1 :]
                break
        else:
            raise ValueError(f"agent prompt has unterminated front matter: {path}")
    return "\n".join(lines).strip()[:MAX_AGENT_PROMPT_CHARS]


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


def _hypothesis_block(hypothesis: Mapping[str, Any] | None) -> str:
    """The hypothesis under test, for the child's context. Empty when unknown.

    Forwarded structurally rather than left to the validator to copy. The
    validator is instructed to paste it and cannot reliably do so: with the
    hypothesis at char ~9,000 of its own payload it told its child "the
    hypothesis text is not visible in this message" and sent only an id, and the
    child then spent its whole budget walking Aurora's filesystem grepping for
    that id (2026-10-21). A child starts with an empty context, so an id is not
    a reference it can resolve -- the text has to travel with the task.
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
        "- The molecules under test are defined ONLY by this block: the candidate design described "
        "here, folded against the target named here (fetch the target from its authoritative "
        "database by accession at run time; never paste it from memory).\n"
        "- If the candidate sequence is given explicitly, use it verbatim. If the hypothesis gives "
        "only a design concept (length, topology, motifs) and no exact sequence, CONSTRUCT or DESIGN "
        "one or a few concrete candidate sequences realizing that architecture, write them to a FASTA "
        "under your workdir, and test those.\n"
        "- NEVER fold, score, or reuse a FASTA/PDB/input file you discovered on the filesystem (e.g. "
        "anything under chai_batch_test, prior outputs, or another campaign's directory). Those are "
        "the wrong molecule and make the result scientifically meaningless. A run on the wrong input "
        "is a failed validation, not evidence.\n"
        "- The decisive experiment is a complex co-fold of candidate binder + target (both chains in "
        "one input), reporting interface confidence (ipTM/ipSAE/pDockQ) and per-residue pLDDT, per "
        "the ENVIRONMENT SETUP recipe.\n"
        "- RUN EXACTLY ONE fold (one candidate vs the target) as the decisive check — not a panel of "
        "many. A single Chai fold takes ~2-3 min. After launching it, POLL until it is actually "
        "finished (batch_status shows complete, or the run_code call returns with the .cif on disk), "
        "then read the real metrics and report them. NEVER report while a run is unfinished: an "
        "unfinished run yields no evidence and wastes the whole round. Budget is one fold per check, "
        "so make the one fold count and wait for it rather than firing many and reporting early."
    )
    return "\n".join(lines)


def _layout_block(base: str, run_id: str, remote: bool = False, batch_root: str = "",
                  plan_id: str = "", hypothesis_id: str = "") -> str:
    """Where a child's files go in its hypothesis workspace (hypothesis_workspace.py).

    On HPC the stop hook cannot list the run's outputs, so a remote child is
    also told to report them as a LOAM_MANIFEST block, which
    hypothesis_workspace.remote_manifest parses from its final report. With a
    batch root it is also told how to name a run_batch and how to harvest its
    results into outputs/rM, which run_batch alone does not do.
    """

    if not run_id:
        return ""
    lines = [
        f"WORKSPACE LAYOUT (enforced by Loam; your run is {run_id}):",
        f"- code and scripts you write: {base}/src/",
        f"- result files: {base}/outputs/{run_id}/",
        f"- temporary files: {base}/scratch/{run_id}/",
        f"- inputs: read from {base}/data/ and never write there",
        "- earlier results of OTHER runs (this hypothesis's or another's): readable only from their "
        "completed runs' outputs/rK/; Loam records each such read as an input of your run",
        "Write nothing anywhere else, and never /tmp: /tmp is local to one node and the next tool call "
        "may run on another. Other runs' outputs are frozen; do not modify them.",
    ]
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
                             hypothesis: Mapping[str, Any] | None = None, layout: str = "") -> str:
    """Context for a code-generator that writes and runs its own code.

    ``layout`` is the _layout_block for the child's reserved run; empty when the
    workspace is off, which leaves the context exactly as before.
    """

    execution = execution if isinstance(execution, Mapping) else {}
    hypothesis_block = _hypothesis_block(hypothesis)
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
    return "\n\n".join(part for part in (
        "Write your own Python for the validator's instruction below and run it on HPC through the "
        "remote MCP tools listed here; do not execute code locally or use any other tool.",
        # Before the validator's instruction: the claim matters more than the
        # wording of the request, and the validator's text may contradict it
        # (e.g. "the hypothesis text is not visible in this message").
        hypothesis_block,
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
    items = tasks if isinstance(tasks, list) else [raw]
    workdir = _workdir(execution if isinstance(execution, Mapping) else None, plan_id, hypothesis_id)
    # Look the hypothesis up here rather than trusting the validator to have
    # copied it into the instruction; see _hypothesis_block. Unknown id -> empty
    # block, and the child's context is exactly as it was before.
    hypothesis = _full_hypotheses(payload).get(hypothesis_id)
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
    # leaves no reserved runs behind.
    run_ids, layout_base = _reserve_child_runs(payload, execution, workdir, plan_id, hypothesis_id, len(parsed))
    remote = isinstance(execution, Mapping) and _text(execution.get("mode")) == "remote_mcp"
    batch_root = hypothesis_workspace.batch_root_for(execution if isinstance(execution, Mapping) else None) if remote else ""
    out = []
    for (goal, instruction), run_id in zip(parsed, run_ids):
        out.append(
            {
                "goal": _child_goal(hypothesis, goal, plan_id, hypothesis_id, run_id),
                "context": _generated_child_context(
                    instruction, execution, workdir, hypothesis, _layout_block(layout_base, run_id, remote, batch_root, plan_id, hypothesis_id)
                ),
                "role": "leaf",
            }
        )
    return {"tasks": out}


def _reserve_child_runs(
    payload: Mapping[str, Any],
    execution: Any,
    workdir: str,
    plan_id: str,
    hypothesis_id: str,
    count: int,
) -> tuple[list[str], str]:
    """Reserve one workspace run per child, and the folder the child should write under.

    Returns empty run ids when the workspace is off or reservation fails; the
    children are then delegated exactly as before, and record_start allocates
    a run for each when it starts. A workspace problem never blocks delegation.
    """

    if hypothesis_workspace.settings(payload) is None:
        return [""] * count, ""
    try:
        root = hypothesis_workspace.workspace_root(payload, plan_id, hypothesis_id)
        run_ids = hypothesis_workspace.reserve_runs(root, count)
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

def tool_policy(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """Block tool use that violates a tagged orchestration role's boundary."""

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
    if not starts and tool_name == "delegate_task" and isinstance(arguments, Mapping):
        tasks = arguments.get("tasks")
        raw_tasks = tasks if isinstance(tasks, list) else [arguments]
        validator_tasks = [
            (index, task)
            for index, task in enumerate(raw_tasks)
            if isinstance(task, Mapping)
            and _marker(_text(task.get("goal")), "loam-agent-role") == "hypothesis-validator"
        ]
        if latest_plan_id and not plan_finished and not active_validators_complete and not validator_tasks:
            return {
                "action": "block",
                "message": (
                    "The active campaign requires one tagged hypothesis-validator task for each "
                    "selected hypothesis; do not dispatch an untagged substitute batch."
                ),
            }
        missing_orchestrator_role = [
            index
            for index, task in validator_tasks
            if _text(task.get("role")) != "orchestrator"
        ]
        if missing_orchestrator_role:
            return {
                "action": "block",
                "message": (
                    "Each tagged hypothesis-validator in a delegate_task batch requires "
                    "per-task role=orchestrator so it can delegate its code-generator child; "
                    f"invalid task indexes: {missing_orchestrator_role}"
                ),
            }
        missing_contract = [
            index
            for index, task in validator_tasks
            if '"loam_validator_contract": "v1"' not in _text(task.get("context"))
        ]
        if missing_contract:
            return {
                "action": "block",
                "message": (
                    "Each tagged hypothesis-validator must preserve the complete injected task context "
                    f"with loam_validator_contract v1; invalid task indexes: {missing_contract}"
                ),
            }
    if not starts and latest_plan_id and not plan_finished and not active_validators_complete and tool_name != "delegate_task":
        return {
            "action": "block",
            "message": (
                "The active campaign plan is pending. Do not execute or inspect work as coordinator; "
                "delegate tagged hypothesis-validator tasks from the injected payload first."
            ),
        }
    if not session_id:
        return None
    if not starts:
        return None
    role = _text(starts[-1].get("role"))
    tool_name = _text(payload.get("tool_name"))
    if role == "reasoner":
        return {"action": "block", "message": "reasoner has no tool capability; return the synthesis to the harness"}
    if role == "code-generator":
        if tool_name == "delegate_task":
            return {"action": "block", "message": "code-generator must not delegate"}
        execution = payload.get("loam_execution")
        if isinstance(execution, Mapping) and execution.get("mode") == "remote_mcp" and _generated(execution):
            allowed = sorted(set(hpc_tool_names(execution).values()))
            if tool_name not in allowed:
                return {"action": "block", "message": f"code-generator may use only the HPC MCP tools {allowed}"}
            return None
        if isinstance(execution, Mapping) and execution.get("mode") == "remote_mcp":
            required_tool = _text(execution.get("tool_name"))
            if tool_name != required_tool:
                return {"action": "block", "message": f"code-generator must use remote MCP tool {required_tool}"}
        return None
    if role != "hypothesis-validator":
        return None
    if tool_name != "delegate_task":
        return {"action": "block", "message": "hypothesis-validator must delegate executable work to code-generator"}
    arguments = payload.get("arguments")
    goals = []
    if isinstance(arguments, Mapping):
        tasks = arguments.get("tasks")
        if isinstance(tasks, list):
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
        requested = len(raw_tasks) if isinstance(raw_tasks, list) else 1
        if used + requested > cap:
            return {
                "action": "block",
                "message": f"code-generator budget for this hypothesis is {cap} and {used} are used; return your report now.",
            }
    return None


def _session_start(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The calling session's latest agent_started row, as tool_policy finds it."""

    session_id = _text(payload.get("session_id")) or _text(payload.get("parent_session_id"))
    if not session_id:
        return None
    return next(
        (
            row
            for row in reversed(_read_rows(_state_path(payload)))
            if row.get("event") == "agent_started" and _text(row.get("child_session_id")) == session_id
        ),
        None,
    )


def workspace_policy(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """Pando-style workspace boundary for a campaign child (hypothesis_workspace.fs_policy)."""

    if hypothesis_workspace.settings(payload) is None:
        return None
    return hypothesis_workspace.fs_policy(payload, _session_start(payload))


def workspace_capture(payload: Mapping[str, Any]) -> None:
    """Record the code an allowed campaign child runs (hypothesis_workspace.capture_code)."""

    if hypothesis_workspace.settings(payload) is not None:
        hypothesis_workspace.capture_code(payload, _session_start(payload))


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
    """Feed the latest reasoner brief into the next Jnana generation goal."""

    rows = _read_rows(_state_path(payload))
    brief = _latest_brief(rows, _text(payload.get("workflow_id")))
    if not brief:
        return
    payload["loam_reasoner_brief"] = brief
    user_message = _text(payload.get("user_message"))
    if user_message:
        payload["user_message"] = (
            f"{user_message}\n\n"
            "Prior reasoner brief (advisory and unvalidated unless independently reconciled):\n"
            f"{brief}"
        )


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


def _validator_directive(plan_id: str, selected: list[dict[str, Any]], registry: Mapping[str, AgentSpec], payload: Mapping[str, Any]) -> str:
    validator = registry["hypothesis-validator"]
    execution = payload.get("loam_execution")
    tasks = []
    for item in selected:
        hypothesis = item.get("hypothesis")
        hypothesis_id = _text(hypothesis.get("hypothesis_id")) if isinstance(hypothesis, Mapping) else ""
        if _generated(execution if isinstance(execution, Mapping) else None):
            tasks.append(_generated_validator_task(plan_id, hypothesis_id, item, execution, validator.prompt_body))  # type: ignore[arg-type]
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
    lines = [
        "Loam orchestration directive: launch every selected hypothesis validator now, in one delegate_task batch.",
        f"Each child's role definition ({validator.prompt_path.name}) is already inlined in its task context as \"agent_definition\" -- the validator cannot read that file, since it may call no tool but delegate_task, so do not replace the definition with a path. Call delegate_task exactly once using the ready-to-run task array below; do not rewrite or replace its goals, contexts, tags, or roles. Every validator task object has its required \"role\": \"orchestrator\" inside the task object; a top-level role is insufficient for batch dispatch. Do not perform experimental code or tool calls yourself.",
        (
            f"Every executable action MUST be delegated by that validator to at most {_max_children(execution)} leaf code-generator children; no other child may execute code."  # type: ignore[arg-type]
            if _generated(execution if isinstance(execution, Mapping) else None)
            else "Every executable action MUST be delegated by that validator to one leaf code-generator child; no other child may execute code."
        ),
        "Required delegate_task tasks payload:",
        json.dumps(tasks, ensure_ascii=False, sort_keys=True),
    ]
    return "\n".join(lines)


def _generated_validator_task(plan_id: str, hypothesis_id: str, item: Mapping[str, Any], execution: Mapping[str, Any], agent_definition: str = "") -> dict[str, Any]:
    """Validator task for generated mode: it designs the experiment itself."""

    cap = _max_children(execution)
    return {
        "goal": (
            f"Validate the supplied hypothesis by designing the smallest decisive experiment and delegating "
            f"it to at most {cap} code-generator leaf children, one at a time, iterating on their actual "
            "results. Do not calculate directly or use any tool other than delegate_task. Then return the "
            "required report headings, including a bare decimal Evidence Score. "
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
                "execution_contract": execution,
                "child_template": {
                    "role": "leaf",
                    "goal_suffix": _child_tags(plan_id, hypothesis_id),
                    "max_children": cap,
                    "workdir": _workdir(execution, plan_id, hypothesis_id),
                },
                "validator_instructions": (
                    "Follow agent_definition above; it is your role specification and is provided in full "
                    "because you may call no tool other than delegate_task. "
                    "Every code-generator child starts with an empty context and cannot see this payload, "
                    "the tournament state or the campaign files, so write selected_hypothesis into each "
                    "child's goal/context in full -- statement, candidate sequence and the falsifying "
                    "prediction. A hypothesis id is a provenance tag, not a reference a child can resolve. "
                    "Restate the hypothesis as a falsifiable claim and design the smallest decisive experiment. "
                    "Delegate each executable step to a code-generator child: role=leaf, goal = what to compute "
                    "and the acceptance condition, ending with child_template.goal_suffix; context = exact "
                    "inputs, parameter ranges and expected outputs. Children write their own code, can reuse "
                    "registered HPC tools, and run large sweeps as batches. At most max_children children in "
                    "total. Never compute or run code yourself. Then return exactly these headings: "
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


def _reasoner_directive(plan_id: str, completions: list[dict[str, Any]], registry: Mapping[str, AgentSpec]) -> str:
    reasoner = registry["reasoner"]
    threads = [
        {
            "hypothesis_id": row.get("hypothesis_id", ""),
            "validator_status": row.get("status"),
            "evidence_score": row.get("evidence_score"),
            "evidence_score_status": row.get("evidence_score_status"),
            "validator_report": _text(row.get("summary"))[:MAX_REASONER_THREAD_CHARS],
        }
        for row in completions
    ]
    return "\n".join(
        (
            "Loam orchestration directive: all selected validator threads are terminal. Delegate exactly one reasoner before responding.",
            f"Pass the reasoner its role definition verbatim in the task context as \"agent_definition\"; it must not execute experimental code. The definition is inlined below rather than referenced as {reasoner.prompt_path.name}, because the reasoner cannot read that file.",
            json.dumps({"agent_definition": reasoner.prompt_body}, ensure_ascii=False),
            f"Tag its goal with [loam-agent-role=reasoner] [loam-plan-id={plan_id}] so its brief is persisted for the next tournament turn.",
            "Completed validator threads are unvalidated orchestration input; preserve their reported uncertainty and do not turn summaries into accepted evidence.",
            json.dumps(threads, ensure_ascii=False, sort_keys=True),
        )
    )


def canonical_delegation_args(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reconstruct pending campaign delegation arguments from persisted state."""

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
                "goal": f"Synthesize validator results [loam-agent-role=reasoner] [loam-plan-id={plan_id}]",
                "context": _reasoner_directive(plan_id, completions, agent_registry(payload)),
                "role": "leaf",
            }
        return None

    directive = context_for(dict(payload))
    marker = "Required delegate_task tasks payload:\n"
    if marker not in directive:
        return None
    try:
        tasks = json.loads(directive.split(marker, 1)[1])
    except json.JSONDecodeError:
        return None
    return {"tasks": tasks} if isinstance(tasks, list) else None


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
            return _reasoner_directive(plan_id, completions, registry)
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
    result = report_value.get("result")
    rankings = result.get("top_hypotheses") if isinstance(result, Mapping) else None
    if not isinstance(rankings, list) or not rankings:
        return ""
    full_hypotheses = _full_hypotheses(payload)
    selected: list[dict[str, Any]] = []
    for rank, ranking in enumerate(rankings, start=1):
        if not isinstance(ranking, Mapping):
            continue
        hypothesis_id = _text(ranking.get("hypothesis_id"))
        hypothesis = full_hypotheses.get(hypothesis_id)
        if not hypothesis:
            return ""
        selected.append({"rank": rank, "elo_rating": ranking.get("elo_rating"), "hypothesis": hypothesis})
    return _validator_directive(plan_id, selected, registry, payload)
