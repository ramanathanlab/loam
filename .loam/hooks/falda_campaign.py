"""Campaign policy for FALDA memory: what Loam writes, recalls and permits.

``falda_memory`` talks to the server; this module decides content. Validator
reports are model output, so atoms written from them are marked unreconciled
and never carry high confidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import falda_flush
import falda_memory

OUTCOMES = ("supported", "contradicted", "inconclusive", "blocked")
_OUTCOME_TAGS = {"contradicted": "negative", "blocked": "negative", "inconclusive": "coverage", "supported": ""}
MAX_ATOM_CHARS = 1_200


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _one_line(value: str) -> str:
    return " ".join(value.split())


def report_section(summary: str, heading: str) -> str:
    """Return the body of ``### <heading>`` in a validator report, or ''."""

    parts = summary.split(f"### {heading}", 1)
    if len(parts) != 2:
        return ""
    rest = parts[1]
    end = rest.find("\n### ")
    return (rest if end < 0 else rest[:end]).strip()


def report_outcome(summary: str) -> str:
    section = report_section(summary, "Outcome")
    first = section.split(None, 1)[0] if section else ""
    word = first.strip("`*.,;:—-").lower()
    if word in OUTCOMES:
        return word
    # Validators sometimes lead with prose; accept a keyword later in the
    # section only when exactly one outcome word appears, so "not
    # contradicted ... supported" stays ambiguous and is skipped.
    found = {match.lower() for match in re.findall(r"\b(" + "|".join(OUTCOMES) + r")\b", section, re.I)}
    return found.pop() if len(found) == 1 else ""


_MAX_REASON_CHARS = 300


def outcome_reason(summary: str) -> str:
    """The validator's one-sentence reason: the Outcome section without its leading keyword.

    It usually says what was actually tested, which is narrower than the hypothesis.
    """

    section = _one_line(re.sub(r"[*`]", "", report_section(summary, "Outcome")))
    first, _, rest = section.partition(" ")
    if first.strip(".,;:—–-").lower() in OUTCOMES:
        section = rest.lstrip(" .,;:—–-")
    return section.rstrip(".")[:_MAX_REASON_CHARS]


def ledger_rows(summary: str) -> list[dict[str, str]]:
    """Parse the Evidence Ledger table; header, separator and short rows are skipped."""

    rows = []
    for line in report_section(summary, "Evidence Ledger").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5 or cells[0].lower() == "id" or set("".join(cells)) <= set("-: "):
            continue
        rows.append(
            {
                "id": cells[0],
                "action": cells[1],
                "observation": cells[2],
                "direction": cells[3].lower(),
                "reliability": cells[4],
            }
        )
    return rows


def _decisive(rows: list[dict[str, str]], direction: str) -> str:
    row = next((row for row in rows if row["direction"].startswith(direction)), None)
    return f"{row['action']}: {row['observation']}" if row else "none recorded"


_MAX_UNTABLED_EVIDENCE_CHARS = 400


def untabled_evidence(summary: str) -> str:
    """Evidence from a ledger written as bullets instead of the template's table.

    Validators often report the child's stdout in a fenced block; that block is
    the observation, so it wins. Otherwise the first three bullets stand in.
    """

    section = report_section(summary, "Evidence Ledger")
    fence = re.search(r"```[a-zA-Z]*\n(.*?)```", section, re.S)
    if fence:
        text = fence.group(1)
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False)
        except ValueError:
            text = _one_line(text)
        return text[:_MAX_UNTABLED_EVIDENCE_CHARS]
    bullets = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith(("- ", "* ")):
            continue
        item = _one_line(re.sub(r"[*`]", "", line[2:])).strip()
        label, _, value = item.partition(":")
        if value.strip():
            bullets.append(f"{label.strip()}: {value.strip()}")
        if len(bullets) == 3:
            break
    return "; ".join(bullets)[:_MAX_UNTABLED_EVIDENCE_CHARS]


def _clip(body: str, suffix: str) -> str:
    room = MAX_ATOM_CHARS - len(suffix) - 1
    if len(body) > room:
        body = body[: room - 1].rstrip() + "…"
    return f"{body} {suffix}"


def outcome_atom(payload: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Map one validator_completed record to an atom, or to the reason it is skipped."""

    summary = _text(record.get("summary"))
    outcome = report_outcome(summary)
    if not outcome:
        return None, "missing_outcome"
    score = record.get("evidence_score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        # Blocked results are the main negative signal and do not use the score.
        if outcome != "blocked":
            return None, "missing_score"
        score = None
    hypothesis_id = _text(record.get("hypothesis_id")) or "unknown"
    hypothesis = _one_line(re.sub(r"[*`]", "", report_section(summary, "Hypothesis"))).rstrip(".") or hypothesis_id
    rows = ledger_rows(summary)
    loose = "" if rows else untabled_evidence(summary)
    actions = "; ".join(row["action"] for row in rows if row["action"]) or "none recorded"
    shown = f"{score:+.2f}" if score is not None else ""
    if outcome == "contradicted":
        execution = payload.get("loam_execution")
        mode = _text(execution.get("mode")) if isinstance(execution, Mapping) else ""
        body = (
            f"Contradicted (evidence score {shown}): {hypothesis}. "
            f"Decisive evidence: {(loose or _decisive(rows, 'contradict'))}. "
            f"Conditions: execution mode {mode or 'unknown'}; actions {actions}."
        )
    elif outcome == "blocked":
        prerequisite = _one_line(report_section(summary, "Next Step")) or "not stated"
        body = f"Blocked: {hypothesis} could not be validated. Missing prerequisite: {prerequisite.rstrip('.')}."
    elif outcome == "inconclusive":
        body = f"Tested, inconclusive ({shown}): {hypothesis}. What was tried: {actions}."
    else:
        evidence = loose or _decisive(rows, "support")
        reason = outcome_reason(summary)
        if reason:
            # A validator tests one case; storing the general claim as "Supported" would overstate it.
            body = (
                f"Supported ({shown}) for what was tested: {reason}. Proposed claim, supported only as far as "
                f"that test reaches: {hypothesis}. Decisive evidence: {evidence}."
            )
        else:
            body = f"Supported ({shown}): {hypothesis}. Decisive evidence: {evidence}."
    plan_id = _text(record.get("plan_id")) or "unknown"
    run_id = _text(record.get("run_id")) or "unknown"
    tags = [
        falda_memory.workflow_tag(payload),
        f"run:{run_id}",
        f"plan:{plan_id}",
        f"hypothesis:{hypothesis_id}",
        "source:validator-report",
    ]
    if _OUTCOME_TAGS[outcome]:
        tags.insert(0, _OUTCOME_TAGS[outcome])
    confidence = "medium" if score is not None and abs(score) >= 0.5 else "low"
    suffix = f"(validator report, unreconciled; plan {plan_id}, run {run_id})"
    return {"type": "fact", "content": _clip(body, suffix), "confidence": confidence, "tags": tags}, ""


def write_validator_outcome(payload: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """Queue one validator outcome for the campaign pool; never raises, never waits on the network."""

    cfg = falda_memory.config(payload)
    if cfg is None or cfg.get("write_validator_outcomes") is not True:
        return
    plan_id = _text(record.get("plan_id"))
    hypothesis_id = _text(record.get("hypothesis_id"))
    try:
        atom, reason = outcome_atom(payload, record)
    except Exception as exc:  # a malformed report must not break record_stop
        atom, reason = None, f"client:{type(exc).__name__}"
    if atom is None:
        falda_memory.log_event(
            payload,
            {"event": "falda_write_skipped", "op": "upsert", "reason": reason, "plan_id": plan_id, "hypothesis_id": hypothesis_id},
        )
        return
    # Spooled, not sent: the detached flusher does the network I/O, so a validator stop never waits on FALDA.
    falda_flush.enqueue(payload, atom, plan_id=plan_id, hypothesis_id=hypothesis_id)
    falda_flush.spawn_flush(payload)


GENERATION_HEADING = "Prior campaign memory (advisory; verify before relying on it):"
GENERATION_SECTIONS = ("What did not work", "Best known results", "Next steps and open questions", "Findings")
# hypothesis_tournament._ensure_initial_hypotheses generates only below this count.
_JNANA_GENERATES_BELOW = 2


def generation_memory(payload: Mapping[str, Any], query: str, hypothesis_count: int | Callable[[], int]) -> str:
    """Recalled memory for Jnana's generation goal; '' when off, when Jnana will not generate, or on failure.

    ``hypothesis_count`` may be a callable so the Jnana state is read only when recall could happen.
    """

    cfg = falda_memory.config(payload)
    if cfg is None or cfg.get("recall_generation") is not True:
        return ""
    if not isinstance(payload.get("loam_hypothesis_tournament"), Mapping):
        return ""
    count = hypothesis_count() if callable(hypothesis_count) else hypothesis_count
    if count >= _JNANA_GENERATES_BELOW:
        return ""
    budget = int(cfg.get("recall_budget_chars") or 1500)
    result = falda_memory.recall(payload, query, consumer="jnana_generation", budget=budget)
    if not result.ok:
        return ""
    text = falda_memory.select_sections(result.context, GENERATION_SECTIONS, budget)
    return f"{GENERATION_HEADING}\n{text}" if text else ""


FALDA_TOOL_PREFIX = "mcp__falda__"
REASONER_CACHE_FILENAME = "falda-reasoner-memory.jsonl"
_MAX_QUERY_CHARS = 2_000


def _reasoner_cache(payload: Mapping[str, Any]) -> Path:
    return falda_memory.calls_path(payload).with_name(REASONER_CACHE_FILENAME)


def _cached_block(payload: Mapping[str, Any], plan_id: str) -> str | None:
    path = _reasoner_cache(payload)
    if not path.is_file():
        return None
    workflow_id = _text(payload.get("workflow_id"))
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("plan_id") == plan_id and row.get("workflow_id") == workflow_id:
            return _text(row.get("block"))
    return None


def _sections_only(context: str) -> str:
    """Drop FALDA's preamble lines; one tells the reader to verify with tools the reasoner cannot use."""

    start = context.find("## ")
    return context[start:] if start >= 0 else ""


def _reasoner_query(payload: Mapping[str, Any], completions: list[Mapping[str, Any]]) -> str:
    lines = [_text(payload.get("user_message"))] if _text(payload.get("user_message")) else []
    for row in completions:
        summary = _text(row.get("summary"))
        claim = _one_line(report_section(summary, "Hypothesis").strip("`")) or _text(row.get("hypothesis_id"))
        lines.append(f"{claim} — {report_outcome(summary) or 'unreported'}")
    return "\n".join(lines)[:_MAX_QUERY_CHARS]


def reasoner_memory_block(payload: Mapping[str, Any], plan_id: str, completions: list[Mapping[str, Any]]) -> str:
    """The reasoner's `Memory interface:` block, recalled once per plan and cached (spec §5.4, §10.3)."""

    cfg = falda_memory.config(payload)
    recall_on = cfg is not None and cfg.get("recall_reasoner") is True
    tools_on = cfg is not None and cfg.get("reasoner_tools") is True
    if not recall_on and not tools_on:
        return ""
    try:
        cached = _cached_block(payload, plan_id)
    except (OSError, ValueError):  # unreadable or non-UTF-8 cache: recall again
        cached = None
    if cached is not None:
        return cached
    tag = falda_memory.workflow_tag(payload)
    recalled = ""
    if recall_on:
        result = falda_memory.recall(
            payload,
            _reasoner_query(payload, completions),
            consumer="reasoner",
            budget=int(cfg.get("recall_budget_chars") or 1500),
            plan_id=plan_id,
        )
        if result.ok:
            recalled = falda_memory.cut_to_lines(
                _sections_only(result.context), int(cfg.get("recall_budget_chars") or 1500)
            )
    # Best effort: a failed, slow or empty recall adds nothing to the reasoner's prompt.
    if not tools_on and not recalled:
        return _cache_block(payload, plan_id, "")
    lines = ["Memory interface:", f"- Shared FALDA pool {cfg['pool']}, scoped to this campaign by the tag {tag}."]
    if tools_on:
        lines.append(
            f"- You may write: call {FALDA_TOOL_PREFIX}falda_remember as described under \"Shared campaign memory\" "
            f"in your instructions, with tags that include {tag}. To read more, call "
            f"{FALDA_TOOL_PREFIX}falda_recall with require_tags [\"{tag}\"]."
        )
    else:
        lines.append("- Writing is disabled for this run: report durable conclusions in your Memory Update section only.")
    if recalled:
        lines.append("- Recalled campaign memory (advisory; verify before relying on it):")
        lines.append(recalled)
        lines.append("- Under Memory Update, cite the recalled items you used by their [mN] tags.")
    return _cache_block(payload, plan_id, "\n".join(lines))


def _cache_block(payload: Mapping[str, Any], plan_id: str, block: str) -> str:
    """Remember this plan's block (even an empty one) so the plan recalls at most once."""

    try:
        path = _reasoner_cache(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            row = {"plan_id": plan_id, "workflow_id": _text(payload.get("workflow_id")), "block": block}
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass
    return block


_RESERVED_TAG = "source:validator-report"
_RESERVED_PREFIXES = ("run:", "plan:", "hypothesis:")  # written by Loam's hooks, never by the reasoner
_ALLOWED_FALDA_TOOLS = frozenset({"falda_remember", "falda_recall", "falda_mark_used"})


def _falda_block_message(payload: Mapping[str, Any], tool_name: str, arguments: Any, role: str) -> str | None:
    cfg = falda_memory.config(payload)
    if cfg is None or cfg.get("reasoner_tools") is not True:
        return "FALDA tools are not enabled for this Loam campaign; continue without them."
    if role != "reasoner":
        return (
            "Only the campaign's tagged reasoner may use FALDA memory; validator outcomes are recorded "
            "automatically. Continue without FALDA tools."
        )
    tool = tool_name[len(FALDA_TOOL_PREFIX):]
    if tool not in _ALLOWED_FALDA_TOOLS:
        return (
            f"{tool_name} is not permitted in a Loam campaign; use only falda_remember, "
            "falda_recall or falda_mark_used."
        )
    args = arguments if isinstance(arguments, Mapping) else {}
    pool = args.get("pool")
    if pool not in (None, "") and _text(pool).lower() != _text(cfg.get("pool")).lower():
        return f"Use only this campaign's pool {cfg.get('pool')}: remove the pool argument or set it to that UUID."
    tag = falda_memory.workflow_tag(payload)
    if tool == "falda_remember":
        tags = args.get("tags")
        if not isinstance(tags, list) or tag not in tags:
            return f'Add the tag "{tag}" to tags and call falda_remember again.'
        if args.get("pinned") is True:
            return "Do not pin campaign memories: call falda_remember again without pinned."
        for other in tags:
            if not isinstance(other, str):
                continue
            if (other.startswith("workflow:") and other != tag) or other == _RESERVED_TAG or other.startswith(_RESERVED_PREFIXES):
                return f'Remove the tag "{other}" from tags (campaign memory takes only "{tag}" and your own topic tags) and call falda_remember again.'
    if tool == "falda_recall":
        required = args.get("require_tags")
        if not isinstance(required, list) or tag not in required:
            return f'Call falda_recall again with require_tags ["{tag}"] so it reads only this campaign.'
    return None


def falda_tool_policy(
    payload: Mapping[str, Any], tool_name: str, arguments: Any, start: Mapping[str, Any] | None
) -> dict[str, str] | None:
    """Allow a FALDA tool call only from the tagged reasoner, in its pool and workflow; log every decision."""

    role = _text(start.get("role")) if isinstance(start, Mapping) else ""
    message = _falda_block_message(payload, tool_name, arguments, role)
    falda_memory.log_event(
        payload,
        {
            "op": "mcp_tool",
            "tool": tool_name,
            "role": role or "untagged",
            "plan_id": _text(start.get("plan_id")) if isinstance(start, Mapping) else "",
            "allowed": message is None,
            "message": message or "",
        },
    )
    return None if message is None else {"action": "block", "message": message}
