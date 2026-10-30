"""Seed and rank a bounded Jnana hypothesis tournament before an LLM call.

This is advisory only. It uses Jnana's internal generation agents to seed an
empty workflow-scoped state, then runs pairwise rankings under a non-blocking
state lock. It returns compact rankings rather than full hypothesis bodies.
Any unavailable configuration, lock contention, or Jnana failure simply
produces no injected context.

Both halves aim at the campaign's goal, not at the turn's message: generation
and ranking read the same goal file the validators and the reasoner are given
(`_research_goal`), and the stored copy is re-set when that file changes
(`_sync_research_goal`).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

# Jnana is intentionally imported only after the configured checkout has been
# added to sys.path. This keeps Loam inert for profiles that do not opt in.
CoScientist: Any = None


@contextmanager
def _try_lock(path: Path) -> Iterator[bool]:
    """Acquire a non-blocking advisory lock, or yield False when busy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            yield False
            return
        yield True
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def _config(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = payload.get("loam_hypothesis_tournament")
    if not isinstance(value, Mapping) or value.get("enabled") is not True:
        return None
    required = ("jnana_root", "state_file", "reports_dir", "provider", "inference_endpoint", "model")
    if any(not isinstance(value.get(key), str) or not value[key].strip() for key in required):
        return None
    return value


def _positive(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _count(value: Any) -> int:
    """A non-negative count, 0 for absent or unreadable: the feature stays off."""

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _base_url(config: Mapping[str, Any]) -> str:
    """Convert an OpenAI-compatible inference URL to the client's base URL."""

    endpoint = str(config["inference_endpoint"]).strip().rstrip("/")
    if config.get("endpoint_is_chat_completions") is True:
        suffix = "/chat/completions"
        if not endpoint.endswith(suffix):
            raise ValueError("endpoint_is_chat_completions requires a URL ending in /chat/completions")
        return endpoint[: -len(suffix)]
    return endpoint


# Keys a per-role block may override. Anything else (jnana_root, state_file,
# match counts, the knowledge pre-pass) stays shared across roles.
_ROLE_OVERRIDE_KEYS = (
    "provider",
    "model",
    "inference_endpoint",
    "endpoint_is_chat_completions",
    "api_key_env",
    "omit_temperature",
    "json_max_tokens",
)


def _role_config(config: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    """Overlay an optional per-role model block on the profile's defaults.

    Jnana uses the tournament's model for two different jobs: generating
    hypotheses and judging pairwise matches. A profile can point each at its own
    provider/model/endpoint by adding a block named after the role, e.g.

        hypothesis_tournament:
          provider: openai            # defaults for every role
          inference_endpoint: ...
          model: ...
          generation:
            model: argo:gpt-5.6-luna
            inference_endpoint: http://127.0.0.1:51900/v1

    Any key left out of the block falls back to the top-level value, so profiles
    that declare no role blocks behave exactly as before.
    """

    override = config.get(role)
    if not isinstance(override, Mapping):
        return config
    merged = dict(config)
    for key in _ROLE_OVERRIDE_KEYS:
        if key not in override:
            continue
        value = override[key]
        # An empty/blank string is a no-op rather than an invalid override: it
        # would otherwise produce an LLMConfig with no model or no endpoint.
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged


def _llm_config(config: Mapping[str, Any], role: str, min_json_tokens: int) -> Any:
    """Build the LLMConfig for one tournament role, honouring its override block."""

    from jnana.protognosis.core.multi_llm_config import LLMConfig

    resolved = _role_config(config, role)
    api_key_env = str(resolved.get("api_key_env") or "").strip()
    api_key = os.environ.get(api_key_env, "").strip() if api_key_env else "local-no-auth"
    return LLMConfig(
        provider=str(resolved["provider"]),
        model=str(resolved["model"]),
        api_key=api_key,
        base_url=_base_url(resolved),
        model_adapter={
            "omit_temperature": resolved.get("omit_temperature") is True,
            "json_max_tokens": max(min_json_tokens, _positive(resolved.get("json_max_tokens"), 1024)),
        },
    )


def _write_report(reports_dir: Path, payload: Mapping[str, Any], result: Mapping[str, Any]) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "loam_jnana_tournament_v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "workflow_id": str(payload.get("workflow_id") or ""),
        "loam_profile": str(payload.get("loam_profile") or ""),
        "session_id": str(payload.get("session_id") or ""),
        "loam_turn": payload.get("loam_turn"),
        "result": dict(result),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = reports_dir / f"tournament-{stamp}-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def _campaign_goal(payload: Mapping[str, Any]) -> tuple[str, str]:
    """The campaign's objective, read from the file every other tier reads.

    Loaded through `campaign_context` rather than re-parsed here, so the
    tournament aims at the same text the validators and the reasoner are given,
    from the same file, with the same validation. A named-but-missing or
    malformed goal file raises out of here exactly as it does for the
    orchestration hook: a tournament quietly ranking against something else is
    the failure this avoids.
    """

    hooks_dir = str(Path(__file__).resolve().parent)
    if hooks_dir not in sys.path:
        sys.path.insert(0, hooks_dir)
    from hypothesis_orchestration import campaign_context

    campaign = campaign_context(payload)
    if campaign is None or not campaign.goal:
        return "", ""
    return campaign.goal, f"campaign:{campaign.goal_name}"


def _research_goal(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Jnana's research goal and where it came from: the campaign, or this turn.

    The campaign's goal wins outright when there is one, and the turn's message
    is left out of it rather than appended. Two reasons. The message is a round
    nudge ("begin the campaign", "dispatch the next round") and carries no
    objective, so mixing it in buys nothing; and it differs every turn, which
    would churn the stored goal and re-parse it -- an LLM call -- on each run.

    Without a campaign goal the behaviour is unchanged: the turn's own request
    is the goal, which is what a bare research profile with no campaign has.
    """

    goal, source = _campaign_goal(payload)
    if goal:
        return goal, source
    return " ".join(str(payload.get("user_message") or "").split()), "user_message"


def _reasoner_brief(payload: Mapping[str, Any]) -> str:
    """The previous round's synthesis, as `prepare_payload` structurally supplies it.

    Read from `loam_reasoner_brief` rather than from the message it is also
    appended to: the message is not the tournament's input any more, and a
    channel the brief shares with a round nudge is a channel that loses it.
    """

    return str(payload.get("loam_reasoner_brief") or "").strip()


def _brief_block(brief: str, goal: str) -> str:
    """The reasoner brief under a heading that says what it is. Empty for none.

    The knowledge pre-pass heads its own brief (`knowledge_brief.HEADER`); this
    one arrives as bare prose, and unheaded it would read as more objective.

    Dropped when the goal already carries it: `prepare_payload` appends the
    brief to the turn's message as well, and that message is the goal for a
    profile that names no campaign. Twice is padding, not emphasis.
    """

    if not brief or brief in goal:
        return ""
    return (
        "Latest round synthesis from the campaign's reasoner (advisory, and "
        "unvalidated unless independently reconciled):\n" + brief
    )


def _goal_with(goal: str, *blocks: str) -> str:
    """The goal plus whichever already-headed blocks have text, in order."""

    return "\n\n".join([goal, *(block for block in blocks if block)])


def _sync_research_goal(coscientist: Any, goal: str) -> str:
    """Store `goal` as the research goal, re-parsing only when it has moved.

    Generation is one-shot -- it is skipped once the state holds two hypotheses
    -- but the stored goal governs every later pairwise match: the judge reads
    `metadata["research_goal"]` and the evaluation criteria parsed from it. So
    the goal has to be re-set when it changes, not only when the state is empty,
    or an edited campaign file (or a fresh reasoner brief) leaves the ranking
    judging against the old one.

    `set_research_goal` costs an LLM call to parse the plan, so the digest of
    the text as stored is recorded beside it and the call is made only when that
    digest moves. "Moved" is also what licenses a pool refresh, so the two stay
    in step: one new round of knowledge, one re-goal, one bounded generation.
    """

    digest = hashlib.sha256(goal.encode("utf-8")).hexdigest()
    metadata = getattr(coscientist.memory, "metadata", None)
    if not isinstance(metadata, dict):
        # A Jnana memory always has one; a stub might not, and an advisory hook
        # should not fail over introspection.
        coscientist.set_research_goal(goal)
        return "set"
    if metadata.get("loam_goal_sha256") == digest and metadata.get("research_goal"):
        return "unchanged"
    coscientist.set_research_goal(goal)
    metadata["loam_goal_sha256"] = digest
    coscientist.memory.save()
    return "set"


def _ensure_initial_hypotheses(
    config: Mapping[str, Any], payload: Mapping[str, Any], llm_config: Any
) -> dict[str, int | str]:
    """Seed an empty Jnana state, or refresh a seeded one against new knowledge.

    Round one generates the pool. Later rounds generate `refresh_hypotheses`
    more, but only when the goal has actually moved -- a fresh reasoner brief or
    an edited campaign file -- so a quiet round costs nothing and a round of
    synthesis enters the pool as candidates that compete rather than as prose
    nobody reads.
    """

    global CoScientist
    if CoScientist is None:
        from jnana.protognosis.core.coscientist import CoScientist as JnanaCoScientist

        CoScientist = JnanaCoScientist

    initial_hypotheses = _positive(config.get("initial_hypotheses"), 3)
    coscientist = CoScientist(
        llm_config=llm_config,
        storage_path=str(Path(str(config["state_file"])).expanduser()),
        max_workers=initial_hypotheses,
    )
    coscientist.start()
    try:
        existing = len(coscientist.get_all_hypotheses())
        goal, goal_source = _research_goal(payload)
        brief = _reasoner_brief(payload)
        if existing >= 2:
            outcome: dict[str, Any] = {"status": "not_needed", "generated_hypotheses": 0}
            if not goal:
                return outcome
            outcome["goal_source"] = goal_source
            # The pool is seeded, but the stored goal still steers every pairwise
            # match, so it is kept in step with the campaign's and the round's.
            outcome["goal_sync"] = _sync_research_goal(coscientist, _goal_with(goal, _brief_block(brief, goal)))
            refresh = _count(config.get("refresh_hypotheses"))
            if refresh and outcome["goal_sync"] == "set":
                coscientist.generate_hypotheses(count=refresh)
                coscientist.wait_for_completion()
                coscientist.memory.save()
                outcome["status"] = "refreshed"
                outcome["refresh_requested"] = refresh
                outcome["generated_hypotheses"] = max(0, len(coscientist.get_all_hypotheses()) - existing)
            return outcome
        if not goal:
            return {"status": "skipped", "reason": "missing_research_goal", "generated_hypotheses": 0}
        requested = max(initial_hypotheses, 2 - existing)
        knowledge = _knowledge_brief(config, goal, llm_config)
        _sync_research_goal(
            coscientist, _goal_with(goal, knowledge.get("brief") or "", _brief_block(brief, goal))
        )
        coscientist.generate_hypotheses(count=requested)
        coscientist.wait_for_completion()
        coscientist.memory.save()
        result: dict[str, Any] = {
            "status": "completed",
            "goal_source": goal_source,
            "generated_hypotheses": max(0, len(coscientist.get_all_hypotheses()) - existing),
        }
        if knowledge:
            result["knowledge"] = {k: knowledge.get(k) for k in
                                   ("status", "tool_calls", "tools_used", "cached", "seconds")}
            result["knowledge"]["brief_chars"] = len(knowledge.get("brief") or "")
        return result
    finally:
        coscientist.stop()


def _knowledge_brief(config: Mapping[str, Any], goal: str, llm_config: Any) -> dict[str, Any]:
    """Run the optional knowledge-graph pre-pass; {} when it is switched off."""

    knowledge = config.get("knowledge")
    if not isinstance(knowledge, Mapping) or knowledge.get("enabled") is not True:
        return {}
    hooks_dir = str(Path(__file__).resolve().parent)
    if hooks_dir not in sys.path:
        sys.path.insert(0, hooks_dir)
    from knowledge_brief import brief_for

    # The pre-pass is a tool-calling research loop, which is a different job from
    # Jnana's JSON generation: a profile may point it at a stronger model on the
    # same endpoint (`knowledge.model`) and otherwise it reuses the tournament's.
    llm = {"base_url": getattr(llm_config, "base_url", None), "api_key": getattr(llm_config, "api_key", None),
           "model": str(knowledge.get("model") or "") or getattr(llm_config, "model", None)}
    try:
        return brief_for(goal, knowledge, llm, Path(str(config["state_file"])).expanduser().parent)
    except Exception as exc:  # noqa: BLE001 - advisory: never block generation
        return {"status": f"failed: {type(exc).__name__}: {exc}", "brief": ""}


def _render(payload: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    if result.get("status") != "completed":
        return ""
    rankings = result.get("top_hypotheses")
    if not isinstance(rankings, list) or not rankings:
        return ""
    lines = [
        "Jnana hypothesis tournament update (advisory; model-mediated ranking, not empirical validation):",
        "- Trigger: Loam turn {turn}; ran {completed}/{requested} pairwise matches.".format(
            turn=payload.get("loam_turn", "?"),
            completed=result.get("matches_completed", 0),
            requested=result.get("matches_requested", 0),
        ),
    ]
    generation = result.get("generation")
    entered = generation.get("generated_hypotheses") if isinstance(generation, Mapping) else 0
    if isinstance(generation, Mapping) and generation.get("status") == "refreshed" and entered:
        # Said out loud because an entrant's 1200.0 is not a ranking: it is the
        # initial rating, and selecting on it picks by storage order.
        lines.append(
            f"- {entered} new hypotheses entered from the last round's synthesis. They start at "
            "1200.0 Elo and are unranked until they have played matches, so do not read an "
            "unplayed entrant's rating as a verdict."
        )
    for index, hypothesis in enumerate(rankings, start=1):
        if not isinstance(hypothesis, Mapping):
            continue
        summary = " ".join(str(hypothesis.get("summary") or "").split())[:180] or "(no summary)"
        lines.append(
            "- #{rank}: {identifier} — {elo:.1f} Elo; {summary}".format(
                rank=index,
                identifier=str(hypothesis.get("hypothesis_id") or "unknown"),
                elo=float(hypothesis.get("elo_rating") or 0.0),
                summary=summary,
            )
        )
    lines.append("Prefer top-ranked candidates for refinement, while independently checking evidence and safety.")
    return "\n".join(lines)


def context_for(payload: Mapping[str, Any]) -> str:
    """Run one configured tournament iteration and return bounded LLM context."""

    config = _config(payload)
    if config is None:
        return ""

    jnana_root = Path(str(config["jnana_root"])).expanduser()
    if not jnana_root.is_dir():
        return ""
    state_file = Path(str(config["state_file"])).expanduser()
    reports_dir = Path(str(config["reports_dir"])).expanduser()

    with _try_lock(state_file.with_suffix(state_file.suffix + ".lock")) as acquired:
        if not acquired:
            return ""
        if str(jnana_root) not in sys.path:
            sys.path.insert(0, str(jnana_root))
        from jnana.protognosis.periodic_tournament import run_incremental_tournament

        # Two roles, each overridable via a `ranking:` / `generation:` block in
        # the profile. Generation agents emit a full mechanism plus predictions,
        # so they get more room than compact pairwise ranking responses.
        llm_config = _llm_config(config, "ranking", 1)
        generation_llm_config = _llm_config(config, "generation", 4096)
        generation = _ensure_initial_hypotheses(config, payload, generation_llm_config)
        result = dict(
            run_incremental_tournament(
                state_file=state_file,
                llm_config=llm_config,
                match_count=_positive(config.get("matches_per_run"), 1),
                top_k=_positive(config.get("top_k"), 3),
            )
        )
        result["generation"] = generation
        _write_report(reports_dir, payload, result)
        return _render(payload, result)
