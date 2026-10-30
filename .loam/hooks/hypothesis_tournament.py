"""Seed and rank a bounded Jnana hypothesis tournament before an LLM call.

This is advisory only. It uses Jnana's internal generation agents to seed an
empty workflow-scoped state, then runs pairwise rankings under a non-blocking
state lock. It returns compact rankings rather than full hypothesis bodies.
Any unavailable configuration, lock contention, or Jnana failure simply
produces no injected context.

Both halves aim at the campaign's goal, not at the turn's message: generation
and ranking read the same goal file the validators and the reasoner are given
(`_research_goal`), and the stored copy is re-set when that file changes
(`_refresh_pool`).
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


def _hooks_on_path() -> None:
    """Make the sibling hook modules importable."""

    hooks_dir = str(Path(__file__).resolve().parent)
    if hooks_dir not in sys.path:
        sys.path.insert(0, hooks_dir)


def _campaign_goal(payload: Mapping[str, Any]) -> tuple[str, str]:
    """The campaign's objective, read from the file every other tier reads.

    Loaded through `campaign_context` rather than re-parsed here, so the
    tournament aims at the same text the validators and the reasoner are given,
    from the same file, with the same validation. A named-but-missing or
    malformed goal file raises out of here exactly as it does for the
    orchestration hook: a tournament quietly ranking against something else is
    the failure this avoids.
    """

    _hooks_on_path()
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


_BRIEF_HEADING = (
    "Latest round synthesis from the campaign's reasoner (advisory, and "
    "unvalidated unless independently reconciled):"
)


def _brief_block(brief: str, goal: str) -> str:
    """The reasoner brief under a heading that says what it is. Empty for none.

    The knowledge pre-pass heads its own brief (`knowledge_brief.HEADER`); this
    one arrives as bare prose, and unheaded it would read as more objective.

    Dropped when the goal already carries it: `prepare_payload` appends the
    brief to the turn's message as well, and that message is the goal for a
    profile that names no campaign. Twice is padding, not emphasis.

    The membership test is made on both texts collapsed, because the goal that
    may already carry the brief is the turn's message, which `_research_goal`
    collapses. A real brief is a multi-line synthesis, so comparing it raw would
    never match it there and would append it a second time.
    """

    if not brief or " ".join(brief.split()) in " ".join(goal.split()):
        return ""
    return f"{_BRIEF_HEADING}\n{brief}"


def _goal_with(goal: str, *blocks: str) -> str:
    """The goal plus whichever already-headed blocks have text, in order."""

    return "\n\n".join([goal, *(block for block in blocks if block)])


def _metadata(coscientist: Any) -> dict[str, Any] | None:
    """Jnana's memory metadata, or None for a stub that has none.

    A Jnana memory always has one and persists it inside the state file; an
    advisory hook should not fail over introspecting a double that does not.
    """

    metadata = getattr(getattr(coscientist, "memory", None), "metadata", None)
    return metadata if isinstance(metadata, dict) else None


def _knowledge_prefix() -> str:
    """The stable leader of the pre-pass's own heading, up to its graph name."""

    _hooks_on_path()
    from knowledge_brief import HEADER

    return HEADER.split("{name}")[0]


def _recover_knowledge(stored_goal: str) -> str:
    """The pre-pass's brief carved back out of a goal stored without the marker.

    States seeded before `loam_knowledge_brief` existed carry the brief only
    inside `metadata["research_goal"]`, where `_goal_with` placed it between the
    objective and the reasoner block. The pre-pass heads it with
    `knowledge_brief.HEADER`, so it is recoverable rather than lost: without
    this, the first re-goal of an existing campaign would drop the brief its
    pool was generated against and spend a refresh on the drop.

    Sliced without stripping, so the result recomposes byte-for-byte and
    `_goal_moved` can see that such a goal has not actually moved.
    """

    start = stored_goal.find(_knowledge_prefix())
    if start < 0:
        return ""
    end = stored_goal.find(f"\n\n{_BRIEF_HEADING}", start)
    return stored_goal[start:] if end < 0 else stored_goal[start:end]


def _stored_knowledge(coscientist: Any) -> str:
    """The knowledge pre-pass's brief as the seeding turn recorded it.

    The pre-pass is a tool-calling research loop, run once when the state is
    seeded, and its brief is part of the goal the pool was generated against. A
    later turn that re-sets the goal has to carry it forward: re-running the
    loop would cost minutes, while leaving it out would both narrow the text the
    pairwise judge reads and register as a goal that moved -- buying a pool
    refresh that no new knowledge asked for.

    An absent marker means a state seeded before it existed, not a state with no
    knowledge, so the brief is recovered from the stored goal rather than
    assumed empty.
    """

    metadata = _metadata(coscientist) or {}
    if "loam_knowledge_brief" in metadata:
        return str(metadata["loam_knowledge_brief"] or "")
    return _recover_knowledge(str(metadata.get("research_goal") or ""))


def _remember_knowledge(coscientist: Any, brief: str) -> None:
    """Record the pre-pass's brief, or its absence, beside the stored goal."""

    metadata = _metadata(coscientist)
    if metadata is not None:
        metadata["loam_knowledge_brief"] = brief


def _digest(goal: str) -> str:
    return hashlib.sha256(goal.encode("utf-8")).hexdigest()


def _goal_moved(coscientist: Any, goal: str) -> str:
    """Whether `goal` differs from the one the state was last set to.

    Generation is one-shot -- it is skipped once the state holds two hypotheses
    -- but the stored goal governs every later pairwise match: the judge reads
    `metadata["research_goal"]` and the evaluation criteria parsed from it. So
    the goal has to be re-set when it changes, not only when the state is empty,
    or an edited campaign file (or a fresh reasoner brief) leaves the ranking
    judging against the old one.

    `set_research_goal` costs an LLM call to parse the plan, so movement is
    measured against a digest recorded beside the stored text. "Moved" is also
    what licenses a pool refresh, so the two stay in step: one new round of
    knowledge, one re-goal, one bounded generation.

    Four readings, because each buys different handling:

    `unverifiable` -- no metadata to record a digest in, so nothing can be
    compared next turn either. Distinct from `moved` because without a digest
    every turn would look like movement, and a refresh must not run on that.

    `unstamped` -- a stored goal whose text matches but which carries no digest
    at all: a state seeded before the digest existed. The text itself shows it
    has not moved, so it is stamped rather than re-parsed, and no refresh is
    spent. Only ever read for a state with no digest key, so it cannot absorb
    the case below, where the text also matches.

    `unchanged` / `moved` -- the digest agrees, or it does not. A refresh
    recorded as still owed for this exact text reads as `moved`, because the
    goal was already stored by a turn whose generation yielded nothing: the text
    has not moved but the hypotheses it was supposed to buy are still missing.
    """

    metadata = _metadata(coscientist)
    if metadata is None:
        return "unverifiable"
    stored = str(metadata.get("research_goal") or "")
    digest = _digest(goal)
    if not stored:
        return "moved"
    if metadata.get("loam_refresh_owed") == digest:
        return "moved"
    if metadata.get("loam_goal_sha256") == digest:
        return "unchanged"
    if stored == goal and "loam_goal_sha256" not in metadata:
        return "unstamped"
    return "moved"


def _commit_goal(coscientist: Any, goal: str) -> None:
    """Record the digest of the goal as stored, so the next turn can compare.

    Deliberately separate from setting the goal. When a refresh is riding on the
    move, this is deferred until the generation has actually yielded something:
    committing first would mark the brief consumed even though it bought no
    hypotheses, and nothing would ever retry it.
    """

    metadata = _metadata(coscientist)
    if metadata is None:
        return
    metadata["loam_goal_sha256"] = _digest(goal)
    metadata.pop("loam_refresh_owed", None)
    coscientist.memory.save()


def _owe_refresh(coscientist: Any, goal: str) -> None:
    """Record that this goal's refresh has not been paid yet.

    The goal itself is stored by now, so its text no longer distinguishes a
    quiet turn from one whose generation came back empty. This does.
    """

    metadata = _metadata(coscientist)
    if metadata is None:
        return
    metadata["loam_refresh_owed"] = _digest(goal)
    coscientist.memory.save()


def _refresh_pool(
    coscientist: Any, config: Mapping[str, Any], goal: str, brief: str, existing: int
) -> dict[str, Any]:
    """Re-goal a seeded state when its goal has moved, and refresh the pool once.

    Returns the fields to merge into the turn's generation outcome. Raises
    whatever the two LLM calls raise; the caller treats that as advisory.
    """

    knowledge = _stored_knowledge(coscientist)
    stored_goal = _goal_with(goal, knowledge, _brief_block(brief, goal))
    moved = _goal_moved(coscientist, stored_goal)
    if moved == "unchanged":
        return {"goal_sync": "unchanged"}
    if moved == "unstamped":
        # The text already matches, so there is nothing to re-parse and nothing
        # new to generate against: stamp it and let the next turn compare.
        _remember_knowledge(coscientist, knowledge)
        _commit_goal(coscientist, stored_goal)
        return {"goal_sync": "unchanged_stamped"}

    # Recorded alongside the goal it is part of, so a state seeded before the
    # marker existed stops needing `_recover_knowledge` after this turn.
    _remember_knowledge(coscientist, knowledge)
    coscientist.set_research_goal(stored_goal)
    outcome: dict[str, Any] = {"goal_sync": "set" if moved == "moved" else "set_unverified"}

    refresh = _count(config.get("refresh_hypotheses"))
    if not refresh or moved != "moved":
        # Nothing rides on the move, so it is final as soon as it is stored.
        _commit_goal(coscientist, stored_goal)
        return outcome

    coscientist.generate_hypotheses(count=refresh)
    coscientist.wait_for_completion()
    entered = max(0, len(coscientist.get_all_hypotheses()) - existing)
    outcome["refresh_requested"] = refresh
    outcome["generated_hypotheses"] = entered
    if not entered:
        # Every generation task failed inside its agent: Jnana logs it, stores
        # nothing and raises nothing. Recording the debt rather than the digest
        # is what makes the next turn try this brief again instead of filing it
        # as spent on a pool it never reached.
        _owe_refresh(coscientist, stored_goal)
        outcome["status"] = "refresh_yielded_nothing"
        return outcome
    coscientist.memory.save()
    _commit_goal(coscientist, stored_goal)
    outcome["status"] = "refreshed"
    return outcome


def _ensure_initial_hypotheses(
    config: Mapping[str, Any], payload: Mapping[str, Any], llm_config: Any
) -> dict[str, int | str]:
    """Seed an empty Jnana state, or refresh a seeded one against new knowledge.

    Round one generates the pool. Later rounds generate `refresh_hypotheses`
    more, but only when the goal has actually moved -- a fresh reasoner brief or
    an edited campaign file -- so a quiet round costs nothing and a round of
    synthesis enters the pool as candidates that compete rather than as prose
    nobody reads. "Moved" is only measurable against a campaign goal, so a
    profile that names none keeps the old behaviour: seed once, then rank.
    """

    global CoScientist
    if CoScientist is None:
        from jnana.protognosis.core.coscientist import CoScientist as JnanaCoScientist

        CoScientist = JnanaCoScientist

    initial_hypotheses = _positive(config.get("initial_hypotheses"), 3)
    coscientist = CoScientist(
        llm_config=llm_config,
        storage_path=str(Path(str(config["state_file"])).expanduser()),
        # Either generation path runs through these workers, and a refresh is
        # usually the larger of the two, so it sizes the pool as well.
        max_workers=max(initial_hypotheses, _count(config.get("refresh_hypotheses"))),
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
            if not goal_source.startswith("campaign:"):
                # Only a campaign goal is stable enough to track. The turn's own
                # message -- the fallback for a profile that names no campaign --
                # differs every turn, so "moved" carries no signal there: syncing
                # would re-parse the plan, an LLM call, and refresh the pool on
                # every single turn. A seeded pool is left ranked, as before.
                outcome["goal_sync"] = "skipped_volatile_goal"
                return outcome
            # The pool is seeded, but the stored goal still steers every pairwise
            # match, so it is kept in step with the campaign's and the round's.
            # The pre-pass's brief is carried forward rather than re-researched
            # or dropped; see `_stored_knowledge`.
            #
            # Guarded because both halves below are LLM calls -- a plan re-parse
            # and a generation -- and they are the first ones this hook makes on
            # a turn that is not the seeding turn. `pre-llm.py` assembles every
            # tier under one `try`, so letting a provider hiccup out of here
            # would discard the campaign's orchestration context along with it.
            try:
                outcome.update(_refresh_pool(coscientist, config, goal, brief, existing))
            except Exception as exc:  # noqa: BLE001 - advisory: never block the turn
                outcome["goal_sync"] = f"failed: {type(exc).__name__}: {exc}"
            return outcome
        if not goal:
            return {"status": "skipped", "reason": "missing_research_goal", "generated_hypotheses": 0}
        requested = max(initial_hypotheses, 2 - existing)
        knowledge = _knowledge_brief(config, goal, llm_config)
        knowledge_block = str(knowledge.get("brief") or "")
        # Recorded, including as absent, so a later re-goal carries exactly what
        # the pre-pass produced for this pool rather than a stale brief.
        _remember_knowledge(coscientist, knowledge_block)
        stored_goal = _goal_with(goal, knowledge_block, _brief_block(brief, goal))
        # Set unconditionally, never gated on the digest. `CoScientist.research_goal`
        # is an instance attribute that starts at None and is restored from a
        # loaded state by nothing; `generate_hypotheses` passes it straight into
        # each generation prompt, while its own guard reads the *persisted*
        # `metadata["research_goal"]`. So a seed turn that reached this point
        # with a matching digest -- a prior turn that stored the goal but left
        # fewer than two hypotheses, which is what a failed or partial first
        # generation leaves behind -- would generate the whole pool against the
        # literal string "None" and pass every guard on the way.
        coscientist.set_research_goal(stored_goal)
        _commit_goal(coscientist, stored_goal)
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
