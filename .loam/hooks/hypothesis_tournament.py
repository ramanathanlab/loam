"""Seed and rank a bounded Jnana hypothesis tournament before an LLM call.

This is advisory only. It uses Jnana's internal generation agents to seed an
empty workflow-scoped state, then runs pairwise rankings under a non-blocking
state lock. It returns compact rankings rather than full hypothesis bodies.
Any unavailable configuration, lock contention, or Jnana failure simply
produces no injected context.
"""

from __future__ import annotations

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


def _base_url(config: Mapping[str, Any]) -> str:
    """Convert an OpenAI-compatible inference URL to the client's base URL."""

    endpoint = str(config["inference_endpoint"]).strip().rstrip("/")
    if config.get("endpoint_is_chat_completions") is True:
        suffix = "/chat/completions"
        if not endpoint.endswith(suffix):
            raise ValueError("endpoint_is_chat_completions requires a URL ending in /chat/completions")
        return endpoint[: -len(suffix)]
    return endpoint


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


def _generation_goal(payload: Mapping[str, Any]) -> str:
    """Return the current user request as Jnana's bounded generation goal."""

    return " ".join(str(payload.get("user_message") or "").split())


def _ensure_initial_hypotheses(
    config: Mapping[str, Any], payload: Mapping[str, Any], llm_config: Any
) -> dict[str, int | str]:
    """Populate an empty Jnana state with bounded internal generation agents."""

    global CoScientist
    goal = _generation_goal(payload)
    if not goal:
        return {"status": "skipped", "reason": "missing_research_goal", "generated_hypotheses": 0}
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
        if existing >= 2:
            return {"status": "not_needed", "generated_hypotheses": 0}
        requested = max(initial_hypotheses, 2 - existing)
        coscientist.set_research_goal(goal)
        coscientist.generate_hypotheses(count=requested)
        coscientist.wait_for_completion()
        coscientist.memory.save()
        return {
            "status": "completed",
            "generated_hypotheses": max(0, len(coscientist.get_all_hypotheses()) - existing),
        }
    finally:
        coscientist.stop()


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
    api_key_env = str(config.get("api_key_env") or "").strip()
    api_key = os.environ.get(api_key_env, "").strip() if api_key_env else "local-no-auth"

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
        from jnana.protognosis.core.multi_llm_config import LLMConfig
        from jnana.protognosis.periodic_tournament import run_incremental_tournament

        llm_config = LLMConfig(
            provider=str(config["provider"]),
            model=str(config["model"]),
            api_key=api_key,
            base_url=_base_url(config),
            model_adapter={
                "omit_temperature": config.get("omit_temperature") is True,
                "json_max_tokens": _positive(config.get("json_max_tokens"), 1024),
            },
        )
        generation_llm_config = LLMConfig(
            provider=str(config["provider"]),
            model=str(config["model"]),
            api_key=api_key,
            base_url=_base_url(config),
            model_adapter={
                "omit_temperature": config.get("omit_temperature") is True,
                # Generation agents emit a full mechanism plus predictions; give
                # them more room than compact pairwise ranking responses.
                "json_max_tokens": max(4096, _positive(config.get("json_max_tokens"), 1024)),
            },
        )
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
