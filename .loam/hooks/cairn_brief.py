"""Read side of Cairn: a per-plan reputation brief appended to campaign delegations."""

from __future__ import annotations

import copy
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cairn_client
import cairn_common as common

SKILL_PATH = "skills/cairn-tool-review/SKILL.md"
PLACEMENT = {
    "code-generator": "Put ### Tool Ratings first in your final report.",
    "hypothesis-validator": (
        "In addition to the required headings, add ### Tool Ratings after ### Outcome and before ### Evidence Score."
    ),
    "reasoner": "Put ### Tool Ratings last in your synthesis.",
}
# Spec 7: the reputation brief goes only where tools are chosen and judged. The reasoner's synthesis
# becomes Jnana steering, so it never sees reputation (at most the rating instructions).
BRIEF_ROLES = frozenset({"code-generator", "hypothesis-validator"})
SILO_NOTE = (
    "These Cairn scores come from this experiment's private silo, not global Cairn: early in the run "
    "most tools will have few or no ratings. Rate honestly after each tool use."
)


def _warn(message: str) -> None:
    print(f"loam cairn brief: {message}", file=sys.stderr)


def _entity_ids(cfg: Mapping[str, Any]) -> list[str]:
    entities = cfg.get("entities")
    return [str(key) for key in entities] if isinstance(entities, Mapping) else []


def _timeout(cfg: Mapping[str, Any]) -> float:
    try:
        return min(10.0, max(0.1, float(cfg.get("timeout_seconds", 5.0))))
    except (TypeError, ValueError):
        return 5.0


def _fetch(cfg: Mapping[str, Any], ids: list[str]) -> tuple[dict[str, dict[str, float]] | None, str]:
    refs = [{"type": "capability", "external_id": ext} for ext in ids]
    project = str(cfg.get("project") or "")
    result = cairn_client.score_batch(str(cfg.get("base_url", "")), refs, _timeout(cfg), project=project)
    if not result.ok:
        return None, str(result.error or f"status {result.status}")
    if result.echo != "ok":  # without the echo these may be production scores, not the experiment's silo
        return None, f"X-Cairn-Project echo {result.echo or 'missing'}"
    data = result.data
    if not isinstance(data, list) or len(data) != len(ids):
        return None, "unexpected score batch response"
    scores: dict[str, dict[str, float]] = {}
    for ext, item in zip(ids, data):
        if not isinstance(item, Mapping):
            return None, "unexpected score batch response"
        try:
            scores[ext] = {
                "composite_score": float(item["composite_score"]),
                "confidence": float(item["confidence"]),
            }
        except (KeyError, TypeError, ValueError):
            return None, "unexpected score batch response"
    return scores, ""


def reputation(payload: Mapping[str, Any], plan_id: str) -> dict[str, dict[str, float]] | None:
    """Scores for every configured entity, fetched at most once per plan."""

    try:
        cfg = common.config(payload)
        ids = _entity_ids(cfg) if cfg else []
        if not cfg or not ids or not plan_id:
            return None
        directory = common.state_dir(payload)
        if directory is None:
            scores, _ = _fetch(cfg, ids)
            return scores
        path = directory / common.REPUTATION
        with common.locked(directory / common.REPUTATION_LOCK) as acquired:
            if acquired:
                for row in reversed(common.read_jsonl(path)):
                    if row.get("plan_id") == plan_id:
                        cached = row.get("scores")
                        if row.get("ok") is True and isinstance(cached, dict):
                            return cached
                        return None
            scores, error = _fetch(cfg, ids)
            if acquired:
                try:
                    common.append_jsonl(path, {
                        "plan_id": plan_id,
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "ok": scores is not None,
                        "scores": scores,
                        "error": error,
                    })
                except OSError as exc:
                    _warn(f"could not cache reputation: {exc}")
            return scores
    except Exception as exc:  # fail open: a Cairn problem never blocks delegation
        _warn(f"reputation unavailable: {exc}")
        return None


def _guidance(payload: Mapping[str, Any]) -> dict[str, str]:
    sections: dict[str, str] = {}
    try:
        text = (Path(str(payload.get("repo_root", ""))) / SKILL_PATH).read_text(encoding="utf-8")
    except OSError:
        return sections
    current = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(lines).strip()
            current, lines = line[3:].strip(), []
        elif current:
            lines.append(line)
    if current:
        sections[current] = "\n".join(lines).strip()
    return sections


SECTION_START = "## Cairn tool reputation (added by Loam)"
SECTION_END = "## End of Cairn tool reputation"


def appendix(payload: Mapping[str, Any], plan_id: str, role: str) -> str:
    cfg = common.config(payload)
    if not cfg:
        return ""
    ids = _entity_ids(cfg)
    guidance = _guidance(payload)
    parts: list[str] = []
    if cfg.get("read_side") is True and role in BRIEF_ROLES:
        scores = reputation(payload, plan_id)
        if scores:
            lines = [SILO_NOTE, "", "Tool reputation from Cairn (read before choosing a tool):"]
            for ext in ids:
                entry = scores.get(ext)
                if not entry or not entry.get("confidence"):
                    lines.append(f"- {ext}: no signal yet")
                else:
                    lines.append(
                        f"- {ext}: composite {entry['composite_score']:.2f}, confidence {entry['confidence']:.2f}"
                    )
            parts.append("\n".join(lines))
            if guidance.get("Reading reputation"):
                parts.append(guidance["Reading reputation"])
    roles = cfg.get("rate_roles")
    if cfg.get("rate_agent_reports") is True and isinstance(roles, list) and role in roles and ids:
        parts.append("Tool ids you may rate: " + ", ".join(ids))
        if guidance.get("Rating the tools you used"):
            parts.append(guidance["Rating the tools you used"])
        if role in PLACEMENT:
            parts.append(PLACEMENT[role])
    body = "\n\n".join(parts)
    if not body:
        return ""
    return SECTION_START + "\n\n" + body + "\n\n" + SECTION_END


def _decorate_task(task: Any, payload: Mapping[str, Any]) -> Any:
    if not isinstance(task, dict):
        return task
    goal, context = task.get("goal"), task.get("context")
    if not isinstance(goal, str) or not isinstance(context, str):
        return task
    role = common.marker(goal, "loam-agent-role")
    plan_id = common.marker(goal, "loam-plan-id")
    if not role or not plan_id:
        return task
    extra = appendix(payload, plan_id, role)
    if not extra:
        return task
    decorated = dict(task)
    decorated["context"] = context + "\n\n" + extra
    return decorated


def decorate_delegation(args: dict[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        if not common.config(payload):
            return args
        out = copy.deepcopy(args)
        if isinstance(out.get("tasks"), list):
            out["tasks"] = [_decorate_task(task, payload) for task in out["tasks"]]
            return out
        return _decorate_task(out, payload)
    except Exception as exc:
        _warn(f"delegation left undecorated: {exc}")
        return args
