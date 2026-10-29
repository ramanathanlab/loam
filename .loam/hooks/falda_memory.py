"""Hosted FALDA client for Loam campaign memory.

Standard library only: hooks may run under Hermes's managed interpreter. Every
call is logged to ``falda-calls.jsonl`` in the run's state directory, which is
the evaluation record of what each round was given. Nothing here raises to its
caller: FALDA is advisory and must never stop a round.

The pool travels in the request body and in ``X-Falda-Pool``; the key comes
from ``falda_keys.resolve_key`` (env, else the cached per-pool key file) and is
never minted here. Recalls sit on the agent's path and are capped in wall-clock
time (``recall_timeout_seconds``, default 2 s); writes go through the spool and
detached flusher in ``falda_flush``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import falda_keys  # noqa: E402
import loam_http  # noqa: E402

CALLS_FILENAME = "falda-calls.jsonl"
# FALDA rejects require_tags entries longer than this.
MAX_TAG_CHARS = 128
# The environment-override profile's placeholder id is shared by every such run: never a campaign tag.
_GENERIC_WORKFLOW_IDS = {"", "environment-workflow"}
Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


@dataclass(frozen=True)
class RecallResult:
    ok: bool
    error: str = ""
    context: str = ""
    recall_id: str = ""
    item_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpsertResult:
    ok: bool
    error: str = ""
    atom_ids: tuple[str, ...] = ()
    status: int = 0  # HTTP status; 0 when no response (network, missing key, client error)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with loam_http.urlopen(request, timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


_transport: Transport = _urllib_transport


def set_transport(transport: Transport | None) -> None:
    """Replace the HTTP transport (tests); None restores urllib."""

    global _transport
    _transport = transport or _urllib_transport


def config(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the enabled FALDA block passed by the adapter, else None."""

    value = payload.get("loam_falda")
    if isinstance(value, Mapping) and value.get("enabled") is True and _text(value.get("pool")):
        return value
    return None


def workflow_tag(payload: Mapping[str, Any]) -> str:
    """The one tag that scopes a campaign's memories; every write, recall and guard uses it.

    The run's workflow id, unless it is missing or the generic environment placeholder; then the
    experiment's CAIRN_PROJECT; then the UTC date, so a hands-free run is still scoped.
    """

    workflow_id = _text(payload.get("workflow_id"))
    if workflow_id in _GENERIC_WORKFLOW_IDS:
        workflow_id = os.environ.get("CAIRN_PROJECT", "").strip()
    if not workflow_id:
        workflow_id = "loam-" + datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"workflow:{workflow_id}"[:MAX_TAG_CHARS]


def calls_path(payload: Mapping[str, Any]) -> Path:
    state_dir = _text(payload.get("loam_state_dir")) or os.environ.get("LOAM_STATE_DIR", "").strip()
    root = Path(state_dir).expanduser() if state_dir else Path(_text(payload.get("repo_root")) or ".") / ".loam" / "state"
    return root / CALLS_FILENAME


def log_event(payload: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    """Append one call-log row. A logging failure is swallowed: the log is evidence, not control."""

    try:
        path = calls_path(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "observed_at": _timestamp(),
            "workflow_id": _text(payload.get("workflow_id")),
            "run_id": _text(payload.get("loam_run_id")),
            **row,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


def _post(
    cfg: Mapping[str, Any], route: str, body: Mapping[str, Any], timeout: float | None = None, key: str = ""
) -> tuple[bool, str, Mapping[str, Any], int]:
    token = key or falda_keys.resolve_key(cfg)
    if not token:
        return False, "missing_token", {}, 0
    pool = _text(cfg.get("pool"))
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Falda-Pool": pool,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = json.dumps({**body, "pool": pool}, ensure_ascii=False).encode("utf-8")
    url = _text(cfg.get("base_url")).rstrip("/") + route
    try:
        status, raw = _transport(url, headers, data, float(timeout or cfg.get("timeout_seconds") or 10))
    except Exception as exc:  # DNS, TLS, refused, timeout. Log the type only: messages may echo request data.
        return False, f"network:{type(exc).__name__}", {}, 0
    try:
        value = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, ValueError):
        value = {}
    if not isinstance(value, Mapping):
        value = {}
    if status >= 400:
        return False, _text(value.get("error")) or f"http_{status}", value, status
    if _text(value.get("error")):
        return False, _text(value.get("error")), value, status
    return True, "", value, status


def _ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_text(item.get("id")) for item in value if isinstance(item, Mapping) and _text(item.get("id")))


def _recall_timeout(cfg: Mapping[str, Any]) -> float:
    try:
        value = float(cfg.get("recall_timeout_seconds") or 2)
    except (TypeError, ValueError):
        value = 2.0
    return value if value > 0 else 2.0


def recall(payload: Mapping[str, Any], query: str, *, consumer: str, budget: int, plan_id: str = "") -> RecallResult:
    """Recall this campaign's memories, filtered to its workflow tag."""

    cfg = config(payload)
    if cfg is None:
        return RecallResult(ok=False, error="disabled")
    tags = [workflow_tag(payload)]
    cap = _recall_timeout(cfg)

    def _call() -> RecallResult:
        ok, error, body, _ = _post(
            cfg, "/recall", {"query": query, "require_tags": tags, "budget": int(budget), "mode": "explicit"}, cap
        )
        return RecallResult(
            ok=ok,
            error=error,
            context=_text(body.get("context")) if ok else "",
            recall_id=_text(body.get("recall_id")),
            item_ids=_ids(body.get("items")) if ok else (),
        )

    try:
        # The socket timeout bounds each read; this bounds the whole recall (a slow drip, DNS stall).
        result = loam_http.bounded(_call, cap, RecallResult(ok=False, error="timeout"))
    except Exception as exc:  # defensive: the client never raises
        result = RecallResult(ok=False, error=f"client:{type(exc).__name__}")
    log_event(
        payload,
        {
            "op": "recall",
            "consumer": consumer,
            "plan_id": plan_id,
            "ok": result.ok,
            "error": result.error,
            "recall_id": result.recall_id,
            "item_ids": list(result.item_ids),
            "chars": len(result.context),
            "require_tags": tags,
        },
    )
    return result


def upsert(
    payload: Mapping[str, Any],
    atoms: list[dict[str, Any]],
    *,
    consumer: str,
    plan_id: str = "",
    hypothesis_id: str = "",
) -> UpsertResult:
    """Write atoms to the campaign pool exactly as given (direct writes keep their tags)."""

    cfg = config(payload)
    if cfg is None:
        return UpsertResult(ok=False, error="disabled")
    try:
        ok, error, body, status = _post(cfg, "/atoms/upsert", {"atoms": atoms})
        result = UpsertResult(ok=ok, error=error, atom_ids=_ids(body.get("items")) if ok else (), status=status)
    except Exception as exc:  # defensive: the client never raises
        result = UpsertResult(ok=False, error=f"client:{type(exc).__name__}")
    log_event(
        payload,
        {
            "op": "upsert",
            "consumer": consumer,
            "plan_id": plan_id,
            "hypothesis_id": hypothesis_id,
            "ok": result.ok,
            "error": result.error,
            "atom_ids": list(result.atom_ids),
            "tags": [atom.get("tags") for atom in atoms if isinstance(atom, Mapping)],
        },
    )
    return result


def select_sections(context: str, titles: tuple[str, ...], limit: int) -> str:
    """Keep the named ``## `` sections of a recall context, in ``titles`` order, cut to ``limit`` characters."""

    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in context.splitlines():
        if line.startswith("## "):
            current = found.setdefault(line[3:].strip(), [])
        elif current is not None and line.strip():
            current.append(line)
    text = "\n\n".join("\n".join([f"## {title}", *found[title]]) for title in titles if found.get(title))
    return cut_to_lines(text, limit)


def cut_to_lines(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters on a line boundary, never ending on a bare ``## `` heading."""

    if len(text) > limit:
        text = text[:limit]
        text = text.rsplit("\n", 1)[0] if "\n" in text else ""
        # A cut must not leave a heading with nothing under it.
        while text and text.rstrip().rsplit("\n", 1)[-1].startswith("## "):
            text = text.rstrip().rsplit("\n", 1)[0] if "\n" in text.rstrip() else ""
    return text.rstrip()
