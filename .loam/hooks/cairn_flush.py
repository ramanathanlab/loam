"""At-most-once batched submission of queued Cairn rating events (standard library only).

Cairn's batch endpoint has no idempotency key, so an event that may have reached Cairn is never
sent twice automatically. An event's state is its last row in ``cairn-submissions.jsonl``; with
none it is pending. ``unsent`` (the request provably never left the host: DNS failure, refused
connection, unreachable network) is retried automatically on the next flush, never twice in one.
``failed`` (an HTTP error response; the batch is atomic so it was not applied) may be resent only
with ``--resubmit-failed``. ``in_flight`` and ``unknown`` are never resent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cairn_common  # noqa: E402
import cairn_keys  # noqa: E402
from cairn_client import submit_batch  # noqa: E402

BATCH_SIZE = 100
MAX_ROUNDS = 3
DEFAULT_IDENTITY_PREFIX = "agent://anl-loam/loam-"
STATUS = "cairn-status.json"
MINT_BACKOFF_SECONDS = 600
MAX_ENTITY_CHARS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_tag_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= 10
        and all(isinstance(v, str) and cairn_common.TAG_RE.fullmatch(v) for v in value)
    )


def _is_metric(name: Any, value: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(cairn_common.METRIC_KEY_RE.fullmatch(name))
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_event(event: Mapping[str, Any]) -> str:
    """Return "" when the event is a valid Cairn ScoreEventIn body, otherwise a reason."""
    if not isinstance(event, Mapping):
        return "event is not an object"
    reviewee = event.get("reviewee")
    if not isinstance(reviewee, Mapping) or reviewee.get("type") != "capability":
        return "reviewee.type must be capability"
    ident = reviewee.get("external_id")
    if not isinstance(ident, str) or not ident.startswith(("mcp://", "tool://")) or ident in ("mcp://", "tool://"):
        return "reviewee.external_id must be an mcp:// or tool:// id"
    if event.get("context") != "general":
        return "context must be general"
    score = event.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        return "score must be a number in [0, 1]"
    weight = event.get("weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 < weight <= 1:
        return "weight must be a number in (0, 1]"
    rationale = event.get("rationale")
    if not isinstance(rationale, str) or len(rationale) > 2000:
        return "rationale must be a string of at most 2000 characters"
    for key in ("task_tags", "failure_modes"):
        if not _is_tag_list(event.get(key)):
            return f"{key} must be a list of at most 10 snake_case tags"
    metrics = event.get("metrics")
    if metrics is not None:
        if not isinstance(metrics, Mapping) or len(metrics) > 20 or not all(_is_metric(k, v) for k, v in metrics.items()):
            return "metrics must map at most 20 snake_case names to finite numbers"
    if "task" in event and (not isinstance(event["task"], str) or len(event["task"]) > 500):
        return "task must be a string of at most 500 characters"
    if not isinstance(event.get("observed_at"), str):
        return "observed_at must be a string"
    return ""


def _states(state: Path) -> dict[str, str]:
    current: dict[str, str] = {}
    for row in cairn_common.read_jsonl(state / cairn_common.SUBMISSIONS):
        event_id, value = row.get("event_id"), row.get("state")
        if isinstance(event_id, str) and isinstance(value, str):
            current[event_id] = value
    return current


def _queued(state: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in cairn_common.read_jsonl(state / cairn_common.QUEUE):
        event_id = row.get("event_id")
        if row.get("status") == "queued" and isinstance(event_id, str) and event_id not in seen:
            seen.add(event_id)
            rows.append(row)
    return rows


def _counts(state: Path) -> dict[str, Any]:
    """Queued events by submission state, plus ``recorded``: rows never meant to be sent, by status."""
    states = _states(state)
    counts: dict[str, Any] = {}
    for row in _queued(state):
        name = states.get(row["event_id"], "pending")
        counts[name] = counts.get(name, 0) + 1
    recorded: dict[str, int] = {}
    seen: set[str] = set()
    for row in cairn_common.read_jsonl(state / cairn_common.QUEUE):
        event_id, status = row.get("event_id"), row.get("status")
        if not isinstance(event_id, str) or event_id in seen or not isinstance(status, str):
            continue
        seen.add(event_id)
        if status != "queued":
            recorded[status] = recorded.get(status, 0) + 1
    if recorded:
        counts["recorded"] = recorded
    unknown: dict[str, int] = {}
    seen = set()
    for row in cairn_common.read_jsonl(state / cairn_common.QUEUE):
        event_id, entity = row.get("event_id"), row.get("entity")
        if (
            row.get("status") == "invalid" and row.get("reason") == "unknown_entity"
            and isinstance(event_id, str) and event_id not in seen and isinstance(entity, str) and entity
        ):
            seen.add(event_id)
            unknown[entity] = unknown.get(entity, 0) + 1
    if unknown:
        counts["unknown_entities"] = unknown
    return counts


def _pending(state: Path, resubmit_failed: bool, attempted: set[str]) -> list[dict[str, Any]]:
    states = _states(state)
    eligible = {None, "unsent", "failed"} if resubmit_failed else {None, "unsent"}
    return [
        row
        for row in _queued(state)
        if states.get(row["event_id"]) in eligible and row["event_id"] not in attempted
    ]


def _transition(state: Path, event_id: str, name: str, error: str = "") -> None:
    cairn_common.append_jsonl(
        state / cairn_common.SUBMISSIONS,
        {"event_id": event_id, "state": name, "at": _now(), "error": error},
    )


def _classify(result: Any) -> tuple[str, str]:
    """Map a batch outcome to a state by HTTP status (the batch is atomic); no response splits on `sent`."""
    status = result.status
    if 200 <= status <= 299:
        return "submitted", result.error or ""  # applied even if the body did not parse
    if 400 <= status <= 499 or status in (500, 501):
        return "failed", f"HTTP {status}: {result.error}"
    if not status and getattr(result, "sent", True) is False:
        return "unsent", result.error or "not sent"
    return "unknown", f"HTTP {status}: {result.error}" if status else (result.error or "no response")


def _drain(state: Path, base_url: str, api_key: str, timeout: float, resubmit_failed: bool, attempted: set[str]) -> None:
    while True:
        batch = _pending(state, resubmit_failed, attempted)[:BATCH_SIZE]
        if not batch:
            return
        sendable: list[dict[str, Any]] = []
        for row in batch:
            attempted.add(row["event_id"])
            reason = validate_event(row.get("event") if isinstance(row.get("event"), Mapping) else {})
            if reason:
                _transition(state, row["event_id"], "invalid", reason)
            else:
                sendable.append(row)
        if not sendable:
            continue
        for row in sendable:
            _transition(state, row["event_id"], "in_flight")
        result = submit_batch(base_url, api_key, [row["event"] for row in sendable], timeout)
        name, error = _classify(result)
        for row in sendable:
            _transition(state, row["event_id"], name, error)


def _read_key(key_file: Path) -> str:
    try:
        return Path(key_file).expanduser().read_text().strip()
    except OSError:
        return ""


def _recent_mint_failure(state: Path) -> str | None:
    """The previous error when cairn-status.json carries a mint_failed_at less than the backoff ago, else None."""
    try:
        previous = json.loads((state / STATUS).read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(previous["mint_failed_at"])).total_seconds()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return str(previous.get("error") or "") if 0 <= age < MINT_BACKOFF_SECONDS else None


def _write_status(
    state: Path, base_url: str, key_file: Path, key: str, error: str, counts: dict[str, Any], mint_failed: bool = False
) -> None:
    """Atomically overwrite cairn-status.json (never the key). Failure is swallowed.

    ``mint_failed_at`` (and its error) is set on a failed mint and carried over from the previous
    status until a mint succeeds or a key is found, so the backoff survives non-minting flushes.
    """
    path = state / STATUS
    tmp = state / f".{STATUS}.{os.getpid()}.tmp"
    try:
        path_key = Path(key_file).expanduser()
        try:
            identity = Path(str(path_key) + ".identity").read_text().strip()
        except OSError:
            identity = ""
        body = {
            "updated_at": _now(),
            "key_file": str(path_key),
            "key": key,
            "identity": identity,
            "base_url": base_url,
            "counts": counts,
            "error": error,
        }
        if mint_failed:
            body["mint_failed_at"] = body["updated_at"]
        elif key not in ("minted", "present"):
            try:
                previous = json.loads((state / STATUS).read_text())
                failed_at = previous["mint_failed_at"]
                if isinstance(failed_at, str):
                    body["mint_failed_at"] = failed_at
                    body["error"] = error or str(previous.get("error") or "")
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
        state.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        try:
            tmp.unlink()
        except OSError:
            pass


def flush(
    state: Path,
    base_url: str,
    key_file: Path,
    timeout: float,
    resubmit_failed: bool = False,
    auto_mint: bool = False,
    identity: str = "",
    identity_prefix: str = DEFAULT_IDENTITY_PREFIX,
) -> dict[str, Any]:
    """Submit pending events at most once; return counts by current state (see ``_counts``).

    With ``auto_mint`` a missing or empty key is minted under the flush lock; on ``busy`` or
    ``failed`` the rows simply stay pending. Every call rewrites ``cairn-status.json``.
    """
    state = Path(state)
    attempted: set[str] = set()
    locked_out = False
    key_state, key_error = "", ""
    backed_off = mint_failed = False
    try:
        for _ in range(MAX_ROUNDS):
            with cairn_common.locked(state / cairn_common.FLUSH_LOCK, blocking=False) as acquired:
                if not acquired:
                    locked_out = True
                    break
                api_key = _read_key(key_file)
                if api_key:
                    key_state = key_state or "present"
                elif auto_mint and (backoff := _recent_mint_failure(state)) is not None:
                    # An earlier mint may have claimed an identity without a reply: do not litter Cairn.
                    key_state, key_error, backed_off = "mint_failed", backoff, True
                elif auto_mint:
                    minted = cairn_keys.ensure_key(
                        Path(key_file), base_url, identity or identity_prefix + secrets.token_hex(4), timeout
                    )
                    key_state = {"minted": "minted", "present": "present", "busy": "mint_busy"}.get(minted.state, "mint_failed")
                    key_error = minted.error
                    mint_failed = key_state == "mint_failed"
                    api_key = _read_key(key_file)
                else:
                    key_state = "missing"
                if not api_key:
                    break
                _drain(state, base_url, api_key, timeout, resubmit_failed, attempted)
            if not _pending(state, resubmit_failed, attempted):
                break
    except Exception as exc:  # noqa: BLE001 - fail open; the lock context manager has already released
        print(f"cairn_flush: {type(exc).__name__}: {exc}", file=sys.stderr)
    counts = _counts(state)
    if locked_out:
        counts["locked"] = 1
    if not key_state:
        key_state = "present" if _read_key(key_file) else "missing"
    if not locked_out and not backed_off:  # a flush that did not run must not overwrite the real one's status
        _write_status(state, base_url, key_file, key_state, key_error, counts, mint_failed)
    return counts


def spawn_flush(payload: Mapping[str, Any]) -> None:
    """Start a detached flusher; silent when Cairn or auto_flush is off. Never raises."""
    try:
        cfg = cairn_common.config(payload)
        state = cairn_common.state_dir(payload)
        if not cfg or not cfg.get("auto_flush") or state is None:
            return
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--state-dir", str(state),
            "--base-url", str(cfg.get("base_url", "")),
            "--key-file", str(Path(str(cfg.get("key_file", ""))).expanduser()),
            "--timeout", str(cfg.get("timeout_seconds", 5)),
        ]
        if cfg.get("auto_mint"):
            argv += ["--auto-mint", "--identity-prefix", str(cfg.get("identity_prefix") or DEFAULT_IDENTITY_PREFIX)]
            if cfg.get("identity"):
                argv += ["--identity", str(cfg["identity"])]
        subprocess.Popen(
            argv,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception:  # noqa: BLE001
        return


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Submit queued Cairn rating events at most once.")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--resubmit-failed", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--auto-mint", action="store_true", help="mint the key if it is missing (never with --report)")
    parser.add_argument("--identity", default="")
    parser.add_argument("--identity-prefix", default=DEFAULT_IDENTITY_PREFIX)
    args = parser.parse_args(argv)
    state = Path(args.state_dir).expanduser()
    if args.report:
        print(json.dumps(_counts(state), sort_keys=True))
        return 0
    flush(
        state, args.base_url, Path(args.key_file), args.timeout, args.resubmit_failed,
        auto_mint=args.auto_mint, identity=args.identity, identity_prefix=args.identity_prefix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
