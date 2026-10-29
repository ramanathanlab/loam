"""FALDA write path: hooks append atoms to a local spool; a detached flusher sends them.

Standard library only. ``enqueue`` does no network I/O, so a validator stop never
waits on FALDA. ``spawn_flush`` starts a detached ``flush``, which takes a
non-blocking lock (a second flusher just exits), gets a key (minting it at most
once across processes, see ``falda_keys``), and sends pending atoms in batches,
stopping at the first failure. Unsent rows wait for the next flush; past
``MAX_PENDING`` the oldest are dropped and counted. Nothing here raises to a hook.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import falda_keys  # noqa: E402
import falda_memory  # noqa: E402

SPOOL = "falda-spool.jsonl"
SENT = "falda-sent.jsonl"
STATUS = "falda-status.json"
LOCK = "falda-flush.lock"
# FALDA takes ~1 s per atom per upsert (measured 2026-10-30): a batch must finish inside the write
# timeout, or FALDA stores it, the flusher times out, and the resend duplicates it.
BATCH_SIZE = 10
MAX_BATCHES = 10
# Under falda_keys.STALE_LOCK_SECONDS, so a slow mint is never mistaken for an abandoned one.
MINT_TIMEOUT_CAP = 15.0
MAX_PENDING = 5000
# Worth retrying later: no response, auth (a key may appear), timeout, rate limit, server trouble.
# Any other 4xx (or a 2xx carrying an error) is FALDA rejecting the batch itself; resending it would
# block every row behind it, so it is dropped and counted as rejected.
_RETRY_STATUSES = frozenset({0, 401, 403, 408, 429})


def _retryable(status: int) -> bool:
    return status in _RETRY_STATUSES or status >= 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _state_dir(payload: Mapping[str, Any]) -> Path:
    return falda_memory.calls_path(payload).parent


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def enqueue(payload: Mapping[str, Any], atom: Mapping[str, Any], *, plan_id: str, hypothesis_id: str) -> None:
    """Spool one atom for the flusher. No network; never raises."""

    try:
        _append(
            _state_dir(payload) / SPOOL,
            {
                "id": uuid.uuid4().hex,
                "atom": dict(atom),
                "plan_id": plan_id,
                "hypothesis_id": hypothesis_id,
                "workflow_id": _text(payload.get("workflow_id")),
                "run_id": _text(payload.get("loam_run_id")),
                "queued_at": _now(),
            },
        )
    except Exception:  # noqa: BLE001 - FALDA is best effort
        pass


def _pending(state: Path) -> list[dict[str, Any]]:
    done = {_text(row.get("id")) for row in _read(state / SENT)}
    return [
        row
        for row in _read(state / SPOOL)
        if _text(row.get("id")) and _text(row.get("id")) not in done and isinstance(row.get("atom"), dict)
    ]


def _report(state: Path) -> dict[str, Any]:
    sent_rows = _read(state / SENT)
    return {
        "pending": len(_pending(state)),
        "sent_total": sum(1 for row in sent_rows if not row.get("dropped") and not row.get("rejected")),
        "dropped_total": sum(1 for row in sent_rows if row.get("dropped")),
        "rejected_total": sum(1 for row in sent_rows if row.get("rejected")),
    }


def _write_status(state: Path, summary: Mapping[str, Any]) -> None:
    try:
        (state / STATUS).write_text(
            json.dumps({**summary, **_report(state), "updated_at": _now()}, sort_keys=True), encoding="utf-8"
        )
    except OSError:
        pass


def _send(state: Path, cfg: Mapping[str, Any], pending: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    for start in range(0, min(len(pending), BATCH_SIZE * MAX_BATCHES), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        first = batch[0]
        payload = {
            "loam_falda": cfg,
            "loam_state_dir": str(state),
            "workflow_id": _text(first.get("workflow_id")),
            "loam_run_id": _text(first.get("run_id")),
        }
        result = falda_memory.upsert(
            payload,
            [row["atom"] for row in batch],
            consumer="validator_outcome",
            plan_id=_text(first.get("plan_id")),
            hypothesis_id=_text(first.get("hypothesis_id")),
        )
        if not result.ok:
            summary["last_error"] = result.error
            if _retryable(result.status):
                return
            for row in batch:
                _append(state / SENT, {"id": row["id"], "rejected": result.error, "rejected_at": _now()})
            summary["rejected"] += len(batch)
            continue
        for row in batch:
            _append(state / SENT, {"id": row["id"], "sent_at": _now()})
        summary["sent"] += len(batch)


def flush(state_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Send pending atoms once; returns a summary (never the key)."""

    state = Path(state_dir)
    try:
        handle = open(state / LOCK, "a+")
    except OSError:
        return {"skipped": "locked"}
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"skipped": "locked"}
        summary: dict[str, Any] = {"sent": 0, "dropped": 0, "rejected": 0, "last_error": "", "key": ""}
        try:
            pending = _pending(state)
            if len(pending) > MAX_PENDING:
                overflow = pending[: len(pending) - MAX_PENDING]
                for row in overflow:
                    _append(state / SENT, {"id": row["id"], "dropped": True, "dropped_at": _now()})
                summary["dropped"] = len(overflow)
                pending = pending[len(overflow) :]
            if pending:
                timeout = min(float(cfg.get("timeout_seconds") or 60), MINT_TIMEOUT_CAP)
                key, summary["key"] = falda_keys.ensure_key(cfg, timeout)
                if key:
                    _send(state, cfg, pending, summary)  # upsert reads the key ensure_key just cached
                else:
                    summary["last_error"] = summary["key"]
        except Exception as exc:  # noqa: BLE001 - never raise from the flusher
            summary["last_error"] = f"client:{type(exc).__name__}"
        _write_status(state, summary)
        return {**summary, "pending": len(_pending(state))}


def spawn_flush(payload: Mapping[str, Any]) -> None:
    """Start a detached flusher; silent when FALDA or auto_flush is off. Never raises."""

    try:
        cfg = falda_memory.config(payload)
        if cfg is None or cfg.get("auto_flush") is not True:
            return
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--state-dir",
            str(_state_dir(payload)),
            "--config-json",
            json.dumps(dict(cfg)),
        ]
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
    parser = argparse.ArgumentParser(description="Send spooled Loam memories to FALDA (best effort).")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--config-json", required=True, help="the falda block as JSON (loam_falda)")
    parser.add_argument("--report", action="store_true", help="print counts only; send nothing")
    args = parser.parse_args(argv)
    try:
        cfg = json.loads(args.config_json)
    except ValueError:
        return 2
    state = Path(args.state_dir).expanduser()
    result = _report(state) if args.report else flush(state, cfg)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
