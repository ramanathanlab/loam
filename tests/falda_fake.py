"""In-memory FALDA transport for tests: records requests, replays queued responses per route."""

from __future__ import annotations

import json
from typing import Any

POOL = "efb99ee4-c412-11f1-8af6-0affe8db6a05"
_DEFAULTS = {
    "/recall": {"recall_id": "r-0", "context": "", "items": [], "total_chars": 0},
    "/atoms/upsert": {"items": [{"id": "atom-0"}], "total_count": 1},
}


def falda_block(**overrides: Any) -> dict[str, Any]:
    """A `loam_falda` payload value as falda_config / payload_values produce it."""

    block = {
        "enabled": True,
        "base_url": "https://falda.example",
        "pool": POOL,
        "token_env": "FALDA_TOKEN",
        "key_dir": "~/.falda/keys",
        "recall_generation": True,
        "recall_reasoner": True,
        "write_validator_outcomes": True,
        "reasoner_tools": True,
        "auto_mint": True,
        "auto_flush": True,
        "recall_budget_chars": 1500,
        "recall_timeout_seconds": 2,
        "reasoner_recall_timeout_seconds": 5,
        "timeout_seconds": 60,
    }
    block.update(overrides)
    return block


class FakeTransport:
    """Callable transport. Queue (status, body) pairs per route; body may be a dict or raw bytes."""

    def __init__(self, recall=None, upsert=None, error: Exception | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.queued = {"/recall": list(recall or []), "/atoms/upsert": list(upsert or [])}
        self.error = error

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "body": json.loads(body), "timeout": timeout})
        if self.error is not None:
            raise self.error
        route = next((name for name in self.queued if url.endswith(name)), "")
        queue = self.queued.get(route, [])
        status, value = queue.pop(0) if queue else (200, _DEFAULTS.get(route, {}))
        return status, value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")

    def bodies(self, route: str) -> list[dict[str, Any]]:
        return [request["body"] for request in self.requests if request["url"].endswith(route)]
