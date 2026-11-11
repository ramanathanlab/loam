"""skill_mcp.core — the thin staging+submit layer behind every skill-MCP tool.

A skill-MCP tool (e.g. chai.fold) does exactly three things, all here:
  1. validate a list of input rows against the skill's manifest row_schema,
  2. stage the skill's run_row runner + the site manifest + inputs.jsonl onto the
     cluster filesystem via the loam-hpc MCP `write_files` (no clone lag — A2),
  3. call the loam-hpc MCP `run_batch` to fan one job per row across the tiles.

It runs NO science and owns NO compute: loam-hpc is the sole executor (its
"no domain logic" boundary). This module is pure stdlib so it can be unit-tested
and imported without FastMCP. See docs/SKILL_MCP_DESIGN.md.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

# skills/skill_mcp/core.py  ->  skills/
SKILLS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HPC_URL = os.environ.get("LOAM_HPC_MCP_URL", "http://127.0.0.1:8000/mcp")
SITE_MANIFEST = SKILLS_ROOT / "site.aurora.json"

_TYPES = {"str": str, "int": int, "float": (int, float), "bool": bool,
          "list": list, "dict": dict, "number": (int, float), "any": object}

# Reserved row keys the validator permits and the runner ignores: identity + the
# provenance fields the artifact registry reads from the staged input.json
# (docs/SKILL_MCP_DESIGN.md §11). They are not tool parameters.
_RESERVED = {"id", "parent", "parents", "round"}


# --------------------------------------------------------------------------- #
# loam-hpc MCP client (streamable-HTTP, stdlib only)
# --------------------------------------------------------------------------- #
class LoamHpcClient:
    """Minimal MCP client for the loam-hpc server: initialize then tools/call."""

    def __init__(self, url: str = DEFAULT_HPC_URL, timeout: float = 1800):
        self.url = url
        self.timeout = timeout
        self.sid: str | None = None

    def _post(self, body: dict):
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if self.sid:
            headers["mcp-session-id"] = self.sid
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=self.timeout)
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self.sid = sid
        raw = resp.read().decode()
        data = None
        for line in raw.splitlines():           # SSE frames: last data: line wins
            if line.startswith("data: "):
                data = line[6:]
        if data:
            return json.loads(data)
        if not raw.strip():                      # 202 Accepted (a notification)
            return None
        return json.loads(raw)

    def connect(self) -> "LoamHpcClient":
        self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "loam-skills", "version": "0"}}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def call(self, name: str, args: dict, rid: int = 2):
        out = self._post({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})
        if not out or "result" not in out:
            raise RuntimeError(f"{name}: bad MCP response {json.dumps(out)[:400]}")
        result = out["result"]
        content = result.get("content") or []
        text = content[0].get("text", "") if content else ""
        if result.get("isError"):
            raise RuntimeError(f"{name} failed: {text[:600]}")
        try:
            return json.loads(text)
        except Exception:
            # loam-hpc returns some failures as a plain "<tool> failed: ..." string
            if " failed" in text.lower():
                raise RuntimeError(text[:600])
            return text


# --------------------------------------------------------------------------- #
# manifests + validation
# --------------------------------------------------------------------------- #
def load_manifests() -> dict:
    """Every skills/**/*.mcp.json, keyed by manifest name, with its dir recorded."""
    manifests: dict[str, dict] = {}
    for path in SKILLS_ROOT.glob("**/*.mcp.json"):
        try:
            m = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(m, dict) or "name" not in m:
            continue
        m["_dir"] = str(path.parent)
        manifests[m["name"]] = m
    return manifests


def validate_rows(manifest: dict, inputs: list, defaults: dict | None = None) -> list[dict]:
    """Fill defaults, reject unknown keys, enforce required keys and types."""
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must be a non-empty list of row objects")
    schema = manifest.get("row_schema", {})
    base = dict(manifest.get("defaults") or {})
    if defaults:
        base.update(defaults)
    required = set(manifest.get("required", []))
    rows: list[dict] = []
    for i, raw in enumerate(inputs):
        if not isinstance(raw, dict):
            raise ValueError(f"row {i} is not an object")
        row = dict(base)
        row.update(raw)
        unknown = [k for k in row if k not in schema and k not in _RESERVED]
        if unknown:
            raise ValueError(f"row {i}: unknown keys {unknown}; allowed: {sorted(schema)}")
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"row {i}: missing required {missing}")
        for k, v in row.items():
            if k in _RESERVED or v is None:
                continue
            kind = schema.get(k)
            if kind and not isinstance(v, _TYPES.get(kind, object)):
                raise ValueError(f"row {i}: {k!r} must be {kind}, got {type(v).__name__}")
        row.setdefault("id", f"{manifest['name']}-{i}")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
def skill_run(op: str, skill: str | None = None, action: str | None = None,
              inputs: list | None = None, from_registry: str | None = None,
              from_hypothesis: str | None = None, defaults: dict | None = None,
              labels: dict | None = None, batch_id: str | None = None,
              type: str | None = None, hypothesis_id: str | None = None,
              limit: int | None = None, hpc_url: str = DEFAULT_HPC_URL) -> dict:
    """ONE unified entry for a loam code-generator (loam gives each child a single
    execution tool). `op` selects the funnel step:
      submit  -> run a skill batch (skill, action; inputs OR from_registry+from_hypothesis)
      query   -> list registry artifacts (type, hypothesis_id)
      harvest -> register a completed batch's outputs (skill, action, batch_id, hypothesis_id)
      status  -> poll a batch (batch_id)
      pool    -> the tile budget
    """
    if op == "submit":
        return submit_skill(skill, action, inputs, defaults=defaults, labels=labels,
                            from_registry=from_registry, from_hypothesis=from_hypothesis,
                            limit=limit, hpc_url=hpc_url)
    if op == "query":
        return registry_call("query", hpc_url=hpc_url, hypothesis_id=hypothesis_id,
                            type=type, limit=limit, unused_only=False)
    if op == "harvest":
        return harvest_batch(skill, action, batch_id, hypothesis_id=hypothesis_id, hpc_url=hpc_url)
    if op in ("status", "pool"):
        cli = LoamHpcClient(hpc_url).connect()
        return cli.call("batch_status", {"batch_id": batch_id}) if op == "status" \
            else cli.call("pool_status", {})
    raise ValueError(f"unknown op {op!r}; use submit|query|harvest|status|pool")


def registry_call(op: str, hpc_url: str = DEFAULT_HPC_URL, **args):
    """Run a registry op WORKER-SIDE: ship registry.py, then run_code it on loam-hpc.

    registry ops take base_dir as their first arg; we resolve it from the worker's
    HPC_BASE_DIR so the laptop never hardcodes a /flare path.
    """
    cli = LoamHpcClient(hpc_url).connect()
    reg_src = (Path(__file__).parent / "registry.py").read_text()
    cli.call("write_files", {"files": {"skill_mcp/registry.py": reg_src}})
    payload = (
        "import os, sys, json\n"
        "BASE = os.environ.get('HPC_BASE_DIR') or os.getcwd()\n"
        "sys.path.insert(0, os.path.join(BASE, 'skill_mcp'))\n"
        "import registry\n"
        f"args = json.loads({json.dumps(json.dumps(args))})\n"
        f"print(json.dumps(getattr(registry, {op!r})(BASE, **args)))\n"
    )
    out = cli.call("run_code", {"payload": payload, "timeout": 300})
    txt = out.get("stdout", "") if isinstance(out, dict) else str(out)
    for line in reversed(txt.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"registry.{op} returned no JSON: {txt[-400:]}")


def harvest_batch(skill: str, action: str, batch_id: str,
                  hypothesis_id: str | None = None, hpc_url: str = DEFAULT_HPC_URL) -> dict:
    """Register a completed batch's outputs as typed artifacts (produces+expects
    from the manifest, so the agent only passes batch_id + skill + action)."""
    manifests = load_manifests()
    m = manifests.get(skill) or {}
    io = (m.get("io") or {}).get(action) or {}
    produces = io.get("produces")
    if not produces:
        raise ValueError(f"{skill}.{action} declares no `produces` type to harvest")
    produces = produces[0] if isinstance(produces, list) else produces
    return registry_call("harvest", hpc_url=hpc_url, batch_id=batch_id, tool=skill,
                         action=action, produces=produces, expects=m.get("expects") or [],
                         hypothesis_id=hypothesis_id)


def _rows_from_registry(m: dict, action: str, from_type: str, hypothesis_id: str,
                        defaults: dict | None, limit: int | None, hpc_url: str) -> list[dict]:
    """Build consumer rows by routing registry artifacts of `from_type`."""
    binding = (m.get("input_binding") or {}).get(action)
    if not binding:
        raise ValueError(f"{m['name']}.{action} has no input_binding; cannot use from_registry")
    consumes = (m.get("io") or {}).get(action, {}).get("consumes")
    allowed = consumes if isinstance(consumes, list) else ([consumes] if consumes else [])
    if allowed and from_type not in allowed:
        raise ValueError(f"{m['name']}.{action} consumes {allowed}, not {from_type!r}")
    r = registry_call("route", hpc_url=hpc_url, hypothesis_id=hypothesis_id,
                      type=from_type, want=binding["format"], limit=limit)
    base_row = dict(defaults or {})
    rows = []
    for i, mat in enumerate(r.get("materialized", [])):
        row = dict(base_row)
        row[binding["field"]] = mat.get("value") if "value" in mat else mat.get("path")
        row["parent"] = mat["artifact_id"]
        row["id"] = f"{m['name']}-{action}-{i}"
        rows.append(row)
    if not rows:
        raise ValueError(f"no {from_type!r} artifacts in hypothesis {hypothesis_id!r} "
                         f"(seeds={r.get('n_seeds')})")
    return rows


def submit_skill(skill: str, action: str, inputs: list | None = None,
                 defaults: dict | None = None, timeout_each: float | None = None,
                 gpus: float | None = None, labels: dict | None = None,
                 from_registry: str | None = None, from_hypothesis: str | None = None,
                 limit: int | None = None, hpc_url: str = DEFAULT_HPC_URL) -> dict:
    """Validate rows, stage them + the runner via loam-hpc write_files, run_batch.

    With `from_registry=<type>`, build the rows by routing registry artifacts of
    that type (materialized into this tool's input format) instead of (or on top
    of) explicit `inputs` — this is how a tool self-wires in a campaign.
    """
    manifests = load_manifests()
    if skill not in manifests:
        raise ValueError(f"unknown skill {skill!r}; have {sorted(manifests)}")
    m = manifests[skill]
    if action not in m.get("actions", []):
        raise ValueError(f"skill {skill!r} has no action {action!r}; has {m.get('actions')}")

    if from_registry:
        hyp = from_hypothesis or (labels or {}).get("hypothesis_id") or "default"
        routed = _rows_from_registry(m, action, from_registry, hyp, defaults, limit, hpc_url)
        inputs = (inputs or []) + routed
        defaults = None  # already folded into the routed rows
    if not inputs:
        raise ValueError("no inputs (pass `inputs` or `from_registry`)")

    rows = validate_rows(m, inputs, defaults)

    runner_rel = m["runner"]
    runner_src = Path(runner_rel) if os.path.isabs(runner_rel) else Path(m["_dir"], runner_rel)
    runner_text = runner_src.read_text()
    runner_name = Path(runner_rel).name
    site_text = SITE_MANIFEST.read_text() if SITE_MANIFEST.exists() else "{}"
    jsonl = "\n".join(json.dumps(r) for r in rows) + "\n"

    slug = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    base_rel = f"skill_mcp/{skill}/{slug}"
    files = {
        f"{base_rel}/{runner_name}": runner_text,
        f"{base_rel}/site.aurora.json": site_text,   # beside the runner, so it resolves
        f"{base_rel}/inputs.jsonl": jsonl,
    }

    cli = LoamHpcClient(hpc_url).connect()
    wf = cli.call("write_files", {"files": files})
    written = {Path(w["path"]).name: w["path"] for w in wf.get("written", [])}
    if runner_name not in written or "inputs.jsonl" not in written:
        raise RuntimeError(f"staging incomplete: {wf}")

    run_args = {
        "script_path": written[runner_name],
        "inputs_path": written["inputs.jsonl"],
        "gpus": gpus if gpus is not None else m.get("gpus", 1),
        "timeout_each": timeout_each if timeout_each is not None else m.get("timeout_each", 900),
    }
    if labels:
        run_args["labels"] = labels
    rb = cli.call("run_batch", run_args, rid=3)
    return {
        "skill": skill, "action": action, "n_rows": len(rows),
        "staged_dir": base_rel, "run_batch": rb,
    }
