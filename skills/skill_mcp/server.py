"""loam-skills MCP server — one typed tool per skill action, over loam-hpc.

Each tool (`chai_fold`, `proteinmpnn_design`, …) takes a LIST of input rows,
validates them against the skill's `*.mcp.json` manifest, stages the skill's
run_row runner + site manifest + inputs.jsonl onto the cluster via the loam-hpc
MCP `write_files`, and submits one `run_batch` job per row. It runs no science
and holds no compute — loam-hpc is the executor (see docs/SKILL_MCP_DESIGN.md).

Config via env:
  LOAM_HPC_MCP_URL   loam-hpc MCP endpoint (default http://127.0.0.1:8000/mcp)
  LOAM_SKILLS_TRANSPORT  'stdio' (default) or 'http'

Run:  python -m skill_mcp.server     (from the `skills/` directory on PYTHONPATH)
"""
from __future__ import annotations

import os

from fastmcp import FastMCP

try:
    from skill_mcp import core
except ImportError:  # run directly from inside skills/skill_mcp
    import core  # type: ignore


def _make_tool(skill: str, action: str, manifest: dict):
    def tool(inputs: list | None = None, defaults: dict | None = None,
             timeout_each: float | None = None, gpus: float | None = None,
             labels: dict | None = None, from_registry: str | None = None,
             from_hypothesis: str | None = None, limit: int | None = None) -> dict:
        return core.submit_skill(skill, action, inputs, defaults=defaults,
                                 timeout_each=timeout_each, gpus=gpus, labels=labels,
                                 from_registry=from_registry, from_hypothesis=from_hypothesis,
                                 limit=limit)

    tool.__name__ = f"{skill}_{action}"
    schema_desc = ", ".join(f"{k}:{v}" for k, v in manifest.get("row_schema", {}).items())
    io = (manifest.get("io") or {}).get(action, {})
    tool.__doc__ = (
        f"Submit a {skill} {action} batch. Give `inputs` (a LIST of rows: "
        f"{{{schema_desc}}}), OR `from_registry=<type>` to auto-build rows from "
        f"registry artifacts of that type (this action consumes {io.get('consumes')}, "
        f"produces {io.get('produces')}); use `labels={{hypothesis_id,plan_id}}` to scope. "
        f"Rows run one-per-tile via run_batch (gpus={manifest.get('gpus', 1)}, "
        f"timeout_each={manifest.get('timeout_each', 900)}s). Returns batch_id; poll "
        f"loam-hpc batch_status, then call skill_harvest to register the outputs. "
        f"{manifest.get('description', '')}"
    ).strip()
    return tool


def _registry_tools():
    def skill_query(type: str, hypothesis_id: str, limit: int | None = None) -> dict:
        """List registry artifacts of a TYPE (sequence/backbone/structure/complex/
        score/trajectory/embedding) for a hypothesis — what is available to route
        into a downstream tool."""
        return core.registry_call("query", hypothesis_id=hypothesis_id, type=type,
                                  limit=limit, unused_only=False)

    def skill_harvest(batch_id: str, skill: str, action: str,
                      hypothesis_id: str | None = None) -> dict:
        """Register a COMPLETED batch's outputs as typed artifacts (produces+expects
        from the manifest). Call after batch_status shows complete, so later tools
        can route from these via from_registry."""
        return core.harvest_batch(skill, action, batch_id, hypothesis_id=hypothesis_id)

    return [skill_query, skill_harvest]


def create_server() -> FastMCP:
    server = FastMCP(
        name="loam-skills",
        instructions=(
            "One tool per skill action (chai_fold, proteinmpnn_design, …). Pass a LIST "
            "of input rows; the tool validates them against the skill manifest, stages "
            "the runner + inputs onto the cluster via the loam-hpc MCP, and submits a "
            "run_batch (one job per row, fanned across tiles). Poll loam-hpc batch_status "
            "with the returned batch_id. These tools do no science themselves — loam-hpc "
            "executes. Size a batch to the loam-hpc pool_status tile budget."
        ),
        on_duplicate="replace",
    )
    manifests = core.load_manifests()
    registered = []
    for name, m in sorted(manifests.items()):
        for action in m.get("actions", []):
            server.add_tool(_make_tool(name, action, m))
            registered.append(f"{name}_{action}")
    for t in _registry_tools():
        server.add_tool(t)
        registered.append(t.__name__)
    server._loam_registered = registered  # for the __main__ banner
    return server


if __name__ == "__main__":
    srv = create_server()
    print(f"[loam-skills] tools: {getattr(srv, '_loam_registered', [])}", flush=True)
    transport = os.environ.get("LOAM_SKILLS_TRANSPORT", "stdio")
    if transport in ("http", "sse"):
        srv.run(transport=transport,
                host=os.environ.get("LOAM_SKILLS_HOST", "127.0.0.1"),
                port=int(os.environ.get("LOAM_SKILLS_PORT", "8010")))
    else:
        srv.run(transport=transport)
