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
    def tool(inputs: list, defaults: dict | None = None,
             timeout_each: float | None = None, gpus: float | None = None,
             labels: dict | None = None) -> dict:
        return core.submit_skill(skill, action, inputs, defaults=defaults,
                                 timeout_each=timeout_each, gpus=gpus, labels=labels)

    tool.__name__ = f"{skill}_{action}"
    schema_desc = ", ".join(f"{k}:{v}" for k, v in manifest.get("row_schema", {}).items())
    tool.__doc__ = (
        f"Submit a {skill} {action} batch. `inputs` is a LIST of rows; each row: "
        f"{{{schema_desc}}}. Rows are validated, staged via loam-hpc write_files, and "
        f"run one-per-tile via run_batch (gpus={manifest.get('gpus', 1)}, "
        f"timeout_each={manifest.get('timeout_each', 900)}s by default). "
        f"Returns the batch_id; poll loam-hpc batch_status. "
        f"{manifest.get('description', '')}"
    ).strip()
    return tool


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
